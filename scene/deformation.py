import torch
import torch.nn as nn
import torch.nn.init as init
from scene.hexplane import HexPlaneField
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        self.args = args

        self.create_net()
    def set_aabb(self, xyz_max, xyz_min):
        self.grid.set_aabb(xyz_max, xyz_min)

    def create_net(self):
        grid_out_dim = self.grid.feat_dim
        self.feature_out = [nn.Linear(grid_out_dim ,self.W)]

        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        mask_deform = []
        test_len = 2
        for i in range(test_len):
            mask_deform.append(nn.ReLU())
            mask_deform.append(nn.Linear(self.W,self.W))
        mask_deform.append(nn.ReLU())
        mask_deform.append(nn.Linear(self.W, 1))
        self.mask_deform = nn.Sequential(*mask_deform)


        self.feature_out = nn.Sequential(*self.feature_out)
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))

        # rotations_deform: always allocated for consistent param count / optimizer
        # layout / init RNG order (do not remove). Output added to rotations only when
        # args.enable_dr_deform is True (additive form).
        self.rotations_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 4))
        # Optional scale / opacity deformation heads (off in the default recipe).
        # Always allocated for consistent param count; outputs used only when
        # args.enable_ds_deform / args.enable_do_deform are True.
        self.scales_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 3))
        self.opacity_deform = nn.Sequential(nn.ReLU(), nn.Linear(self.W, self.W), nn.ReLU(), nn.Linear(self.W, 1))

    def query_time(self, rays_pts_emb, time_emb):

        grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
        hidden = grid_feature

        hidden = self.feature_out(hidden)


        return hidden
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None,shs_emb=None, time_emb=None):
        return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, shs_emb, time_emb)

    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, shs_emb, time_emb):
        hidden = self.query_time(rays_pts_emb, time_emb)
        dx = self.pos_deform(hidden)
        # dx_deform_divisor: 10 damps the position offset; 1 = full magnitude
        _dx_div = getattr(self.args, 'dx_deform_divisor', 10)
        pts = rays_pts_emb[:,:3] + dx / _dx_div

        if getattr(self.args, 'enable_ds_deform', False):
            ds = self.scales_deform(hidden)
            scales = scales_emb[:,:3] + ds   # optional scale deformation
        else:
            scales = scales_emb[:,:3]
        if getattr(self.args, 'enable_dr_deform', False):
            dr = self.rotations_deform(hidden)
            rotations = rotations_emb[:,:4] + dr   # additive rotation deformation
        else:
            rotations = rotations_emb[:,:4]
        if getattr(self.args, 'enable_do_deform', False):
            do = self.opacity_deform(hidden)
            opacity = opacity_emb[:,:1] + do   # optional opacity deformation
        else:
            opacity = opacity_emb[:,:1]

        ## new mask
        shs_temp = shs_emb
        mask_deform = self.mask_deform(hidden)
        shs_temp[:,-1, 2] +=  mask_deform[:,-1]
        shs = shs_temp

        return pts, scales, rotations, opacity, shs
    def get_mlp_parameters_cano(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name and 'mask_deform' not in name:
                parameter_list.append(param)
        return parameter_list
    def get_mlp_parameters_others(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if 'mask_deform' not in name:
                continue
            if "grid" not in name:
                parameter_list.append(param)
        return parameter_list

    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list

class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        times_ch = 2*timebase_pe+1
        # timenet is vestigial in the current forward path but kept: it is part of the
        # optimizer/checkpoint layout and its init draws precede deformation_net's.
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)

    def forward(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        return self.forward_dynamic(point, scales, rotations, opacity, shs, times_sel)

    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, shs=None, times_sel=None):
        point_emb = poc_fre(point,self.pos_poc)
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
        means3D, scales, rotations, opacity, shs = self.deformation_net( point_emb,
                                                  scales_emb,
                                                rotations_emb,
                                                opacity,
                                                shs,
                                                times_sel)
        return means3D, scales, rotations, opacity, shs
    def get_mlp_parameters_cano(self):
        return self.deformation_net.get_mlp_parameters_cano() + list(self.timenet.parameters())
    def get_mlp_parameters_others(self):
        return self.deformation_net.get_mlp_parameters_others()
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()



def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb
