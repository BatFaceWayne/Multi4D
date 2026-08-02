#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
"""
This module defines the GaussianModel and GaussianModel_dynamic classes, which are
central to the 3D Gaussian Splatting representation. These classes manage the
attributes of the Gaussians (e.g., position, color, scale, rotation) and provide
methods for their creation, optimization, and adaptive densification.
"""

import torch
import numpy as np
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from scene.deformation import deform_network
from scene.regulation import compute_plane_smoothness

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_rotation_4d


class GaussianBranchBase:
    """Shared surface of the three Gaussian branches (FG dynamic / BG static /
    TR transient): activation wiring + basic parameter getters. Bodies are the
    identical copies extracted from the three classes (2026-07-29); the genuinely
    divergent machinery (optimizer surgery, densify/prune internals, 4D fields,
    ply IO) deliberately stays per-class."""

    def setup_functions(self):
        """
        Set up the activation functions for Gaussian attributes.

        This function defines the transformations applied to the raw learnable
        parameters to obtain the final Gaussian attributes (e.g., scaling,
        opacity). Using activations like `exp` for scaling ensures positivity.
        """

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log


        self.opacity_activation = torch.sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self):
        """
        Returns the scaling parameters of the Gaussians.

        Returns:
            torch.Tensor: The scaling parameters of the Gaussians.
        """
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        """
        Returns the rotation parameters of the Gaussians.

        Returns:
            torch.Tensor: The rotation parameters of the Gaussians.
        """
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        """
        Returns the positions of the Gaussians.

        Returns:
            torch.Tensor: The positions of the Gaussians.
        """
        return self._xyz

    @property
    def get_features(self):
        """
        Concatenates the DC and rest features of the Gaussians.

        Returns:
            torch.Tensor: The concatenated features.
        """
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        """
        Returns the opacity parameters of the Gaussians.

        Returns:
            torch.Tensor: The opacity parameters of the Gaussians.
        """
        return self.opacity_activation(self._opacity)


class PersistentBranchMixin:
    """Methods shared verbatim by the two persistent branches (FG dynamic + BG
    static) but not the transient branch (which has its own density control)."""

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    def densify(self, max_grad, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

    def prune(self, min_opacity):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


class GaussianModel_dynamic(PersistentBranchMixin, GaussianBranchBase):
    """
    Represents a dynamic set of 3D Gaussians.

    This class manages the core attributes of dynamic Gaussians, including a
    deformation network (`deform_network`) to model motion over time. It handles
    the creation from point clouds, optimization setup, and the adaptive control
    mechanisms like densification and pruning which are essential for capturing
    complex dynamic scenes.
    """


    def __init__(self, sh_degree: int, args):
        """
        Initialize the Dynamic Gaussian Model.

        Args:
            sh_degree (int): The maximum degree of Spherical Harmonics to use for
                             representing view-dependent colors.
            args: A namespace or dictionary containing configuration for the
                  deformation network.
        """
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._deformation = deform_network(args)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.mean_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._deformation_table = torch.empty(0)
        self.sum_dx        = torch.empty(0)   # [N, 3]
        self.sum_dx_sq     = torch.empty(0)   # [N, 3]
        self.count_dx      = torch.empty(0)   # [N]
        self.setup_functions()

    def capture(self):
        """
        Captures the current state of the GaussianModel_dynamic.

        Returns:
            tuple: A tuple containing the current state of the model, including
                   active SH degree, positions, deformation network state,
                   deformation table, features, scaling, rotation, opacity,
                   max radii, gradient accumulator, denominator, optimizer state,
                   and spatial learning rate scale.
        """
        return (
            self.active_sh_degree,
            self._xyz,
            self._deformation.state_dict(),
            self._deformation_table,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        """
        Restores the state of the GaussianModel_dynamic from a saved model.

        Args:
            model_args (tuple): A tuple containing the restored state, including
                                active SH degree, positions, deformation network
                                state, deformation table, features, scaling,
                                rotation, opacity, max radii, gradient
                                accumulator, denominator, optimizer state, and
                                spatial learning rate scale.
            training_args: A namespace or dictionary containing configuration
                           for the deformation network.
        """
        (self.active_sh_degree,
         self._xyz,
         deform_state,
         self._deformation_table,

         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self._deformation.load_state_dict(deform_state)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)


    def oneupSHdegree(self):
        """
        Increases the active SH degree if it's less than the maximum.
        """
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """
        Creates Gaussians from a point cloud.

        Args:
            pcd (BasicPointCloud): The input point cloud.
            spatial_lr_scale (float): The spatial learning rate scale.
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2 + 1)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1


        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._deformation = self._deformation.to("cuda")

        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]), device="cuda"), 0)

    def training_setup(self, training_args):
        """
        Sets up the optimizer and learning rate schedulers for training.

        Args:
            training_args: A namespace or dictionary containing configuration
                           for the deformation network.
        """
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        n = self.get_xyz.shape[0]
        self.sum_dx        = torch.zeros(n, 3, device="cuda")
        self.sum_dx_sq     = torch.zeros(n, 3, device="cuda")
        self.count_dx      = torch.zeros(n, device="cuda")
        ##### 0.01
        self.downscale_mask_form = training_args.downscale_mask_deform_lr
        print('self.spatial_lr_scale', self.spatial_lr_scale)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self._deformation.get_mlp_parameters_cano()),
             'lr': training_args.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self._deformation.get_grid_parameters()),
             'lr': training_args.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / training_args.SH_lr_downscaling_start, "name": "f_rest"},
            {'params': list(self._deformation.get_mlp_parameters_others()),
             'lr': training_args.deformation_lr_init * self.spatial_lr_scale * self.downscale_mask_form,
             "name": "mask_update"},

            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},

            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}

        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.deformation_lr_init * self.spatial_lr_scale,
            lr_final=training_args.deformation_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.deformation_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init * self.spatial_lr_scale,
                                                     lr_final=training_args.grid_lr_final * self.spatial_lr_scale,
                                                     lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                     max_steps=training_args.position_lr_max_steps)
        self.mask_update_args = get_expon_lr_func(
            lr_init=training_args.deformation_lr_init * self.spatial_lr_scale * self.downscale_mask_form,
            lr_final=training_args.deformation_lr_final * self.spatial_lr_scale * self.downscale_mask_form,
            lr_delay_mult=training_args.deformation_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)
        self.feature_update_args = get_expon_lr_func(lr_init=training_args.feature_lr / training_args.SH_lr_downscaling_start,
                                                     lr_final=training_args.feature_lr / training_args.SH_lr_downscaling_end,
                                                     lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                     max_steps=training_args.position_lr_max_steps)
    def compute_transfer_state(self, time, compute_velocity=True):
        """Run the deformation network and return the same dict that
        render_foreground writes into save_key (a.k.a. deformed_gs), without
        invoking the rasterizer. Used by FG->TR organic transfer when the
        normal three-render path is not available (e.g. Phase 3 / one-render).

        Convention matches render_foreground exactly:
          means3D_final     deformed xyz (raw)
          scales_final      raw scaling (pre log/exp activation)
          rotations_final   raw rotation (pre normalization)
          opacity_final     raw opacity (pre sigmoid)
          shs_full          raw SH features
          time              scalar viewpoint time
          dynamic_score     sigmoid(shs_full[:, -1, 2])
          velocity, deformation_score (if compute_velocity)
        """
        with torch.no_grad():
            means3D = self.get_xyz
            scales = self._scaling
            rotations = self._rotation
            opacity = self._opacity
            shs = self.get_features

            time_tensor = torch.tensor(time).to(means3D.device).repeat(means3D.shape[0], 1)

            means3D_final, scales_final, rotations_final, opacity_final, shs_full = self._deformation(
                means3D, scales, rotations, opacity, shs, time_tensor
            )

            save_key = {
                'means3D_final':   means3D_final.clone().detach(),
                'scales_final':    scales_final.clone().detach(),
                'rotations_final': rotations_final.clone().detach(),
                'opacity_final':   opacity_final.clone().detach(),
                'shs_full':        shs_full.clone().detach(),
                'time':            time,
                'dynamic_score':   torch.sigmoid(shs_full[:, -1, 2]).clone().detach(),
            }

            if compute_velocity:
                epsilon = 1 / 300
                time_next = time_tensor + epsilon
                means3D_next, _, _, _, _ = self._deformation(
                    means3D, scales, rotations, opacity, shs, time_next
                )
                velocity = (means3D_next - means3D_final) / epsilon
                save_key['velocity'] = velocity.detach()
                save_key['deformation_score'] = torch.norm(velocity, dim=-1).detach()

        return save_key

    def select_fg_to_transient(self, deformed_state, dynamic_score_threshold=0.05,
                               max_transfer=2000, time_duration=None,
                               camera_center=None, bias_factor=0.0):
        """Select FG points whose dynamic_score exceeds the threshold and copy
        their deformed state into a dict that transient_gaussian.add_points_from_dict
        can ingest. Returns None if no points are selected.

        When the qualifying pool exceeds max_transfer, a uniform random subsample
        (randperm) is taken.
        """
        dynamic_score = deformed_state['dynamic_score']
        prune_mask = dynamic_score > dynamic_score_threshold

        print('all points', prune_mask.shape[0], 'pruned points to transient:', prune_mask.sum())

        if prune_mask.sum() == 0:
            return None

        current_count = prune_mask.sum().item()
        if current_count > max_transfer:
            true_indices = torch.nonzero(prune_mask).squeeze()
            perm = torch.randperm(current_count, device=true_indices.device)[:max_transfer]
            keep_indices = true_indices[perm]
            new_mask = torch.zeros_like(prune_mask)
            new_mask[keep_indices] = True
            prune_mask = new_mask

        xyz      = deformed_state['means3D_final'][prune_mask, :]
        rotation = deformed_state['rotations_final'][prune_mask, :]
        scaling  = deformed_state['scales_final'][prune_mask, :]
        opacity  = deformed_state['opacity_final'][prune_mask, :]

        features_full = deformed_state['shs_full']
        features_dc   = features_full[prune_mask, 0:1, :]
        features_rest = features_full[prune_mask, 1:16, :]

        velocity = None
        if 'velocity' in deformed_state:
            velocity = deformed_state['velocity'][prune_mask]
            # Foreground time is [0,1]; transient time is [time_duration[0], time_duration[1]].
            if time_duration is not None:
                duration = time_duration[1] - time_duration[0]
                if duration > 1e-6:
                    velocity = velocity / duration

        if camera_center is not None and bias_factor > 0:
            cam_center = camera_center.to(xyz.device)
            xyz = xyz + (cam_center - xyz) * bias_factor

        if time_duration is not None and 'time' in deformed_state:
            t_ref = time_duration[0] + deformed_state['time'] * (time_duration[1] - time_duration[0])
        elif time_duration is not None:
            t_ref = (time_duration[0] + time_duration[1]) / 2.0
        else:
            t_ref = 0.0

        # Boost opacity so the transferred points contribute immediately.
        current_prob = torch.sigmoid(opacity)
        opacity = inverse_sigmoid(torch.clamp(current_prob + 0.05, max=0.999))

        return {
            "xyz": xyz,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "scaling": scaling,
            "rotation": rotation,
            "opacity": opacity,
            "t_ref": t_ref,
            "velocity": velocity,
        }

    def weight_prune(self, weight_threshold: float, mean_ratio):
        """
        Prune points whose tracked _max_weight is below the given threshold.
        After pruning, reset remaining weights to zero (e.g., to re-accumulate).
        """
        prune_mask = ((self.max_radii2D * (1 - mean_ratio) + self.mean_radii2D * mean_ratio) < weight_threshold)

        self.prune_points(prune_mask)
        # reset remaining weights for next accumulation window
        self.max_radii2D = self.max_radii2D * 0.0
        self.mean_radii2D = self.mean_radii2D * 0
        torch.cuda.empty_cache()

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "deformation":
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "mask_update":
                lr = self.mask_update_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "f_rest":
                lr = self.feature_update_args(iteration)
                param_group['lr'] = lr


    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l


    def load_model(self, path):
        print("loading model from exists{}".format(path))
        weight_dict = torch.load(os.path.join(path, "deformation.pth"), map_location="cuda")
        self._deformation.load_state_dict(weight_dict)
        self._deformation = self._deformation.to("cuda")
        self._deformation_table = torch.gt(torch.ones((self.get_xyz.shape[0]), device="cuda"), 0)
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        if os.path.exists(os.path.join(path, "deformation_table.pth")):
            self._deformation_table = torch.load(os.path.join(path, "deformation_table.pth"), map_location="cuda")
        if os.path.exists(os.path.join(path, "deformation_accum.pth")):
            self._deformation_accum = torch.load(os.path.join(path, "deformation_accum.pth"), map_location="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.mean_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def save_deformation(self, path):
        torch.save(self._deformation.state_dict(), os.path.join(path, "deformation.pth"))
        torch.save(self._deformation_table, os.path.join(path, "deformation_table.pth"))
        torch.save(self._deformation_accum, os.path.join(path, "deformation_accum.pth"))

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def reset_opacity_partially_small(self):
        ##### a feature to force reset
        opacities_new = self.get_opacity
        opacities_new_replace = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))

        opacities_mask = np.array(list(range(0, self.get_opacity.shape[0])))

        opacities_mask_select = np.random.choice(opacities_mask, int(opacities_mask.shape[0] * 0.5), replace=False)
        opacities_new[opacities_mask_select] = opacities_new_replace[opacities_mask_select]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * ((self.max_sh_degree + 1) ** 2 + 1) - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 + 1 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree


    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._deformation_accum = self._deformation_accum[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self._deformation_table = self._deformation_table[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.mean_radii2D = self.mean_radii2D[valid_points_mask]
        if self.sum_dx.shape[0] > 0:
            self.sum_dx    = self.sum_dx[valid_points_mask]
            self.sum_dx_sq = self.sum_dx_sq[valid_points_mask]
            self.count_dx  = self.count_dx[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1: continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_deformation_table, new_max_radii2D, new_mean_radii2D):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], -1)
        self.mean_radii2D = torch.cat([self.mean_radii2D, new_mean_radii2D], -1)
        self._deformation_table = torch.cat([self._deformation_table, new_deformation_table], -1)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self._deformation_accum = torch.zeros((self.get_xyz.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # Pad per-point accumulators for new points — keeps history for existing
        # points while giving new points a fresh count of 0.
        n_new = new_xyz.shape[0]
        if self.sum_dx.shape[0] > 0:
            self.sum_dx    = torch.cat([self.sum_dx,    torch.zeros(n_new, 3, device=self.sum_dx.device)],    dim=0)
            self.sum_dx_sq = torch.cat([self.sum_dx_sq, torch.zeros(n_new, 3, device=self.sum_dx_sq.device)], dim=0)
            self.count_dx  = torch.cat([self.count_dx,  torch.zeros(n_new,    device=self.count_dx.device)],  dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)
        if not selected_pts_mask.any():
            return
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_deformation_table = self._deformation_table[selected_pts_mask].repeat(N)
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)
        new_mean_radii2D = self.mean_radii2D[selected_pts_mask].repeat(N)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_deformation_table, new_max_radii2D, new_mean_radii2D)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        grads_accum_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.logical_and(grads_accum_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_deformation_table = self._deformation_table[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]
        new_mean_radii2D = self.mean_radii2D[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_deformation_table, new_max_radii2D, new_mean_radii2D)


    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [0, 1, 3]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total

    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return total

    def _l1_regulation(self):
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total

    def compute_regulation(self, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()

class GaussianModel(PersistentBranchMixin, GaussianBranchBase):
    """
    Represents a static set of 3D Gaussians.

    This class is a simplified version of `GaussianModel_dynamic`, tailored for
    static scenes. It manages the same core attributes (position, color, scale,
    etc.) but does not include the deformation network, making it more efficient
    for modeling the static background components of a scene.
    """


    def __init__(self, sh_degree: int):
        """
        Initialize the Static Gaussian Model.

        Args:
            sh_degree (int): The maximum degree of Spherical Harmonics.
        """
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        """
        Captures the current state of the GaussianModel.

        Returns:
            tuple: A tuple containing the current state of the model, including
                   active SH degree, positions, features, scaling, rotation,
                   opacity, max radii, gradient accumulator, denominator,
                   optimizer state, and spatial learning rate scale.
        """
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        """
        Restores the state of the GaussianModel from a saved model.

        Args:
            model_args (tuple): A tuple containing the restored state, including
                                active SH degree, positions, features, scaling,
                                rotation, opacity, max radii, gradient
                                accumulator, denominator, optimizer state, and
                                spatial learning rate scale.
            training_args: A namespace or dictionary containing configuration
                           for the deformation network.
        """
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)


    def oneupSHdegree(self):
        """
        Increases the active SH degree if it's less than the maximum.
        """
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def reset_opacity_partially(self):
        """
        Partially resets the opacity of Gaussians to a small value.
        """
        ##### a feature to force reset
        opacities_new = self.get_opacity
        opacities_new_replace = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        opacities_mask = np.array(list(range(0, self.get_opacity.shape[0])))
        opacities_mask_select = np.random.choice(opacities_mask, int(opacities_mask.shape[0] * 0.7), replace=False)
        opacities_new[opacities_mask_select] = opacities_new_replace[opacities_mask_select]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        """
        Creates Gaussians from a point cloud.

        Args:
            pcd (BasicPointCloud): The input point cloud.
            spatial_lr_scale (float): The spatial learning rate scale.
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        """
        Sets up the optimizer and learning rate schedulers for training.

        Args:
            training_args: A namespace or dictionary containing configuration
                           for the deformation network.
        """
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)


    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_max_radii2D):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Follow the same pattern as GaussianModel_dynamic - concatenate max_radii2D instead of resetting to zeros
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], -1)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_max_radii2D)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_max_radii2D)


    def weight_prune(self, weight_threshold: float):
        """
        Prune points whose tracked _max_weight is below the given threshold.
        After pruning, reset remaining weights to zero (e.g., to re-accumulate).
        """
        prune_mask = (self.max_radii2D < weight_threshold)
        print('all points', prune_mask.shape[0], 'pruned points:', prune_mask.sum())
        self.prune_points(prune_mask)

        # reset remaining weights for next accumulation window
        self.max_radii2D = self.max_radii2D * 0.0
        torch.cuda.empty_cache()


class GaussianModelTransient(GaussianBranchBase):
    """Transient branch (G_t): 4D spatiotemporal Gaussians with a temporal extent,
    seeded only by velocity-aware lifting from the persistent-dynamic branch.
    Carries its own density control (densify_and_prune / add_densification_stats_grad /
    weight_prune) and 4D fields, so it does NOT inherit PersistentBranchMixin."""


    def __init__(self, sh_degree: int, gaussian_dim: int = 3, time_duration: list = [-0.5, 0.5], rot_4d: bool = False,
                 sh_degree_t: int = 0):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.mean_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.gaussian_dim = gaussian_dim
        self._t = torch.empty(0)
        self._scaling_t = torch.empty(0)
        self.time_duration = time_duration
        self.rot_4d = rot_4d
        self._rotation_r = torch.empty(0)
        self.t_gradient_accum = torch.empty(0)
        if self.rot_4d:
            assert self.gaussian_dim == 4

        self.active_sh_degree_t = 0
        self.max_sh_degree_t = sh_degree_t

        self.setup_functions()


    @property
    def get_scaling_t(self):
        return self.scaling_activation(self._scaling_t)

    @property
    def get_scaling_xyzt(self):
        return self.scaling_activation(torch.cat([self._scaling, self._scaling_t], dim=1))


    @property
    def get_rotation_r(self):
        return self.rotation_activation(self._rotation_r)


    @property
    def get_t(self):
        return self._t

    @property
    def get_xyzt(self):
        return torch.cat([self._xyz, self._t], dim=1)


    @property
    def get_max_sh_channels(self):
        return (self.max_sh_degree + 1) ** 2 * (self.max_sh_degree_t + 1)


    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
        elif self.max_sh_degree_t and self.active_sh_degree_t < self.max_sh_degree_t:
            self.active_sh_degree_t += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, self.get_max_sh_channels)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        if self.gaussian_dim == 4:
            if pcd.time is None:
                fused_times = (torch.rand(fused_point_cloud.shape[0], 1, device="cuda") * 1.2 - 0.1) * (
                        self.time_duration[1] - self.time_duration[0]) + self.time_duration[0]
            else:
                fused_times = torch.from_numpy(pcd.time).cuda().float()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        if self.gaussian_dim == 4:
            dist_t = torch.zeros_like(fused_times, device="cuda") + (self.time_duration[1] - self.time_duration[0]) / 5
            scales_t = torch.log(torch.sqrt(dist_t))
            if self.rot_4d:
                rots_r = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
                rots_r[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.gaussian_dim == 4:
            self._t = nn.Parameter(fused_times.requires_grad_(True))
            self._scaling_t = nn.Parameter(scales_t.requires_grad_(True))
            if self.rot_4d:
                self._rotation_r = nn.Parameter(rots_r.requires_grad_(True))


    def add_points(self, xyz, features_dc, features_rest, scaling, rotation, opacity, t_ref, velocity=None, scaling_t_div=10, velocity_scale=1.0):
        num_points = xyz.shape[0]
        new_t = None
        new_scaling_t = None
        new_rotation_r = None

        if self.gaussian_dim == 4:
            new_t = torch.ones((num_points, 1), device="cuda") * t_ref
            dist_t = torch.zeros_like(new_t) + (self.time_duration[1] - self.time_duration[0]) / scaling_t_div
            new_scaling_t = torch.log(torch.sqrt(dist_t))

            if self.rot_4d:
                new_rotation_r = torch.zeros((num_points, 4), device="cuda")
                new_rotation_r[:, 0] = 1
                from utils.general_utils import velocity_to_rotation_r
                if velocity is not None:
                    # velocity_scale damps the inherited FG velocity at TR init.
                    # 1.0 = full velocity (current behavior); 0.0 = no inherited motion (rotation_r stays identity).
                    if velocity_scale != 1.0:
                        velocity = velocity * float(velocity_scale)
                    if float(velocity_scale) != 0.0:
                        # Pass the copied FG rotation as rotation_l so the resulting
                        # 4D rotation is internally consistent: build_rotation_4d
                        # uses both q_l and q_r at render time, so q_r must be
                        # derived for the actual q_l (here = the FG's deformed
                        # rotation), not for the historical default q_l = identity.
                        new_rotation_r = velocity_to_rotation_r(velocity, rotation_l=rotation)
                    # else: keep identity rotation_r initialized above

            # Pad features_rest if dimensions mismatch (e.g. 3D input -> 4D model)
            # 3D SH (16 coeffs total) has 15 rest coeffs. 4D SH (48 coeffs total) has 47 rest coeffs.
            if features_rest.shape[1] < self._features_rest.shape[1]:
                padding_size = self._features_rest.shape[1] - features_rest.shape[1]
                padding = torch.zeros((num_points, padding_size, 3), device="cuda")
                features_rest = torch.cat([features_rest, padding], dim=1)

        self.densification_postfix_nodenom(xyz, features_dc, features_rest, opacity, scaling, rotation,
                                   new_t, new_scaling_t, new_rotation_r)
    def densification_postfix_nodenom(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_t, new_scaling_t, new_rotation_r):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             }

        d["t"] = new_t
        d["scaling_t"] = new_scaling_t
        if self.rot_4d:
            d["rotation_r"] = new_rotation_r

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._t = optimizable_tensors['t']
        self._scaling_t = optimizable_tensors['scaling_t']
        if self.rot_4d:
            self._rotation_r = optimizable_tensors['rotation_r']

        num_new = new_xyz.shape[0]
        self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum, torch.zeros((num_new, 1), device="cuda")), dim=0)
        self.denom = torch.cat((self.denom, torch.zeros((num_new, 1), device="cuda")), dim=0)
        # New TR points (FG→TR donors, dyn_score_clone seeds, dx_var/opacity_var transfers)
        # get max_radii2D = 0.1 (above the 0.005→0.020 weight_prune threshold ramp ceiling)
        # so they survive a same-iter weight_prune. weight_prune zeros max_radii2D for
        # all survivors after pruning, so this only protects the immediate same-iter prune;
        # the donor still earns its accumulation window over the next weight_prune_fre_transient
        # iters before being judged on real signal. Mirrors how densify children inherit
        # parent's max_radii2D for the same protection (see densify_and_clone/split).
        self.max_radii2D = torch.cat((self.max_radii2D, torch.full((num_new,), 0.1, device="cuda")), dim=0)
        self.mean_radii2D = torch.cat((self.mean_radii2D, torch.zeros((num_new), device="cuda")), dim=0)
        self.t_gradient_accum = torch.cat((self.t_gradient_accum, torch.zeros((num_new, 1), device="cuda")), dim=0)

    def add_points_from_dict(self, data_dict, velocity_scale=1.0):
        if data_dict is None:
            return
        self.add_points(
            data_dict["xyz"],
            data_dict["features_dc"],
            data_dict["features_rest"],
            data_dict["scaling"],
            data_dict["rotation"],
            data_dict["opacity"],
            data_dict["t_ref"],
            velocity=data_dict.get("velocity", None),
            velocity_scale=velocity_scale,
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # Temporal-axis LR mirrors the positional LR.
        training_args.position_t_lr_init = training_args.position_lr_init
        self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        l.append({'params': [self._t], 'lr': training_args.position_t_lr_init, "name": "t"})
        l.append({'params': [self._scaling_t], 'lr': training_args.scaling_lr, "name": "scaling_t"})

        l.append({'params': [self._rotation_r], 'lr': training_args.rotation_lr   , "name": "rotation_r"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale ,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        lr_xyz = None
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr_xyz = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr_xyz

        return lr_xyz

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.mean_radii2D = self.mean_radii2D[valid_points_mask]

        if self.gaussian_dim == 4:
            self._t = optimizable_tensors['t']
            self._scaling_t = optimizable_tensors['scaling_t']
            if self.rot_4d:
                self._rotation_r = optimizable_tensors['rotation_r']
            self.t_gradient_accum = self.t_gradient_accum[valid_points_mask]

    def weight_prune(self, weight_threshold: float):
        """
        Prune points whose tracked _max_weight is below the given threshold.
        After pruning, reset remaining weights to zero (e.g., to re-accumulate).
        """
        prune_mask = (self.max_radii2D < weight_threshold)

        self.prune_points(prune_mask)
        # reset remaining weights for next accumulation window
        self.max_radii2D = self.max_radii2D * 0.0
        self.mean_radii2D = self.mean_radii2D * 0
        torch.cuda.empty_cache()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_t, new_scaling_t, new_rotation_r, new_mean_radii2D, new_max_radii2D):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             }

        d["t"] = new_t
        d["scaling_t"] = new_scaling_t
        if self.rot_4d:
            d["rotation_r"] = new_rotation_r

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self._t = optimizable_tensors['t']
        self._scaling_t = optimizable_tensors['scaling_t']
        if self.rot_4d:
            self._rotation_r = optimizable_tensors['rotation_r']
        self.t_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], -1)
        self.mean_radii2D = torch.cat([self.mean_radii2D, new_mean_radii2D], -1)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        if not self.rot_4d:
            stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_t = None
            new_scaling_t = None
            new_rotation_r = None
            if self.gaussian_dim == 4:
                stds_t = self.get_scaling_t[selected_pts_mask].repeat(N, 1)
                means_t = torch.zeros((stds_t.size(0), 1), device="cuda")
                samples_t = torch.normal(mean=means_t, std=stds_t)
                new_t = samples_t + self.get_t[selected_pts_mask].repeat(N, 1)
                new_scaling_t = self.scaling_inverse_activation(
                    self.get_scaling_t[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        else:
            stds = self.get_scaling_xyzt[selected_pts_mask].repeat(N, 1)
            means = torch.zeros((stds.size(0), 4), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation_4d(self._rotation[selected_pts_mask], self._rotation_r[selected_pts_mask]).repeat(N,
                                                                                                                    1,
                                                                                                                    1)
            new_xyzt = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyzt[selected_pts_mask].repeat(N,
                                                                                                                    1)
            new_xyz = new_xyzt[..., 0:3]
            new_t = new_xyzt[..., 3:4]
            new_scaling_t = self.scaling_inverse_activation(
                self.get_scaling_t[selected_pts_mask].repeat(N, 1) / (0.8 * N))
            new_rotation_r = self._rotation_r[selected_pts_mask].repeat(N, 1)

        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)
        new_mean_radii2D = self.mean_radii2D[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation,
                                   new_t, new_scaling_t, new_rotation_r, new_mean_radii2D, new_max_radii2D)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_t = None
        new_scaling_t = None
        new_rotation_r = None
        if self.gaussian_dim == 4:
            new_t = self._t[selected_pts_mask]
            new_scaling_t = self._scaling_t[selected_pts_mask]
            if self.rot_4d:
                new_rotation_r = self._rotation_r[selected_pts_mask]

        new_max_radii2D = self.max_radii2D[selected_pts_mask]
        new_mean_radii2D = self.mean_radii2D[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_t, new_scaling_t, new_rotation_r, new_mean_radii2D, new_max_radii2D)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_t=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask =torch.logical_or(prune_mask, big_points_ws)

        if prune_mask.numel() > 1000:
            # only prune if more than 1000 pts
            self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats_grad(self, viewspace_point_grad, update_filter, avg_t_grad=None):
        self.xyz_gradient_accum[update_filter] += viewspace_point_grad[update_filter]
        self.denom[update_filter] += 1
        if self.gaussian_dim == 4 and avg_t_grad is not None:
            self.t_gradient_accum[update_filter] += avg_t_grad[update_filter]

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels including the total SHs
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))

        # 4D attributes
        if self.gaussian_dim == 4:
            l.append('t')
            for i in range(self._scaling_t.shape[1]):
                l.append('scale_t_{}'.format(i))
            if self.rot_4d:
                for i in range(self._rotation_r.shape[1]):
                    l.append('rot_r_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        attributes = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]

        if self.gaussian_dim == 4:
            t = self._t.detach().cpu().numpy()
            scale_t = self._scaling_t.detach().cpu().numpy()
            attributes.extend([t, scale_t])
            if self.rot_4d:
                rotation_r = self._rotation_r.detach().cpu().numpy()
                attributes.append(rotation_r)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(attributes, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        """Inverse of save_ply for the 4D transient branch. Mirrors the spatial loader used
        by the other branches, then loads the temporal attributes (t, scale_t_*, rot_r_*).
        Prefixes overlap (scale_ vs scale_t_, rot_ vs rot_r_), so the spatial filters
        explicitly exclude the temporal names."""
        plydata = PlyData.read(path)
        prop_names = [p.name for p in plydata.elements[0].properties]

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = sorted([p for p in prop_names if p.startswith("f_rest_")],
                               key=lambda x: int(x.split('_')[-1]))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, len(extra_f_names) // 3))

        # spatial scale (scale_*) — exclude the temporal scale_t_*
        scale_names = sorted([p for p in prop_names if p.startswith("scale_") and not p.startswith("scale_t")],
                             key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # spatial rotation (rot_*) — exclude the 4D rot_r_*
        rot_names = sorted([p for p in prop_names if p.startswith("rot_") and not p.startswith("rot_r")],
                           key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        if self.gaussian_dim == 4:
            t = np.asarray(plydata.elements[0]["t"])[..., np.newaxis]
            self._t = nn.Parameter(torch.tensor(t, dtype=torch.float, device="cuda").requires_grad_(True))
            scale_t_names = sorted([p for p in prop_names if p.startswith("scale_t_")],
                                   key=lambda x: int(x.split('_')[-1]))
            scales_t = np.zeros((xyz.shape[0], len(scale_t_names)))
            for idx, attr_name in enumerate(scale_t_names):
                scales_t[:, idx] = np.asarray(plydata.elements[0][attr_name])
            self._scaling_t = nn.Parameter(torch.tensor(scales_t, dtype=torch.float, device="cuda").requires_grad_(True))
            if self.rot_4d:
                rot_r_names = sorted([p for p in prop_names if p.startswith("rot_r_")],
                                     key=lambda x: int(x.split('_')[-1]))
                rots_r = np.zeros((xyz.shape[0], len(rot_r_names)))
                for idx, attr_name in enumerate(rot_r_names):
                    rots_r[:, idx] = np.asarray(plydata.elements[0][attr_name])
                self._rotation_r = nn.Parameter(torch.tensor(rots_r, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
