import os
import torch
import numpy as np
import hdbscan
import multiprocessing
import sys
import argparse
import math
import torchvision
from argparse import ArgumentParser, Namespace  # Namespace: cfg_args is eval()'d below
from os import makedirs
from scene import Scene2gs_mixed
from scene.gaussian_model import GaussianModel, GaussianModel_dynamic
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from ext.grounded_sam import grouned_sam_output, load_model_hf
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
from tqdm import tqdm
import pytorch3d.ops as ops

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def generate_grid_index(depth):
    h, w = depth.shape
    grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid = torch.stack(grid, dim=-1)
    return grid

def postprocessing(features, query_feature, score_threshold=0.8):
    features /= features.norm(dim=-1, keepdim=True)
    query_feature /= query_feature.norm(dim=-1, keepdim=True)
    query_feature = query_feature.unsqueeze(-1)
    scores = features.half() @ query_feature.half()
    scores = scores[:, 0]
    mask = (scores >= score_threshold)
    return mask

def render_rgb_with_depth(viewpoint_camera, pc_foreground, pc_background, pipe, bg_color, override_color=None, mask_fg=None, mask_bg=None):
    time = torch.tensor(viewpoint_camera.time).to(pc_foreground.get_xyz.device).repeat(pc_foreground.get_xyz.shape[0], 1)
    
    means3D_fg, scales_fg, rotations_fg, opacity_fg, shs_fg = pc_foreground._deformation(
        pc_foreground.get_xyz, pc_foreground._scaling, pc_foreground._rotation,
        pc_foreground._opacity, pc_foreground.get_features, time
    )
    
    opacity_fg = pc_foreground.opacity_activation(opacity_fg)
    scales_fg = pc_foreground.scaling_activation(scales_fg)
    rotations_fg = pc_foreground.rotation_activation(rotations_fg)
    
    means3D_bg = pc_background.get_xyz
    opacity_bg = pc_background.opacity_activation(pc_background._opacity)
    scales_bg = pc_background.scaling_activation(pc_background._scaling)
    rotations_bg = pc_background.rotation_activation(pc_background._rotation)
    shs_bg = pc_background.get_features
    
    if mask_fg is not None:
        means3D_fg = means3D_fg[mask_fg]
        scales_fg = scales_fg[mask_fg]
        rotations_fg = rotations_fg[mask_fg]
        opacity_fg = opacity_fg[mask_fg]
        shs_fg = shs_fg[mask_fg]
        
    if mask_bg is not None:
        means3D_bg = means3D_bg[mask_bg]
        scales_bg = scales_bg[mask_bg]
        rotations_bg = rotations_bg[mask_bg]
        opacity_bg = opacity_bg[mask_bg]
        shs_bg = shs_bg[mask_bg]
        
    means3D = torch.cat([means3D_fg, means3D_bg], dim=0)
    opacity = torch.cat([opacity_fg, opacity_bg], dim=0)
    scales = torch.cat([scales_fg, scales_bg], dim=0)
    rotations = torch.cat([rotations_fg, rotations_bg], dim=0)
    if shs_fg.shape[1] != shs_bg.shape[1]:
        shs_fg = shs_fg[:, :shs_bg.shape[1], :]
    shs = torch.cat([shs_fg, shs_bg], dim=0)
    
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc_foreground.active_sh_degree, 
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
        bwd_depth=False,
        bwd_dynamic=True
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=False, device="cuda")
    dynamics = torch.ones((means2D.shape[0], 1, means2D.shape[1]), dtype=means2D.dtype,
                            device=means2D.device)

    # For overriding color (e.g., when rendering binary mask)
    colors_precomp = None
    if override_color is not None:
        colors_precomp = override_color
        shs = None

    rendered_image, radii, depth, dynamics_map, max_weight_t = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        dynamics=dynamics,
        cov3D_precomp=None)
        
    return {"render": rendered_image, "depth": depth, "means3D_deformed": means3D_fg}

def render_cluster_by_text(dataset, hypernetwork, iteration, pipeline, opt, args):
    with torch.no_grad():
        dataset.sh_degree = 3
        foreground_gaussians = GaussianModel_dynamic(dataset.sh_degree, hypernetwork)
        background_gaussians = GaussianModel(dataset.sh_degree)
        
        custom_checkpoint_path = None
        if args.render_checkpoint and os.path.isdir(args.render_checkpoint):
             custom_checkpoint_path = args.render_checkpoint
             scene_load_path = None 
        else:
             scene_load_path = args.render_checkpoint

        scene = Scene2gs_mixed(dataset, foreground_gaussians, load_iteration=scene_load_path, gaussians_second=background_gaussians, gaussians_transient=None)
        
        if custom_checkpoint_path:
            pc_dir = os.path.join(custom_checkpoint_path, "point_cloud")
            if os.path.exists(pc_dir):
                iters = [int(fol.split("_")[-1]) for fol in os.listdir(pc_dir) if "iteration_" in fol]
                if iters:
                    max_iter = max(iters)
                    load_dir = os.path.join(pc_dir, f"iteration_{max_iter}")
                    if os.path.exists(os.path.join(load_dir, "point_cloud.ply")):
                         foreground_gaussians.load_ply(os.path.join(load_dir, "point_cloud.ply"))
                    if os.path.exists(os.path.join(load_dir, "point_cloud_second.ply")):
                         background_gaussians.load_ply(os.path.join(load_dir, "point_cloud_second.ply"))
                    if os.path.exists(os.path.join(load_dir, "deformation.pth")):
                         deform_state = torch.load(os.path.join(load_dir, "deformation.pth"), map_location="cuda")
                         foreground_gaussians._deformation.load_state_dict(deform_state)
                         foreground_gaussians._deformation.to("cuda")

        # LOAD SEMANTIC FEATURES
        semantic_load_dir = os.path.join(args.saving_folder, args.expname)
        # Prefer final features (run root); fall back to newest semantic_iteration_* checkpoint.
        sem_dir = semantic_load_dir
        if not os.path.exists(os.path.join(sem_dir, "foreground_semantics.pt")):
            ckpts = sorted([d for d in os.listdir(semantic_load_dir) if d.startswith("semantic_iteration_")],
                           key=lambda d: int(d.split("_")[-1])) if os.path.isdir(semantic_load_dir) else []
            if ckpts:
                sem_dir = os.path.join(semantic_load_dir, ckpts[-1])
        fg_sem_path = os.path.join(sem_dir, "foreground_semantics.pt")
        bg_sem_path = os.path.join(sem_dir, "background_semantics.pt")
        
        has_fg, has_bg = False, False
        if os.path.exists(fg_sem_path):
            foreground_gaussians.semantic_feature = torch.load(fg_sem_path, map_location="cuda")
            has_fg = True
        if os.path.exists(bg_sem_path):
            background_gaussians.semantic_feature = torch.load(bg_sem_path, map_location="cuda")
            has_bg = True

        if not has_fg and not has_bg:
            print("No semantic features found.")
            return

        # Empty side inherits the feature dim of the present side (partial checkpoints).
        feat_dim = (foreground_gaussians.semantic_feature if has_fg else background_gaussians.semantic_feature).shape[-1]
        fg_feats = foreground_gaussians.semantic_feature if has_fg else torch.empty(0, feat_dim, device="cuda")
        bg_feats = background_gaussians.semantic_feature if has_bg else torch.empty(0, feat_dim, device="cuda")
        
        if has_fg and len(fg_feats.shape) == 3: fg_feats = fg_feats.squeeze(1)
        if has_bg and len(bg_feats.shape) == 3: bg_feats = bg_feats.squeeze(1)

        all_feats = torch.cat([fg_feats, bg_feats], dim=0)
        normed_point_features = torch.nn.functional.normalize(all_feats, dim = -1, p = 2)
        
        print(f"Running HDBSCAN Clustering on combined {all_feats.shape[0]} features...")
        percent = 0.02
        mask = torch.rand(all_feats.shape[0]) > (1 - percent)
        sampled_point_features = all_feats[mask]
        normed_sampled_point_features = sampled_point_features / torch.norm(sampled_point_features, dim = -1, keepdim = True)
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01, allow_single_cluster = False, core_dist_n_jobs=multiprocessing.cpu_count())
        cluster_labels = clusterer.fit_predict(normed_sampled_point_features.detach().cpu().numpy())
        
        num_clusters = len(np.unique(cluster_labels))
        print(f"Found {num_clusters} clusters.")

        # Build one center per real cluster label (>= 0), indexed by label value.
        # Noise (-1) gets no center row, so it never competes in the argmax and
        # noise points are absorbed by the nearest real cluster.
        valid_labels = [l for l in sorted(np.unique(cluster_labels)) if l >= 0]
        num_centers = (max(valid_labels) + 1) if valid_labels else 1  # 1 zero-row fallback if all noise
        cluster_centers = torch.zeros(num_centers, normed_sampled_point_features.shape[-1])
        for lbl in valid_labels:
            mask_c = (cluster_labels == lbl)
            cluster_centers[lbl] = torch.nn.functional.normalize(normed_sampled_point_features[mask_c].mean(dim=0), dim=-1)

        seg_score = torch.einsum('nc,bc->bn', cluster_centers.cpu(), normed_point_features.cpu())
        cluster_ids = seg_score.argmax(dim=-1).numpy()  # center row index IS the hdbscan label
        
        num_fg = fg_feats.shape[0]

        views = scene.getTestCameras()
        bg_color = [1,1,1] if getattr(dataset, 'white_background', False) else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(semantic_load_dir, f"ours_{iteration}_text", "renders")
        pred_masks_path = os.path.join(semantic_load_dir, f"ours_{iteration}_text", "pred_masks")
        makedirs(render_path, exist_ok=True)
        makedirs(pred_masks_path, exist_ok=True)

        if args.text_prompt != '':
            print("Text prompt detected: ", args.text_prompt)
            ckpt_repo_id = "ShilongLiu/GroundingDINO"
            ckpt_filename = "groundingdino_swinb_cogcoor.pth"
            ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
            groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)
            sam_checkpoint = 'dependency/sam_vit_h_4b8939.pth'
            sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
            sam.to(device='cuda')
            sam_predictor = SamPredictor(sam)

        text_masked_cls_id = None
        
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            res = render_rgb_with_depth(view, foreground_gaussians, background_gaussians, pipeline, background)
            rendering = res["render"]
            depth = res["depth"].squeeze()

            # Deformed FG + Static BG matching all_feats order
            means3D = torch.cat([res["means3D_deformed"], background_gaussians.get_xyz], dim=0)

            if idx == 0 and args.text_prompt != '':
                text_mask, annotated_frame_with_mask = grouned_sam_output(groundingdino_model, sam_predictor, args.text_prompt, to8b(rendering).transpose(1,2,0))
                del sam_predictor
                del groundingdino_model
                Image.fromarray(annotated_frame_with_mask).save(os.path.join(render_path[:-8],'grounded-sam---' + args.text_prompt + '.png'))
                Image.fromarray(text_mask.detach().cpu().numpy()).save(os.path.join(render_path[:-8],'binary-grounded-sam---' + args.text_prompt + '.png'))

                grid_index = generate_grid_index(depth).cuda()
                z = view.zfar / (view.zfar - view.znear) * depth[text_mask] - view.zfar * view.znear / (view.zfar - view.znear)

                uvz = torch.cat(((((grid_index[text_mask, :][:, 1] - 0.5) / view.image_width * 2 - 1) * depth[text_mask]).unsqueeze(-1),
                                (((grid_index[text_mask, :][:, 0] - 0.5) / view.image_height * 2 - 1) * depth[text_mask]).unsqueeze(-1),
                                z.unsqueeze(-1),
                                depth[text_mask].unsqueeze(-1)), 1)
                
                text_masked_points_in_3D = uvz @ (torch.inverse(view.full_proj_transform.cuda()))[:, :3]
                
                # KNN between mask points and current Deformed GS Centers
                knn_obj = ops.knn_points(
                    text_masked_points_in_3D.unsqueeze(0).float(),
                    means3D.detach().unsqueeze(0).float(),
                    K=1,
                )
                ijs = knn_obj.idx.squeeze(0).squeeze(-1)
                
                text_masked_points_cls = torch.tensor(cluster_ids)[ijs.cpu()].int()
                _bincount = torch.bincount(text_masked_points_cls)
                _cls_thr = args.cls_point_threshold
                print("cluster point counts (>500):", {int(i): int(c) for i, c in enumerate(_bincount) if c > 500})
                text_masked_cls_id = torch.where(_bincount > _cls_thr, 1, 0).nonzero()
                print("Text prompt cls id (thr=%d): " % _cls_thr, text_masked_cls_id)
            if args.text_prompt != '' and text_masked_cls_id is not None:
                segmented_mask = None
                for id in text_masked_cls_id.squeeze(-1):
                    id = id.item()
                    pre_mask = torch.tensor(cluster_ids == id, device="cuda")
                    # postprocessing expects features and query_feature
                    # we will query the specific cluster mean
                    mean_feat = all_feats[pre_mask].mean(dim=0)
                    filtered_mask = postprocessing(all_feats.clone(), mean_feat, score_threshold=args.score_threshold)
                    post_mask = pre_mask & filtered_mask
                    if segmented_mask is None:
                        segmented_mask = post_mask
                    else:
                        segmented_mask |= post_mask
                
                # Render Binary Mask
                # Colors: 1 everywhere. Mask controls what gets rendered.
                
                if segmented_mask is not None:
                    mask_fg = segmented_mask[:num_fg] if has_fg else None
                    mask_bg = segmented_mask[num_fg:] if has_bg else None
                    
                    bg_col_bin = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
                    
                    # Compute # of points we are rendering
                    cnt = 0
                    if mask_fg is not None: cnt += mask_fg.sum().item()
                    if mask_bg is not None: cnt += mask_bg.sum().item()
                    
                    if cnt > 0:
                        # Override color = ones restricted to the SELECTED points. No occlusion:
                        # only the selected gaussians are rendered (mask_fg/mask_bg passed through),
                        # all white -> target silhouette regardless of occluding geometry.
                        override_color = torch.ones((means3D.shape[0], 3), device="cuda").float()
                        if mask_fg is not None: override_color_fg = override_color[:num_fg][mask_fg]
                        else: override_color_fg = None
                        if mask_bg is not None: override_color_bg = override_color[num_fg:][mask_bg]
                        else: override_color_bg = None

                        over_col = torch.cat([fc for fc in [override_color_fg, override_color_bg] if fc is not None], dim=0)

                        # Render actual RGB appearance of selected clusters
                        rgb_res = render_rgb_with_depth(view, foreground_gaussians, background_gaussians, pipeline, bg_col_bin, override_color=None, mask_fg=mask_fg, mask_bg=mask_bg)
                        torchvision.utils.save_image(rgb_res["render"].cpu(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

                        if args.occlusion_aware:
                            # Occlusion-aware: rasterize EVERY gaussian, painting the
                            # selected ones white and the rest black. Alpha compositing
                            # then does the occlusion -- a nearer unselected (black)
                            # gaussian covers a selected one behind it, so the mask is
                            # the object's *visible* extent.
                            occ_col = torch.zeros((means3D.shape[0], 3), device="cuda").float()
                            if mask_fg is not None: occ_col[:num_fg][mask_fg] = 1.0
                            if mask_bg is not None: occ_col[num_fg:][mask_bg] = 1.0
                            bin_res = render_rgb_with_depth(view, foreground_gaussians, background_gaussians, pipeline, bg_col_bin, override_color=occ_col)
                        else:
                            # Render Binary Mask (NO occlusion): only selected gaussians, all white.
                            bin_res = render_rgb_with_depth(view, foreground_gaussians, background_gaussians, pipeline, bg_col_bin, override_color=over_col, mask_fg=mask_fg, mask_bg=mask_bg)
                        buffer_image = bin_res["render"]

                        buffer_image[buffer_image < 0.5] = 0
                        buffer_image[buffer_image != 0] = 1

                        torchvision.utils.save_image(buffer_image.cpu(), os.path.join(pred_masks_path, '{0:05d}'.format(idx) + ".png"))
                    else:
                        # Empty mask
                        torchvision.utils.save_image(torch.zeros(3, view.image_height, view.image_width), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                        torchvision.utils.save_image(torch.zeros(3, view.image_height, view.image_width), os.path.join(pred_masks_path, '{0:05d}'.format(idx) + ".png"))
                
if __name__ == "__main__":
    parser = ArgumentParser(description="Render Cluster by Text")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--render_checkpoint", type=str, default=None)
    
    parser.add_argument('--text_prompt', type=str, default='Person')
    # Minimum unprojected mask-points for a cluster to be selected. Every cluster
    # over this is unioned into the mask, so the value has to sit clear of the
    # background clusters that the depth unprojection catches at the silhouette
    # edge. Measured on Neu3D: the object cluster holds 62-73k points in every
    # scene, stray clusters <=5k -- a wide empty gap. The old default of 4000 sat
    # in the noise and pulled a 4999-point background cluster into coffee_martini,
    # collapsing it to mIoU 0.18; at 20000 the same scene scores 0.91.
    parser.add_argument("--cls_point_threshold", default=20000, type=int)
    parser.add_argument("--score_threshold", default=0.95, type=float)  # cosine filter on the selected cluster
    # Default (off) renders the selected gaussians alone, giving the object's full
    # silhouette through any occluder. Turn on to composite them against the rest
    # of the scene instead, so the mask is only the *visible* extent -- which is
    # what an occlusion-aware GT expects.
    parser.add_argument("--occlusion_aware", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
        
    if args.render_checkpoint:
         cfgfilepath = os.path.join(args.render_checkpoint, "cfg_args")
         if os.path.exists(cfgfilepath):
             with open(cfgfilepath) as cfg_file:
                 cfgfile_string = cfg_file.read()
                 args_cfgfile = eval(cfgfile_string)
                 merged_dict = vars(args_cfgfile).copy()
                 for k, v in vars(args).items():
                     if v is not None:
                         merged_dict[k] = v
                 args = argparse.Namespace(**merged_dict)

    if not hasattr(args, 'source_path') or args.source_path is None:
        args.source_path = ""
    if not args.render_checkpoint:
        args.render_checkpoint = args.model_path
    args.model_path = args.render_checkpoint

    render_cluster_by_text(model.extract(args), hp.extract(args), args.iteration, pipeline.extract(args), op.extract(args), args)
