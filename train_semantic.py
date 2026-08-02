
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

import torch
import torch.nn as nn
import sys
from scene import GaussianModel, GaussianModel_dynamic, Scene2gs_mixed
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
import os
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import torchvision
import numpy as np
import pytorch3d.ops as ops # For KNN smoothing

# TRASE Utils
from utils.trase_utils import (
    get_sample_pixel_and_mask, get_pixel_weights, 
    get_pixel_mask_correspondence_matrix, get_features_correspondence_matrix,
    positive_pixel_pair_loss, negative_pixel_pair_loss
)
import psutil

def smooth_semantics(xyz, features, K=16, precomputed_idx=None):
    """
    Apply KNN smoothing to semantic features.
    Matches TRASE get_smoothed_gaussian_features.
    """
    if K <= 1:
        return features
        
    # Normalized features before smoothing (TRASE logic)
    # Note: TRASE normalizes inside get_smoothed_gaussian_features
    normed_features = torch.nn.functional.normalize(features, dim=-1, p=2)
    
    if precomputed_idx is not None:
        idx = precomputed_idx
    else:
        # KNN
        # xyz: (N, 3) -> (1, N, 3)
        with torch.no_grad():
            knn_res = ops.knn_points(xyz.unsqueeze(0), xyz.unsqueeze(0), K=K)
            idx = knn_res.idx.squeeze(0) # (N, K)
    
    # Gather neighbors
    # features: (N, C)
    # neighbor_feats: (N, K, C)
    # efficient gather? 
    # normed_features[idx] -> (N, K, C)
    neighbor_feats = normed_features[idx]
    
    # Mean
    
    # Dropout (TRASE logic: get_smoothed_gaussian_features)
    # Default dropout is 0.5 in TRASE rendering
    dropout = 0.5
    if dropout > 0 and dropout < 1:
        # Select random subset of neighbors
        # For efficiency, TRASE does: select_point = torch.randperm(K)[:int(K*dropout)]
        # This selects the SAME subset of neighbor indices for ALL points?
        # TRASE code:
        # select_point = torch.randperm(K)[ : int(K*dropout)]
        # select_idx = self.feature_smooth_map["m"][:, select_point]
        # So yes, it selects the same 'k-th' neighbors for all points. e.g. always the 1st, 3rd, 5th nearest neighbor.
        
        num_sel = int(K * dropout)
        if num_sel > 0:
            perm = torch.randperm(K, device=xyz.device)[:num_sel]
            neighbor_feats = neighbor_feats[:, perm, :]
    
    smoothed = neighbor_feats.mean(dim=1)
    return smoothed

def training(dataset, opt, pipe, hyper, args):
    first_iter = 0
    prepare_output_and_logger(dataset)  # side effects: output dir + cfg_args dump
    
    # Initialize Gaussian Models (fixed geometry)
    dataset.sh_degree = 3
    foreground_gaussians = GaussianModel_dynamic(dataset.sh_degree, hyper)
    background_gaussians = GaussianModel(dataset.sh_degree)
    
    # Load checkpoint
    if args.start_checkpoint:
        if os.path.isfile(args.start_checkpoint):
            print("Loading checkpoint from file:", args.start_checkpoint)
            (model_params, first_iter) = torch.load(args.start_checkpoint)
            foreground_gaussians.restore(model_params, opt)
        elif os.path.isdir(args.start_checkpoint):
            print("Checkpoint is a directory, deferring loading to Scene init:", args.start_checkpoint)
        else:
            print("Checkpoint path not found:", args.start_checkpoint)
    
    # Initialize Scene - No transient
    scene = Scene2gs_mixed(dataset, foreground_gaussians, load_iteration=args.start_checkpoint, gaussians_second=background_gaussians, gaussians_transient=None)

    # Freezing Geometry
    # (No FG/BG training_setup here: both models are frozen below and their
    #  optimizers were never stepped — only the semantic Adam below trains.)

    # Freeze Foreground Parameters
    for param_name in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
        if hasattr(foreground_gaussians, param_name):
            attr = getattr(foreground_gaussians, param_name)
            if isinstance(attr, torch.Tensor):
                attr.requires_grad = False
    
    # Freeze Deformation Network if it exists
    if hasattr(foreground_gaussians, "_deformation") and isinstance(foreground_gaussians._deformation, nn.Module):
        for param in foreground_gaussians._deformation.parameters():
            param.requires_grad = False

    # Freeze Background Parameters (Standard GaussianModel)
    for param_name in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
        if hasattr(background_gaussians, param_name):
            attr = getattr(background_gaussians, param_name)
            if isinstance(attr, torch.Tensor):
                attr.requires_grad = False

    # Semantic Setup (TRASE Mode)
    if args.mode == 'semantic':
        print("Initializing Semantic Features...")
        semantic_dim = args.semantic_dim
        
        # Initialize semantics (random init?)
        # TRASE uses standard parameter init. 
        # Foreground
        num_fg = foreground_gaussians.get_xyz.shape[0]
        # TRASE uses RGB2SH(torch.rand), effectively centered uniform scaled by 1/C0
        C0 = 0.28209479177387814
        foreground_gaussians.semantic_feature = nn.Parameter((torch.rand(num_fg, semantic_dim, device="cuda") - 0.5) / C0)
        
        # Background
        num_bg = background_gaussians.get_xyz.shape[0]
        background_gaussians.semantic_feature = nn.Parameter((torch.rand(num_bg, semantic_dim, device="cuda") - 0.5) / C0)
            
        # Optimizer
        l = [
            {'params': [foreground_gaussians.semantic_feature], 'lr': args.semantic_lr, "name": "semantic_fg"},
            {'params': [background_gaussians.semantic_feature], 'lr': args.semantic_lr, "name": "semantic_bg"},
        ]
        optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # NOTE: KNN for smoothing is NOT precomputed: smooth_semantics runs on the
    # concatenated deformed-FG + static-BG means3D, and the FG positions depend
    # on viewpoint_cam.time, so neighbor indices vary across iterations.

    for iteration in range(first_iter, opt.iterations + 1):
        # TIMERS SETUP — only when the %100 timing print below will fire
        do_timing = (iteration % 100 == 0)
        if do_timing:
            render_start = torch.cuda.Event(enable_timing=True)
            render_end = torch.cuda.Event(enable_timing=True)
            dataload_start = torch.cuda.Event(enable_timing=True)
            dataload_end = torch.cuda.Event(enable_timing=True)
            sampling_start = torch.cuda.Event(enable_timing=True)
            sampling_end = torch.cuda.Event(enable_timing=True)
            corr_start = torch.cuda.Event(enable_timing=True)
            corr_end = torch.cuda.Event(enable_timing=True)
            loss_start = torch.cuda.Event(enable_timing=True)
            loss_end = torch.cuda.Event(enable_timing=True)
            back_start = torch.cuda.Event(enable_timing=True)
            back_end = torch.cuda.Event(enable_timing=True)

        # Pick a random Viewpoint
        if not viewpoint_stack:
            viewpoint_stack = list(scene.getTrainCameras())
        viewpoint_cam = viewpoint_stack.pop(torch.randint(0, len(viewpoint_stack), (1,)).item())

        # Semantic Render
        if do_timing:
            render_start.record()
        render_pkg = render_semantic(
            viewpoint_cam, foreground_gaussians, background_gaussians, pipe, background, args
        )
        rendered_features = render_pkg["render"]
        if do_timing:
            render_end.record()
            dataload_start.record()

        # Load Masks logic (TRASE style)
        sam_masks = None
        if True:
            # 1. Parsing Image Path Info (e.g. .../dynerf/sear_steak/cam01/images/0001.png)
            img_basename = os.path.basename(viewpoint_cam.image_path) # 0001.png
            img_name_no_ext = os.path.splitext(img_basename)[0] # 0001
            
            parent_dir = os.path.dirname(viewpoint_cam.image_path) # .../cam01/images
            grandparent_dir = os.path.dirname(parent_dir) # .../cam01
            cam_name = os.path.basename(grandparent_dir) # cam01 ("cam01" from folder name)

            # 2. Strategy A: TRASE centralized mask root (--trase_root):
            # <trase_root>/{scene}/masks/camXX_XXXX.pt, scene inferred from source_path.
            if args.source_path:
                exp_name_from_path = os.path.basename(os.path.normpath(args.source_path)) # sear_steak
            else:
                exp_name_from_path = "sear_steak" # Fallback/Default

            trase_root = args.trase_root
            # Construct filename: cam01_0001.pt
            trase_filename = f"{cam_name}_{img_name_no_ext}.pt"
            trase_mask_path = os.path.join(trase_root, exp_name_from_path, "masks", trase_filename)
            
            # 3. Strategy B: Local Parallel Folder (User Snippet)
            # Path: .../cam01/masks/0001.pt
            local_mask_dir = os.path.join(grandparent_dir, "masks")
            local_mask_path = os.path.join(local_mask_dir, f"{img_name_no_ext}.pt")

            mask_path = None
            if os.path.exists(trase_mask_path):
                mask_path = trase_mask_path
                # print(f"Found mask at TRASE path: {mask_path}")
            elif os.path.exists(local_mask_path):
                mask_path = local_mask_path
                # print(f"Found mask at local path: {mask_path}")
            
            if mask_path and os.path.exists(mask_path):
                 try:
                     masks_pkg = torch.load(mask_path, weights_only=False)
                 except TypeError:
                      # If weights_only is not supported (older pytorch), try without
                     masks_pkg = torch.load(mask_path)
                 
                 # TRASE stores as dict: {'masks': ..., 'N': ..., 'H': ..., 'W': ...}
                 if isinstance(masks_pkg, dict) and 'masks' in masks_pkg:
                     sam_masks = torch.from_numpy(np.array(masks_pkg['masks'].tolist())).reshape(masks_pkg['N'], masks_pkg['H'], masks_pkg['W']).cuda()
                 else:
                     # Direct tensor?
                     sam_masks = masks_pkg.cuda()
            else:
                 # Debug print only occasionally to avoid spam
                 if iteration % 100 == 0:
                    print(f"Mask not found. Checked:\n  {trase_mask_path}\n  {local_mask_path}")
        if do_timing:
            dataload_end.record()
        
        loss = 0.0
        
        if sam_masks is not None:
             # TRASE Sampling and Contrastive Loss
             # 1. Sample Pixels
             if do_timing:
                 sampling_start.record()
             num_sampled_pixels = args.num_sampled_pixels # e.g. 5000
             num_sampled_masks = args.num_sampled_masks # e.g. 50, but typically constrained by mask count
             
             sampled_pixel, sampled_mask = get_sample_pixel_and_mask(sam_masks, num_sampled_pixels, num_sampled_masks)
             if do_timing:
                 sampling_end.record()
             
             # 2. Regularization Loss for Feature Norm
             rendered_feature_norm = rendered_features.norm(dim=0, p=2).mean()
             rendered_feature_norm_reg = (1 - rendered_feature_norm) ** 2
             
             # 3. Correspondence Matrices
             if do_timing:
                 corr_start.record()
             # Interpolate features to mask size if needed (usually same size)
             if rendered_features.shape[1] != sam_masks.shape[1] or rendered_features.shape[2] != sam_masks.shape[2]:
                  rendered_features = torch.nn.functional.interpolate(rendered_features.unsqueeze(0), sam_masks.shape[-2:], mode='bilinear').squeeze(0)
             
             C_matrix = get_pixel_mask_correspondence_matrix(sam_masks, sampled_pixel, sampled_mask)
             C_F_matrix = get_features_correspondence_matrix(rendered_features, sampled_pixel)
             
             pixel_weights = get_pixel_weights(sam_masks, sampled_pixel)
             if do_timing:
                 corr_end.record()
             
             # 4. Contrastive Loss — paper Sec 3.8:  L_sem = L_pos + L_neg
             #    (soft-mined contrastive objective on the rendered semantic map
             #     S-hat vs the 2D SAM masks M_sam). `rfn` adds a feature-norm
             #     regularizer on top of the paper's two-term L_sem.
             # opt.contrastive_mode default 'soft'
             # opt.rfn default 1.0

             if do_timing:
                 loss_start.record()
             if args.contrastive_mode == 'all':
                 # 'all' mode has no positive/negative thresholds by design
                 L_pos = positive_pixel_pair_loss['all'](C=C_matrix, C_F=C_F_matrix, weights=pixel_weights)
                 L_neg = negative_pixel_pair_loss['all'](C=C_matrix, C_F=C_F_matrix, weights=pixel_weights)
             else:
                 L_pos = positive_pixel_pair_loss[args.contrastive_mode](
                     C=C_matrix, C_F=C_F_matrix, positive_th=args.hard_positive_th, weights=pixel_weights
                 )
                 L_neg = negative_pixel_pair_loss[args.contrastive_mode](
                     C=C_matrix, C_F=C_F_matrix, negative_th=args.hard_negative_th, weights=pixel_weights
                 )

             L_sem = L_pos + L_neg + args.rfn * rendered_feature_norm_reg
             loss = L_sem   # generic handle used by backward()/EMA logging below
             if do_timing:
                 loss_end.record()

        else:
             print("No masks found, skipping loss")
             # Dummy records to avoid errors
             if do_timing:
                 sampling_start.record(); sampling_end.record()
                 corr_start.record(); corr_end.record()
                 loss_start.record(); loss_end.record()

        # BACKWARD
        if do_timing:
            back_start.record()
        if not isinstance(loss, float):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
        if do_timing:
            back_end.record()

        torch.cuda.synchronize()

        with torch.no_grad():
            if not isinstance(loss, float):
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                # Calculate GB values
                cuda_peak = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
                cuda_curr = torch.cuda.memory_allocated() / (1024.0 ** 3)
                cpu_mem = psutil.Process(os.getpid()).memory_info().rss  / (1024.0 ** 3)

                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "GPU_Peak": f"{cuda_peak:.2f}GB",
                    "GPU_Curr": f"{cuda_curr:.2f}GB",
                    "RAM": f"{cpu_mem:.2f}GB"
                })
                progress_bar.update(10)

            # if iteration % 10 == 0:
            #     progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            #     progress_bar.update(10)

            if iteration % 2000 == 0:
                print(f"\n[Iter {iteration}] Saving semantic checkpoint...")
                ckpt_dir = os.path.join(args.saving_folder, args.expname, f"semantic_iteration_{iteration}")
                os.makedirs(ckpt_dir, exist_ok=True)
                
                fg_ckpt_path = os.path.join(ckpt_dir, "foreground_semantics.pt")
                bg_ckpt_path = os.path.join(ckpt_dir, "background_semantics.pt")
                
                torch.save(foreground_gaussians.semantic_feature.data, fg_ckpt_path)
                torch.save(background_gaussians.semantic_feature.data, bg_ckpt_path)
                print(f"Saved checkpoint to {ckpt_dir}")

            if iteration == opt.iterations:
                progress_bar.close()

                # Save optimized semantic features (Restored)
                print("\nTraining complete. Saving semantic features...")
                save_dir = os.path.join(args.saving_folder, args.expname)
                os.makedirs(save_dir, exist_ok=True)
                
                fg_path = os.path.join(save_dir, "foreground_semantics.pt")
                bg_path = os.path.join(save_dir, "background_semantics.pt")
                
                torch.save(foreground_gaussians.semantic_feature.data, fg_path)
                torch.save(background_gaussians.semantic_feature.data, bg_path)
                print(f"Saved foreground features to {fg_path}")
                print(f"Saved background features to {bg_path}")


            if do_timing:
                print(f"[Iter {iteration}] Timings (ms): "
                      f"Render={render_start.elapsed_time(render_end):.2f}, "
                      f"Load={dataload_start.elapsed_time(dataload_end):.2f}, "
                      f"Sample={sampling_start.elapsed_time(sampling_end):.2f}, "
                      f"Corr={corr_start.elapsed_time(corr_end):.2f}, "
                      f"Loss={loss_start.elapsed_time(loss_end):.2f}, "
                      f"Back={back_start.elapsed_time(back_end):.2f}")
                dump_path = os.path.join(args.saving_folder, args.expname, "semantic_vis")
                os.makedirs(dump_path, exist_ok=True)
                # Visualizing first 3 channels normalized
                vis_image = rendered_features[:3].detach().cpu()
                # Normalize for vis?
                vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-9)
                torchvision.utils.save_image(vis_image, os.path.join(dump_path, f"{iteration:05d}.png"))
                # print(f"Iteration {iteration}: Loss {loss.item()}")

def render_semantic(viewpoint_camera, pc_foreground, pc_background, pipe, bg_color, args):
    # render_direct Style Rendering for Semantics    
    
    # 1. Deform Foreground
    time = torch.tensor(viewpoint_camera.time).to(pc_foreground.get_xyz.device).repeat(pc_foreground.get_xyz.shape[0], 1)
    
    means3D_fg, scales_fg, rotations_fg, opacity_fg, _ = pc_foreground._deformation(
        pc_foreground.get_xyz, pc_foreground._scaling, pc_foreground._rotation, 
        pc_foreground._opacity, pc_foreground.get_features, time
    )
    
    # Activations
    opacity_fg = pc_foreground.opacity_activation(opacity_fg)
    scales_fg = pc_foreground.scaling_activation(scales_fg)
    rotations_fg = pc_foreground.rotation_activation(rotations_fg)
    
    # 2. Prepare Background
    means3D_bg = pc_background.get_xyz
    opacity_bg = pc_background.opacity_activation(pc_background._opacity)
    scales_bg = pc_background.scaling_activation(pc_background._scaling)
    rotations_bg = pc_background.rotation_activation(pc_background._rotation)
    
    # 3. Concatenate
    means3D = torch.cat([means3D_fg, means3D_bg], dim=0)
    opacity = torch.cat([opacity_fg, opacity_bg], dim=0)
    scales = torch.cat([scales_fg, scales_bg], dim=0)
    rotations = torch.cat([rotations_fg, rotations_bg], dim=0)
    
    
    sem_fg = pc_foreground.semantic_feature
    sem_bg = pc_background.semantic_feature
    
    semantics = torch.cat([sem_fg, sem_bg], dim=0)
    
    # Apply Smoothing if requested (TRASE logic) - NOW ON COMBINED & DEFORMED GEOMETRY
    if args.smooth_K > 1:
        # Use deformed means3D for smoothing
        semantics = smooth_semantics(means3D, semantics, K=args.smooth_K, precomputed_idx=None)

    
    # --- TRASE Feature Normalization (Before Rasterization) ---
    # In TRASE render(): sh_objs = sh_objs / (sh_objs.norm(dim=2, keepdim=True) + 1e-9)
    # Our `semantics` shape: (N, C)
    semantics = semantics / (semantics.norm(dim=1, keepdim=True) + 1e-9)
    
    # 4. Rasterizer Setup
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
        sh_degree=0, 
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
        bwd_depth=False,
        bwd_dynamic=True
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=False, device="cuda")
    
    # 5. Multi-pass Rendering
    semantic_dim = semantics.shape[1]
    rendered_semantics = []
    
    for i in range(0, semantic_dim, 3):
        chunk = semantics[:, i:min(i+3, semantic_dim)]
        
        if chunk.shape[1] < 3:
            pad = torch.zeros(chunk.shape[0], 3 - chunk.shape[1], device="cuda")
            colors_precomp = torch.cat([chunk, pad], dim=1)
        else:
            colors_precomp = chunk
            

        dynamics = torch.ones((means2D.shape[0], 1, means2D.shape[1]), dtype=means2D.dtype,
                              device=means2D.device)  # * 0.5
        rendered_image, radii, depth, dynamics_map, max_weight_t = rasterizer(
            # rendered_image, radii, depth = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            dynamics=dynamics,
            cov3D_precomp=None)  # <-- NEW PARAMETER HERE )
        valid_ch = min(3, semantic_dim - i)
        rendered_semantics.append(rendered_image[:valid_ch])
        
    final_image = torch.cat(rendered_semantics, dim=0)
    
    return {"render": final_image}


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

if __name__ == "__main__":
    parser = ArgumentParser(description="Semantic Training Script")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument('--mode', default='semantic', type=str)
    parser.add_argument('--trase_root', type=str, default="./data/TRASE/neu3d",
                        help="Root of the TRASE centralized SAM-mask store: <trase_root>/<scene>/masks/camXX_XXXX.pt")
    parser.add_argument('--start_checkpoint', type=str, default = None)
    parser.add_argument('--semantic_dim', type=int, default=16)
    parser.add_argument('--semantic_lr', type=float, default=0.0025)
    
    # TRASE Compatibility
    parser.add_argument('--contrastive_mode', type=str, default='soft')
    parser.add_argument('--num_sampled_masks', type=int, default=50)
    parser.add_argument('--num_sampled_pixels', type=int, default=10000)
    parser.add_argument('--smooth_K', type=int, default=16)
    parser.add_argument('--hard_positive_th', type=float, default=0.75)
    parser.add_argument('--hard_negative_th', type=float, default=0.5)
    parser.add_argument('--rfn', type=float, default=1.0)
    
    # Arguments for compatibility with specific user workflow (matches render_3pc.py)
    parser.add_argument('--expname', type=str, default="")
    parser.add_argument('--configs', type=str, default="") 
    parser.add_argument('--render_checkpoint', type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    # MMCV Config Parsing Logic
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    # Compat logic: If start_checkpoint is not set, use render_checkpoint
    if not args.start_checkpoint and args.render_checkpoint:
        args.start_checkpoint = args.render_checkpoint

    # The training loop's optimizer only exists in semantic mode; fail fast otherwise.
    assert args.mode == 'semantic', \
        f"train_semantic.py only supports --mode semantic (got '{args.mode}')"

    training(lp.extract(args), op.extract(args), pp.extract(args), hp.extract(args), args)
