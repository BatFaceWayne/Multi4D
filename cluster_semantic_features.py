
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


def render_rgb(viewpoint_camera, pc_foreground, pc_background, pipe, bg_color, mask_fg=None, mask_bg=None):
    """
    Render RGB image with optional masks for foreground and background.
    """
    # 1. Deform Foreground
    # We assume 'time' is needed if deformation exists
    time = torch.tensor(viewpoint_camera.time).to(pc_foreground.get_xyz.device).repeat(pc_foreground.get_xyz.shape[0], 1)
    
    means3D_fg, scales_fg, rotations_fg, opacity_fg, shs_fg = pc_foreground._deformation(
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
    shs_bg = pc_background.get_features
    
    # 3. Apply Masks if provided
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
        
    # 4. Concatenate
    means3D = torch.cat([means3D_fg, means3D_bg], dim=0)
    opacity = torch.cat([opacity_fg, opacity_bg], dim=0)
    scales = torch.cat([scales_fg, scales_bg], dim=0)
    rotations = torch.cat([rotations_fg, rotations_bg], dim=0)
    # FG carries an extra dynamic SH channel (17 vs BG's 16); drop it for the static composite RGB render
    if shs_fg.shape[1] != shs_bg.shape[1]:
        shs_fg = shs_fg[:, :shs_bg.shape[1], :]
    shs = torch.cat([shs_fg, shs_bg], dim=0)
    
    # 5. Rasterizer Setup
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

    # 6. Render
    rendered_image, radii, depth, dynamics_map, max_weight_t = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        dynamics=dynamics,
        cov3D_precomp=None)
        
    return rendered_image

def cluster_semantic_features(dataset, hypernetwork, iteration, pipeline, opt, args):
    with torch.no_grad():
        dataset.sh_degree = 3
        foreground_gaussians = GaussianModel_dynamic(dataset.sh_degree, hypernetwork)
        background_gaussians = GaussianModel(dataset.sh_degree)
        
        # Initialize Scene (Loads geometry from checkpoints)
        custom_checkpoint_path = None
        if args.render_checkpoint and os.path.isdir(args.render_checkpoint):
             custom_checkpoint_path = args.render_checkpoint
             scene_load_path = None 
        else:
             scene_load_path = args.render_checkpoint

        # We load the scene to ensure geometry is correct
        scene = Scene2gs_mixed(dataset, foreground_gaussians, load_iteration=scene_load_path, gaussians_second=background_gaussians, gaussians_transient=None)

        # Iteration the geometry was actually loaded from (used for the output dir,
        # so we never fabricate point_cloud/iteration_-1 from the CLI default).
        loaded_iter = scene.loaded_iter

        # Manual Loading for Directory Checkpoints if Scene didn't handle it fully or for custom path
        if custom_checkpoint_path:
            print(f"Processing directory checkpoint: {custom_checkpoint_path}")
            pc_dir = os.path.join(custom_checkpoint_path, "point_cloud")
            if os.path.exists(pc_dir):
                iters = [int(fol.split("_")[-1]) for fol in os.listdir(pc_dir) if "iteration_" in fol]
                if iters:
                    max_iter = max(iters)
                    loaded_iter = max_iter
                    load_dir = os.path.join(pc_dir, f"iteration_{max_iter}")
                    print(f"Loading geometry/deformation from {load_dir}")
                    
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
        print(f"Loading Semantic Features from {semantic_load_dir}...")
        
        # Prefer the final features (saved at the run root at end of training); fall back
        # to the newest periodic semantic_iteration_* checkpoint.
        sem_dir = semantic_load_dir
        if not os.path.exists(os.path.join(sem_dir, "foreground_semantics.pt")):
            ckpts = sorted([d for d in os.listdir(semantic_load_dir) if d.startswith("semantic_iteration_")],
                           key=lambda d: int(d.split("_")[-1])) if os.path.isdir(semantic_load_dir) else []
            if ckpts:
                sem_dir = os.path.join(semantic_load_dir, ckpts[-1])
                print(f"Final features not found; using latest checkpoint {ckpts[-1]}")
        fg_sem_path = os.path.join(sem_dir, "foreground_semantics.pt")
        bg_sem_path = os.path.join(sem_dir, "background_semantics.pt")
        
        has_fg = False
        has_bg = False
        
        if os.path.exists(fg_sem_path):
            foreground_gaussians.semantic_feature = torch.load(fg_sem_path, map_location="cuda")
            print("Loaded FG Semantics: ", foreground_gaussians.semantic_feature.shape)
            has_fg = True
        else:
            print(f"WARNING: FG Semantics not found at {fg_sem_path}")
            
        if os.path.exists(bg_sem_path):
            background_gaussians.semantic_feature = torch.load(bg_sem_path, map_location="cuda")
            print("Loaded BG Semantics: ", background_gaussians.semantic_feature.shape)
            has_bg = True
        else:
             print(f"WARNING: BG Semantics not found at {bg_sem_path}")

        if not has_fg and not has_bg:
            print("No semantic features found. Exiting.")
            return

        # Prepare Features for Clustering (empty side inherits the feature dim of the present side)
        feat_dim = (foreground_gaussians.semantic_feature if has_fg else background_gaussians.semantic_feature).shape[-1]
        fg_feats = foreground_gaussians.semantic_feature if has_fg else torch.empty(0, feat_dim, device="cuda")
        bg_feats = background_gaussians.semantic_feature if has_bg else torch.empty(0, feat_dim, device="cuda")
        
        # Ensure dimensions match (they should be N x FeatureDim)
        # render_semantic_features says they are saved as parameters, so likely already N x C or N x C x 1
        # gui_standalone.py DBSCAN usage:
        # point_features = self.gaussians.get_gaussian_features.squeeze(1)
        # normed_point_features = torch.nn.functional.normalize(point_features, dim = -1, p = 2)
        
        if has_fg and len(fg_feats.shape) == 3:
             fg_feats = fg_feats.squeeze(1)
        if has_bg and len(bg_feats.shape) == 3:
             bg_feats = bg_feats.squeeze(1)

        print(f"FG Feats Shape: {fg_feats.shape}")
        print(f"BG Feats Shape: {bg_feats.shape}")

        # Concatenate Features
        all_feats = torch.cat([fg_feats, bg_feats], dim=0)
        print(f"Combined Feats Shape: {all_feats.shape}")
        
        # DBSCAN Clustering
        print("Running DBSCAN Clustering on Combined Features...")
        
        # Parameters (from gui_standalone.py or custom)
        percent = 0.02 # Sample 2% for training
        
        normed_point_features = torch.nn.functional.normalize(all_feats, dim = -1, p = 2)
        
        # Sampling
        num_points = all_feats.shape[0]
        mask = torch.rand(num_points) > (1 - percent)
        sampled_point_features = all_feats[mask]
        normed_sampled_point_features = sampled_point_features / torch.norm(sampled_point_features, dim = -1, keepdim = True)
        
        print(f"Training HDBSCAN on {sampled_point_features.shape[0]} sampled points...")
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01, allow_single_cluster = False, core_dist_n_jobs=multiprocessing.cpu_count())
        
        cluster_labels = clusterer.fit_predict(normed_sampled_point_features.detach().cpu().numpy())
        
        num_clusters = len(np.unique(cluster_labels))
        print(f"Found {num_clusters} clusters (including noise).")

        # Build one center per real cluster label (>= 0), indexed by label value.
        # HDBSCAN labels are contiguous 0..K-1 plus optional -1 noise; the noise
        # label gets no center row, so it never competes in the argmax below and
        # noise points are absorbed by the nearest real cluster.
        valid_labels = [l for l in sorted(np.unique(cluster_labels)) if l >= 0]
        num_centers = (max(valid_labels) + 1) if valid_labels else 1  # 1 zero-row fallback if all noise
        cluster_centers = torch.zeros(num_centers, normed_sampled_point_features.shape[-1])
        for lbl in valid_labels:
            mask_c = (cluster_labels == lbl)
            cluster_centers[lbl] = torch.nn.functional.normalize(
                normed_sampled_point_features[mask_c].mean(dim=0), dim=-1
            )

        seg_score = torch.einsum('nc,bc->bn', cluster_centers.cpu(), normed_point_features.cpu())

        # Determine IDs via argmax; the center row index IS the hdbscan label.
        cluster_ids = seg_score.argmax(dim=-1).numpy()

        print("Cluster assignment done.")
        print("Unique assigned IDs (Indices into centers):", np.unique(cluster_ids))

        
        # Split IDs to FG/BG for rendering
        num_fg = fg_feats.shape[0]
        ids_fg = cluster_ids[:num_fg] if has_fg else np.array([])
        ids_bg = cluster_ids[num_fg:] if has_bg else np.array([])
        
        # --- RENDERING ---
        print("\nStarting Cluster Rendering...")
        
        test_cameras = scene.getTestCameras()
        if not test_cameras:
            print("No test cameras found!")
        else:
            # First and Last Frame
            cam_list = [test_cameras[0], test_cameras[-1]]
            frame_indices = [0, len(test_cameras)-1]
            
            bg_color = [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            
            output_dir = os.path.join(args.saving_folder, args.expname, "render_cluster")
            makedirs(output_dir, exist_ok=True)
            
            unique_ids = np.unique(cluster_ids)
            print(f"Rendering {len(unique_ids)} clusters for {len(cam_list)} views...")
            
            for cam_idx, (cam, frame_num) in enumerate(zip(cam_list, frame_indices)):
                frame_dir = os.path.join(output_dir, f"frame_{frame_num:03d}")
                makedirs(frame_dir, exist_ok=True)
                print(f"Rendering Frame {frame_num:03d}...")
                
                for cid in unique_ids:
                    # Create Masks
                    mask_fg = torch.tensor(ids_fg == cid, device="cuda") if has_fg else None
                    mask_bg = torch.tensor(ids_bg == cid, device="cuda") if has_bg else None
                    
                    # Optimization: Skip if cluster has < X points?
                    cnt = 0
                    if mask_fg is not None: cnt += mask_fg.sum().item()
                    if mask_bg is not None: cnt += mask_bg.sum().item()
                    
                    if cnt == 0:
                        continue
                        
                    # Render
                    image = render_rgb(cam, foreground_gaussians, background_gaussians, pipeline, background, mask_fg=mask_fg, mask_bg=mask_bg)
                    
                    # Save
                    save_path = os.path.join(frame_dir, f"cluster_{cid:03d}.jpg")
                    torchvision.utils.save_image(image, save_path)
                    
            print(f"Rendering done. Saved to {output_dir}")
        
        # Generate Colors
        label_to_color = np.random.rand(num_clusters + 1, 3) # +1 just in case
        cluster_colors = torch.from_numpy(label_to_color[cluster_ids]).float().cuda()
        
        clusters_fg = None
        clusters_bg = None
        
        # Derive from the actually-loaded iteration; fall back to the CLI value only
        # if no checkpoint iteration was resolved (degenerate no-geometry case).
        out_iter = loaded_iter if loaded_iter is not None else iteration
        output_dir_gs = os.path.join(dataset.model_path, f"point_cloud/iteration_{out_iter}")
        makedirs(output_dir_gs, exist_ok=True)
        
        if has_fg:
            cols_fg = cluster_colors[:num_fg]
            clusters_fg = {"id": torch.from_numpy(ids_fg).unsqueeze(-1).float().cuda(), "rgb": cols_fg}
            
            # Save FG
            save_path = os.path.join(output_dir_gs, "clusters_fg.pt")
            torch.save(clusters_fg, save_path)
            print(f"Saved FG clusters to {save_path}")
            
        if has_bg:
            cols_bg = cluster_colors[num_fg:]
            clusters_bg = {"id": torch.from_numpy(ids_bg).unsqueeze(-1).float().cuda(), "rgb": cols_bg}
            
            # Save BG
            save_path = os.path.join(output_dir_gs, "clusters_bg.pt")
            torch.save(clusters_bg, save_path)
            print(f"Saved BG clusters to {save_path}")
            
        # Save Combined
        clusters_combined = {"id": torch.from_numpy(cluster_ids).unsqueeze(-1).float().cuda(), "rgb": cluster_colors}
        save_path_comb = os.path.join(output_dir_gs, "clusters_combined.pt")
        torch.save(clusters_combined, save_path_comb)
        print(f"Saved Combined clusters to {save_path_comb}")
        
        print("Done.")

if __name__ == "__main__":
    parser = ArgumentParser(description="Cluster Semantic Features")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    
    # Semantic Args
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--render_checkpoint", type=str, default=None)


    args = parser.parse_args(sys.argv[1:])
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
        
    print("Semantic Clustering script started...")

    # Handle model_path / render_checkpoint logic for get_combined_args compatibility
    # If render_checkpoint is set but model_path is not, use it to find cfg_args
    
    if args.render_checkpoint:
         cfgfilepath = os.path.join(args.render_checkpoint, "cfg_args")
         if os.path.exists(cfgfilepath):
             print(f"Loading config from {cfgfilepath}")
             with open(cfgfilepath) as cfg_file:
                 cfgfile_string = cfg_file.read()
                 args_cfgfile = eval(cfgfile_string)
                 
                 # Merge: cmdline args override config args
                 merged_dict = vars(args_cfgfile).copy()
                 for k, v in vars(args).items():
                     if v is not None:
                         merged_dict[k] = v
                 args = argparse.Namespace(**merged_dict)
         else:
             print("Config file cfg_args not found in checkpoint.")

    # Fix validation for source_path
    if not hasattr(args, 'source_path') or args.source_path is None:
        args.source_path = ""
    if not args.render_checkpoint:
        args.render_checkpoint = args.model_path
    args.model_path = args.render_checkpoint

    cluster_semantic_features(model.extract(args), hp.extract(args), args.iteration, pipeline.extract(args), op.extract(args), args)
