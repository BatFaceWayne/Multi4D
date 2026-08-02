
import torch
from scene import Scene2gs_mixed
import os
import sys
from os import makedirs
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from scene.gaussian_model import GaussianModel, GaussianModel_dynamic
from torch import nn
from train_semantic import render_semantic  # Reuse from training script

def render_semantic_set(model_path, name, iteration, views, foreground_gaussians, background_gaussians, pipeline, background, args):
    # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "semantic_renders")
    render_path = os.path.join(args.saving_folder, args.expname,  name, "ours_{}".format(iteration), "semantic_renders")

    makedirs(render_path, exist_ok=True)

    print("Rendering Semantic {} set with {} views".format(name, len(views)))

    # (KNN smoothing happens inside render_semantic on the combined deformed cloud.)
    for idx, view in enumerate(views):
        with torch.no_grad():
            render_pkg = render_semantic(
                view, foreground_gaussians, background_gaussians, pipeline, background, args
            )
            
            ren = render_pkg["render"]
            
            # Visualize: Standard is PCA or just first 3 channels normalized
            # Here we just save the first 3 channels normalized as a quick check
            # For real usage, user might want the full tensor (saved via torch.save?)
            # But render_set usually saves images.
            
            vis_image = ren[:3].detach().cpu()
            vis_image = (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-9)
            
            torchvision.utils.save_image(vis_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset, hypernetwork, iteration, pipeline, opt, skip_train, skip_test, load_path, args):
    with torch.no_grad():
        dataset.sh_degree = 3
        foreground_gaussians = GaussianModel_dynamic(dataset.sh_degree, hypernetwork)
        background_gaussians = GaussianModel(dataset.sh_degree)
        
        # Initialize Scene (Loads geometry from checkpoints)
        # Note: We need to handle the directory checkpoint logic here too if Scene doesn't
        # But for rendering, we expect a valid path or we use the custom logic we added to train_semantic.py
        # Actually simplest to duplicate the custom load logic.
        
        custom_checkpoint_path = None
        if load_path and os.path.isdir(load_path):
             custom_checkpoint_path = load_path
             scene_load_path = None # Let Scene init from colmap/random
        else:
             scene_load_path = load_path

        scene = Scene2gs_mixed(dataset, foreground_gaussians, load_iteration=scene_load_path, gaussians_second=background_gaussians, gaussians_transient=None)
        
        # Manual Loading for Directory Checkpoints (if needed)
        if custom_checkpoint_path:
            print(f"Processing directory checkpoint: {custom_checkpoint_path}")
            pc_dir = os.path.join(custom_checkpoint_path, "point_cloud")
            if os.path.exists(pc_dir):
                iters = [int(fol.split("_")[-1]) for fol in os.listdir(pc_dir) if "iteration_" in fol]
                if iters:
                    max_iter = max(iters)
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

        # Load the semantic features from where train_semantic.py saved them:
        # args.saving_folder/args.expname/{foreground,background}_semantics.pt
        
        semantic_load_dir = os.path.join(args.saving_folder, args.expname)
        print(f"Loading Semantic Features from {semantic_load_dir}...")
        
        fg_sem_path = os.path.join(semantic_load_dir, "foreground_semantics.pt")
        bg_sem_path = os.path.join(semantic_load_dir, "background_semantics.pt")
        
        if os.path.exists(fg_sem_path):
            foreground_gaussians.semantic_feature = nn.Parameter(torch.load(fg_sem_path, map_location="cuda"))
            print("Loaded FG Semantics")
        else:
            print(f"WARNING: FG Semantics not found at {fg_sem_path}")
            
        if os.path.exists(bg_sem_path):
            background_gaussians.semantic_feature = nn.Parameter(torch.load(bg_sem_path, map_location="cuda"))
            print("Loaded BG Semantics")
        else:
             print(f"WARNING: BG Semantics not found at {bg_sem_path}")

        bg_color = [1,1,1] if getattr(dataset, 'white_background', False) else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_semantic_set(dataset.model_path, "train", scene.loaded_iter or 0, scene.getTrainCameras(), foreground_gaussians, background_gaussians, pipeline, background, args)

        if not skip_test:
             render_semantic_set(dataset.model_path, "test", scene.loaded_iter or 0, scene.getTestCameras(), foreground_gaussians, background_gaussians, pipeline, background, args)

if __name__ == "__main__":
    parser = ArgumentParser(description="Semantic Rendering Script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    hp = ModelHiddenParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    # Semantic Args
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--render_checkpoint", type=str, default=None)
    parser.add_argument('--smooth_K', type=int, default=0) # 0 to disable override or match training

    args = parser.parse_args(sys.argv[1:])
    
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
        
    print("Semantic Rendering script started...")
    
    # Ensure smooth_K is set if not passed (default from config might be used)
    if not hasattr(args, 'smooth_K'):
        args.smooth_K = 0
    args.model_path = args.render_checkpoint

    render_sets(model.extract(args), hp.extract(args), args.iteration, pipeline.extract(args), op.extract(args), args.skip_train, args.skip_test, args.render_checkpoint, args)
