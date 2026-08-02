"""Evaluation, final test dump, and training-progress visualisation.

Split out of train.py (2026-07-30): these produce metrics and images from a
training state and are called from exactly one place each in the training loop.
Reported metrics are PSNR + both SSIM conventions (ssim1 = GraphDECO/torch,
ssim2 = skimage) + both LPIPS nets. DSSIM_k = (1 - SSIM_k) / 2 if you need it.
"""
import os
import json
import numpy as np
import torch
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import lpips
from skimage.metrics import structural_similarity as sk_ssim

from utils.loss_utils import ssim as gs_ssim
from utils.image_utils import psnr as gs_psnr
from gaussian_renderer import render_background, render_foreground, render_mask, render_rawall3pc


_lpips_cache = {}


def _get_lpips(net):
    if net not in _lpips_cache:
        _lpips_cache[net] = lpips.LPIPS(net=net).cuda()
    return _lpips_cache[net]


def run_evaluation(
    iteration, scene, foreground_gaussians, background_gaussians, transient_gaussian, pipeline_config,
    optimization_params, expname, stage, current_best_psnr=-1.0
):
    print(f"\n[ITER {iteration}] Running Evaluation...")

    # --- Match OPPO/GraphDECO metric code style ---


    test_cams = scene.getTestCameras()

    # Setup cameras for evaluation

    viewpoint_stack = [i for i in test_cams]

    # --- Metrics accumulators ---
    psnr_list = []
    ssim1_list = []    # GraphDECO/torch SSIM  (the SSIM1 convention)
    ssim2_list = []    # skimage SSIM         (the SSIM2 convention)      # GraphDECO SSIM
    lpips_alex_list = []
    lpips_vgg_list = []
    # render_3d (persistent: FG+BG only, no TR) — diagnostic only
    # Per-frame records (dumped to iteration_NNNN/per_frame_metrics.json
    # ONLY if this iter is selected for image-saving).
    per_frame_records = []

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    lpips_alex_model = _get_lpips('alex')
    lpips_vgg_model = _get_lpips('vgg')

    # --- PASS 1: METRICS ONLY (match OPPO / 3DGS style) ---
    with torch.no_grad():
        for viewpoint_cam in tqdm(viewpoint_stack, desc="Eval metrics"):
            # Render Full Scene
            render_pkg = render_rawall3pc(
                viewpoint_cam, foreground_gaussians, background_gaussians, transient_gaussian,
                pipeline_config,
                bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
                stage=stage
            )
            rendering = render_pkg["render"]

            # --- Clamp rendering to [0,1] before metrics, as in OPPO code ---
            rendering = torch.clamp(rendering, 0.0, 1.0)

            # OPPO/3DGS metrics use RGB only; training cameras store 0..255
            gt = (
                viewpoint_cam.original_image[0:3, :, :]  # CxHxW
                .cuda()
                .float()
                / 255.0
            )
            gt = torch.clamp(gt, 0.0, 1.0)

            # --- GraphDECO metrics expect NCHW ---
            r_b = rendering.unsqueeze(0)   # 1xCxHxW
            g_b = gt.unsqueeze(0)

            psnr_val = gs_psnr(r_b, g_b).mean().item()
            ssim1_val = gs_ssim(r_b, g_b).mean().item()

            # SSIM2: skimage on HxWxC numpy in [0,1] (the other convention used in
            # the literature; report both so numbers are comparable either way).
            r_np = r_b[0].permute(1, 2, 0).cpu().numpy()
            g_np = g_b[0].permute(1, 2, 0).cpu().numpy()
            try:
                ssim2_val = sk_ssim(r_np, g_np, channel_axis=-1, data_range=1.0)
            except TypeError:      # older skimage
                ssim2_val = sk_ssim(r_np, g_np, multichannel=True, data_range=1.0)

            lp_alex = lpips_alex_model(r_b, g_b).mean().item()
            lp_vgg  = lpips_vgg_model(r_b, g_b).mean().item()

            psnr_list.append(psnr_val)
            ssim1_list.append(ssim1_val)
            ssim2_list.append(float(ssim2_val))
            lpips_alex_list.append(lp_alex)
            lpips_vgg_list.append(lp_vgg)

            # Per-frame record for optional dump alongside image saves.
            per_frame_records.append({
                "frame": viewpoint_cam.image_name,
                "psnr": float(psnr_val),
                "ssim1": float(ssim1_val),
                "ssim2": float(ssim2_val),
                "lpips_alex": float(lp_alex),
                "lpips_vgg": float(lp_vgg),
            })

            # cleanup
            del rendering, gt, r_b, g_b, render_pkg
            torch.cuda.empty_cache()

    # Reported metrics: PSNR + both SSIM conventions + both LPIPS nets.
    # DSSIM is not stored: DSSIM_k = (1 - SSIM_k) / 2, derive it when tabulating.
    metrics = {
        "iteration": int(iteration),
        "psnr": float(np.mean(psnr_list)),
        "ssim1": float(np.mean(ssim1_list)),
        "ssim2": float(np.mean(ssim2_list)),
        "lpips_alex": float(np.mean(lpips_alex_list)),
        "lpips_vgg": float(np.mean(lpips_vgg_list)),
    }
    print(f"[ITER {iteration}] Eval Results: "
          f"PSNR={metrics['psnr']:.4f}, "
          f"SSIM1={metrics['ssim1']:.4f}, SSIM2={metrics['ssim2']:.4f}, "
          f"LPIPS(Alex)={metrics['lpips_alex']:.4f}, "
          f"LPIPS(VGG)={metrics['lpips_vgg']:.4f}")

    # --- Best PSNR logic stays the same ---
    new_best_psnr = current_best_psnr
    if metrics["psnr"] > current_best_psnr and iteration > 16000:
        new_best_psnr = metrics["psnr"]
        print(f"[ITER {iteration}] New best PSNR: {new_best_psnr:.4f}. RE-RENDERING for Image Saving...")

        # Single "best" location, overwritten on each new best.
        _best_dir = os.path.join(optimization_params.saving_folder, expname, "best")
        render_path = os.path.join(_best_dir, "render")
        gt_path = os.path.join(_best_dir, "gt")
        render_3d_path = os.path.join(_best_dir, "static_persistent")
        render_4d_path = os.path.join(_best_dir, "transient")

        os.makedirs(render_path, exist_ok=True)
        os.makedirs(gt_path, exist_ok=True)
        os.makedirs(render_3d_path, exist_ok=True)
        os.makedirs(render_4d_path, exist_ok=True)

        # Dump per-frame metrics computed in pass 1 (consistent with what the
        # saved images will show, since rasterizer is deterministic at this iter).
        per_frame_path = os.path.join(_best_dir, "per_frame_metrics.json")
        with open(per_frame_path, "w") as _pf:
            json.dump({
                "iteration": int(iteration),
                "expname": expname,
                "n_frames": len(per_frame_records),
                "per_frame": per_frame_records,
            }, _pf, indent=2)

        def clamp_if_not_none(x):
            return torch.clamp(x, 0.0, 1.0).cpu() if x is not None else None

        with torch.no_grad():
            for viewpoint_cam in tqdm(viewpoint_stack, desc="Saving images"):
                render_pkg = render_rawall3pc(
                    viewpoint_cam, foreground_gaussians, background_gaussians, transient_gaussian,
                    pipeline_config,
                    bg_color=bg,
                    stage=stage
                )
                rendering = render_pkg["render"]

                rendering = torch.clamp(rendering, 0.0, 1.0)
                gt = torch.clamp(viewpoint_cam.original_image.float().cuda() / 255.0, 0.0, 1.0)

                render_3d = render_pkg.get("render_3d", None)
                render_4d = render_pkg.get("render_4d", None)

                torchvision.utils.save_image(rendering.cpu(), os.path.join(render_path, f"{viewpoint_cam.image_name}.png"))
                torchvision.utils.save_image(gt.cpu(), os.path.join(gt_path, f"{viewpoint_cam.image_name}.png"))

                if render_3d is not None:
                    torchvision.utils.save_image(clamp_if_not_none(render_3d),
                                                 os.path.join(render_3d_path, f"{viewpoint_cam.image_name}.png"))
                if render_4d is not None:
                    torchvision.utils.save_image(clamp_if_not_none(render_4d),
                                                 os.path.join(render_4d_path, f"{viewpoint_cam.image_name}.png"))

                del rendering, gt, render_pkg, render_3d, render_4d
                torch.cuda.empty_cache()

        print(f"[ITER {iteration}] Saving Scene (Point Clouds) due to new best PSNR...")
        # "best" instead of iteration_N: overwrites the single best checkpoint.
        # per_frame_metrics.json in the best/ dir records which iteration it came from.
        scene.save("best", stage)
    else:
        print(f"[ITER {iteration}] PSNR {metrics['psnr']:.4f} did not beat best {current_best_psnr:.4f}. Skipping image save.")

    # Append JSONL
    json_path = os.path.join(optimization_params.saving_folder, expname, "debug_metrics.json")
    with open(json_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")

    return new_best_psnr


def final_test_dump(test_cams, foreground_gaussians, background_gaussians, transient_gaussian,
                    pipeline_config, background, optimization_params, expname, stage):
    """After fine-stage training: render every test camera once and write a
    labelled 2x5 debug panel per camera into train_cams/. Scores are written
    separately by run_evaluation; full per-branch renders live in best/."""
    for viewpoint_cam in test_cams:
        viewpoint_cams = [viewpoint_cam]

        with torch.no_grad():
            images = []
            gt_images = []
            images_second = []
            images_4d_debug = []
            motion_masks = []
            hybrid_full_images = []
            for viewpoint_cam in viewpoint_cams:
                render_pkg_dynamic_pers = render_foreground(viewpoint_cam, foreground_gaussians, pipeline_config,
                                                            background, stage=stage)

                image = render_pkg_dynamic_pers["render"]
                images.append(image.unsqueeze(0))

                gt_image = viewpoint_cam.original_image.float().cuda() / 255
                render_pkg_second = render_background(viewpoint_cam, background_gaussians, pipeline_config,
                                                      background, stage='coarse')
                image_second = render_pkg_second["render"]

                render_pkg_motion = render_mask(viewpoint_cam, foreground_gaussians, pipeline_config, background,
                                                stage=stage)

                motion_mask = render_pkg_motion["render"]

                render_pkg_hybrid = render_rawall3pc(viewpoint_cam, foreground_gaussians, background_gaussians,
                                                     transient_gaussian,
                                                     pipeline_config, background, stage=stage)
                image_transient_hybrid = render_pkg_hybrid["render"]

                hybrid_full_images.append(image_transient_hybrid.unsqueeze(0))
                images_4d_debug.append(render_pkg_hybrid["render_4d"].unsqueeze(0))

                motion_masks.append(motion_mask.unsqueeze(0))
                images_second.append(image_second.unsqueeze(0))
                gt_images.append(gt_image.unsqueeze(0))

            motion_masks = torch.cat(motion_masks, 0)
            image_tensor_4d_debug = torch.cat(images_4d_debug, 0)
            image_tensor_hybrid_full = torch.cat(hybrid_full_images, 0)
            image_tensor_first = torch.cat(images, 0)
            gt_image_tensor = torch.cat(gt_images, 0)
            image_tensor_second = torch.cat(images_second, 0)
            motion_masks = torch.clamp(motion_masks, 1e-9, 1 - 1e-9)
            motion_pro_first = motion_masks[:, 2:3, :, :] + 1e-6

            image_second_to_show = image_tensor_second.clone().detach().cpu()

            motion_masks_first = motion_pro_first + 1e-6
            motion_masks_second = 1 - motion_pro_first + 1e-6

            image_dy = image_tensor_first * motion_masks_first
            image_sta = image_tensor_second * motion_masks_second
            image_tensor = image_dy + image_sta

            # The after-training pass emits only the labelled 2x5 panel per test
            # camera, into train_cams/. Per-branch PNGs are not dumped here:
            # best/ already holds render / gt / static_persistent / transient for
            # every test frame, at the best checkpoint rather than the last one.
            out_debug_depth_dir = os.path.join(optimization_params.saving_folder, expname,
                                               'train_cams')
            os.makedirs(out_debug_depth_dir, exist_ok=True)


            _, ax = plt.subplots(2, 5, figsize=(30, 12))
            plt.rcParams['font.family'] = "sans-serif"

            ax[0, 0].imshow(gt_image_tensor.clone().detach().cpu()[0].permute(1, 2, 0).numpy())
            ax[0, 0].set_title("Ground Truth")
            ax[1, 0].imshow(image_tensor.clone().detach().cpu()[0].permute(1, 2, 0).numpy())
            ax[1, 0].set_title("Persistent Composition Render")
            ax[0, 1].imshow(image_tensor_first[0].clone().detach().cpu().permute(1, 2, 0).numpy())
            ax[0, 1].set_title("Persistent Dynamic (raw)")

            ax[1, 1].imshow(image_dy[0].clone().detach().cpu().permute(1, 2, 0).numpy())
            ax[1, 1].set_title("Persistent Dynamic (gated)")

            ax[0, 2].imshow(image_second_to_show[0].permute(1, 2, 0).numpy())
            ax[0, 2].set_title("Static (raw)")
            ax[1, 2].imshow(image_sta[0].clone().detach().cpu().permute(1, 2, 0).numpy())
            ax[1, 2].set_title("Static (gated)")

            ax[0, 3].imshow(
                (motion_masks[0, 2:3, :, :]).clone().detach().cpu().permute(1, 2,
                                                                            0).numpy(),
                cmap='jet', vmin=0, vmax=1)
            ax[0, 3].set_title("Allocation prob (dynamic)")
            # motion_masks channel 1 = raw alpha of the persistent-dynamic branch
            # (where G_d actually has coverage). Channel 2 is the allocation prob.
            ax[1, 3].imshow(
                (motion_masks[0, 1:2, :, :]).clone().detach().cpu().permute(1, 2, 0).numpy(),
                cmap='jet', vmin=0, vmax=1)
            ax[1, 3].set_title("Persistent Dynamic Alpha")
            ax[0, 4].imshow(image_tensor_4d_debug[0].clone().detach().cpu().permute(1, 2, 0).numpy())
            ax[0, 4].set_title("Transient")
            ax[1, 4].imshow(
                image_tensor_hybrid_full[0].clone().detach().cpu().permute(1, 2, 0).numpy())
            ax[1, 4].set_title("Full Render")

            plt.savefig(
                os.path.join(out_debug_depth_dir, viewpoint_cam.image_name + "_t.jpg"))
            plt.close()


def save_train_debug_viz(iteration, stage, expname, saving_folder, phase3_start_iter,
                         gt_image_tensor, image_tensor, image_tensor_first, image_dy,
                         image_second_to_show, image_sta, motion_masks,
                         images_4d_debug, image_tensor_hybrid_full=None):
    """Periodic training-progress panel (jpg per 100 iters): 1x2 in Phase 3,
    else a 2x5 grid mirroring the after-training panel in final_test_dump.
    Panels use the paper's vocabulary: Static (G_s), Persistent Dynamic (G_d),
    Transient (G_t). No depth subplots."""
    out_debug_dir = os.path.join(saving_folder, expname)
    os.makedirs(out_debug_dir, exist_ok=True)

    if iteration >= phase3_start_iter:  # Simplified panel for Phase 3
        _, ax = plt.subplots(1, 2, figsize=(12, 6))
        plt.rcParams['font.family'] = "sans-serif"

        ax[0].imshow(gt_image_tensor.clone().detach().cpu()[0].permute(1, 2, 0).numpy())
        ax[0].set_title("Ground Truth")
        ax[1].imshow(image_tensor.clone().detach().cpu()[0].permute(1, 2, 0).numpy())
        ax[1].set_title("Full Render")

        plt.savefig(os.path.join(out_debug_dir, stage + '_' + str(iteration).zfill(6) + ".jpg"))
        plt.close()
    else:
        # 2x5 layout mirroring final_test_dump's after-training panel.
        _, ax = plt.subplots(2, 5, figsize=(30, 12))
        plt.rcParams['font.family'] = "sans-serif"

        ax[0, 0].imshow(gt_image_tensor.clone().detach().cpu()[0].permute(1, 2, 0).numpy())
        ax[0, 0].set_title("Ground Truth")
        ax[1, 0].imshow(image_tensor.clone().detach().cpu()[0].permute(1, 2, 0).numpy())
        ax[1, 0].set_title("Persistent Composition Render")

        ax[0, 1].imshow(image_tensor_first[0].clone().detach().cpu().permute(1, 2, 0).numpy())
        ax[0, 1].set_title("Persistent Dynamic (raw)")
        ax[1, 1].imshow(image_dy[0].clone().detach().cpu().permute(1, 2, 0).numpy())
        ax[1, 1].set_title("Persistent Dynamic (gated)")

        ax[0, 2].imshow(image_second_to_show[0].permute(1, 2, 0).numpy())
        ax[0, 2].set_title("Static (raw)")
        ax[1, 2].imshow(image_sta[0].clone().detach().cpu().permute(1, 2, 0).numpy())
        ax[1, 2].set_title("Static (gated)")

        ax[0, 3].imshow(
            (motion_masks[0, 2:3, :, :]).clone().detach().cpu().permute(1, 2, 0).numpy(),
            cmap='jet', vmin=0, vmax=1)
        ax[0, 3].set_title("Allocation prob (dynamic)")
        ax[1, 3].imshow(
            (motion_masks[0, 1:2, :, :]).clone().detach().cpu().permute(1, 2, 0).numpy(),
            cmap='jet', vmin=0, vmax=1)
        ax[1, 3].set_title("Persistent Dynamic Alpha")

        ax[0, 4].imshow(images_4d_debug[0].clone().detach().cpu().permute(1, 2, 0).numpy())
        ax[0, 4].set_title("Transient")
        if image_tensor_hybrid_full is not None:
            ax[1, 4].imshow(
                image_tensor_hybrid_full[0].clone().detach().cpu().permute(1, 2, 0).numpy())
            ax[1, 4].set_title("Full Render")
        else:
            ax[1, 4].axis("off")

        plt.savefig(os.path.join(out_debug_dir, stage + '_' + str(iteration).zfill(6) + ".jpg"))
        plt.close()
