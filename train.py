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
Dynamic Gaussian Splatting training script (release version).
This refactor only adds concise documentation and removes duplicate imports
while keeping the original variable names and overall structure intact.
"""

import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, structural_ssim, \
    ssim_raw, EdgeAwareTV
from gaussian_renderer import render_background, render_foreground, render_mask, render_rawall3pc
from scene import GaussianModel, GaussianModel_dynamic, Scene2gs_mixed
from scene.gaussian_model import GaussianModelTransient
from utils.general_utils import safe_state
from utils.eval_utils import run_evaluation, final_test_dump, save_train_debug_viz
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
import copy




def maintain_foreground(iteration, stage, optimization_params, scene, foreground_gaussians,
                        viewspace_point_tensor_grad, visibility_filter, last_max_weight,
                        weight_prune_freq, phase3_start_iter):
    """FG adaptive density control: densification stats, threshold schedule,
    densify / weight_prune / prune, opacity reset."""
    # ---- Adaptive Gaussian Densification & Pruning ----
    if iteration < optimization_params.densify_until_iter and iteration < phase3_start_iter:

        # Track foreground importance (using probability instead of radii)
        foreground_gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

        foreground_gaussians.max_radii2D = torch.max(foreground_gaussians.max_radii2D, last_max_weight)
        if iteration <= 1:
            foreground_gaussians.mean_radii2D = (last_max_weight / weight_prune_freq)
        else:
            foreground_gaussians.mean_radii2D = foreground_gaussians.mean_radii2D + (
                    last_max_weight / weight_prune_freq)

        # ---- Threshold Scheduling ----
        if stage == "coarse":
            opacity_threshold = optimization_params.opacity_threshold_coarse
            densify_threshold = optimization_params.densify_grad_threshold_coarse
        else:
            # Progressive threshold scheduling for fine stage
            opacity_threshold = optimization_params.opacity_threshold_fine_init - iteration * (
                    optimization_params.opacity_threshold_fine_init - optimization_params.opacity_threshold_fine_after) / (
                                    optimization_params.densify_until_iter)
            if iteration < optimization_params.densify_until_iter * 0.5:
                # First half: gradually reduce threshold
                densify_threshold = optimization_params.densify_grad_threshold_fine_init - iteration * (
                        optimization_params.densify_grad_threshold_fine_init - optimization_params.densify_grad_threshold_after) / (
                                        optimization_params.densify_until_iter)
            else:
                # Second half: use minimum threshold
                densify_threshold = optimization_params.densify_grad_threshold_after

        # ---- Densification (Add Gaussians) ----
        if iteration > optimization_params.densify_from_iter and iteration % optimization_params.densification_interval == 0 and \
                foreground_gaussians.get_xyz.shape[0] < optimization_params.max_gaussian_foreground:
            foreground_gaussians.densify(densify_threshold, scene.cameras_extent)

        if (
                iteration) % weight_prune_freq == 0 and iteration > optimization_params.transient_merging_iter and iteration < phase3_start_iter:
            # FG weight-prune aggressiveness. mean_ratio blends max vs mean accumulated weight in
            # the prune score (max>=mean, so higher mean_ratio => lower score => prunes MORE).
            # 1.0 is a more aggressive alternative (prunes harder); see fg_mean_ratio.
            mean_ratio = getattr(optimization_params, 'fg_mean_ratio', 0.8)
            # Calculate linear progress (0.0 to 1.0)
            progress = (iteration - optimization_params.transient_merging_iter) / (
                    phase3_start_iter - optimization_params.transient_merging_iter)
            progress = max(0.0, min(1.0, progress))
            # Interpolate the prune threshold from 0.01 up to the ceiling over the
            # window; a higher ceiling (e.g. 0.2) prunes the FG branch harder.
            _fg_ceil = getattr(optimization_params, 'fg_prune_thresh_ceil', 0.1)
            prune_thresh = 0.01 + progress * (_fg_ceil - 0.01)

            foreground_gaussians.weight_prune(prune_thresh, mean_ratio)

        # ---- Pruning (Remove Gaussians) ----
        if iteration > optimization_params.pruning_from_iter and iteration % optimization_params.pruning_interval == 0 and \
                foreground_gaussians.get_xyz.shape[0] > 1000:
            foreground_gaussians.prune(opacity_threshold)


        # ---- Opacity Reset (prevent degradation; stops at Phase 3) ----
        if iteration % optimization_params.opacity_reset_interval == 0 and iteration < phase3_start_iter:
            foreground_gaussians.reset_opacity_partially_small()


def maintain_background(iteration, stage, optimization_params, scene, background_gaussians,
                        viewspace_point_tensor_grad_second, visibility_filter_second,
                        last_max_weight_second, phase3_start_iter):
    """BG adaptive density control: densify / prune / opacity reset / weight_prune."""
    # ---- Background Model Management ----
    if iteration < optimization_params.densify_until_iter:
        visibility_filter = visibility_filter_second

        # Track background importance
        background_gaussians.max_radii2D = torch.max(background_gaussians.max_radii2D, last_max_weight_second)

        background_gaussians.add_densification_stats(viewspace_point_tensor_grad_second, visibility_filter)

        # ---- Background Threshold Configuration ----
        if stage == "coarse":
            opacity_threshold = optimization_params.opacity_threshold_coarse
            densify_threshold = optimization_params.densify_grad_threshold_coarse
        else:
            opacity_threshold = optimization_params.opacity_threshold_fine_init - iteration * (
                    optimization_params.opacity_threshold_fine_init - optimization_params.opacity_threshold_fine_after) / (
                                    optimization_params.densify_until_iter)
            # Background uses coarse threshold x2 (less aggressive)
            densify_threshold = optimization_params.densify_grad_threshold_coarse * 2

        # ---- Background Densification ----
        if iteration > optimization_params.densify_from_iter and iteration % optimization_params.densification_interval == 0 and \
                background_gaussians.get_xyz.shape[0] < optimization_params.max_gaussian_background:
            background_gaussians.densify(densify_threshold, scene.cameras_extent)

        # ---- Background Pruning ----
        if iteration > optimization_params.pruning_from_iter and iteration % optimization_params.pruning_interval == 0 and \
                background_gaussians.get_xyz.shape[0] > 10000:
            background_gaussians.prune(opacity_threshold)

        # ---- Background Opacity Reset (no early stop) ----
        if iteration % optimization_params.opacity_reset_interval == 0:
            background_gaussians.reset_opacity_partially()
        # BG weight_prune (every 500 iters): stop at phase3_start_iter — after that
        # max_radii2D is zeroed and pruning would catastrophically prune all BG.
        if iteration % 500 == 0 and iteration > optimization_params.transient_merging_iter and iteration < phase3_start_iter:
            background_gaussians.weight_prune(0.001)



def maintain_transient(iteration, optimization_params, scene, transient_gaussian, render_pkg_hybrid,
                       viewspace_point_tensor_grad_transient, visibility_filter_transient,
                       viewspace_point_tensor_list_transient,
                       fg_to_tr_active, prune_data, weight_prune_fre_transient, phase3_start_iter):
    """TR density control: densification stats (+time grad), single-cam weight
    accumulation, densify_and_prune, FG->TR add_points, weight_prune ramp, reset."""
    # ---- Transient Model Management ----
    if iteration < optimization_params.densify_until_iter:
        if visibility_filter_transient is not None:
            visibility_filter = visibility_filter_transient

            # Track transient importance

            # Add densification stats
            if len(viewspace_point_tensor_list_transient) > 0:
                # Check if gradient was computed

                avg_t_grad = None
                if transient_gaussian.gaussian_dim == 4 and transient_gaussian._t.grad is not None:
                    # Extract time gradient similar to reference: batch_t_grad_4d = gaussians_4d._t.grad.clone()[:,0].detach()
                    avg_t_grad = transient_gaussian._t.grad.clone()[:, 0].detach().unsqueeze(1)
                    avg_t_grad /= len(viewspace_point_tensor_list_transient)

                # Compute norm of the gradient for densification stats
                transient_gaussian.add_densification_stats_grad(viewspace_point_tensor_grad_transient,
                                                                visibility_filter, avg_t_grad)

            # ---- Transient Weight Accumulation (single-cam BY DESIGN) ----
            # Uses the LAST batch cam, not a batch-aware max like FG/BG. This
            # halves the per-iter increment, which is what calibrates the
            # weight_prune threshold floor — making it batch-aware breaks the
            # prune dynamics. Changing it requires halving
            # weight_prune_fre_transient or doubling the floor.
            # Must run BEFORE any FG->TR transfer (sized to pre-transfer TR).
            if "max_weights_t" in render_pkg_hybrid:
                max_weights_t = render_pkg_hybrid["max_weights_t"]
                transient_gaussian.max_radii2D = torch.max(transient_gaussian.max_radii2D, max_weights_t)
                if iteration <= 1:
                    transient_gaussian.mean_radii2D = (max_weights_t / weight_prune_fre_transient)
                else:
                    transient_gaussian.mean_radii2D = transient_gaussian.mean_radii2D + (
                            max_weights_t / weight_prune_fre_transient)

            # ---- Transient Threshold Configuration ----

            opacity_threshold = optimization_params.opacity_threshold_coarse
            densify_threshold = optimization_params.densify_grad_threshold_coarse

            # ---- Transient Densification ----
            # When enable_transient_densify is False (monocular NeRF-DS), TR grows
            # only via the capped FG->TR transfer below.
            if getattr(optimization_params, 'enable_transient_densify', True) and \
                    iteration > optimization_params.densify_from_iter and iteration % optimization_params.densification_interval == 0 and \
                    transient_gaussian.get_xyz.shape[0] < getattr(optimization_params, 'max_gaussian_transient', 500000):
                size_threshold = 20 if iteration > optimization_params.opacity_reset_interval else None
                transient_gaussian.densify_and_prune(densify_threshold / getattr(optimization_params, 'tr_densify_divisor', 6), opacity_threshold,
                                                     scene.cameras_extent,
                                                     size_threshold, None)

            if fg_to_tr_active:
                _vscale = float(getattr(optimization_params, 'fg_to_tr_velocity_scale', 1.0))
                transient_gaussian.add_points_from_dict(prune_data, velocity_scale=_vscale)

            # Start gate for TR weight_prune. The threshold ramp is anchored to
            # phase3_start_iter regardless, so an earlier start still prunes at
            # the gentle floor until Phase 3 opens.
            tr_prune_start_iter = int(getattr(optimization_params, 'tr_weight_prune_start_iter', phase3_start_iter))
            if iteration % weight_prune_fre_transient == 0 and iteration > tr_prune_start_iter and \
                    transient_gaussian.get_xyz.shape[0] > 1000:
                # Anchored to phase3_start_iter, NOT tr_prune_start_iter:
                # 0.005 until Phase 3, then ramps to 0.02 by densify_until_iter.
                end_ramp_iter = optimization_params.densify_until_iter
                progress_t = (iteration - phase3_start_iter) / (end_ramp_iter - phase3_start_iter)
                progress_t = max(0.0, min(1.0, progress_t))

                # TR-prune threshold ramp 0.005 -> 0.02 by densify_until_iter.
                transient_prune_thresh = 0.005 + progress_t * (0.02 - 0.005)

                transient_gaussian.weight_prune(transient_prune_thresh)



def scene_reconstruction(optimization_params, hypernetwork_config, pipeline_config,
                                 saving_iterations,
                                 checkpoint_iterations, checkpoint,
                                 foreground_gaussians, scene, stage, train_iter,
                                 background_gaussians=None,
                                 transient_gaussian=None,
                                 expname='debug_2gs'):
    # ---------------------------------------------------------------------
    #  ➤  COARSE / FINE TRAINING LOOP (Foreground & Background Gaussians)
    # ---------------------------------------------------------------------
    """Two-stage (coarse→fine) training loop for Multi4D dynamic Gaussian splatting.

    This routine orchestrates optimisation of *foreground* (dynamic) and
    *background* (static) Gaussian models by alternating between:

    1.  Coarse stage – establishes scene geometry, aggressive densification.
    2.  Fine stage  – refines colour/opacity, enforces temporal & depth priors.

    It supports mixed-resolution datasets, gradient accumulation, adaptive
    densification/pruning, brightness control, and a rich loss cocktail.

    Parameters
    ----------
    opt : argparse.Namespace
        All optimisation hyper-parameters.
    hyper : ModelHiddenParams
        Deformation & temporal smoothness coefficients.
    pipe : PipelineParams
        Rendering pipeline configuration.
    saving_iterations / checkpoint_iterations : list[int]
        Iteration indices for lightweight save / full checkpoint.
    checkpoint : str | None
        Path to an existing checkpoint to resume from.
    gaussians : GaussianModel_dynamic
        Foreground (dynamic) Gaussian representation.
    scene : Scene2gs_mixed
        Wrapper that holds cameras + both Gaussian sets.
    stage : {"coarse", "fine"}
    train_iter : int
        Number of optimisation iterations for this stage.
    gaussians_second : GaussianModel, optional
        Background (static) Gaussian representation.
    expname : str
        Experiment tag used for output folders.
    """
    first_iter = 0

    # ---- Model Setup ----
    foreground_gaussians.training_setup(optimization_params)
    background_gaussians.training_setup(optimization_params)
    transient_gaussian.training_setup(optimization_params)

    # ---- Checkpoint Resume Logic ----
    if checkpoint:
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            return
        if stage in checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
            foreground_gaussians.restore(model_params, optimization_params)

        # ---- Background Colors (near-black to avoid numerical issues) ----
    bg_color = [1e-7, 1e-7, 1e-7]
    # No PSNR floor, so sub-28 dB datasets still save a best checkpoint.
    best_psnr = -1.0

    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ---- CUDA Timing Events ----
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # ---- Training State Initialization ----
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    ema_lossl1_for_log = 0.0

    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    # ---- Camera Data Loading ----
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    viewpoint_stack_index = list(range(len(train_cams)))
    if not viewpoint_stack:
        # Manual sampling mode - copy camera list
        viewpoint_stack = [i for i in train_cams]
        viewpoint_stack_index_save = copy.deepcopy(viewpoint_stack_index)

    batch_size = optimization_params.batch_size
    print("data loading done")

    # =========================================================================
    # TRAINING SCHEDULE. Windows are fine-stage iterations (coarse is a separate
    # scene_reconstruction call that restarts the counter). Reference config:
    # coarse 2000, merging 3000, phase3 10000, densify_until 16000, end 20000.
    #
    #   Phase 2   1 .. phase3_start     four renders/iter; composite+hybrid losses
    #   Phase 3   phase3_start .. end   one render/iter; simplified loss + depth
    #   "merged"  merging .. phase3     TR joins the loss; FG->TR donors flow
    #
    #   mechanism            window                cadence
    #   ------------------   -------------------   --------------------------
    #   FG/BG densify        500 .. see note       100  (densification_interval)
    #   FG/BG opacity prune  500 .. see note       600  (pruning_interval)
    #   FG/BG opacity reset  .. see note           2000 (opacity_reset_interval)
    #     note: FG stops at phase3_start, BG at densify_until.
    #   FG weight-prune      merging .. phase3     100  (hardcoded)
    #   BG weight-prune      merging .. phase3     500  (hardcoded)
    #   TR densify/prune     500 .. densify_until  100
    #   TR weight-prune      tr_prune_start ..     tr_weight_prune_freq (200)
    #                        densify_until
    #   FG->TR donors        merging .. phase3     fg_to_tr_transfer_freq (50)
    #   soft_dxv stats       iter 1 (incl coarse)  every iter; zeroed every
    #                        .. phase3             dx_var_reset_interval (1000).
    #                                              The LOSS waits for its start iter.
    #   eval                 fine only             500
    #   best-save            fine, iter > 16000    on new best PSNR
    #   debug viz            all                   100
    # =========================================================================
    phase3_start_iter = getattr(optimization_params, 'phase3_start_iter', 10000)
    # FG->TR organic transfer window: phase 2 (transient_merging_iter .. phase3_start_iter).
    fg_to_tr_start_iter = optimization_params.transient_merging_iter
    fg_to_tr_end_iter = phase3_start_iter
    fg_to_tr_freq = max(1, getattr(optimization_params, 'fg_to_tr_transfer_freq', 50))
    # Constants (never mutated in the loop).
    weight_prune_freq = 100        # FG weight-prune cadence (iters between fires)
    weight_prune_fre_transient = int(getattr(optimization_params, 'tr_weight_prune_freq', 100))
    depth_reg = EdgeAwareTV()  # stateless module, constructed once
    if stage == "fine":
        _phase_msg = (
            f"[PhaseTiming] "
            f"phase3_start={phase3_start_iter}; "
            f"TR organic seed={fg_to_tr_start_iter}-{fg_to_tr_end_iter} step={fg_to_tr_freq}; "
            f"TR merge loss>{optimization_params.transient_merging_iter}; "
            f"FG densify=500-{phase3_start_iter}; "
            f"BG/TR densify=500-{optimization_params.densify_until_iter}; "
            f"FG prune={optimization_params.transient_merging_iter}-{phase3_start_iter}; "
            f"soft_dxv_start={getattr(optimization_params, 'soft_dxv_start_iter', None)} "
            f"enabled={getattr(optimization_params, 'use_soft_dxv_loss', False)}; "
            f"depth_order=lam{float(getattr(optimization_params, 'lambda_depth_order', 0.0))} (P3 only); "
            f"vscale={float(getattr(optimization_params, 'fg_to_tr_velocity_scale', 1.0))}; "
        )
        print(_phase_msg)

    # Bound only on pre-Phase-3 iterations (see below); pre-set so the debug viz
    # arg is always defined regardless of where training starts.
    image_tensor_hybrid_full = None
    for iteration in range(first_iter, final_iter + 1):
        iter_start.record()

        # Named phase predicates (see the schedule table above). Pure aliases —
        # each is exactly the comparison it replaces, computed once per iteration.
        in_phase3 = iteration >= phase3_start_iter      # one-render regime
        pre_phase3 = not in_phase3                      # composite/hybrid regime
        after_merge = iteration > optimization_params.transient_merging_iter  # TR in loss

        # Learning rate scheduling
        foreground_gaussians.update_learning_rate(iteration)
        background_gaussians.update_learning_rate(iteration)

        # Spherical harmonics degree progression
        if iteration % optimization_params.foreground_oneupshinterval == 0:
            foreground_gaussians.oneupSHdegree()
        if iteration % optimization_params.background_oneupshinterval == 0:
            background_gaussians.oneupSHdegree()
        # Update SH degree for transient model (same interval as foreground)
        if iteration % optimization_params.foreground_oneupshinterval == 0:
            transient_gaussian.oneupSHdegree()

        # ---- Camera Batch Sampling (uniform random, without replacement) ----
        idx = 0
        viewpoint_cams = []

        while idx < batch_size:
            viewpoint_cam_idx = viewpoint_stack_index.pop(randint(0, len(viewpoint_stack_index) - 1))
            viewpoint_cam = viewpoint_stack[viewpoint_cam_idx]
            if not viewpoint_stack_index:
                # Reset for next epoch
                viewpoint_stack_index = viewpoint_stack_index_save.copy()
            viewpoint_cams.append(viewpoint_cam)
            idx += 1

        if len(viewpoint_cams) == 0:
            continue

        # ---- Batch Rendering Setup ----
        images = []
        gt_images = []
        images_second = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        visibility_filter_list_second = []
        viewspace_point_tensor_list_second = []
        motion_masks = []

        depth_images_dy = []
        last_max_weight_list = []
        last_max_weight_list_second = []
        images_transient = []
        visibility_filter_list_transient = []
        viewspace_point_tensor_list_transient = []
        images_4d_debug = []
        L_diversity_terms = []   # paper Eq.8 L_diversity: SSIM(C_t, C_p), Sec 3.3


        depth_4d_list = []
        depth_3d_list = []
        dynamic_3d = []

        # ---- Multi-Model Rendering Loop ----
        for viewpoint_cam in viewpoint_cams:
            if in_phase3:
                # Phase 3: render_rawall3pc only (background first, for stats).
                render_pkg_hybrid = render_rawall3pc(viewpoint_cam, foreground_gaussians, background_gaussians,
                                                     transient_gaussian,
                                                     pipeline_config, background, stage=stage, dropout_use=True)

                # Populate main lists
                image = render_pkg_hybrid["render"]
                images.append(image.unsqueeze(0))

                gt_image = viewpoint_cam.original_image.float().cuda() / 255
                gt_images.append(gt_image.unsqueeze(0))

                # Populate Stats lists (Needed for densification/pruning)
                # Foreground Stats
                visibility_filter_list.append(render_pkg_hybrid['visibility_filter_fg'].unsqueeze(0))
                viewspace_point_tensor_list.append(render_pkg_hybrid['viewspace_points_fg'])
                # Placeholder for motion

                # Background Stats
                # Background Stats
                visibility_filter_list_second.append(render_pkg_hybrid['visibility_filter_bg'].unsqueeze(0))
                viewspace_point_tensor_list_second.append(render_pkg_hybrid['screenspace_points_bg'])
                dynamic_3d.append(render_pkg_hybrid['render_3d'].unsqueeze(0))
                # Transient Stats
                visibility_filter_list_transient.append(render_pkg_hybrid['visibility_filter'].unsqueeze(0))
                viewspace_point_tensor_list_transient.append(render_pkg_hybrid['viewspace_points'])

                # Placeholders / Secondary Outputs to prevent crashes
                images_second.append(torch.zeros_like(image).unsqueeze(0))  # Background placeholder
                motion_masks.append(
                    torch.zeros(3, image.shape[1], image.shape[2], device="cuda").unsqueeze(0))  # Placeholder
                depth_images_dy.append(render_pkg_hybrid['depth'].unsqueeze(0))

                if "depth_4d" in render_pkg_hybrid and "depth_3d" in render_pkg_hybrid:
                    depth_4d_list.append(render_pkg_hybrid['depth_4d'].unsqueeze(0))
                    depth_3d_list.append(render_pkg_hybrid['depth_3d'].unsqueeze(0))

                # Probs/Weights Placeholders (Phase 3: per-branch stats not used here; mechanisms
                # depending on standalone-render stats are gated on iter < phase3_start_iter.)
                last_max_weight_list.append(torch.zeros_like(render_pkg_hybrid['max_weights_fg']).unsqueeze(0))
                last_max_weight_list_second.append(torch.zeros_like(render_pkg_hybrid['max_weights_bg']).unsqueeze(0))

                # Transient auxiliary
                if iteration % 100 == 0:
                    images_4d_debug.append(render_pkg_hybrid["render_4d"])


                render_4d = render_pkg_hybrid["render_4d"].unsqueeze(0)
                render_3d = render_pkg_hybrid["render_3d"].unsqueeze(0).clone().detach()
                alpha_hybrid = render_pkg_hybrid["alpha"].unsqueeze(0) > 0
                sim = structural_ssim(render_4d, render_3d)
                L_diversity_terms.append((sim * alpha_hybrid).mean())

                continue

            # Render foreground (dynamic) Gaussians
            render_pkg_motion = render_mask(viewpoint_cam, foreground_gaussians, pipeline_config, background,
                                            stage=stage, dropout_use=True)

            motion_mask = render_pkg_motion["render"]

            mask_render = motion_mask[2:3, :, :] + 1e-6

            foreground_mask_prob = (mask_render > 0.5).float()

            render_pkg_dynamic_pers = render_foreground(viewpoint_cam, foreground_gaussians, pipeline_config,
                                                        background,
                                                        stage=stage, prob_mask=(foreground_mask_prob),
                                                        dropout_use=True)

            image, viewspace_point_tensor, visibility_filter = render_pkg_dynamic_pers["render"], \
                render_pkg_dynamic_pers["viewspace_points"], render_pkg_dynamic_pers["visibility_filter"]
            images.append(image.unsqueeze(0))
            depth_images_dy.append(render_pkg_dynamic_pers['depth'].unsqueeze(0))

            # Accumulate per-Gaussian displacement variance for the soft_dxv loss
            if getattr(optimization_params, 'use_soft_dxv_loss', False):
                _dx_reset = getattr(optimization_params, 'dx_var_reset_interval', 0)
                if (_dx_reset > 0 and iteration % _dx_reset == 0
                        and foreground_gaussians.sum_dx.numel() > 0):
                    foreground_gaussians.sum_dx.zero_()
                    foreground_gaussians.sum_dx_sq.zero_()
                    foreground_gaussians.count_dx.zero_()
                deformed_xyz = render_pkg_dynamic_pers['deformed_gs']['means3D_final']  # [N,3]
                dx = (deformed_xyz - foreground_gaussians.get_xyz.detach()).detach()     # [N,3]
                foreground_gaussians.sum_dx    += dx
                foreground_gaussians.sum_dx_sq += dx.pow(2)
                foreground_gaussians.count_dx  += 1

            # Ground truth image
            gt_image = viewpoint_cam.original_image.float().cuda() / 255

            # Render background (static) Gaussians
            render_pkg_second = render_background(viewpoint_cam, background_gaussians, pipeline_config, background,
                                                  stage='coarse', prob_mask=(1 - foreground_mask_prob),
                                                  dropout_use=True)

            # Render transient Gaussians
            render_pkg_hybrid = render_rawall3pc(viewpoint_cam, foreground_gaussians, background_gaussians,
                                                 transient_gaussian,
                                                 pipeline_config, background, stage=stage, dropout_use=True)

            image_transient = render_pkg_hybrid["render"]
            dynamic_3d.append(render_pkg_hybrid['render_3d'].unsqueeze(0))

            images_transient.append(image_transient.unsqueeze(0))
            viewspace_point_tensor_list_transient.append(render_pkg_hybrid['viewspace_points'])
            visibility_filter_list_transient.append(render_pkg_hybrid['visibility_filter'].unsqueeze(0))
            if iteration % 100 == 0:
                images_4d_debug.append(render_pkg_hybrid["render_4d"])

            render_4d = render_pkg_hybrid["render_4d"].unsqueeze(0)
            render_3d = render_pkg_hybrid["render_3d"].unsqueeze(0).clone().detach()
            alpha_hybrid = render_pkg_hybrid["alpha"].unsqueeze(0)
            sim = structural_ssim(render_4d, render_3d)
            L_diversity_terms.append((sim * alpha_hybrid).mean())


            # Render motion probability mask

            image_second = render_pkg_second["render"]

            motion_masks.append(motion_mask.unsqueeze(0))
            images_second.append(image_second.unsqueeze(0))
            last_max_weight_list.append(render_pkg_dynamic_pers["max_weight_t"].unsqueeze(0))
            last_max_weight_list_second.append(render_pkg_second["max_weight_t"].unsqueeze(0))


            gt_images.append(gt_image.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            viewspace_point_tensor_list_second.append(render_pkg_second['viewspace_points'])
            visibility_filter_list_second.append(render_pkg_second['visibility_filter'].unsqueeze(0))

        # ---- Batch Tensor Consolidation ----
        motion_masks = torch.cat(motion_masks, 0)
        last_max_weight = torch.cat(last_max_weight_list, 0).max(dim=0).values
        last_max_weight_second = torch.cat(last_max_weight_list_second, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        render_3d_tensor = torch.cat(dynamic_3d, 0)

        visibility_filter_second = torch.cat(visibility_filter_list_second).any(dim=0)

        image_tensor_first = torch.cat(images, 0)
        gt_image_tensor = torch.cat(gt_images, 0)
        image_tensor_second = torch.cat(images_second, 0)
        depth_images_dy_tensor = torch.cat(depth_images_dy, 0)

        # Hybrid (TR-composite) batch tensor exists only pre-Phase-3; its only
        # consumers (the hybrid L1/SSIM losses) are gated iteration < phase3_start_iter.
        if pre_phase3:
            image_tensor_hybrid_full_batch = torch.cat(images_transient, 0)

        visibility_filter_transient_count = torch.cat(visibility_filter_list_transient).sum(
            dim=0)
        visibility_filter_transient = torch.cat(visibility_filter_list_transient).any(dim=0)

        # ---- Motion Probability Processing ----
        # Channel 2: foreground routing/occupancy mask (mask_render)
        # Channel 1: raw FG-branch alpha render (fg_alpha_render)
        if stage == "coarse":
            mask_render = motion_masks[:, 2:3, :, :] - 0.25
            # fg_alpha_render's only consumer (L_reg_alpha) is gated iteration > 3000,
            # and coarse ends at 2000 — so it is never read in coarse.
            fg_alpha_render = None
        else:
            mask_render = motion_masks[:, 2:3, :, :] + 1e-6
            fg_alpha_render = motion_masks[:, 1:2, :, :] + 1e-6

        # ---- Region weights for FG/BG losses ----
        fg_region_weight = mask_render
        bg_region_weight = (1 - mask_render)


        # ---- Image Composition ----
        # Only needed pre-Phase-3: the P3 loss uses image_tensor_first directly and
        # the P3 (1x2) viz branch never touches image_dy/image_sta.
        if pre_phase3:
            image_dy = image_tensor_first * fg_region_weight
            image_sta = image_tensor_second * bg_region_weight
        else:
            image_dy = image_sta = None

        vis_thresh = 0.49

        # Final composite image (with transient component if available)

        # =====================================================================
        # LOSS MAP — every term, its weight, and its gate. Paper Eq. 8 gives the
        # idealized 4-term objective; the code carries more. Gates use the
        # in_phase3 / pre_phase3 / after_merge predicates (see schedule table).
        # All terms accumulate into one `loss`; one backward() consumes it.
        #
        #   term                        weight                gate
        #   -------------------------   -------------------   -------------------
        #   L_color L1                  lambda_main_loss (4)  always (per-stage:
        #                                                     coarse 0.5/0.5, P2
        #                                                     composite, P3 hybrid)
        #   L_color SSIM                lambda_ssim (0.2)     always
        #   hybrid-full L1              1                     pre_phase3
        #   hybrid-full SSIM            lambda_ssim (0.2)     pre_phase3 & merged
        #   L_soft_dxv                  lambda_soft_dxv       soft_dxv_start ..
        #   L_reg_depth_order           lambda_depth_order    in_phase3
        #   L_reg_tv (plane/time)       ModelHiddenParams     fine stage
        #   L_sep_fgbg                  0.01                  pre_phase3
        #   coarse SSIM pair ×2         0.2 (bare literals)   coarse only
        #   L_diversity                 0.1                   when TR rendered
        #   render_3d anchor L1         0.1 flat              always
        #   L_reg_alpha (paper L_α)     lambda_fg_alpha_mask  fg_alpha_mask_start
        #                                                     .. phase3
        #   composite region L1 ×2      1                     pre_phase3
        #   composite region SSIM ×2    lambda_ssim (0.2)     pre_phase3
        #   scale-cap reg ×2            0.01                  always
        #   aspect-ratio reg ×2         0.1, fires every 10   always — the %10
        #                                                     duty cycle IS part
        #                                                     of the weight
        #   depth TV                    lambda_phase3_depth_tv  iteration > phase3
        # =====================================================================

        # ---- Stage-Specific Loss ----
        # Applies to the six lambda_ssim sites; the two coarse SSIM terms keep
        # their 0.2 literals.
        _lam_ssim = float(optimization_params.lambda_ssim)
        if pre_phase3:
            image_tensor = image_dy + image_sta
            image_tensor_hybrid_full = image_tensor_hybrid_full_batch

        if in_phase3:
            # Phase 3: simplified composite L1+SSIM on render_rawall3pc output.
            image_tensor = image_tensor_first
            Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
            ssim_notdense = _lam_ssim * (
                    1.0 - ssim(image_tensor, gt_image_tensor[:, :3, :, :]))
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()

            loss = optimization_params.lambda_main_loss * Ll1 + ssim_notdense  # paper: L_color
        elif stage == "coarse":
            # Coarse stage: background + mixed component loss
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
            ssim_color = _lam_ssim * (
                    1.0 - ssim(image_tensor, gt_image_tensor[:, :3, :, :]))

            ll2 = l1_loss(image_tensor_second, gt_image_tensor[:, :3, :, :])
            # Fixed 0.5/0.5 composition against GT: regularizes foreground
            # learning while still allowing structural modeling.
            Ll1 = l1_loss(
                image_tensor_first * (0.5) + image_tensor_second.clone().detach() * (
                    0.5)
                , gt_image_tensor[:, :3, :, :])
            loss = (optimization_params.lambda_main_loss * Ll1
                    + optimization_params.lambda_main_loss * ll2
                    + ssim_color)  # paper: L_color


        else:
            # Fine stage (Phase 2): composite L1 + composite SSIM.
            Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
            ssim_color = _lam_ssim * (
                    1.0 - ssim(image_tensor, gt_image_tensor[:, :3, :, :]))
            loss = optimization_params.lambda_main_loss * Ll1 + ssim_color  # paper: L_color


        # ---- Soft routing losses (continuous, gradient-driven; Direction 1) ----
        if (getattr(optimization_params, 'use_soft_dxv_loss', False)
                and iteration > getattr(optimization_params, 'soft_dxv_start_iter', 3000)
                and foreground_gaussians.count_dx.numel() > 0
                and foreground_gaussians.count_dx.max() >= 20):
            _c = foreground_gaussians.count_dx.unsqueeze(-1).clamp(min=1)
            _mean_dx    = foreground_gaussians.sum_dx    / _c
            _mean_dx_sq = foreground_gaussians.sum_dx_sq / _c
            _var_dx = (_mean_dx_sq - _mean_dx.pow(2)).clamp(min=0).sum(-1)   # [N]
            _thr = max(getattr(optimization_params, 'soft_dxv_threshold', 0.001), 1e-8)
            static_score = torch.exp(-_var_dx / _thr).detach()               # [N] ~1 for static
            _fg_opa = foreground_gaussians.get_opacity.squeeze(-1)           # [N]
            if static_score.shape[0] == _fg_opa.shape[0]:
                # soft_dxv loss: opacity penalty on persistent-dynamic (G_d)
                # Gaussians whose displacement variance says they are static.
                # An implementation-level regularizer, not a paper-named term --
                # the paper's alpha term is L_reg_alpha, further down.
                L_soft_dxv = (static_score * _fg_opa).mean()
                loss += optimization_params.lambda_soft_dxv * L_soft_dxv

        # Depth-order loss: penalize 4D-Gaussian depth > 3D-Gaussian depth (transient behind static).
        # The shipped configs set lambda_depth_order=0.01 — an ACTIVE term (Phase-3 only in
        # practice: depth_4d/depth_3d lists are populated only on the one-render path).
        _l_depth_order = float(getattr(optimization_params, 'lambda_depth_order', 0.0))
        if _l_depth_order > 0 and len(depth_4d_list) > 0 and len(depth_3d_list) > 0:
            depth_4d_tensor = torch.cat(depth_4d_list, 0)
            depth_3d_tensor = torch.cat(depth_3d_list, 0)
            # paper L_reg (Sec 3.7) — depth-order regularizer
            L_reg_depth_order = torch.relu(depth_4d_tensor - depth_3d_tensor.clone().detach()).mean()
            loss += _l_depth_order * L_reg_depth_order

        # Hybrid (TR-composite) photometric loss: L1 always (pre-Phase-3),
        # SSIM joins once TR merging is active.
        if pre_phase3:
            loss += l1_loss(image_tensor_hybrid_full, gt_image_tensor)
            if after_merge:
                ssim_notdense2 = _lam_ssim * (
                        1.0 - ssim(image_tensor_hybrid_full, gt_image_tensor[:, :3, :, :]))
                loss += ssim_notdense2

        # ---- Stage-Specific SSIM Loss ----
        if stage == 'coarse':
            ssim_loss_temp = ssim(image_tensor_second, gt_image_tensor)
            ssim_loss_temp1 = ssim(image_tensor_first, gt_image_tensor)
            loss += 0.2 * (1.0 - ssim_loss_temp)
            loss += 0.2 * (1.0 - ssim_loss_temp1)

        # ---- L_reg (paper Eq.8, Sec 3.7): temporal-smoothness / plane-TV on HexPlane ----
        if stage == "fine" and hypernetwork_config.time_smoothness_weight != 0:
            L_reg_tv = foreground_gaussians.compute_regulation(hypernetwork_config.time_smoothness_weight,
                                                              hypernetwork_config.l1_time_planes,
                                                              hypernetwork_config.plane_tv_weight)
            loss += L_reg_tv

        # ---- L_sep (paper Eq.8, Sec 3.4): dynamic-static separation ----
        # SSIM(C_d, C_s) inside the FG region, so G_d and G_s do not model the
        # same content. Distinct from L_diversity (transient-vs-persistent).
        # Pre-Phase-3 only: in Phase 3 the region mask is all-False anyway.
        if pre_phase3:
            dynamic_segment = image_tensor_first
            static_disposed = (image_tensor_second).clone().detach().cuda()
            d_ssim_str = (structural_ssim(dynamic_segment, static_disposed)) * (
                    fg_region_weight > 1 - vis_thresh)

            L_sep_fgbg = d_ssim_str.mean()
            loss += 0.01 * L_sep_fgbg

        # ---- L_diversity (paper Eq.8, Sec 3.3): SSIM(C_t, C_p) ----
        # Transient Contribution vs Persistent Render, accumulated per-view above.

        if len(L_diversity_terms) > 0:
            loss += 0.1 * torch.stack(L_diversity_terms).mean()

        loss += 0.1 * l1_loss(render_3d_tensor, gt_image_tensor)

        _fga_start = int(optimization_params.fg_alpha_mask_start_iter)
        if iteration > _fga_start and pre_phase3 and fg_alpha_render is not None:
            # L_reg_alpha -- the paper's alpha term (L_α, part of L_reg Eq.8):
            # pull the raw G_d alpha render toward the detached dynamic mask M_d.
            # Set lambda_fg_alpha_mask=0 to disable (skips the compute).
            _lam_fg_alpha = float(getattr(optimization_params, 'lambda_fg_alpha_mask', 0.01))
            if _lam_fg_alpha > 0:
                L_reg_alpha = l1_loss(fg_alpha_render, mask_render.clone().detach())
                loss += _lam_fg_alpha * L_reg_alpha

        # ---- Foreground/Background Component Losses ----
        if pre_phase3:

            # High-confidence region losses: same SSIM + L1 pair applied to the
            # persistent-dynamic render over its region, then the static render
            # over the complementary region.
            for _render, _region in ((image_tensor_first, fg_region_weight),
                                     (image_tensor_second, bg_region_weight)):
                _mask = _region > 1 - vis_thresh
                _l1 = l1_loss(_render * _mask, gt_image_tensor * _mask)
                loss += _lam_ssim * ((1.0 - ssim_raw(_render, gt_image_tensor)) * _mask).mean()
                loss += _l1

        # ---- L_reg (paper Eq.8, Sec 3.7): Gaussian scale regularization ----
        # exp(_scaling) is shared by the scale-cap and aspect-ratio terms below.
        max_scale = 0.1 * scene.cameras_extent
        _max_scale_t = torch.tensor(max_scale).cuda()
        scale_exps = (torch.exp(foreground_gaussians._scaling),
                      torch.exp(background_gaussians._scaling))
        for scale_exp in scale_exps:
            loss += 0.01 * (torch.maximum(scale_exp.amax(dim=-1),
                                          _max_scale_t) - max_scale).mean()

        # ---- L_reg (paper Eq.8, Sec 3.7): Gaussian aspect-ratio regularization ----
        # NOTE: the %10 duty cycle is part of the calibration — in expectation it
        # scales the 0.1 weight to ~0.01 per iteration. Do not change the cadence
        # without treating it as a loss-weight change (see LOSS MAP).
        if iteration % 10 == 0:
            for scale_exp in scale_exps:
                loss += 0.1 * (torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(5)) - 5).mean()

        # Phase-3 depth TV smoothness on persistent (FG+BG) depth.
        # Set lambda_phase3_depth_tv=0 to disable (skips the compute).
        _lam_depth_tv = float(getattr(optimization_params, 'lambda_phase3_depth_tv', 0.01))
        if iteration > phase3_start_iter and _lam_depth_tv > 0:  # strict >: skips the phase3 boundary iter
            depth_scaled = depth_images_dy_tensor / scene.cameras_extent
            loss_smooth_sta = depth_reg(depth_scaled.permute(0, 2, 3, 1), gt_image_tensor.permute(0, 2, 3, 1))
            loss += _lam_depth_tv * loss_smooth_sta

        # ---- Backpropagation ----

        loss.backward()

        # Emergency restart on NaN loss
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        # ---- Gradient Collection from Multi-Model Rendering ----
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor_list[0])
        viewspace_point_tensor_grad_second = torch.zeros_like(viewspace_point_tensor_list_second[0])
        viewspace_point_tensor_grad_transient = torch.zeros_like(viewspace_point_tensor_list_transient[0])[:, 0]

        for idx in range(0, len(viewspace_point_tensor_list)):
            if viewspace_point_tensor_list[idx].grad is not None:
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            if viewspace_point_tensor_list_second[idx].grad is not None:
                viewspace_point_tensor_grad_second = viewspace_point_tensor_grad_second + \
                                                     viewspace_point_tensor_list_second[idx].grad
            if viewspace_point_tensor_list_transient[idx].grad is not None:
                viewspace_point_tensor_grad_transient = viewspace_point_tensor_grad_transient + \
                                                        torch.norm(viewspace_point_tensor_list_transient[idx].grad[:, :2],
                                                                   dim=-1)

        viewspace_point_tensor_grad_transient[visibility_filter_transient] = viewspace_point_tensor_grad_transient[
                                                                                 visibility_filter_transient] * 2 / \
                                                                             visibility_filter_transient_count[
                                                                                 visibility_filter_transient]
        viewspace_point_tensor_grad_transient = viewspace_point_tensor_grad_transient.unsqueeze(1)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            ema_lossl1_for_log = 0.4 * Ll1.item() + 0.6 * ema_lossl1_for_log
            total_point = foreground_gaussians._xyz.shape[0]
            total_point_second = background_gaussians._xyz.shape[0]
            total_point_transient = transient_gaussian.get_xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "ll1": f"{ema_lossl1_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point": f"{total_point}",
                                          "point_second": f"{total_point_second}",
                                          "point_transient": f"{total_point_transient}"})
                progress_bar.update(10)
            if iteration == optimization_params.iterations:
                progress_bar.close()

            # ---- Logging and Saving ----
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)

            # ---- Debug Visualization (every 100 iterations) ----
            if iteration % 100 == 0:
                # These copies feed ONLY the viz below — computed here to avoid a
                # per-iteration D2H transfer / GPU cat at 1% duty cycle.
                image_second_to_show = image_tensor_second.clone().detach().cpu()
                save_train_debug_viz(
                    iteration=iteration, stage=stage, expname=expname,
                    saving_folder=optimization_params.saving_folder,
                    phase3_start_iter=phase3_start_iter,
                    gt_image_tensor=gt_image_tensor, image_tensor=image_tensor,
                    image_tensor_first=image_tensor_first, image_dy=image_dy,
                    image_second_to_show=image_second_to_show, image_sta=image_sta,
                    motion_masks=motion_masks,
                    images_4d_debug=images_4d_debug,
                    image_tensor_hybrid_full=image_tensor_hybrid_full)


            # FG->TR organic transfer is gated identically at both donor extraction
            # and add_points_from_dict, so compute once.
            fg_to_tr_active = (iteration % fg_to_tr_freq == 0
                               and iteration > fg_to_tr_start_iter
                               and iteration < fg_to_tr_end_iter)

            # Outside the FG densify block so it runs in both Phase 2 and 3;
            # uses the deformation-only helper, not render_foreground.
            prune_data = None
            if fg_to_tr_active:
                _max_transfer = int(getattr(optimization_params, 'fg_to_tr_max_transfer_value', 2000))
                deformed_state = foreground_gaussians.compute_transfer_state(
                    viewpoint_cam.time, compute_velocity=True)
                # Dynamic-score threshold for FG→TR donor pre-filter (default 0.05).
                _ds_thresh = float(getattr(optimization_params, 'fg_to_tr_dynamic_score_threshold', 0.05))
                prune_data = foreground_gaussians.select_fg_to_transient(
                    deformed_state,
                    dynamic_score_threshold=_ds_thresh,
                    max_transfer=_max_transfer,
                    time_duration=transient_gaussian.time_duration,
                    camera_center=viewpoint_cam.camera_center,
                    bias_factor=1e-6,
                )
                if iteration % 500 == 0:
                    # candidate count forces a device sync — compute only when printing.
                    _candidate_count = int((deformed_state['dynamic_score'] > _ds_thresh).sum().item())
                    _added = 0 if prune_data is None else int(prune_data['xyz'].shape[0])
                    _trc   = int(transient_gaussian.get_xyz.shape[0])
                    print(f"[FG2TR] iter={iteration} candidates={_candidate_count} "
                          f"max_transfer={_max_transfer} added={_added} tr_count={_trc}")

            # ---- Adaptive Gaussian Densification & Pruning ----
            if iteration < optimization_params.densify_until_iter and pre_phase3:
                maintain_foreground(
                    iteration, stage, optimization_params, scene, foreground_gaussians,
                    viewspace_point_tensor_grad, visibility_filter, last_max_weight,
                    weight_prune_freq, phase3_start_iter)
            # ---- Foreground Model Optimization ----
            if iteration < optimization_params.iterations:
                foreground_gaussians.optimizer.step()
                foreground_gaussians.optimizer.zero_grad(set_to_none=True)

            # ---- Background Model Management ----
            if iteration < optimization_params.densify_until_iter:
                maintain_background(iteration, stage, optimization_params, scene, background_gaussians,
                                    viewspace_point_tensor_grad_second, visibility_filter_second,
                                    last_max_weight_second, phase3_start_iter)
            # ---- Background Model Optimization ----
            if iteration < optimization_params.iterations:
                background_gaussians.optimizer.step()
                background_gaussians.optimizer.zero_grad(set_to_none=True)

            # ---- Transient Model Management ----
            if iteration < optimization_params.densify_until_iter:
                maintain_transient(iteration, optimization_params, scene, transient_gaussian,
                                   render_pkg_hybrid,
                                   viewspace_point_tensor_grad_transient, visibility_filter_transient,
                                   viewspace_point_tensor_list_transient,
                                   fg_to_tr_active, prune_data, weight_prune_fre_transient,
                                   phase3_start_iter)
            # ---- Transient Model Optimization ----
            if iteration < optimization_params.iterations:
                transient_gaussian.optimizer.step()
                transient_gaussian.optimizer.zero_grad(set_to_none=True)
                # Update learning rate for transient model
                transient_gaussian.update_learning_rate(iteration)

            # ---- Checkpoint Saving ----
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((foreground_gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + f"_{stage}_" + str(iteration) + ".pth")
                torch.save((background_gaussians.capture(), iteration),
                           scene.model_path + "/chkpnt" + f"_{stage}_gs2_" + str(iteration) + ".pth")

            # ---- Periodic Evaluation (Every 500 iterations, fine stage only:
            # coarse evals could never save images and only appended JSON) ----
            if stage == "fine" and iteration % 500 == 0:
                best_psnr = run_evaluation(iteration, scene, foreground_gaussians, background_gaussians,
                                           transient_gaussian,
                                           pipeline_config, optimization_params, expname, stage, best_psnr)

    # ---- Post-Training Evaluation (Fine Stage Only) ----
    if stage == "fine":
        final_test_dump(test_cams, foreground_gaussians, background_gaussians, transient_gaussian,
                        pipeline_config, background, optimization_params, expname, stage)


def training(dataset, hypernetwork_config, optimization_params, pipeline_config, saving_iterations,
             checkpoint_iterations, checkpoint, expname):
    """Entry-point that prepares data structures and launches the two-stage training."""
    prepare_output_folder(expname)
    dataset.sh_degree = 3
    foreground_gaussians = GaussianModel_dynamic(dataset.sh_degree, hypernetwork_config)
    background_gaussians = GaussianModel(dataset.sh_degree)
    # Initialize transient Gaussian as 4D model (like reference)
    time_duration = list(getattr(optimization_params, 'transient_time_duration', [0.0, 10.0]))
    print(f"[TR init] time_duration={time_duration}, max_gaussian_transient={getattr(optimization_params, 'max_gaussian_transient', 500000)}")
    rot_4d = True  # Set to False to match FreetimeGS snippet (no 4D rotation)
    sh_degree_t = 2  # Set to 0 to match FreetimeGS snippet (standard SHs)
    transient_gaussian = GaussianModelTransient(dataset.sh_degree, gaussian_dim=4, time_duration=time_duration,
                                                rot_4d=rot_4d, sh_degree_t=sh_degree_t)
    dataset.model_path = args.model_path

    import shutil
    os.makedirs(os.path.join(optimization_params.saving_folder, expname), exist_ok=True)
    # cfg_args was just written to args.model_path/cfg_args at module-init time.
    # Copy to saving_folder/expname/ for eval-side reproducibility.
    shutil.copyfile(os.path.join(args.model_path, 'cfg_args'),
                    os.path.join(optimization_params.saving_folder, expname, 'cfg_args'))

    scene = Scene2gs_mixed(dataset, foreground_gaussians, gaussians_second=background_gaussians,
                           gaussians_transient=transient_gaussian)

    scene_reconstruction(optimization_params, hypernetwork_config, pipeline_config,
                                 saving_iterations,
                                 checkpoint_iterations, checkpoint,
                                 foreground_gaussians, scene, "coarse",
                                 optimization_params.coarse_iterations,
                                 background_gaussians=background_gaussians,
                                 transient_gaussian=scene.gaussians_transient, expname=expname)

    foreground_gaussians.max_radii2D = torch.zeros_like(foreground_gaussians.max_radii2D).cuda()
    foreground_gaussians.mean_radii2D = torch.zeros_like(foreground_gaussians.mean_radii2D).cuda()
    background_gaussians.max_radii2D = torch.zeros_like(background_gaussians.max_radii2D).cuda()

    scene_reconstruction(optimization_params, hypernetwork_config, pipeline_config,
                                 saving_iterations,
                                 checkpoint_iterations, checkpoint,
                                 foreground_gaussians, scene, "fine", optimization_params.iterations,
                                 background_gaussians=background_gaussians,
                                 transient_gaussian=scene.gaussians_transient, expname=expname)

    from distutils.dir_util import copy_tree
    # Mirror the saved point clouds next to the renders, so an eval run needs
    # only saving_folder.
    copy_tree(os.path.join(args.model_path, "point_cloud"),
              os.path.join(optimization_params.saving_folder, expname, 'point_cloud'))


def prepare_output_folder(expname):
    """Create the run folder and write cfg_args for reproducibility.

    Parameters
    ----------
    expname : str
        Human-readable experiment identifier. The folder `./output/<expname>`
        will be created and the parsed command-line arguments written to
        `cfg_args` for reproducibility.
    """
    if not args.model_path:
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def setup_seed(seed):
    """Deterministic behaviour across PyTorch / NumPy / random for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Set up command line argument parser
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[19999, 39999, 79999, 99999, 119999])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams

        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # ---- Architecture: HexPlane spatiotemporal grid ----

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args),
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.expname)

    print("\nTraining complete.")
