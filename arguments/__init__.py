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

from argparse import ArgumentParser
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self.eval = True
        # Add a 50k-point sky sphere to the dynerf init point cloud. Helps wide outdoor
        # scenes, hurts sky-less indoor ones — only the dynerf_w config enables it.
        self.use_dynerf_sky_pc = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 128
        self.timebase_pe = 4
        self.defor_depth = 1
        self.posebase_pe = 10
        self.scale_rotation_pe = 2
        self.opacity_pe = 2
        self.timenet_width = 64
        # self.timenet_width = 512
        # self.timenet_output = 128
        self.timenet_output = 32
        self.bounds = 1.6
        self.plane_tv_weight = 0.0002
        self.time_smoothness_weight = 0.001
        self.l1_time_planes = 0.0001
        # 4th entry ~= half the sequence length (150 for 300-frame dynerf clips).
        self.kplanes_config = {
            'grid_dimensions': 2,
            'input_coordinate_dim': 4,
            'output_coordinate_dim': 16,
            'resolution': [64, 64, 64, 150]
        }
        self.multires = [1, 2, 4]


        # ---- Which deformations the network applies ----
        # When True, the rotations_deform head is added to the persistent-dynamic
        # rotations (additive form). When False, rotation passes through unchanged.
        self.enable_dr_deform = False    # rotation passthrough; True = rotation-deformation head

        # Position/scale/opacity deformation. The wide-scene config uses the fully
        # expressive setting: dx_deform_divisor=1 with the scale and opacity heads on.
        self.dx_deform_divisor = 10      # current: dx/10. Set 1 for full-magnitude position deform.
        self.enable_ds_deform = False    # current: scale passthrough. True = scale-deformation MLP.
        self.enable_do_deform = False    # current: opacity passthrough. True = opacity-deformation MLP.

        super().__init__(parser, "ModelHiddenParams")


# NOTE: the values below ARE the shipped defaults — the former
# arguments/video_dataset/default.py was merged in here (2026-07-30), so there is
# exactly one place to read a default from. The 4 dataset configs in arguments/
# override only what they actually change.
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 20_000
        self.coarse_iterations = 1000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 20_000
        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01
        self.grid_lr_init = 0.0016
        self.grid_lr_final = 0.00016

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.opacity_reset_interval = 2000
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 16_000
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002
        self.pruning_from_iter = 500
        self.pruning_interval = 600
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005
        self.batch_size = 2
        self.enable_transient_densify = True  # False => TR grows only via capped FG->TR transfer (v14/monocular; avoids TR explosion)
        self.tr_densify_divisor = 6        # TR densify threshold = FG_threshold / N (smaller N = larger threshold = less TR growth)
        # 0 = no upper bound (reset fires until end of training).
        # Set to phase3_start_iter (10000) to mirror FG (last fire iter 8000 with interval=2000).

        # Damping factor for inherited FG velocity at FG→TR transfer init.
        # 1.0 (default) = velocity-to-rotation_r uses the full FG velocity (current behavior post-velocity-fix).
        # 0.5 = TR's temporal axis is half-aligned with FG motion.
        # 0.1 = TR's temporal axis is nearly motion-free.
        # 0.0 = TR's temporal axis is identity (no motion alignment); pre-velocity-fix-style behavior.
        # Tests whether the strict velocity-aligned rotation_r makes TR too similar to FG donors.
        self.fg_to_tr_velocity_scale = 1.0


        # ---- FG -> TR lifting (donor flow) ----
        # Number of persistent-dynamic donors lifted into the transient branch per transfer.
        self.fg_to_tr_max_transfer_value = 2000
        # TR weight-prune cadence (iters between fires) — the only transient outflow.
        self.tr_weight_prune_freq = 200
        # TR weight-prune gate: first iteration at which it fires.
        self.tr_weight_prune_start_iter = 6000


        #### training set up
        self.foreground_oneupshinterval = 200
        self.background_oneupshinterval = 200
        self.saving_folder = './results/'
        self.lambda_main_loss = 1
        self.max_gaussian_foreground = 90000
        self.max_gaussian_background = 960000
        # ---- TR temporal extent + capacity ----
        # transient_time_duration: time window covered by each TR Gaussian's temporal extent.
        # Wider = each TR Gaussian represents more time = fewer TR points needed for coverage.
        # max_gaussian_transient: hard cap on TR count for densification gating.
        self.transient_time_duration = [0.0, 10.0]
        self.max_gaussian_transient = 500000
        # ---- Depth-order loss gate ----
        # Default 0 = OFF.
        # When > 0: loss += lambda_depth_order * relu(depth_4d - depth_3d.detach()).mean()
        # Phase-3-only — depth_4d/depth_3d lists are populated only inside
        # the iter >= phase3_start_iter one-render branch (matches sweep-era code).
        self.lambda_depth_order = 0.0

        # ---- FG alpha-mask loss: the paper's alpha term ----
        # L_reg_alpha (paper L_α, part of L_reg Eq.8).
        # Weight on L1(fg_alpha_render, dynamic mask M_d), active 3000 <= iter < phase3.
        # Every knob MUST be declared here: merge_hparams only applies a config key
        # `if hasattr(args, key)`, so an undeclared knob is silently ignored.
        # Raising this weight past ~5x degrades quality and seeds static floaters.
        self.lambda_fg_alpha_mask = 0.01
        # First iter at which L_reg_alpha becomes active (gate is strict >).
        # 3000 = shipped. Kept as its own knob so a reparametrized schedule
        # (e.g. a 2x-stretched run) can move it with the other phase points
        # instead of silently desyncing from transient_merging_iter.
        self.fg_alpha_mask_start_iter = 3000
        # Phase-3 edge-aware depth-TV weight (0 disables).
        self.lambda_phase3_depth_tv = 0.01
        # FG->TR donor pre-filter: minimum dynamic score to be eligible for lifting.
        self.fg_to_tr_dynamic_score_threshold = 0.05
        # Unified SSIM loss weight. 0.2 is the shipped setting: PSNR-neutral with about
        # 25% fewer Gaussians than 0.4, at a small SSIM/LPIPS cost. See the loss map
        # in train.py for every site this weight feeds.
        self.lambda_ssim = 0.2
        self.downscale_mask_deform_lr = 0.1
        self.SH_lr_downscaling_start = 8
        self.SH_lr_downscaling_end = 20
        # FG weight-prune aggressiveness. fg_mean_ratio=1.0 with ceil=0.2 prunes much
        # harder (used by the NeRF-DS config). See the FG weight_prune block in train.py.
        self.fg_mean_ratio = 0.8
        self.fg_prune_thresh_ceil = 0.1
        self.transient_merging_iter = 6000
        # Iter at which Phase 3 begins: switch to one-render (render_rawall3pc only),
        # simplified composite loss, FG densify/self-prune ends, TR weight-prune starts.
        self.phase3_start_iter = 10000
        # FG->TR organic transfer window. Sentinel value 0 means "use the default":
        #   start_iter=0 -> fall back to transient_merging_iter
        #   end_iter=0   -> fall back to phase3_start_iter
        # When end_iter > phase3_start_iter, the donor state must come from a
        # render-independent helper (see GaussianModel_dynamic.compute_transfer_state).
        self.fg_to_tr_transfer_freq = 50




        self.dx_var_reset_interval      = 0      # if >0, rolling window reset every N iters


        # ---- Soft routing losses (continuous, gradient-driven; replace hard transfers) ----
        self.use_soft_dxv_loss     = False   # FG opacity penalty weighted by "static score" from Var(Δx)
        self.lambda_soft_dxv       = 0.005
        self.soft_dxv_threshold    = 0.001   # exp(-var/threshold) saturation scale
        self.soft_dxv_start_iter   = 3000








        super().__init__(parser, "Optimization Parameters")


