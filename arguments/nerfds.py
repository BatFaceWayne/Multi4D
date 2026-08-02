# =============================================================================
# nerfds — NeRF-DS scenes: basin, bell, cup, plate, press, sieve
# =============================================================================
# Pairs with nerfds_as.py, which handles the `as` scene; the two-config split is
# deliberate. NeRF-DS is monocular, so this configuration is deliberately
# smaller than the dynerf one: a coarser temporal grid, fewer multi-resolution
# levels, a shallower and narrower deformation MLP, stronger grid
# regularization, single-camera batches, and an aggressive foreground
# weight-prune that keeps the persistent-dynamic branch compact.
# =============================================================================

ModelParams = dict(
    use_dynerf_sky_pc = False,   # no sky sphere in the init cloud (hurts sky-less
                                 # scenes; only dynerf_w.py turns it on)
)

# ---- Backbone: HexPlane spatiotemporal grid + deformation MLP -----------------
# HexPlane factors (x,y,z,t) into six 2D planes at multiple spatial resolutions.
# The deformation net queries it at (position, time) and predicts per-Gaussian
# offsets; which offsets are enabled is the enable_*_deform switches below.
ModelHiddenParams = dict(
    # six-plane grid: 2D planes over a 4D (x,y,z,t) domain, 32 features per plane,
    # spatial res 64^3 and 50 temporal bins (tuned for the shorter monocular clips).
    kplanes_config = {'grid_dimensions': 2, 'input_coordinate_dim': 4,
                      'output_coordinate_dim': 32, 'resolution': [64, 64, 64, 50]},
    multires       = [1, 2],        # fewer levels: cheaper + better on sparse monocular input
    defor_depth    = 0,             # no extra hidden layer (shallower net for NeRF-DS)
    net_width      = 64,

    # grid regularizers (plane total-variation + temporal smoothness + L1 on time planes)
    plane_tv_weight        = 0.0001,
    time_smoothness_weight = 0.01,
    l1_time_planes         = 0.0001,

    # ---- which deformations the net actually applies ----
    dx_deform_divisor = 10,           # position deform damped to dx/10
    enable_dr_deform  = True,         # additive rotation deformation
    enable_ds_deform  = False,        # NO scale deformation
    enable_do_deform  = False,        # NO opacity deformation
)

# ---- Training recipe ---------------------------------------------------------
OptimizationParams = dict(
    coarse_iterations                = 2000,
    densify_grad_threshold_coarse    = 0.0004,
    densify_grad_threshold_fine_init = 0.0004,
    densify_grad_threshold_after     = 0.0004,
    grid_lr_init                     = 0.0008,
    grid_lr_final                    = 0.000016,
    downscale_mask_deform_lr         = 0.1,
    lambda_main_loss                 = 1,                 # monocular regime
    transient_merging_iter           = 3000,              # TR joins the loss after this
    fg_to_tr_velocity_scale          = 0.1,               # damp inherited FG velocity at lifting
    fg_to_tr_max_transfer_value      = 1000,
    tr_densify_divisor               = 2,                 # TR densify threshold = FG / 2
    tr_weight_prune_freq             = 200,               # the only TR outflow
    tr_weight_prune_start_iter       = 6000,
    max_gaussian_transient           = 1_000_000,
    use_soft_dxv_loss                = True,              # soft_dxv loss; load-bearing (-0.22 PSNR if dropped)
    lambda_soft_dxv                  = 0.005,
    soft_dxv_threshold               = 0.001,
    soft_dxv_start_iter              = 3000,
    dx_var_reset_interval            = 1000,
    lambda_depth_order               = 0.01,              # Phase-3 depth-order loss

    # ---- dataset-specific overrides ----
    batch_size                       = 1,                 # monocular regime
    enable_transient_densify         = False,             # TR grows only via capped FG->TR transfer
    transient_time_duration          = [0.0, 1.0],
    fg_to_tr_transfer_freq           = 100,
    fg_mean_ratio                    = 1.0,               # aggressive FG weight-prune (FG ~25k -> ~3k)
    fg_prune_thresh_ceil             = 0.2,
)
