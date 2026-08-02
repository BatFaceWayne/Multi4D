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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from .diff_gaussian_rasterization_hybrid import GaussianRasterizationSettings_hybrid, GaussianRasterizer_hybrid


def render_background(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                      stage="fine", prob_mask=None, dropout_use=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration

    means3D = pc.get_xyz
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        bwd_depth=True,
        bwd_dynamic=False,
        debug=pipe.debug
    )
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # Covariance is computed from scaling / rotation by the rasterizer (cov3D_precomp=None).
    scales = pc._scaling
    rotations = pc._rotation
    cov3D_precomp = None
    try:
        if "coarse" in stage:
            means3D_final, scales_final, rotations_final, opacity_final, shs_final = (
                means3D,
                scales,
                rotations,
                opacity,
                shs,
            )
        elif "fine" in stage:
            means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
                means3D, scales, rotations, opacity, shs, time
            )
        else:
            raise NotImplementedError
    except:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    if dropout_use:
        compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")

        # Apply DropGaussian with compensation
        drop_rate = 0.1
        d = torch.nn.Dropout(p=drop_rate)
        compensation = d(compensation)
        opacity = opacity * compensation[:, None]
    # SH -> RGB conversion is done by the rasterizer (no python-side precompute).
    colors_precomp = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    dynamics = torch.ones((means2D.shape[0], 1, means2D.shape[1]), dtype=means2D.dtype, device=means2D.device)

    rendered_image, radii, depth, dynamics_map, max_weight_t = rasterizer(
        means3D=means3D_final,
        means2D=means2D,
        shs=shs_final,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales_final,
        rotations=rotations_final,
        dynamics=dynamics,
        cov3D_precomp=cov3D_precomp,
        prob_mask=prob_mask)
    # Those Gaussians that were frustum culled or had a radi    us of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "max_weight_t": max_weight_t}


def render_foreground(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                      stage="fine", prob_mask=None, dropout_use=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration

    means3D = pc.get_xyz
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        bwd_depth=True,
        bwd_dynamic=False,
        prefiltered=False,
        debug=pipe.debug
    )
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # Covariance is computed from scaling / rotation by the rasterizer (cov3D_precomp=None).
    scales = pc._scaling
    rotations = pc._rotation
    cov3D_precomp = None

    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_full = (
            means3D,
            scales,
            rotations,
            opacity,
            shs,
        )
    elif "fine" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_full = pc._deformation(
            means3D, scales, rotations, opacity, shs, time
        )


    else:
        raise NotImplementedError

    save_key = {}

    # Deformed positions, exposed for the soft_dxv displacement-variance
    # accumulator in train.py (the only consumer of 'deformed_gs').
    # Detach to prevent gradient interference.
    with torch.no_grad():
        save_key['means3D_final'] = means3D_final.clone().detach()

    shs_final = shs_full[:, :16, :]
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    # (Assume the rest of the 3DGS pipeline is already set up)
    # Create initial compensation factor (1 for each Gaussian)
    if dropout_use:
        compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")

        # Apply DropGaussian with compensation
        drop_rate = 0.1
        d = torch.nn.Dropout(p=drop_rate)
        compensation = d(compensation)
        opacity = opacity * compensation[:, None]
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # SH -> RGB conversion is done by the rasterizer (colors_precomp=None below).
    dynamics = torch.ones((means2D.shape[0], 1, means2D.shape[1]), dtype=means2D.dtype,
                          device=means2D.device)

    rendered_image, radii_i, depth, dynamics_map, max_weight_t_f = rasterizer(
        means3D=means3D_final,
        means2D=means2D,
        shs=shs_final,
        colors_precomp=None,
        opacities=opacity,
        scales=scales_final,
        rotations=rotations_final,
        dynamics=dynamics,
        cov3D_precomp=cov3D_precomp,
        prob_mask=prob_mask
    )
    radii = torch.zeros((means3D_final.shape[0],), device=radii_i.device, dtype=torch.int32)
    radii[:] = radii_i

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    max_weight_t_f_orig = torch.zeros((means3D_final.shape[0],), device=radii_i.device, dtype=max_weight_t_f.dtype)
    max_weight_t_f_orig[:] = max_weight_t_f
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            # Standard 3DGS render output. Densification here is driven by
            # max_weight, not radii, so this field is diagnostic only.
            "radii": radii,
            "depth": depth,
            "max_weight_t": max_weight_t_f_orig,
            'deformed_gs': save_key}




def render_mask(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                stage="fine", dropout_use=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration

    means3D = pc.get_xyz

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        bwd_depth=False,
        bwd_dynamic=True,
        prefiltered=False,
        debug=pipe.debug
    )
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0], 1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    # Covariance is computed from scaling / rotation by the rasterizer (cov3D_precomp=None).
    scales = pc._scaling
    rotations = pc._rotation
    cov3D_precomp = None
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = (
            means3D,
            scales,
            rotations,
            opacity,
            shs,
        )
    elif "fine" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
            means3D, scales, rotations, opacity, shs, time
        )
    else:
        raise NotImplementedError

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    if dropout_use:
        compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")

        # Apply DropGaussian with compensation
        drop_rate = 0.1
        d = torch.nn.Dropout(p=drop_rate)
        compensation = d(compensation)
        opacity = opacity * compensation[:, None]

    colors_precomp = torch.zeros_like(pc.get_xyz)
    mask_dy = torch.sigmoid(shs_final[:, -1, 2])

    #### add temperature to seperation prediction favoring the seperation
    light_var = torch.sigmoid(shs_final[:, -1, 1])

    colors_precomp[..., 0] = light_var
    colors_precomp[..., 1] = torch.ones_like(light_var)  # mask_dy
    colors_precomp[..., -1] = mask_dy

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    dynamics = torch.ones((means2D.shape[0], 1, means2D.shape[1]), dtype=means2D.dtype, device=means2D.device)

    rendered_image, radii, depth, dynamics_map, max_weight_t_f = rasterizer(
        means3D=means3D_final,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales_final,
        rotations=rotations_final,
        dynamics=dynamics,
        cov3D_precomp=cov3D_precomp)

    return {"render": rendered_image}


def render_rawall3pc(viewpoint_camera, pc_foreground: GaussianModel, pc_background: GaussianModel,
                     pc_transient: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                     stage="fine", dropout_use=False):
    """
    Render all 3 sets of Gaussians:
    - Foreground (Deformed) + Background (Static) -> Combined Static input for Hybrid Rasterizer
    - Transient (4D) -> Dynamic input for Hybrid Rasterizer
    """

    # --- 1. Prepare Screenspace Points (Gradients) ---
    # We need separate screenspace points for each model to track gradients separately for densification

    # Transient
    screenspace_points_transient = torch.zeros_like(pc_transient.get_xyz, dtype=pc_transient.get_xyz.dtype,
                                                    requires_grad=True, device="cuda")
    try:
        screenspace_points_transient.retain_grad()
    except:
        pass

    # Foreground
    screenspace_points_fg = torch.zeros_like(pc_foreground.get_xyz, dtype=pc_foreground.get_xyz.dtype,
                                             requires_grad=True, device="cuda")
    try:
        screenspace_points_fg.retain_grad()
    except:
        pass

    # Background
    screenspace_points_bg = torch.zeros_like(pc_background.get_xyz, dtype=pc_background.get_xyz.dtype,
                                             requires_grad=True, device="cuda")
    try:
        screenspace_points_bg.retain_grad()
    except:
        pass

    # --- 2. Setup Rasterizer ---
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    time_to_scale = pc_transient.time_duration[0] + viewpoint_camera.time * (pc_transient.time_duration[1] - pc_transient.time_duration[0])

    raster_settings = GaussianRasterizationSettings_hybrid(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc_transient.active_sh_degree,
        sh_degree_t=pc_transient.active_sh_degree_t,
        campos=viewpoint_camera.camera_center.cuda(),
        timestamp=time_to_scale,
        time_duration=pc_transient.time_duration[1] - pc_transient.time_duration[0],
        rot_4d=pc_transient.rot_4d,
        gaussian_dim=pc_transient.gaussian_dim,
        force_sh_3d=False,
        prefiltered=False,
        opa_threshold=0.005,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer_hybrid(raster_settings=raster_settings)

    # --- 3. Prepare Transient (Dynamic) Input ---
    means3D = pc_transient.get_xyz
    means2D = screenspace_points_transient
    opacity = pc_transient.get_opacity

    scales_t = None
    rotations_r = None
    ts = None
    cov3D_precomp = None

    scales = pc_transient.get_scaling
    rotations = pc_transient.get_rotation
    if pc_transient.gaussian_dim == 4:
        scales_t = pc_transient.get_scaling_t
        ts = pc_transient.get_t
        if pc_transient.rot_4d:
            rotations_r = pc_transient.get_rotation_r

    shs = None
    colors_precomp = None
    shs = pc_transient.get_features
    if pc_transient.gaussian_dim == 4 and ts is None:
        ts = pc_transient.get_t

    flow_2d = torch.zeros_like(pc_transient.get_xyz[:, :2])


    # --- 4. Prepare Foreground (Deformed) ---
    means3D_fg_raw = pc_foreground.get_xyz
    opacity_fg = pc_foreground._opacity
    shs_fg_full = pc_foreground.get_features
    scales_fg = pc_foreground._scaling
    rotations_fg = pc_foreground._rotation

    if "coarse" in stage:
        means3D_fg, scales_fg, rotations_fg, opacity_fg, shs_fg_full = (
            means3D_fg_raw,
            scales_fg,
            rotations_fg,
            opacity_fg,
            shs_fg_full,
        )
    elif "fine" in stage:
        time_fg = torch.tensor(viewpoint_camera.time).to(means3D_fg_raw.device).repeat(means3D_fg_raw.shape[0], 1)
        means3D_fg, scales_fg, rotations_fg, opacity_fg, shs_fg_full = pc_foreground._deformation(
            means3D_fg_raw, scales_fg, rotations_fg, opacity_fg, shs_fg_full, time_fg
        )
    else:
        raise NotImplementedError

    screenspace_points_fg_filtered = screenspace_points_fg

    opacity_fg = pc_foreground.opacity_activation(opacity_fg)
    scales_fg = pc_foreground.scaling_activation(scales_fg)
    rotations_fg = pc_foreground.rotation_activation(rotations_fg)
    shs_fg = shs_fg_full[:, :16, :]

    # --- 5. Prepare Background (Static) ---
    means3D_bg = pc_background.get_xyz
    opacity_bg = pc_background.get_opacity  # Already activated
    scales_bg = pc_background.get_scaling  # Already activated
    rotations_bg = pc_background.get_rotation  # Already activated
    shs_bg_full = pc_background.get_features
    shs_bg = shs_bg_full[:, :16, :]

    # --- 6. Concatenate Foreground and Background ---
    means3D_static = torch.cat([means3D_fg, means3D_bg], dim=0)
    # Concatenate screenspace points for the rasterizer
    means2D_static = torch.cat([screenspace_points_fg_filtered, screenspace_points_bg], dim=0)

    opacity_static = torch.cat([opacity_fg, opacity_bg], dim=0)
    scales_static = torch.cat([scales_fg, scales_bg], dim=0)
    rotations_static = torch.cat([rotations_fg, rotations_bg], dim=0)
    shs_static = torch.cat([shs_fg, shs_bg], dim=0)

    # --- 7. Rasterize ---
    if dropout_use:
        compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")
        compensation2 = torch.ones(opacity_static.shape[0], dtype=torch.float32, device="cuda")

        # Apply DropGaussian with compensation
        drop_rate = 0.1
        d = torch.nn.Dropout(p=drop_rate)
        compensation = d(compensation)
        opacity = opacity * compensation[:, None]
        compensation2 = d(compensation2)
        opacity_static = opacity_static * compensation2[:, None]

    rendered_image, radii, depth, alpha, flow, covs_com, radii_static, color_4d, color_3d, invdepth, depth_4d, depth_3d, max_weight, max_weight_static = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        flow_2d=flow_2d,
        opacities=opacity,
        ts=ts,
        scales=scales,
        scales_t=scales_t,
        rotations=rotations,
        rotations_r=rotations_r,
        cov3D_precomp=cov3D_precomp,
        means3D_static=means3D_static,
        means2D_static=means2D_static,
        shs_static=shs_static,
        opacities_static=opacity_static,  # Use real opacity
        scales_static=scales_static,
        rotations_static=rotations_static,
        prob_mask=torch.ones((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)), dtype=torch.float32, device="cuda")
    )

    radii_all = radii

    # Split radii and weights for separate visibility filtering
    num_fg = means3D_fg.shape[0]
    radii_fg = radii_static[:num_fg]
    radii_bg = radii_static[num_fg:]

    radii_fg_full = torch.zeros((means3D_fg_raw.shape[0],), device=radii_fg.device, dtype=torch.int32)
    radii_fg_full[:] = radii_fg

    # Split max_weight_static into fg and bg components; map fg back to full size
    mw_fg_filtered = max_weight_static[:num_fg]
    mw_bg = max_weight_static[num_fg:]
    max_weights_fg_full = torch.zeros((means3D_fg_raw.shape[0],), device=mw_fg_filtered.device, dtype=torch.float32)
    max_weights_fg_full[:] = mw_fg_filtered

    return {"render": rendered_image,
            "viewspace_points": screenspace_points_transient,
            # Mapped to "viewspace_points" for backward compatibility if needed
            "visibility_filter": radii_all > 0,
            "depth": depth,
            "alpha": alpha,
            "render_4d": color_4d,
            "render_3d": color_3d,
            "depth_4d": depth_4d,
            "depth_3d": depth_3d,

            # Additional keys for separate models
            "viewspace_points_fg": screenspace_points_fg,
            "visibility_filter_fg": radii_fg_full > 0,

            "screenspace_points_bg": screenspace_points_bg,
            "visibility_filter_bg": radii_bg > 0,
            "max_weights_t": max_weight,
            "max_weights_fg": max_weights_fg_full,
            "max_weights_bg": mw_bg,
            }
