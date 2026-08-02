/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD {
// Perform initial steps for each Gaussian prior to rasterization.
void preprocess(int P, int D, int D_t, int M, const float *orig_points,
                float *out_means3D, const float *ts, const glm::vec3 *scales,
                const float *scales_t, const float scale_modifier,
                const glm::vec4 *rotations, const glm::vec4 *rotations_r,
                const float *opacities, const float *shs, bool *clamped,
                const float *cov3D_precomp, const float *colors_precomp,
                const float *viewmatrix, const float *projmatrix,
                const glm::vec3 *cam_pos, const float timestamp,
                const float time_duration, const bool rot_4d,
                const int gaussian_dim, const bool force_sh_3d, const int W,
                int H, const float focal_x, float focal_y, const float tan_fovx,
                float tan_fovy, int *radii, float opa_threshold,
                float2 *points_xy_image, float *depths, float *cov3Ds,
                float *colors, float4 *conic_opacity, const dim3 grid,
                uint32_t *tiles_touched, bool prefiltered);

// Main rasterization method.
void render(int P, const dim3 grid, dim3 block, const uint2 *ranges,
            const uint32_t *point_list, const uint32_t *per_tile_bucket_offset,
            uint32_t *bucket_to_tile, float *sampled_T, float *sampled_ar,
            float *sampled_ard, int W, int H, const float2 *means2D,
            const float *colors, const float *colors_static, const float *flows,
            const float *depths, const float4 *conic_opacity,
            const float4 *conic_opacity_static, float *final_T,
            uint32_t *n_contrib, uint32_t *max_contrib, const float *bg_color,
            float *out_color, float *out_invdepth, float *out_flow,
            float *out_depth, float *out_depth_4D, float *out_depth_3D,
            float *out_4D, float *out_3D, float *out_T, float *max_weight,
            float *max_weight_static, const float *prob_mask);

void preprocess_static(
    int P, int P_static, int D, int M, const float *means3D_static,
    const glm::vec3 *scales_static, const float scale_modifier,
    const glm::vec4 *rotations_static, const float *opacities_static,
    const float *shs_static, bool *clamped, const float *viewmatrix,
    const float *projmatrix, const glm::vec3 *cam_pos, const int W, int H,
    const float focal_x, float focal_y, const float tan_fovx, float tan_fovy,
    int *radii_static, float2 *means2D, float *depths, float *cov3Ds_static,
    float *rgb_static, float4 *conic_opacity_static, const dim3 grid,
    uint32_t *tiles_touched, bool prefiltered);
} // namespace FORWARD

#endif