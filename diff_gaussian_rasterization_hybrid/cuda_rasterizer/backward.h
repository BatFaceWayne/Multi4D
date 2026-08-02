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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD {
void render(int P, const dim3 grid, const dim3 block, const uint2 *ranges,
            const uint32_t *point_list, int W, int H, int R, int B,
            const uint32_t *per_bucket_tile_offset,
            const uint32_t *bucket_to_tile, const float *sampled_T,
            const float *sampled_ar, const float *sampled_ard,
            const float *bg_color, const float2 *means2D,
            const float4 *conic_opacity, const float4 *conic_opacity_static,
            const float *colors, const float *colors_static,
            const float *depths, const float *flows_2d, const float *final_Ts,
            const uint32_t *n_contrib, const uint32_t *max_contrib,
            const float *pixel_colors, const float *pixel_indepths,
            const float *dL_dpixels, const float *dL_invdepths,
            const float *dL_depths, const float *dL_depths_4d,
            const float *dL_depths_3d, const float *dL_masks,
            const float *dL_dpix_flow, const float *dL_dpix_4d,
            const float *dL_dpix_3d, float3 *dL_dmean2D, float4 *dL_dconic2D,
            float *dL_dopacity, float *dL_dcolors, float *dL_dflows,
            float3 *dL_dmean2D_static, float4 *dL_dconic2D_static,
            float *dL_dopacity_static, float *dL_dcolors_static);

void preprocess(int P, int D, int D_t, int M, const float3 *means,
                const int *radii, const float *shs, const float *ts,
                const float *opacities, const bool *clamped,
                const uint32_t *tiles_touched, const glm::vec3 *scales,
                const float *scales_t, const glm::vec4 *rotations,
                const glm::vec4 *rotations_r, const float scale_modifier,
                const float *cov3Ds, const float *view, const float *proj,
                const float focal_x, float focal_y, const float tan_fovx,
                float tan_fovy, const glm::vec3 *campos, const float timestamp,
                const float time_duration, const bool rot_4d,
                const int gaussian_dim, const bool force_sh_3d,
                const float3 *dL_dmean2D, const float *dL_dconics,
                glm::vec3 *dL_dmeans, float *dL_dcolor, float *dL_dcov3D,
                float *dL_dsh, float *dL_dts, glm::vec3 *dL_dscale,
                float *dL_dscale_t, glm::vec4 *dL_drot, glm::vec4 *dL_drot_r,
                float *dL_dopacity);

void preprocess_static(
    int P, int D, int D_t, int M, const float3 *means_static,
    const int *radii_static, const float *shs_static,
    const float *opacities_static, const bool *clamped,
    const uint32_t *tiles_touched, const glm::vec3 *scales_static,
    const glm::vec4 *rotations_static, const float scale_modifier,
    const float *cov3Ds_static, const float *view, const float *proj,
    const float focal_x, float focal_y, const float tan_fovx, float tan_fovy,
    const glm::vec3 *campos, const float3 *dL_dmean2D_static,
    const float *dL_dconics_static, glm::vec3 *dL_dmeans_static,
    float *dL_dcolor_static, float *dL_dcov3D_static, float *dL_dsh_Static,
    glm::vec3 *dL_dscale_static, glm::vec4 *dL_drot_static,
    float *dL_dopacity_static);
} // namespace BACKWARD

#endif