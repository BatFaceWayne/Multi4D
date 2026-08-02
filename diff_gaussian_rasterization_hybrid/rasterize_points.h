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

#pragma once
#include <cstdio>
#include <string>
#include <torch/extension.h>
#include <tuple>

std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor &background, const torch::Tensor &means3D,
    const torch::Tensor &colors, const torch::Tensor &flows,
    const torch::Tensor &opacity, const torch::Tensor &ts,
    const torch::Tensor &scales, const torch::Tensor &scales_t,
    const torch::Tensor &rotations, const torch::Tensor &rotations_r,
    const torch::Tensor &means3D_static, const torch::Tensor &sh_static,
    const torch::Tensor &opacities_static, const torch::Tensor &scales_static,
    const torch::Tensor &rotations_static, const float scale_modifier,
    const torch::Tensor &cov3D_precomp, const torch::Tensor &viewmatrix,
    const torch::Tensor &projmatrix, const float tan_fovx, const float tan_fovy,
    const int image_height, const int image_width, const torch::Tensor &sh,
    const int degree, const int degree_t, const torch::Tensor &campos,
    const float timestamp, const float time_duration, const bool rot_4d,
    const int gaussian_dim, const bool force_sh_3d, const bool prefiltered,
    const float opa_threshold, const torch::Tensor &prob_mask,
    const bool debug = false);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor &background, const torch::Tensor &means3D,
    const torch::Tensor &out_means3D, const torch::Tensor &radii,
    const torch::Tensor &colors, const torch::Tensor &flows_2d,
    const torch::Tensor &opacities, const torch::Tensor &ts,
    const torch::Tensor &scales, const torch::Tensor &scales_t,
    const torch::Tensor &rotations, const torch::Tensor &rotations_r,
    const torch::Tensor &means3D_static, const torch::Tensor &radii_static,
    const torch::Tensor &sh_static, const torch::Tensor &opacities_static,
    const torch::Tensor &scales_static, const torch::Tensor &rotations_static,
    const float scale_modifier, const torch::Tensor &cov3D_precomp,
    const torch::Tensor &viewmatrix, const torch::Tensor &projmatrix,
    const float tan_fovx, const float tan_fovy,
    const torch::Tensor &dL_dout_color, const torch::Tensor &dL_dout_depth,
    const torch::Tensor &dL_dout_depth_4d,
    const torch::Tensor &dL_dout_depth_3d, const torch::Tensor &dL_dout_mask,
    const torch::Tensor &dL_dout_flow, const torch::Tensor &sh,
    const torch::Tensor &dL_dout_invdepth,
    const torch::Tensor &dL_dout_color_4d,
    const torch::Tensor &dL_dout_color_3d, const int degree, const int degree_t,
    const torch::Tensor &campos, const float timestamp,
    const float time_duration, const bool rot_4d, const int gaussian_dim,
    const bool force_sh_3d, const torch::Tensor &geomBuffer, const int R,
    const torch::Tensor &binningBuffer, const torch::Tensor &imageBuffer,
    const int B, const torch::Tensor &sampleBuffer, const bool debug);

torch::Tensor markVisible(torch::Tensor &means3D, torch::Tensor &viewmatrix,
                          torch::Tensor &projmatrix);