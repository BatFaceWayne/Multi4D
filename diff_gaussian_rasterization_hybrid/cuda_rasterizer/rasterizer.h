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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <functional>
#include <vector>

namespace CudaRasterizer {
class Rasterizer {
public:
  static void markVisible(int P, float *means3D, float *viewmatrix,
                          float *projmatrix, bool *present);

  static std::tuple<int, int>
  forward(std::function<char *(size_t)> geometryBuffer,
          std::function<char *(size_t)> binningBuffer,
          std::function<char *(size_t)> imageBuffer,
          std::function<char *(size_t)> sampleBuffer, const int P,
          const int P_static, const int P_total, int D, int D_t, int M,
          int M_static, const float *background, const int width, int height,
          const float *means3D, float *out_means3D, const float *shs,
          const float *colors_precomp, const float *flows_precomp,
          const float *opacities, const float *ts, const float *scales,
          const float *scales_t, const float scale_modifier,
          const float *rotations, const float *rotations_r,
          const float *cov3D_precomp, const float *viewmatrix,
          const float *projmatrix, const float *cam_pos, const float timestamp,
          const float time_duration, const bool rot_4d, const int gaussian_dim,
          const bool force_sh_3d, const float tan_fovx, float tan_fovy,
          const bool prefiltered, float *out_color, float *out_invdepth,
          float *out_flow, float *out_depth, float *out_depth_4D,
          float *out_depth_3D, float *out_4D, float *out_3D, float *out_T,
          float opa_threshold, const float *means3D_static,
          const float *opacities_static, const float *scales_static,
          const float *rotations_static, const float *sh_static,
          int *radii = nullptr, int *radii_static = nullptr,
          float *max_weight = nullptr, float *max_weight_static = nullptr,
          const float *prob_mask = nullptr, bool debug = false);

  static void backward(
      const int P, const int P_static, int D, int D_t, int M, int M_static,
      int R, int B, const float *background, const int width, int height,
      const float *out_means3D, const float *shs, const float *colors_precomp,
      const float *flows_2d, const float *opacities, const float *ts,
      const float *scales, const float *scales_t, const float scale_modifier,
      const float *rotations, const float *rotations_r,
      const float *cov3D_precomp, const float *viewmatrix,
      const float *projmatrix, const float *campos, const float timestamp,
      const float time_duration, const bool rot_4d, const int gaussian_dim,
      const bool force_sh_3d, const float tan_fovx, float tan_fovy,
      const int *radii, const float *means3D_static, const float *shs_static,
      const float *opacities_static, const float *scales_static,
      const float *rotations_static, const int *radii_static, char *geom_buffer,
      char *binning_buffer, char *image_buffer, char *sample_buffer,
      const float *dL_dpix, const float *dL_invdepths, const float *dL_depths,
      const float *dL_depths_4d, const float *dL_depths_3d,
      const float *dL_masks, const float *dL_dpix_flow, const float *dL_dpix_4d,
      const float *dL_dpix_3d, float *dL_dmean2D, float *dL_dconic,
      float *dL_dopacity, float *dL_dcolor, float *dL_dmean3D, float *dL_dcov3D,
      float *dL_dsh, float *dL_dflows, float *dL_dts, float *dL_dscale,
      float *dL_dscale_t, float *dL_drot, float *dL_drot_r,
      float *dL_dmean2D_static, float *dL_dconic_static,
      float *dL_dopacity_static, float *dL_dmean3D_static,
      float *dL_dcov3D_static, float *dL_dcolor_static, float *dL_sh_static,
      float *dL_dscale_static, float *dL_drot_static, bool debug);
};
}; // namespace CudaRasterizer

#endif