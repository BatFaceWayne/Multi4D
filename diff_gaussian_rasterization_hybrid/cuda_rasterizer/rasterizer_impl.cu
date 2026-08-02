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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "rasterizer_impl.h"
#include <algorithm>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <numeric>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "backward.h"
#include "forward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n) {
  uint32_t msb = sizeof(n) * 4;
  uint32_t step = msb;
  while (step > 1) {
    step /= 2;
    if (n >> msb)
      msb += step;
    else
      msb -= step;
  }
  if (n >> msb)
    msb++;
  return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P, const float *orig_points,
                             const float *viewmatrix, const float *projmatrix,
                             bool *present) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P)
    return;

  float3 p_view;
  float3 orig_point = {orig_points[3 * idx], orig_points[3 * idx + 1],
                       orig_points[3 * idx + 2]};
  present[idx] = in_frustum(orig_point, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps.
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(int P, int P_total, const float2 *points_xy,
                                  const float *depths, const uint32_t *offsets,
                                  uint64_t *gaussian_keys_unsorted,
                                  uint32_t *gaussian_values_unsorted,
                                  int *radii, int *radii_static, dim3 grid) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P_total)
    return;

  // Generate no key/value pair for invisible Gaussians
  if (((idx < P) && (radii[idx] > 0)) ||
      ((idx >= P) && (radii_static[idx - P] > 0))) {
    // Find this Gaussian's offset in buffer for writing keys/values.
    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
    uint2 rect_min, rect_max;

    if (idx < P)
      getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
    else
      getRect(points_xy[idx], radii_static[idx - P], rect_min, rect_max, grid);

    // For each tile that the bounding rect overlaps, emit a
    // key/value pair. The key is |  tile ID  |      depth      |,
    // and the value is the ID of the Gaussian. Sorting the values
    // with this key yields Gaussian IDs in a list, such that they
    // are first sorted by tile and then by depth.
    for (int y = rect_min.y; y < rect_max.y; y++) {
      for (int x = rect_min.x; x < rect_max.x; x++) {
        uint64_t key = y * grid.x + x;
        key <<= 32;
        key |= *((uint32_t *)&depths[idx]);
        gaussian_keys_unsorted[off] = key;
        gaussian_values_unsorted[off] = idx;
        off++;
      }
    }
  }
}

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t *point_list_keys,
                                   uint2 *ranges) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= L)
    return;

  // Read tile ID from key. Update start/end of tile range if at limit.
  uint64_t key = point_list_keys[idx];
  uint32_t currtile = key >> 32;
  bool valid_tile = currtile != (uint32_t)-1;

  if (idx == 0)
    ranges[currtile].x = 0;
  else {
    uint32_t prevtile = point_list_keys[idx - 1] >> 32;
    if (currtile != prevtile) {
      ranges[prevtile].y = idx;
      if (valid_tile)
        ranges[currtile].x = idx;
    }
  }
  if (idx == L - 1 && valid_tile)
    ranges[currtile].y = L;
}

// for each tile, see how many buckets/warps are needed to store the state
__global__ void perTileBucketCount(int T, uint2 *ranges,
                                   uint32_t *bucketCount) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= T)
    return;

  uint2 range = ranges[idx];
  int num_splats = range.y - range.x;
  int num_buckets = (num_splats + 31) / 32;
  bucketCount[idx] = (uint32_t)num_buckets;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(int P, float *means3D,
                                             float *viewmatrix,
                                             float *projmatrix, bool *present) {
  checkFrustum<<<(P + 255) / 256, 256>>>(P, means3D, viewmatrix, projmatrix,
                                         present);
}

CudaRasterizer::GeometryState
CudaRasterizer::GeometryState::fromChunk(char *&chunk, size_t P,
                                         size_t P_static, size_t P_total) {
  GeometryState geom;
  obtain(chunk, geom.depths, P_total, 128);
  obtain(chunk, geom.clamped, P * 3, 128);
  obtain(chunk, geom.internal_radii, P, 128);
  obtain(chunk, geom.internal_radii_static, P_static, 128);
  obtain(chunk, geom.means2D, P_total, 128);
  obtain(chunk, geom.cov3D, P * 6, 128);
  obtain(chunk, geom.conic_opacity, P, 128);
  obtain(chunk, geom.rgb, P * 3, 128);
  obtain(chunk, geom.tiles_touched, P_total, 128);
  cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched,
                                geom.tiles_touched, P_total);
  obtain(chunk, geom.scanning_space, geom.scan_size, 128);
  obtain(chunk, geom.point_offsets, P_total, 128);
  obtain(chunk, geom.conic_opacity_static, P_static, 128);
  obtain(chunk, geom.rgb_static, P_static * 3, 128);
  obtain(chunk, geom.cov3D_static, P_static * 6, 128);
  return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char *&chunk,
                                                                 size_t N) {
  ImageState img;
  obtain(chunk, img.accum_alpha, N, 128);
  obtain(chunk, img.n_contrib, N, 128);
  obtain(chunk, img.ranges, N, 128);
  int *dummy = nullptr;
  int *wummy = nullptr;
  cub::DeviceScan::InclusiveSum(nullptr, img.scan_size, dummy, wummy, N);
  obtain(chunk, img.contrib_scan, img.scan_size, 128);

  obtain(chunk, img.max_contrib, N, 128);
  obtain(chunk, img.pixel_colors, N * NUM_CHANNELS, 128);
  obtain(chunk, img.pixel_invDepths, N, 128);
  obtain(chunk, img.bucket_count, N, 128);
  obtain(chunk, img.bucket_offsets, N, 128);
  cub::DeviceScan::InclusiveSum(nullptr, img.bucket_count_scan_size,
                                img.bucket_count, img.bucket_count, N);
  obtain(chunk, img.bucket_count_scanning_space, img.bucket_count_scan_size,
         128);

  return img;
}

CudaRasterizer::BinningState
CudaRasterizer::BinningState::fromChunk(char *&chunk, size_t P) {
  BinningState binning;
  obtain(chunk, binning.point_list, P, 128);
  obtain(chunk, binning.point_list_unsorted, P, 128);
  obtain(chunk, binning.point_list_keys, P, 128);
  obtain(chunk, binning.point_list_keys_unsorted, P, 128);
  cub::DeviceRadixSort::SortPairs(
      nullptr, binning.sorting_size, binning.point_list_keys_unsorted,
      binning.point_list_keys, binning.point_list_unsorted, binning.point_list,
      P);
  obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
  return binning;
}

CudaRasterizer::SampleState CudaRasterizer::SampleState::fromChunk(char *&chunk,
                                                                   size_t C) {
  SampleState sample;
  obtain(chunk, sample.bucket_to_tile, C * BLOCK_SIZE, 128);
  obtain(chunk, sample.T, C * BLOCK_SIZE, 128);
  obtain(chunk, sample.ar, NUM_CHANNELS * C * BLOCK_SIZE, 128);
  obtain(chunk, sample.ard, C * BLOCK_SIZE, 128);
  return sample;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
std::tuple<int, int> CudaRasterizer::Rasterizer::forward(
    std::function<char *(size_t)> geometryBuffer,
    std::function<char *(size_t)> binningBuffer,
    std::function<char *(size_t)> imageBuffer,
    std::function<char *(size_t)> sampleBuffer, const int P, const int P_static,
    const int P_total, int D, int D_t, int M, int M_static,
    const float *background, const int width, int height, const float *means3D,
    float *out_means3D, const float *shs, const float *colors_precomp,
    const float *flows_precomp, const float *opacities, const float *ts,
    const float *scales, const float *scales_t, const float scale_modifier,
    const float *rotations, const float *rotations_r,
    const float *cov3D_precomp, const float *viewmatrix,
    const float *projmatrix, const float *cam_pos, const float timestamp,
    const float time_duration, const bool rot_4d, const int gaussian_dim,
    const bool force_sh_3d, const float tan_fovx, float tan_fovy,
    const bool prefiltered, float *out_color, float *invdepth, float *out_flow,
    float *out_depth, float *out_depth_4D, float *out_depth_3D, float *out_4D,
    float *out_3D, float *out_T, float opa_threshold,
    const float *means3D_static, const float *opacities_static,
    const float *scales_static, const float *rotations_static,
    const float *shs_static, int *radii, int *radii_static, float *max_weight,
    float *max_weight_static, const float *prob_mask, bool debug) {
  const float focal_y = height / (2.0f * tan_fovy);
  const float focal_x = width / (2.0f * tan_fovx);

  size_t chunk_size = required_geom<GeometryState>(P, P_static, P_total);
  char *chunkptr = geometryBuffer(chunk_size);
  GeometryState geomState =
      GeometryState::fromChunk(chunkptr, P, P_static, P_total);

  if (radii == nullptr) {
    radii = geomState.internal_radii;
  }
  if (radii_static == nullptr) {
    radii_static = geomState.internal_radii_static;
  }

  dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X,
                 (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  dim3 block(BLOCK_X, BLOCK_Y, 1);

  // Dynamically resize image-based auxiliary buffers during training
  size_t img_chunk_size = required<ImageState>(width * height);
  char *img_chunkptr = imageBuffer(img_chunk_size);
  ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

  if (NUM_CHANNELS != 3 && colors_precomp == nullptr) {
    throw std::runtime_error(
        "For non-RGB, provide precomputed Gaussian colors!");
  }

  // Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs
  // to RGB)
  CHECK_CUDA(FORWARD::preprocess(
                 P, D, D_t, M, means3D, out_means3D, ts, (glm::vec3 *)scales,
                 scales_t, scale_modifier, (glm::vec4 *)rotations,
                 (glm::vec4 *)rotations_r, opacities, shs, geomState.clamped,
                 cov3D_precomp, colors_precomp, viewmatrix, projmatrix,
                 (glm::vec3 *)cam_pos, timestamp, time_duration, rot_4d,
                 gaussian_dim, force_sh_3d, width, height, focal_x, focal_y,
                 tan_fovx, tan_fovy, radii, opa_threshold, geomState.means2D,
                 geomState.depths, geomState.cov3D, geomState.rgb,
                 geomState.conic_opacity, tile_grid, geomState.tiles_touched,
                 prefiltered),
             debug)

  // If we have static Gaussians, preprocess them as well
  if (P_static > 0) {
    CHECK_CUDA(FORWARD::preprocess_static(
                   P, P_static, D, M_static, means3D_static,
                   (glm::vec3 *)scales_static, scale_modifier,
                   (glm::vec4 *)rotations_static, opacities_static, shs_static,
                   geomState.clamped, viewmatrix, projmatrix,
                   (glm::vec3 *)cam_pos, width, height, focal_x, focal_y,
                   tan_fovx, tan_fovy, radii_static, geomState.means2D,
                   geomState.depths, geomState.cov3D_static,
                   geomState.rgb_static, geomState.conic_opacity_static,
                   tile_grid, geomState.tiles_touched, prefiltered),
               debug)
  }

  // Compute prefix sum over full list of touched tile counts by Gaussians
  // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
  CHECK_CUDA(cub::DeviceScan::InclusiveSum(
                 geomState.scanning_space, geomState.scan_size,
                 geomState.tiles_touched, geomState.point_offsets, P_total),
             debug)

  // Retrieve total number of Gaussian instances to launch and resize aux
  // buffers
  int num_rendered;
  CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P_total - 1,
                        sizeof(int), cudaMemcpyDeviceToHost),
             debug);

  size_t binning_chunk_size = required<BinningState>(num_rendered);
  char *binning_chunkptr = binningBuffer(binning_chunk_size);
  BinningState binningState =
      BinningState::fromChunk(binning_chunkptr, num_rendered);

  // For each instance to be rendered, produce adequate [ tile | depth ] key
  // and corresponding dublicated Gaussian indices to be sorted
  duplicateWithKeys<<<(P_total + 255) / 256, 256>>>(
      P, P_total, geomState.means2D, geomState.depths, geomState.point_offsets,
      binningState.point_list_keys_unsorted, binningState.point_list_unsorted,
      radii, radii_static, tile_grid) CHECK_CUDA(, debug)

      // int bit = getHigherMsb(tile_grid.x * tile_grid.y);
      int bit = 32;

  // Sort complete list of (duplicated) Gaussian indices by keys
  CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
                 binningState.list_sorting_space, binningState.sorting_size,
                 binningState.point_list_keys_unsorted,
                 binningState.point_list_keys, binningState.point_list_unsorted,
                 binningState.point_list, num_rendered, 0, 32 + bit),
             debug)

  CHECK_CUDA(
      cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)),
      debug);

  // Identify start and end of per-tile workloads in sorted list
  if (num_rendered > 0)
    identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
        num_rendered, binningState.point_list_keys, imgState.ranges);
  CHECK_CUDA(, debug)

  // bucket count
  int num_tiles = tile_grid.x * tile_grid.y;
  perTileBucketCount<<<(num_tiles + 255) / 256, 256>>>(
      num_tiles, imgState.ranges, imgState.bucket_count);
  CHECK_CUDA(cub::DeviceScan::InclusiveSum(imgState.bucket_count_scanning_space,
                                           imgState.bucket_count_scan_size,
                                           imgState.bucket_count,
                                           imgState.bucket_offsets, num_tiles),
             debug)
  unsigned int bucket_sum;
  CHECK_CUDA(cudaMemcpy(&bucket_sum, imgState.bucket_offsets + num_tiles - 1,
                        sizeof(unsigned int), cudaMemcpyDeviceToHost),
             debug);
  // create a state to store. size is number is the total number of buckets *
  // block_size
  size_t sample_chunk_size = required<SampleState>(bucket_sum);
  char *sample_chunkptr = sampleBuffer(sample_chunk_size);
  SampleState sampleState = SampleState::fromChunk(sample_chunkptr, bucket_sum);

  // Let each tile blend its range of Gaussians independently in parallel
  const float *feature_ptr =
      colors_precomp != nullptr ? colors_precomp : geomState.rgb;
  const float *flow_ptr = flows_precomp;
  CHECK_CUDA(FORWARD::render(
                 P, tile_grid, block, imgState.ranges, binningState.point_list,
                 imgState.bucket_offsets, sampleState.bucket_to_tile,
                 sampleState.T, sampleState.ar, sampleState.ard, width, height,
                 geomState.means2D, feature_ptr, geomState.rgb_static, flow_ptr,
                 geomState.depths, geomState.conic_opacity,
                 geomState.conic_opacity_static, imgState.accum_alpha,
                 imgState.n_contrib, imgState.max_contrib, background,
                 out_color, invdepth, out_flow, out_depth, out_depth_4D,
                 out_depth_3D, out_4D, out_3D, out_T, max_weight,
                 max_weight_static, prob_mask),
             debug)

  CHECK_CUDA(cudaMemcpy(out_T, imgState.accum_alpha,
                        width * height * sizeof(float),
                        cudaMemcpyDeviceToDevice),
             debug);
  CHECK_CUDA(cudaMemcpy(imgState.pixel_colors, out_color,
                        sizeof(float) * width * height * NUM_CHANNELS,
                        cudaMemcpyDeviceToDevice),
             debug);
  CHECK_CUDA(cudaMemcpy(imgState.pixel_invDepths, invdepth,
                        sizeof(float) * width * height,
                        cudaMemcpyDeviceToDevice),
             debug);
  return std::make_tuple(num_rendered, bucket_sum);
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
    const int P, const int P_static, int D, int D_t, int M, int M_static, int R,
    int B, const float *background, const int width, int height,
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
    char *binning_buffer, char *img_buffer, char *sample_buffer,
    const float *dL_dpix, const float *dL_invdepths, const float *dL_depths,
    const float *dL_depths_4d, const float *dL_depths_3d, const float *dL_masks,
    const float *dL_dpix_flow, const float *dL_dpix_4d, const float *dL_dpix_3d,
    float *dL_dmean2D, float *dL_dconic, float *dL_dopacity, float *dL_dcolor,
    float *dL_dmean3D, float *dL_dcov3D, float *dL_dsh, float *dL_dflows,
    float *dL_dts, float *dL_dscale, float *dL_dscale_t, float *dL_drot,
    float *dL_drot_r, float *dL_dmean2D_static, float *dL_dconic_static,
    float *dL_dopacity_static, float *dL_dmean3D_static,
    float *dL_dcov3D_static, float *dL_dcolor_static, float *dL_dsh_static,
    float *dL_dscale_static, float *dL_drot_static, bool debug) {
  GeometryState geomState =
      GeometryState::fromChunk(geom_buffer, P, P_static, P + P_static);
  BinningState binningState = BinningState::fromChunk(binning_buffer, R);
  ImageState imgState = ImageState::fromChunk(img_buffer, width * height);
  SampleState sampleState = SampleState::fromChunk(sample_buffer, B);

  if (radii == nullptr) {
    radii = geomState.internal_radii;
  }

  const float focal_y = height / (2.0f * tan_fovy);
  const float focal_x = width / (2.0f * tan_fovx);

  const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X,
                       (height + BLOCK_Y - 1) / BLOCK_Y, 1);
  const dim3 block(BLOCK_X, BLOCK_Y, 1);

  // Compute loss gradients w.r.t. 2D mean position, conic matrix,
  // opacity and RGB of Gaussians from per-pixel loss gradients.
  // If we were given precomputed colors and not SHs, use them.
  const float *color_ptr =
      (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
  const float *depth_ptr = geomState.depths;
  CHECK_CUDA(BACKWARD::render(
                 P, tile_grid, block, imgState.ranges, binningState.point_list,
                 width, height, R, B, imgState.bucket_offsets,
                 sampleState.bucket_to_tile, sampleState.T, sampleState.ar,
                 sampleState.ard, background, geomState.means2D,
                 geomState.conic_opacity, geomState.conic_opacity_static,
                 color_ptr, geomState.rgb_static, depth_ptr, flows_2d,
                 imgState.accum_alpha, imgState.n_contrib, imgState.max_contrib,
                 imgState.pixel_colors, imgState.pixel_invDepths, dL_dpix,
                 dL_invdepths, dL_depths, dL_depths_4d, dL_depths_3d, dL_masks,
                 dL_dpix_flow, dL_dpix_4d, dL_dpix_3d, (float3 *)dL_dmean2D,
                 (float4 *)dL_dconic, dL_dopacity, dL_dcolor, dL_dflows,
                 (float3 *)dL_dmean2D_static, (float4 *)dL_dconic_static,
                 dL_dopacity_static, dL_dcolor_static),
             debug)

  // Take care of the rest of preprocessing. Was the precomputed covariance
  // given to us or a scales/rot pair? If precomputed, pass that. If not,
  // use the one we computed ourselves.
  const float *cov3D_ptr =
      (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
  CHECK_CUDA(BACKWARD::preprocess(
                 P, D, D_t, M, (float3 *)out_means3D, radii, shs, ts, opacities,
                 geomState.clamped, geomState.tiles_touched,
                 (glm::vec3 *)scales, scales_t, (glm::vec4 *)rotations,
                 (glm::vec4 *)rotations_r, scale_modifier, cov3D_ptr,
                 viewmatrix, projmatrix, focal_x, focal_y, tan_fovx, tan_fovy,
                 (glm::vec3 *)campos, timestamp, time_duration, rot_4d,
                 gaussian_dim, force_sh_3d, (float3 *)dL_dmean2D, dL_dconic,
                 (glm::vec3 *)dL_dmean3D, dL_dcolor, dL_dcov3D, dL_dsh, dL_dts,
                 (glm::vec3 *)dL_dscale, dL_dscale_t, (glm::vec4 *)dL_drot,
                 (glm::vec4 *)dL_drot_r, dL_dopacity),
             debug)
  if (P_static > 0) {
    CHECK_CUDA(BACKWARD::preprocess_static(
                   P, P_static, D, M_static, (float3 *)means3D_static,
                   radii_static, shs_static, opacities_static,
                   geomState.clamped, geomState.tiles_touched,
                   (glm::vec3 *)scales_static, (glm::vec4 *)rotations_static,
                   scale_modifier, geomState.cov3D_static, viewmatrix,
                   projmatrix, focal_x, focal_y, tan_fovx, tan_fovy,
                   (glm::vec3 *)campos, (float3 *)dL_dmean2D_static,
                   dL_dconic_static, (glm::vec3 *)dL_dmean3D_static,
                   dL_dcolor_static, dL_dcov3D_static, dL_dsh_static,
                   (glm::vec3 *)dL_dscale_static, (glm::vec4 *)dL_drot_static,
                   dL_dopacity_static),
               debug)
  }
}