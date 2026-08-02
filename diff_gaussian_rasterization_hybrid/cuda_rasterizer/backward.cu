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

#include "auxiliary.h"
#include "backward.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs,
                                   const glm::vec3 *means, glm::vec3 campos,
                                   const float *shs, const bool *clamped,
                                   const glm::vec3 *dL_dcolor,
                                   glm::vec3 *dL_dmeans, glm::vec3 *dL_dshs) {
  // Compute intermediate values, as it is done during forward
  glm::vec3 pos = means[idx];
  glm::vec3 dir_orig = pos - campos;
  glm::vec3 dir = dir_orig / glm::length(dir_orig);

  glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs;

  // Use PyTorch rule for clamping: if clamping was applied,
  // gradient becomes 0.
  glm::vec3 dL_dRGB = dL_dcolor[idx];
  dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
  dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
  dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

  glm::vec3 dRGBdx(0, 0, 0);
  glm::vec3 dRGBdy(0, 0, 0);
  glm::vec3 dRGBdz(0, 0, 0);
  float x = dir.x;
  float y = dir.y;
  float z = dir.z;

  // Target location for this Gaussian to write SH gradients to
  glm::vec3 *dL_dsh = dL_dshs + idx * max_coeffs;

  // No tricks here, just high school-level calculus.
  float dRGBdsh0 = SH_C0;
  dL_dsh[0] = dRGBdsh0 * dL_dRGB;
  if (deg > 0) {
    float dRGBdsh1 = -SH_C1 * y;
    float dRGBdsh2 = SH_C1 * z;
    float dRGBdsh3 = -SH_C1 * x;
    dL_dsh[1] = dRGBdsh1 * dL_dRGB;
    dL_dsh[2] = dRGBdsh2 * dL_dRGB;
    dL_dsh[3] = dRGBdsh3 * dL_dRGB;

    dRGBdx = -SH_C1 * sh[3];
    dRGBdy = -SH_C1 * sh[1];
    dRGBdz = SH_C1 * sh[2];

    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;

      float dRGBdsh4 = SH_C2[0] * xy;
      float dRGBdsh5 = SH_C2[1] * yz;
      float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
      float dRGBdsh7 = SH_C2[3] * xz;
      float dRGBdsh8 = SH_C2[4] * (xx - yy);
      dL_dsh[4] = dRGBdsh4 * dL_dRGB;
      dL_dsh[5] = dRGBdsh5 * dL_dRGB;
      dL_dsh[6] = dRGBdsh6 * dL_dRGB;
      dL_dsh[7] = dRGBdsh7 * dL_dRGB;
      dL_dsh[8] = dRGBdsh8 * dL_dRGB;

      dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] +
                SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
      dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] +
                SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
      dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] +
                SH_C2[3] * x * sh[7];

      if (deg > 2) {
        float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
        float dRGBdsh10 = SH_C3[1] * xy * z;
        float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
        float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
        float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
        float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
        float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
        dL_dsh[9] = dRGBdsh9 * dL_dRGB;
        dL_dsh[10] = dRGBdsh10 * dL_dRGB;
        dL_dsh[11] = dRGBdsh11 * dL_dRGB;
        dL_dsh[12] = dRGBdsh12 * dL_dRGB;
        dL_dsh[13] = dRGBdsh13 * dL_dRGB;
        dL_dsh[14] = dRGBdsh14 * dL_dRGB;
        dL_dsh[15] = dRGBdsh15 * dL_dRGB;

        dRGBdx += (SH_C3[0] * sh[9] * 3.f * 2.f * xy + SH_C3[1] * sh[10] * yz +
                   SH_C3[2] * sh[11] * -2.f * xy +
                   SH_C3[3] * sh[12] * -3.f * 2.f * xz +
                   SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
                   SH_C3[5] * sh[14] * 2.f * xz +
                   SH_C3[6] * sh[15] * 3.f * (xx - yy));

        dRGBdy +=
            (SH_C3[0] * sh[9] * 3.f * (xx - yy) + SH_C3[1] * sh[10] * xz +
             SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
             SH_C3[3] * sh[12] * -3.f * 2.f * yz +
             SH_C3[4] * sh[13] * -2.f * xy + SH_C3[5] * sh[14] * -2.f * yz +
             SH_C3[6] * sh[15] * -3.f * 2.f * xy);

        dRGBdz += (SH_C3[1] * sh[10] * xy + SH_C3[2] * sh[11] * 4.f * 2.f * yz +
                   SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
                   SH_C3[4] * sh[13] * 4.f * 2.f * xz +
                   SH_C3[5] * sh[14] * (xx - yy));
      }
    }
  }

  // The view direction is an input to the computation. View direction
  // is influenced by the Gaussian's mean, so SHs gradients
  // must propagate back into 3D position.
  glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB),
                    glm::dot(dRGBdz, dL_dRGB));

  // Account for normalization of direction
  float3 dL_dmean = dnormvdv(float3{dir_orig.x, dir_orig.y, dir_orig.z},
                             float3{dL_ddir.x, dL_ddir.y, dL_ddir.z});

  // Gradients of loss w.r.t. Gaussian means, but only the portion
  // that is caused because the mean affects the view-dependent color.
  // Additional mean gradient is accumulated in below methods.
  dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void
computeColorFromSH_4D(int idx, int deg, int deg_t, int max_coeffs,
                      const glm::vec3 *means, glm::vec3 campos,
                      const float *shs, const bool *clamped, const float *ts,
                      const float timestamp, const float time_duration,
                      const glm::vec3 *dL_dcolor, glm::vec3 *dL_dmeans,
                      glm::vec3 *dL_dshs, float *dL_dts) {
  // Compute intermediate values, as it is done during forward
  const float dir_t = ts[idx] - timestamp;
  glm::vec3 pos = means[idx];
  glm::vec3 dir_orig = pos - campos;
  glm::vec3 dir = dir_orig / glm::length(dir_orig);

  glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs;

  // Use PyTorch rule for clamping: if clamping was applied,
  // gradient becomes 0.
  glm::vec3 dL_dRGB = dL_dcolor[idx];
  dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
  dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
  dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

  glm::vec3 dRGBdx(0, 0, 0);
  glm::vec3 dRGBdy(0, 0, 0);
  glm::vec3 dRGBdz(0, 0, 0);
  glm::vec3 dRGBdt(0, 0, 0);

  // Target location for this Gaussian to write SH gradients to
  glm::vec3 *dL_dsh = dL_dshs + idx * max_coeffs;

  // No tricks here, just high school-level calculus.
  float l0m0 = SH_C0;

  float dRGBdsh0 = l0m0;
  dL_dsh[0] = dRGBdsh0 * dL_dRGB;

  if (deg > 0) {
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    float l1m1 = -1 * SH_C1 * y;
    float l1m0 = SH_C1 * z;
    float l1p1 = -1 * SH_C1 * x;

    float dl1m1_dy = -1 * SH_C1;
    float dl1m0_dz = SH_C1;
    float dl1p1_dx = -1 * SH_C1;

    dL_dsh[1] = l0m0 * dL_dRGB;
    dL_dsh[2] = l1m0 * dL_dRGB;
    dL_dsh[3] = l1p1 * dL_dRGB;

    dRGBdx = dl1p1_dx * sh[3];
    dRGBdy = dl1m1_dy * sh[1];
    dRGBdz = dl1m0_dz * sh[2];

    if (deg > 1) {
      float xx = x * x, yy = y * y, zz = z * z;
      float xy = x * y, yz = y * z, xz = x * z;

      float l2m2 = SH_C2[0] * xy;
      float l2m1 = SH_C2[1] * yz;
      float l2m0 = SH_C2[2] * (2.0 * zz - xx - yy);
      float l2p1 = SH_C2[3] * xz;
      float l2p2 = SH_C2[4] * (xx - yy);

      float dl2m2_dx = SH_C2[0] * y;
      float dl2m2_dy = SH_C2[0] * x;
      float dl2m1_dy = SH_C2[1] * z;
      float dl2m1_dz = SH_C2[1] * y;
      float dl2m0_dx = -2 * SH_C2[2] * x;
      float dl2m0_dy = -2 * SH_C2[2] * y;
      float dl2m0_dz = 4 * SH_C2[2] * z;
      float dl2p1_dx = SH_C2[3] * z;
      float dl2p1_dz = SH_C2[3] * x;
      float dl2p2_dx = 2 * SH_C2[4] * x;
      float dl2p2_dy = -2 * SH_C2[4] * y;

      dL_dsh[4] = l2m2 * dL_dRGB;
      dL_dsh[5] = l2m1 * dL_dRGB;
      dL_dsh[6] = l2m0 * dL_dRGB;
      dL_dsh[7] = l2p1 * dL_dRGB;
      dL_dsh[8] = l2p2 * dL_dRGB;

      dRGBdx += (dl2m2_dx * sh[4] + dl2m0_dx * sh[6] + dl2p1_dx * sh[7] +
                 dl2p2_dx * sh[8]);
      dRGBdy += (dl2m2_dy * sh[4] + dl2m1_dy * sh[5] + dl2m0_dy * sh[6] +
                 dl2p2_dy * sh[8]);
      dRGBdz += (dl2m1_dz * sh[5] + dl2m0_dz * sh[6] + dl2p1_dz * sh[7]);

      if (deg > 2) {
        float l3m3 = SH_C3[0] * y * (3 * xx - yy);
        float l3m2 = SH_C3[1] * xy * z;
        float l3m1 = SH_C3[2] * y * (4 * zz - xx - yy);
        float l3m0 = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
        float l3p1 = SH_C3[4] * x * (4 * zz - xx - yy);
        float l3p2 = SH_C3[5] * z * (xx - yy);
        float l3p3 = SH_C3[6] * x * (xx - 3 * yy);

        float dl3m3_dx = SH_C3[0] * y * 6 * x;
        float dl3m3_dy = SH_C3[0] * (3 * xx - 3 * yy);
        float dl3m2_dx = SH_C3[1] * yz;
        float dl3m2_dy = SH_C3[1] * xz;
        float dl3m2_dz = SH_C3[1] * xy;
        float dl3m1_dx = -SH_C3[2] * y * 2 * x;
        float dl3m1_dy = SH_C3[2] * (4 * zz - xx - 3 * yy);
        float dl3m1_dz = SH_C3[2] * y * 8 * z;
        float dl3m0_dx = -SH_C3[3] * z * 6 * x;
        float dl3m0_dy = -SH_C3[3] * z * 6 * y;
        float dl3m0_dz = SH_C3[3] * (6 * zz - 3 * xx - 3 * yy);
        float dl3p1_dx = SH_C3[4] * (4 * zz - 3 * xx - yy);
        float dl3p1_dy = -SH_C3[4] * x * 2 * y;
        float dl3p1_dz = SH_C3[4] * x * 8 * z;
        float dl3p2_dx = SH_C3[5] * z * 2 * x;
        float dl3p2_dy = -SH_C3[5] * z * 2 * y;
        float dl3p2_dz = SH_C3[5] * (xx - yy);
        float dl3p3_dx = SH_C3[6] * (3 * xx - 3 * yy);
        float dl3p3_dy = -SH_C3[6] * x * 6 * y;

        dL_dsh[9] = l3m3 * dL_dRGB;
        dL_dsh[10] = l3m2 * dL_dRGB;
        dL_dsh[11] = l3m1 * dL_dRGB;
        dL_dsh[12] = l3m0 * dL_dRGB;
        dL_dsh[13] = l3p1 * dL_dRGB;
        dL_dsh[14] = l3p2 * dL_dRGB;
        dL_dsh[15] = l3p3 * dL_dRGB;

        dRGBdx += (dl3m3_dx * sh[9] + dl3m2_dx * sh[10] + dl3m1_dx * sh[11] +
                   dl3m0_dx * sh[12] + dl3p1_dx * sh[13] + dl3p2_dx * sh[14] +
                   dl3p3_dx * sh[15]);

        dRGBdy += (dl3m3_dy * sh[9] + dl3m2_dy * sh[10] + dl3m1_dy * sh[11] +
                   dl3m0_dy * sh[12] + dl3p1_dy * sh[13] + dl3p2_dy * sh[14] +
                   dl3p3_dy * sh[15]);

        dRGBdz += (dl3m2_dz * sh[10] + dl3m1_dz * sh[11] + dl3m0_dz * sh[12] +
                   dl3p1_dz * sh[13] + dl3p2_dz * sh[14]);

        if (deg_t > 0) {
          float t1 = cos(2 * MY_PI * dir_t / time_duration);
          float dt1_dt = sin(2 * MY_PI * dir_t / time_duration) * 2 * MY_PI /
                         time_duration;

          dL_dsh[16] = t1 * l0m0 * dL_dRGB;
          dL_dsh[17] = t1 * l1m1 * dL_dRGB;
          dL_dsh[18] = t1 * l1m0 * dL_dRGB;
          dL_dsh[19] = t1 * l1p1 * dL_dRGB;
          dL_dsh[20] = t1 * l2m2 * dL_dRGB;
          dL_dsh[21] = t1 * l2m1 * dL_dRGB;
          dL_dsh[22] = t1 * l2m0 * dL_dRGB;
          dL_dsh[23] = t1 * l2p1 * dL_dRGB;
          dL_dsh[24] = t1 * l2p2 * dL_dRGB;
          dL_dsh[25] = t1 * l3m3 * dL_dRGB;
          dL_dsh[26] = t1 * l3m2 * dL_dRGB;
          dL_dsh[27] = t1 * l3m1 * dL_dRGB;
          dL_dsh[28] = t1 * l3m0 * dL_dRGB;
          dL_dsh[29] = t1 * l3p1 * dL_dRGB;
          dL_dsh[30] = t1 * l3p2 * dL_dRGB;
          dL_dsh[31] = t1 * l3p3 * dL_dRGB;

          dRGBdt =
              dt1_dt *
              (l0m0 * sh[16] + l1m1 * sh[17] + l1m0 * sh[18] + l1p1 * sh[19] +
               l2m2 * sh[20] + l2m1 * sh[21] + l2m0 * sh[22] + l2p1 * sh[23] +
               l2p2 * sh[24] + l3m3 * sh[25] + l3m2 * sh[26] + l3m1 * sh[27] +
               l3m0 * sh[28] + l3p1 * sh[29] + l3p2 * sh[30] + l3p3 * sh[31]);

          dRGBdx +=
              t1 * (dl1p1_dx * sh[19] + dl2m2_dx * sh[20] + dl2m0_dx * sh[22] +
                    dl2p1_dx * sh[23] + dl2p2_dx * sh[24] + dl3m3_dx * sh[25] +
                    dl3m2_dx * sh[26] + dl3m1_dx * sh[27] + dl3m0_dx * sh[28] +
                    dl3p1_dx * sh[29] + dl3p2_dx * sh[30] + dl3p3_dx * sh[31]);

          dRGBdy +=
              t1 * (dl1m1_dy * sh[17] + dl2m2_dy * sh[20] + dl2m1_dy * sh[21] +
                    dl2m0_dy * sh[22] + dl2p2_dy * sh[24] + dl3m3_dy * sh[25] +
                    dl3m2_dy * sh[26] + dl3m1_dy * sh[27] + dl3m0_dy * sh[28] +
                    dl3p1_dy * sh[29] + dl3p2_dy * sh[30] + dl3p3_dy * sh[31]);

          dRGBdz +=
              t1 * (dl1m0_dz * sh[18] + dl2m1_dz * sh[21] + dl2m0_dz * sh[22] +
                    dl2p1_dz * sh[23] + dl3m2_dz * sh[26] + dl3m1_dz * sh[27] +
                    dl3m0_dz * sh[28] + dl3p1_dz * sh[29] + dl3p2_dz * sh[30]);

          if (deg_t > 1) {
            float t2 = cos(2 * MY_PI * dir_t * 2 / time_duration);
            float dt2_dt = sin(2 * MY_PI * dir_t * 2 / time_duration) * 2 *
                           MY_PI * 2 / time_duration;

            dL_dsh[32] = t2 * l0m0 * dL_dRGB;
            dL_dsh[33] = t2 * l1m1 * dL_dRGB;
            dL_dsh[34] = t2 * l1m0 * dL_dRGB;
            dL_dsh[35] = t2 * l1p1 * dL_dRGB;
            dL_dsh[36] = t2 * l2m2 * dL_dRGB;
            dL_dsh[37] = t2 * l2m1 * dL_dRGB;
            dL_dsh[38] = t2 * l2m0 * dL_dRGB;
            dL_dsh[39] = t2 * l2p1 * dL_dRGB;
            dL_dsh[40] = t2 * l2p2 * dL_dRGB;
            dL_dsh[41] = t2 * l3m3 * dL_dRGB;
            dL_dsh[42] = t2 * l3m2 * dL_dRGB;
            dL_dsh[43] = t2 * l3m1 * dL_dRGB;
            dL_dsh[44] = t2 * l3m0 * dL_dRGB;
            dL_dsh[45] = t2 * l3p1 * dL_dRGB;
            dL_dsh[46] = t2 * l3p2 * dL_dRGB;
            dL_dsh[47] = t2 * l3p3 * dL_dRGB;

            dRGBdt =
                dt2_dt *
                (l0m0 * sh[32] + l1m1 * sh[33] + l1m0 * sh[34] + l1p1 * sh[35] +
                 l2m2 * sh[36] + l2m1 * sh[37] + l2m0 * sh[38] + l2p1 * sh[39] +
                 l2p2 * sh[40] + l3m3 * sh[41] + l3m2 * sh[42] + l3m1 * sh[43] +
                 l3m0 * sh[44] + l3p1 * sh[45] + l3p2 * sh[46] + l3p3 * sh[47]);

            dRGBdx +=
                t2 *
                (dl1p1_dx * sh[35] + dl2m2_dx * sh[36] + dl2m0_dx * sh[38] +
                 dl2p1_dx * sh[39] + dl2p2_dx * sh[40] + dl3m3_dx * sh[41] +
                 dl3m2_dx * sh[42] + dl3m1_dx * sh[43] + dl3m0_dx * sh[44] +
                 dl3p1_dx * sh[45] + dl3p2_dx * sh[46] + dl3p3_dx * sh[47]);

            dRGBdy +=
                t2 *
                (dl1m1_dy * sh[33] + dl2m2_dy * sh[36] + dl2m1_dy * sh[37] +
                 dl2m0_dy * sh[38] + dl2p2_dy * sh[40] + dl3m3_dy * sh[41] +
                 dl3m2_dy * sh[42] + dl3m1_dy * sh[43] + dl3m0_dy * sh[44] +
                 dl3p1_dy * sh[45] + dl3p2_dy * sh[46] + dl3p3_dy * sh[47]);

            dRGBdz +=
                t2 *
                (dl1m0_dz * sh[34] + dl2m1_dz * sh[37] + dl2m0_dz * sh[38] +
                 dl2p1_dz * sh[39] + dl3m2_dz * sh[42] + dl3m1_dz * sh[43] +
                 dl3m0_dz * sh[44] + dl3p1_dz * sh[45] + dl3p2_dz * sh[46]);
          }
        }
      }
    }
  }

  // The view direction is an input to the computation. View direction
  // is influenced by the Gaussian's mean, so SHs gradients
  // must propagate back into 3D position.
  glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB),
                    glm::dot(dRGBdz, dL_dRGB));

  // Account for normalization of direction
  float3 dL_dmean = dnormvdv(float3{dir_orig.x, dir_orig.y, dir_orig.z},
                             float3{dL_ddir.x, dL_ddir.y, dL_ddir.z});

  // Gradients of loss w.r.t. Gaussian means, but only the portion
  // that is caused because the mean affects the view-dependent color.
  // Additional mean gradient is accumulated in below methods.
  dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
  dL_dts[idx] += glm::dot(dRGBdt, dL_dRGB);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P, const float3 *means, const int *radii,
                                 const float *cov3Ds, const float h_x,
                                 float h_y, const float tan_fovx,
                                 float tan_fovy, const float *view_matrix,
                                 const float *dL_dconics,
                                 const float3 *dL_dmean2D, float3 *dL_dmeans,
                                 float *dL_dcov) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P || !(radii[idx] > 0))
    return;

  // Reading location of 3D covariance for this Gaussian
  const float *cov3D = cov3Ds + 6 * idx;

  // Fetch gradients, recompute 2D covariance and relevant
  // intermediate forward results needed in the backward.
  float3 mean = means[idx];
  float3 dL_dconic = {dL_dconics[4 * idx], dL_dconics[4 * idx + 1],
                      dL_dconics[4 * idx + 3]};
  float3 t = transformPoint4x3(mean, view_matrix);

  const float limx = 1.3f * tan_fovx;
  const float limy = 1.3f * tan_fovy;
  const float txtz = t.x / t.z;
  const float tytz = t.y / t.z;
  t.x = min(limx, max(-limx, txtz)) * t.z;
  t.y = min(limy, max(-limy, tytz)) * t.z;

  const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
  const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

  glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z), 0.0f,
                          h_y / t.z, -(h_y * t.y) / (t.z * t.z), 0, 0, 0);

  glm::mat3 W = glm::mat3(view_matrix[0], view_matrix[4], view_matrix[8],
                          view_matrix[1], view_matrix[5], view_matrix[9],
                          view_matrix[2], view_matrix[6], view_matrix[10]);

  glm::mat3 Vrk = glm::mat3(cov3D[0], cov3D[1], cov3D[2], cov3D[1], cov3D[3],
                            cov3D[4], cov3D[2], cov3D[4], cov3D[5]);

  glm::mat3 T = W * J;

  glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

  // Use helper variables for 2D covariance entries. More compact.
  float a = cov2D[0][0] += 0.3f;
  float b = cov2D[0][1];
  float c = cov2D[1][1] += 0.3f;

  float denom = a * c - b * b;
  float dL_da = 0, dL_db = 0, dL_dc = 0;
  float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

  if (denom2inv != 0) {
    // Gradients of loss w.r.t. entries of 2D covariance matrix,
    // given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
    // e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
    dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y +
                         (denom - a * c) * dL_dconic.z);
    dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y +
                         (denom - a * c) * dL_dconic.x);
    dL_db = denom2inv * 2 *
            (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y +
             a * b * dL_dconic.z);

    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
    // given gradients w.r.t. 2D covariance matrix (diagonal).
    // cov2D = transpose(T) * transpose(Vrk) * T;
    dL_dcov[6 * idx + 0] =
        (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db +
         T[1][0] * T[1][0] * dL_dc);
    dL_dcov[6 * idx + 3] =
        (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db +
         T[1][1] * T[1][1] * dL_dc);
    dL_dcov[6 * idx + 5] =
        (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db +
         T[1][2] * T[1][2] * dL_dc);

    // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
    // given gradients w.r.t. 2D covariance matrix (off-diagonal).
    // Off-diagonal elements appear twice --> double the gradient.
    // cov2D = transpose(T) * transpose(Vrk) * T;
    dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da +
                           (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db +
                           2 * T[1][0] * T[1][1] * dL_dc;
    dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da +
                           (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db +
                           2 * T[1][0] * T[1][2] * dL_dc;
    dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da +
                           (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db +
                           2 * T[1][1] * T[1][2] * dL_dc;
  } else {
    for (int i = 0; i < 6; i++)
      dL_dcov[6 * idx + i] = 0;
  }

  // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
  // cov2D = transpose(T) * transpose(Vrk) * T;
  float dL_dT00 =
      2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) *
          dL_da +
      (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
  float dL_dT01 =
      2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) *
          dL_da +
      (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
  float dL_dT02 =
      2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) *
          dL_da +
      (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
  float dL_dT10 =
      2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) *
          dL_dc +
      (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
  float dL_dT11 =
      2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) *
          dL_dc +
      (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
  float dL_dT12 =
      2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) *
          dL_dc +
      (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

  // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
  // T = W * J
  float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
  float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
  float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
  float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

  float tz = 1.f / t.z;
  float tz2 = tz * tz;
  float tz3 = tz2 * tz;

  // Gradients of loss w.r.t. transformed Gaussian mean t
  float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
  float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
  float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 +
                 (2 * h_x * t.x) * tz3 * dL_dJ02 +
                 (2 * h_y * t.y) * tz3 * dL_dJ12;

  // Account for transformation of mean to t
  // t = transformPoint4x3(mean, view_matrix);
  // float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty,
  // dL_dtz+dL_dmean2D[idx].z }, view_matrix);
  float3 dL_dmean = transformVec4x3Transpose(
      {dL_dtx, dL_dty, dL_dtz - (dL_dmean2D[idx].z * tz2)}, view_matrix);

  // Gradients of loss w.r.t. Gaussian means, but only the portion
  // that is caused because the mean affects the covariance matrix.
  // Additional mean gradient is accumulated in BACKWARD::preprocess.
  dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a
// 3D covariance matrix for each Gaussian.
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod,
                             const glm::vec4 rot, const float *dL_dcov3Ds,
                             glm::vec3 *dL_dscales, glm::vec4 *dL_drots) {
  // Recompute (intermediate) results for the 3D covariance computation.
  glm::vec4 q = rot; // / glm::length(rot);
  float r = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;

  glm::mat3 R = glm::mat3(1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z),
                          2.f * (x * z + r * y), 2.f * (x * y + r * z),
                          1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
                          2.f * (x * z - r * y), 2.f * (y * z + r * x),
                          1.f - 2.f * (x * x + y * y));

  glm::mat3 S = glm::mat3(1.0f);

  glm::vec3 s = mod * scale;
  S[0][0] = s.x;
  S[1][1] = s.y;
  S[2][2] = s.z;

  glm::mat3 M = S * R;

  const float *dL_dcov3D = dL_dcov3Ds + 6 * idx;

  // glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
  // glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2],
  // dL_dcov3D[4]);

  // Convert per-element covariance loss gradients to matrix form
  glm::mat3 dL_dSigma =
      glm::mat3(dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
                0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
                0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]);

  // Compute loss gradient w.r.t. matrix M
  // dSigma_dM = 2 * M
  glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

  glm::mat3 Rt = glm::transpose(R);
  glm::mat3 dL_dMt = glm::transpose(dL_dM);

  // Gradients of loss w.r.t. scale
  glm::vec3 *dL_dscale = dL_dscales + idx;
  dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
  dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
  dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

  dL_dMt[0] *= s.x;
  dL_dMt[1] *= s.y;
  dL_dMt[2] *= s.z;

  // Gradients of loss w.r.t. normalized quaternion
  glm::vec4 dL_dq;
  dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) +
            2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) +
            2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
  dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) +
            2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) +
            2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) -
            4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
  dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) +
            2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) +
            2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) -
            4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
  dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) +
            2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) +
            2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) -
            4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

  // Gradients of loss w.r.t. unnormalized quaternion
  float4 *dL_drot = (float4 *)(dL_drots + idx);
  *dL_drot = float4{dL_dq.x, dL_dq.y, dL_dq.z,
                    dL_dq.w}; // dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w },
                              // float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass for the conversion of scale and rotation to a
// 3D covariance matrix for each Gaussian.
__device__ void computeCov3D_conditional(
    int idx, const glm::vec3 scale, const float scale_t, float mod,
    const glm::vec4 rot, const glm::vec4 rot_r, const float t,
    const float timestamp, const float opacity, bool &mask,
    const float *dL_dcov3Ds, const glm::vec3 *dL_dmeans, float *dL_dopacity,
    float *dL_dts, glm::vec3 *dL_dscales, float *dL_dscales_t,
    glm::vec4 *dL_drots, glm::vec4 *dL_drots_r) {
  float dt = timestamp - t;
  glm::mat4 S = glm::mat4(1.0f);
  S[0][0] = mod * scale.x;
  S[1][1] = mod * scale.y;
  S[2][2] = mod * scale.z;
  S[3][3] = mod * scale_t;

  float a = rot.x;
  float b = rot.y;
  float c = rot.z;
  float d = rot.w;

  float p = rot_r.x;
  float q = rot_r.y;
  float r = rot_r.z;
  float s = rot_r.w;

  // glm::mat4 M_l = glm::mat4(
  // 	a, -b, -c, -d,
  // 	b, a,-d, c,
  // 	c, d, a,-b,
  // 	d,-c, b, a
  // );

  // glm::mat4 M_r = glm::mat4(
  // 	p, q, r, s,
  // 	-q, p,-s, r,
  // 	-r, s, p,-q,
  // 	-s,-r, q, p
  // );

  glm::mat4 M_l =
      glm::mat4(a, b, -c, d, -b, a, d, c, c, -d, a, b, -d, -c, -b, a);

  glm::mat4 M_r =
      glm::mat4(p, q, -r, -s, -q, p, s, -r, r, -s, p, -q, s, r, q, p);
  // glm stores in column major
  glm::mat4 R = M_r * M_l;
  glm::mat4 M = S * R;

  glm::mat4 Sigma = glm::transpose(M) * M;

  float cov_t = Sigma[3][3];
  float marginal_t = __expf(-0.5 * dt * dt / cov_t);
  mask = marginal_t > 0.05;
  if (!mask)
    return;

  glm::mat3 cov11 = glm::mat3(Sigma);
  glm::vec3 cov12 = glm::vec3(Sigma[3][0], Sigma[3][1], Sigma[3][2]);

  const float *dL_dcov3D = dL_dcov3Ds + 6 * idx;

  glm::vec3 dL_dcov12 =
      -glm::vec3(dL_dcov3D[0] * cov12[0] + dL_dcov3D[1] * cov12[1] * 0.5 +
                     dL_dcov3D[2] * cov12[2] * 0.5,
                 dL_dcov3D[1] * cov12[0] * 0.5 + dL_dcov3D[3] * cov12[1] +
                     dL_dcov3D[4] * cov12[2] * 0.5,
                 dL_dcov3D[2] * cov12[0] * 0.5 + dL_dcov3D[4] * cov12[1] * 0.5 +
                     dL_dcov3D[5] * cov12[2]) *
      2.0f / cov_t;

  float dL_dcovt =
      (cov12[0] * cov12[0] * dL_dcov3D[0] + cov12[0] * cov12[1] * dL_dcov3D[1] +
       cov12[0] * cov12[2] * dL_dcov3D[2] + cov12[1] * cov12[1] * dL_dcov3D[3] +
       cov12[1] * cov12[2] * dL_dcov3D[4] +
       cov12[2] * cov12[2] * dL_dcov3D[5]) /
      (cov_t * cov_t);

  // from opacity
  float dL_dmarginal_t = dL_dopacity[idx] * opacity;
  dL_dopacity[idx] *= marginal_t;
  float dmarginalt_dcovt = marginal_t * dt * dt / 2 / (cov_t * cov_t);
  float dmarginalt_dt = marginal_t * dt / cov_t;
  dL_dcovt += dmarginalt_dcovt * dL_dmarginal_t;
  float dL_dt = dL_dmarginal_t * dmarginalt_dt;

  // from means
  glm::vec3 dL_ddeltamean = dL_dmeans[idx];
  dL_dcov12 += dL_ddeltamean / cov_t * dt;
  float dL_ddeltamean_dot_cov12 = glm::dot(dL_ddeltamean, cov12);
  dL_dcovt += -dL_ddeltamean_dot_cov12 / (cov_t * cov_t) * dt;
  dL_dt += -dL_ddeltamean_dot_cov12 / cov_t;
  dL_dts[idx] += dL_dt;

  glm::mat4 dL_dSigma = glm::mat4(
      dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
      0.5f * dL_dcov12[0], 0.5f * dL_dcov3D[1], dL_dcov3D[3],
      0.5f * dL_dcov3D[4], 0.5f * dL_dcov12[1], 0.5f * dL_dcov3D[2],
      0.5f * dL_dcov3D[4], dL_dcov3D[5], 0.5f * dL_dcov12[2],
      0.5f * dL_dcov12[0], 0.5f * dL_dcov12[1], 0.5f * dL_dcov12[2], dL_dcovt);
  // Compute loss gradient w.r.t. matrix M
  // dSigma_dM = 2 * M
  glm::mat4 dL_dM = 2.0f * M * dL_dSigma;

  glm::mat4 Rt = glm::transpose(R);
  glm::mat4 dL_dMt = glm::transpose(dL_dM);

  // Gradients of loss w.r.t. scale
  glm::vec3 *dL_dscale = dL_dscales + idx;
  dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
  dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
  dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);
  dL_dscales_t[idx] = glm::dot(Rt[3], dL_dMt[3]);

  dL_dMt[0] *= mod * scale.x;
  dL_dMt[1] *= mod * scale.y;
  dL_dMt[2] *= mod * scale.z;
  dL_dMt[3] *= mod * scale_t;

  glm::mat4 dL_dml_t = dL_dMt * M_r;
  // 	dL_dml_t = glm::transpose(dL_dml_t);
  glm::vec4 dL_drot(0.0f);
  // dL_drot.x = dL_dml_t[0][0] + dL_dml_t[1][1] + dL_dml_t[2][2] +
  // dL_dml_t[3][3]; dL_drot.y = dL_dml_t[0][1] - dL_dml_t[1][0] +
  // dL_dml_t[2][3] - dL_dml_t[3][2]; dL_drot.z = dL_dml_t[0][2] -
  // dL_dml_t[1][3] - dL_dml_t[2][0] + dL_dml_t[3][1]; dL_drot.w =
  // dL_dml_t[0][3] + dL_dml_t[1][2] - dL_dml_t[2][1] - dL_dml_t[3][0];
  dL_drot.x = dL_dml_t[0][0] + dL_dml_t[1][1] + dL_dml_t[2][2] + dL_dml_t[3][3];
  dL_drot.y =
      -dL_dml_t[0][1] + dL_dml_t[1][0] - dL_dml_t[2][3] + dL_dml_t[3][2];
  dL_drot.z = dL_dml_t[0][2] - dL_dml_t[1][3] - dL_dml_t[2][0] + dL_dml_t[3][1];
  dL_drot.w =
      -dL_dml_t[0][3] - dL_dml_t[1][2] + dL_dml_t[2][1] + dL_dml_t[3][0];

  glm::mat4 dL_dmr_t = M_l * dL_dMt;
  glm::vec4 dL_drot_r(0.0f);
  // dL_drot_r.x = dL_dmr_t[0][0] + dL_dmr_t[1][1] + dL_dmr_t[2][2] +
  // dL_dmr_t[3][3]; dL_drot_r.y = -dL_dmr_t[0][1] + dL_dmr_t[1][0] +
  // dL_dmr_t[2][3] - dL_dmr_t[3][2]; dL_drot_r.z = -dL_dmr_t[0][2] -
  // dL_dmr_t[1][3] + dL_dmr_t[2][0] + dL_dmr_t[3][1]; dL_drot_r.w =
  // -dL_dmr_t[0][3] + dL_dmr_t[1][2] - dL_dmr_t[2][1] + dL_dmr_t[3][0];
  dL_drot_r.x =
      dL_dmr_t[0][0] + dL_dmr_t[1][1] + dL_dmr_t[2][2] + dL_dmr_t[3][3];
  dL_drot_r.y =
      -dL_dmr_t[0][1] + dL_dmr_t[1][0] + dL_dmr_t[2][3] - dL_dmr_t[3][2];
  dL_drot_r.z =
      dL_dmr_t[0][2] + dL_dmr_t[1][3] - dL_dmr_t[2][0] - dL_dmr_t[3][1];
  dL_drot_r.w =
      dL_dmr_t[0][3] - dL_dmr_t[1][2] + dL_dmr_t[2][1] - dL_dmr_t[3][0];

  dL_drots[idx] = dL_drot;
  dL_drots_r[idx] = dL_drot_r;
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template <int C>
__global__ void preprocessCUDA(
    int P, int D, int D_t, int M, const float3 *means, const int *radii,
    const float *shs, const float *ts, const float *opacities,
    const bool *clamped, const uint32_t *tiles_touched, const glm::vec3 *scales,
    const float *scales_t, const glm::vec4 *rotations,
    const glm::vec4 *rotations_r, const float scale_modifier, const float *proj,
    const glm::vec3 *campos, const float timestamp, const float time_duration,
    const bool rot_4d, const int gaussian_dim, const bool force_sh_3d,
    const float3 *dL_dmean2D, glm::vec3 *dL_dmeans, float *dL_dcolor,
    float *dL_dcov3D, float *dL_dsh, float *dL_dts, glm::vec3 *dL_dscale,
    float *dL_dscale_t, glm::vec4 *dL_drot, glm::vec4 *dL_drot_r,
    float *dL_dopacity) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P || !(radii[idx] > 0))
    return;
  if (tiles_touched[idx] == 0)
    return;

  float3 m = means[idx];

  // Taking care of gradients from the screenspace points
  float4 m_hom = transformPoint4x4(m, proj);
  float m_w = 1.0f / (m_hom.w + 0.0000001f);

  // Compute loss gradient w.r.t. 3D means due to gradients of 2D means
  // from rendering procedure
  glm::vec3 dL_dmean;
  float mul1 =
      (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
  float mul2 =
      (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
  dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x +
               (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
  dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x +
               (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
  dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x +
               (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

  // That's the second part of the mean gradient. Previous computation
  // of cov2D and following SH conversion also affects it.
  dL_dmeans[idx] += dL_dmean;

  // Compute gradient updates due to computing colors from SHs
  if (shs) {
    if (gaussian_dim == 3 || force_sh_3d) {
      computeColorFromSH(idx, D, M, (glm::vec3 *)means, *campos, shs, clamped,
                         (glm::vec3 *)dL_dcolor, (glm::vec3 *)dL_dmeans,
                         (glm::vec3 *)dL_dsh);
    } else {
      computeColorFromSH_4D(idx, D, D_t, M, (glm::vec3 *)means, *campos, shs,
                            clamped, ts, timestamp, time_duration,
                            (glm::vec3 *)dL_dcolor, (glm::vec3 *)dL_dmeans,
                            (glm::vec3 *)dL_dsh, dL_dts);
    }
  }
  // Compute gradient updates due to computing covariance from scale/rotation
  if (scales) {
    if (rot_4d) {
      bool time_mask = true;
      computeCov3D_conditional(idx, scales[idx], scales_t[idx], scale_modifier,
                               rotations[idx], rotations_r[idx], ts[idx],
                               timestamp, opacities[idx], time_mask, dL_dcov3D,
                               dL_dmeans, dL_dopacity, dL_dts, dL_dscale,
                               dL_dscale_t, dL_drot, dL_drot_r);
      if (!time_mask)
        return;
    } else {
      computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D,
                   dL_dscale, dL_drot);
      if (gaussian_dim == 4) {
        // marginal opacity
      }
    }
  }
}

template <int C>
__global__ void preprocessCUDA_static(
    int P, int P_static, int D, int M, const float3 *means_static,
    const int *radii_static, const float *shs_static,
    const float *opacities_static, const bool *clamped,
    const uint32_t *tiles_touched, const glm::vec3 *scales_static,
    const glm::vec4 *rotations_static, const float scale_modifier,
    const float *proj, const glm::vec3 *campos, const float3 *dL_dmean2D_static,
    glm::vec3 *dL_dmeans_static, float *dL_dcolor_static,
    float *dL_dcov3D_static, float *dL_dsh_static, glm::vec3 *dL_dscale_static,
    glm::vec4 *dL_drot_static, float *dL_dopacity_static) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P_static || !(radii_static[idx] > 0))
    return;
  if (tiles_touched[idx + P] == 0)
    return;

  float3 m = means_static[idx];

  // Taking care of gradients from the screenspace points
  float4 m_hom = transformPoint4x4(m, proj);
  float m_w = 1.0f / (m_hom.w + 0.0000001f);

  // Compute loss gradient w.r.t. 3D means due to gradients of 2D means
  // from rendering procedure
  glm::vec3 dL_dmean;
  float mul1 =
      (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
  float mul2 =
      (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
  dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D_static[idx].x +
               (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D_static[idx].y;
  dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D_static[idx].x +
               (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D_static[idx].y;
  dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D_static[idx].x +
               (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D_static[idx].y;

  // That's the second part of the mean gradient. Previous computation
  // of cov2D and following SH conversion also affects it.
  dL_dmeans_static[idx] += dL_dmean;

  // Compute gradient updates due to computing colors from SHs

  computeColorFromSH(idx, D, M, (glm::vec3 *)means_static, *campos, shs_static,
                     clamped, (glm::vec3 *)dL_dcolor_static,
                     (glm::vec3 *)dL_dmeans_static, (glm::vec3 *)dL_dsh_static);

  // Compute gradient updates due to computing covariance from scale/rotation
  if (scales_static) {
    computeCov3D(idx, scales_static[idx], scale_modifier, rotations_static[idx],
                 dL_dcov3D_static, dL_dscale_static, dL_drot_static);
  }
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y) renderCUDA(
    int P, const uint2 *__restrict__ ranges,
    const uint32_t *__restrict__ point_list, int W, int H,
    const float *__restrict__ bg_color,
    const float2 *__restrict__ points_xy_image,
    const float4 *__restrict__ conic_opacity,
    const float4 *__restrict__ conic_opacity_static,
    const float *__restrict__ colors, const float *__restrict__ colors_static,
    const float *__restrict__ depths, const float *__restrict__ flows_2d,
    const float *__restrict__ final_Ts, const uint32_t *__restrict__ n_contrib,
    const float *__restrict__ dL_dpixels, const float *__restrict__ dL_depths,
    const float *__restrict__ dL_masks, const float *__restrict__ dL_dpix_flow,
    float3 *__restrict__ dL_dmean2D, float4 *__restrict__ dL_dconic2D,
    float *__restrict__ dL_dopacity, float *__restrict__ dL_dcolors,
    float *__restrict__ dL_dflows, float3 *__restrict__ dL_dmean2D_static,
    float4 *__restrict__ dL_dconic2D_static,
    float *__restrict__ dL_dopacity_static,
    float *__restrict__ dL_dcolors_static) {
  // We rasterize again. Compute necessary block info.
  auto block = cg::this_thread_block();
  const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 pix_min = {block.group_index().x * BLOCK_X,
                         block.group_index().y * BLOCK_Y};
  const uint2 pix_max = {min(pix_min.x + BLOCK_X, W),
                         min(pix_min.y + BLOCK_Y, H)};
  const uint2 pix = {pix_min.x + block.thread_index().x,
                     pix_min.y + block.thread_index().y};
  const uint32_t pix_id = W * pix.y + pix.x;
  const float2 pixf = {(float)pix.x, (float)pix.y};

  const bool inside = pix.x < W && pix.y < H;
  const uint2 range =
      ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

  const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

  bool done = !inside;
  int toDo = range.y - range.x;

  __shared__ int collected_id[BLOCK_SIZE];
  __shared__ float2 collected_xy[BLOCK_SIZE];
  __shared__ float4 collected_conic_opacity[BLOCK_SIZE];
  __shared__ float collected_colors[C * BLOCK_SIZE];
  __shared__ float collected_depths[BLOCK_SIZE];
  __shared__ float collected_flows[2 * BLOCK_SIZE];

  // In the forward, we stored the final value for T, the
  // product of all (1 - alpha) factors.
  const float T_final = inside ? final_Ts[pix_id] : 0;
  float T = T_final;

  // We start from the back. The ID of the last contributing
  // Gaussian is known from each pixel from the forward.
  uint32_t contributor = toDo;
  const int last_contributor = inside ? n_contrib[pix_id] : 0;

  float accum_rec[C] = {0};
  float accum_flow_rec[2] = {0};
  float dL_dpixel[C] = {0};
  float dL_dflow[2] = {0};
  float dL_depth = 0;
  float dL_mask = 0;
  float accum_depth_rec = 0;
  float accum_mask_rec = 0;
  if (inside) {
    for (int i = 0; i < C; i++) {
      dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
    }
    for (int i = 0; i < 2; i++) {
      dL_dflow[i] = dL_dpix_flow[i * H * W + pix_id];
    }
    dL_depth = dL_depths[pix_id];
    dL_mask = dL_masks[pix_id];
  }

  float last_alpha = 0;
  float last_color[C] = {0};
  float last_flow[2] = {0};
  float last_depth = 0;

  // Gradient of pixel coordinate w.r.t. normalized
  // screen-space viewport corrdinates (-1 to 1)
  const float ddelx_dx = 0.5 * W;
  const float ddely_dy = 0.5 * H;

  // Traverse all Gaussians
  for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
    // Load auxiliary data into shared memory, start in the BACK
    // and load them in revers order.
    block.sync();
    const int progress = i * BLOCK_SIZE + block.thread_rank();
    if (range.x + progress < range.y) {
      const int coll_id = point_list[range.y - progress - 1];
      collected_id[block.thread_rank()] = coll_id;
      collected_xy[block.thread_rank()] = points_xy_image[coll_id];
      if (coll_id < P) {
        collected_depths[block.thread_rank()] = depths[coll_id];
        collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
        for (int i = 0; i < C; i++)
          collected_colors[i * BLOCK_SIZE + block.thread_rank()] =
              colors[coll_id * C + i];
        for (int i = 0; i < 2; i++)
          collected_flows[i * BLOCK_SIZE + block.thread_rank()] =
              flows_2d[coll_id * 2 + i];
      } else {
        collected_conic_opacity[block.thread_rank()] =
            conic_opacity_static[coll_id - P];
        for (int i = 0; i < C; i++)
          collected_colors[i * BLOCK_SIZE + block.thread_rank()] =
              colors_static[(coll_id - P) * C + i];
      }
    }
    block.sync();

    // Iterate over Gaussians
    for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
      // Keep track of current Gaussian ID. Skip, if this one
      // is behind the last contributor for this pixel.
      contributor--;
      if (contributor >= last_contributor)
        continue;

      // Compute blending values, as before.
      const float2 xy = collected_xy[j];
      const float2 d = {xy.x - pixf.x, xy.y - pixf.y};
      const float4 con_o = collected_conic_opacity[j];
      const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) -
                          con_o.y * d.x * d.y;
      if (power > 0.0f)
        continue;

      const float G = exp(power);
      const float alpha = min(0.99f, con_o.w * G);
      if (alpha < 1.0f / 255.0f)
        continue;

      T = T / (1.f - alpha);
      const float dchannel_dcolor = alpha * T;

      // Propagate gradients to per-Gaussian colors and keep
      // gradients w.r.t. alpha (blending factor for a Gaussian/pixel
      // pair).
      float dL_dalpha = 0.0f;
      const int global_id = collected_id[j];
      for (int ch = 0; ch < C; ch++) {
        const float c = collected_colors[ch * BLOCK_SIZE + j];
        // Update last color (to be used in the next iteration)
        accum_rec[ch] =
            last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
        last_color[ch] = c;

        const float dL_dchannel = dL_dpixel[ch];
        dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
        // Update the gradients w.r.t. color of the Gaussian.
        // Atomic, since this pixel is just one of potentially
        // many that were affected by this Gaussian.
        if (global_id < P)
          atomicAdd(&(dL_dcolors[global_id * C + ch]),
                    dchannel_dcolor * dL_dchannel);
        else
          atomicAdd(&(dL_dcolors_static[(global_id - P) * C + ch]),
                    dchannel_dcolor * dL_dchannel);
      }

      if (global_id < P) {
        for (int ch = 0; ch < 2; ch++) {
          const float flow = collected_flows[ch * BLOCK_SIZE + j];
          // Update last color (to be used in the next iteration)
          accum_flow_rec[ch] = last_alpha * last_flow[ch] +
                               (1.f - last_alpha) * accum_flow_rec[ch];
          last_flow[ch] = flow;

          const float dL_dchannelflow = dL_dflow[ch];
          dL_dalpha += (flow - accum_flow_rec[ch]) * dL_dchannelflow;
          // Update the gradients w.r.t. color of the Gaussian.
          // Atomic, since this pixel is just one of potentially
          // many that were affected by this Gaussian.
          atomicAdd(&(dL_dflows[global_id * 2 + ch]),
                    dchannel_dcolor * dL_dchannelflow);
        }
      }
      // Propagate gradients to per-Gaussian depths
      const float c_d = collected_depths[j];
      accum_depth_rec =
          last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
      last_depth = c_d;
      dL_dalpha += ((c_d - accum_depth_rec) * dL_depth);

      // Propagate gradients from masks
      accum_mask_rec = last_alpha + (1.f - last_alpha) * accum_mask_rec;
      dL_dalpha += ((1.0 - accum_mask_rec) * dL_mask);

      dL_dalpha *= T;
      // Update last alpha (to be used in the next iteration)
      last_alpha = alpha;

      // Account for fact that alpha also influences how much of
      // the background color is added if nothing left to blend
      float bg_dot_dpixel = 0;
      for (int i = 0; i < C; i++)
        bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
      dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

      // Helpful reusable temporary variables
      const float dL_dG = con_o.w * dL_dalpha;
      const float gdx = G * d.x;
      const float gdy = G * d.y;
      const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
      const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

      if (global_id < P) {
        // Update gradients w.r.t. 2D mean position of the Gaussian
        atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
        atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
        atomicAdd(&dL_dmean2D[global_id].z, dL_depth * dchannel_dcolor);

        // Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
        atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
        atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
        atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

        // Update gradients w.r.t. opacity of the Gaussian
        atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
      } else {
        // Update gradients w.r.t. 2D mean position of the Gaussian
        atomicAdd(&dL_dmean2D_static[global_id - P].x,
                  dL_dG * dG_ddelx * ddelx_dx);
        atomicAdd(&dL_dmean2D_static[global_id - P].y,
                  dL_dG * dG_ddely * ddely_dy);

        // Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
        atomicAdd(&dL_dconic2D_static[global_id - P].x,
                  -0.5f * gdx * d.x * dL_dG);
        atomicAdd(&dL_dconic2D_static[global_id - P].y,
                  -0.5f * gdx * d.y * dL_dG);
        atomicAdd(&dL_dconic2D_static[global_id - P].w,
                  -0.5f * gdy * d.y * dL_dG);

        // Update gradients w.r.t. opacity of the Gaussian
        atomicAdd(&(dL_dopacity_static[global_id - P]), G * dL_dalpha);
      }
    }
  }
}

template <uint32_t C>
__global__ void PerGaussianRenderCUDA(
    int P, const uint2 *__restrict__ ranges,
    const uint32_t *__restrict__ point_list, int W, int H, int B,
    const uint32_t *__restrict__ per_tile_bucket_offset,
    const uint32_t *__restrict__ bucket_to_tile,
    const float *__restrict__ sampled_T, const float *__restrict__ sampled_ar,
    const float *__restrict__ sampled_ard, const float *__restrict__ bg_color,
    const float2 *__restrict__ points_xy_image,
    const float4 *__restrict__ conic_opacity,
    const float4 *__restrict__ conic_opacity_static,
    const float *__restrict__ colors, const float *__restrict__ colors_static,
    const float *__restrict__ depths, const float *__restrict__ flows_2d,
    const float *__restrict__ final_Ts, const uint32_t *__restrict__ n_contrib,
    const uint32_t *__restrict__ max_contrib,
    const float *__restrict__ pixel_colors,
    const float *__restrict__ pixel_invDepths,
    const float *__restrict__ dL_dpixels,
    const float *__restrict__ dL_invdepths, const float *__restrict__ dL_depths,
    const float *__restrict__ dL_depths_4d,
    const float *__restrict__ dL_depths_3d, const float *__restrict__ dL_masks,
    const float *__restrict__ dL_dpix_flow,
    const float *__restrict__ dL_dpix_4d, const float *__restrict__ dL_dpix_3d,
    float3 *__restrict__ dL_dmean2D, float4 *__restrict__ dL_dconic2D,
    float *__restrict__ dL_dopacity, float *__restrict__ dL_dcolors,
    float *__restrict__ dL_dflows, float3 *__restrict__ dL_dmean2D_static,
    float4 *__restrict__ dL_dconic2D_static,
    float *__restrict__ dL_dopacity_static,
    float *__restrict__ dL_dcolors_static) {
  // global_bucket_idx = warp_idx
  auto block = cg::this_thread_block();
  auto my_warp = cg::tiled_partition<32>(block);
  uint32_t global_bucket_idx =
      block.group_index().x * my_warp.meta_group_size() +
      my_warp.meta_group_rank();
  bool valid_bucket = global_bucket_idx < (uint32_t)B;
  if (!valid_bucket)
    return;

  bool valid_splat = false;

  uint32_t tile_id, bbm;
  uint2 range;
  int num_splats_in_tile, bucket_idx_in_tile;
  int splat_idx_in_tile, splat_idx_global;

  tile_id = bucket_to_tile[global_bucket_idx];
  range = ranges[tile_id];
  num_splats_in_tile = range.y - range.x;
  // What is the number of buckets before me? what is my offset?
  bbm = tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
  bucket_idx_in_tile = global_bucket_idx - bbm;
  splat_idx_in_tile = bucket_idx_in_tile * 32 + my_warp.thread_rank();
  splat_idx_global = range.x + splat_idx_in_tile;
  valid_splat = (splat_idx_in_tile < num_splats_in_tile);

  // if first gaussian in bucket is useless, then others are also useless
  if (bucket_idx_in_tile * 32 >= max_contrib[tile_id]) {
    return;
  }

  // Load Gaussian properties into registers
  int gaussian_idx = 0;
  float2 xy = {0.0f, 0.0f};
  float4 con_o = {0.0f, 0.0f, 0.0f, 0.0f};
  float c[C] = {0.0f};
  float invd = 0.f;
  if (valid_splat) {
    gaussian_idx = point_list[splat_idx_global];
    xy = points_xy_image[gaussian_idx];
    invd = 1.f / depths[gaussian_idx];
    if (gaussian_idx < P) {
      con_o = conic_opacity[gaussian_idx];
      for (int ch = 0; ch < C; ++ch)
        c[ch] = colors[gaussian_idx * C + ch];

    } else {
      con_o = conic_opacity_static[gaussian_idx - P];
      for (int ch = 0; ch < C; ++ch)
        c[ch] = colors_static[(gaussian_idx - P) * C + ch];
    }
  }

  // Gradient accumulation variables
  float Register_dL_dmean2D_x = 0.0f;
  float Register_dL_dmean2D_y = 0.0f;
  float Register_dL_dconic2D_x = 0.0f;
  float Register_dL_dconic2D_y = 0.0f;
  float Register_dL_dconic2D_w = 0.0f;
  float Register_dL_dopacity = 0.0f;
  float Register_dL_dcolors[C] = {0.0f};
  float Register_dL_dinvdepths = 0.0f;

  // tile metadata
  const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
  const uint2 tile = {tile_id % horizontal_blocks, tile_id / horizontal_blocks};
  const uint2 pix_min = {tile.x * BLOCK_X, tile.y * BLOCK_Y};

  // values useful for gradient calculation
  float T;
  float T_final;
  float last_contributor;
  float ar[C];
  float ard;
  float dL_dpixel[C];
  float dL_invdepth;
  float dL_depth;
  float dL_mask;
  const float ddelx_dx = 0.5 * W;
  const float ddely_dy = 0.5 * H;

  // iterate over all pixels in the tile
  for (int i = 0; i < BLOCK_SIZE + 31; ++i) {
    // SHUFFLING

    // At this point, T already has my (1 - alpha) multiplied.
    // So pass this ready-made T value to next thread.
    T = my_warp.shfl_up(T, 1);
    last_contributor = my_warp.shfl_up(last_contributor, 1);
    T_final = my_warp.shfl_up(T_final, 1);
    for (int ch = 0; ch < C; ++ch) {
      ar[ch] = my_warp.shfl_up(ar[ch], 1);
      dL_dpixel[ch] = my_warp.shfl_up(dL_dpixel[ch], 1);
    }
    ard = my_warp.shfl_up(ard, 1);
    dL_invdepth = my_warp.shfl_up(dL_invdepth, 1);
    dL_depth = my_warp.shfl_up(dL_depth, 1);
    dL_mask = my_warp.shfl_up(dL_mask, 1);
    // dL_depth_4d/3d are deliberately NOT in this shuffle chain: like
    // dL_dpixel_4d/3d, every thread re-reads them per pixel below, so
    // propagating them along the warp would be redundant.

    // which pixel index should this thread deal with?
    int idx = i - my_warp.thread_rank();
    const uint2 pix = {pix_min.x + idx % BLOCK_X, pix_min.y + idx / BLOCK_X};
    const uint32_t pix_id = W * pix.y + pix.x;
    const float2 pixf = {(float)pix.x, (float)pix.y};
    bool valid_pixel = pix.x < W && pix.y < H;

    // every 32nd thread should read the stored state from memory
    float dL_dpixel_4d[C] = {0.0f};
    float dL_dpixel_3d[C] = {0.0f};
    float dL_depth_4d = 0.0f;
    float dL_depth_3d = 0.0f;
    if (valid_splat && valid_pixel && my_warp.thread_rank() == 0 &&
        idx < BLOCK_SIZE) {
      T = sampled_T[global_bucket_idx * BLOCK_SIZE + idx];
      for (int ch = 0; ch < C; ++ch)
        ar[ch] = -pixel_colors[ch * H * W + pix_id] +
                 sampled_ar[global_bucket_idx * BLOCK_SIZE * C +
                            ch * BLOCK_SIZE + idx];
      ard = -pixel_invDepths[pix_id] +
            sampled_ard[global_bucket_idx * BLOCK_SIZE + idx];
      T_final = final_Ts[pix_id];
      last_contributor = n_contrib[pix_id];
      for (int ch = 0; ch < C; ++ch) {
        dL_dpixel[ch] = dL_dpixels[ch * H * W + pix_id];
      }
      dL_invdepth = dL_invdepths[pix_id];
      dL_invdepth = dL_invdepths[pix_id];
      dL_depth = dL_depths[pix_id];
      if (dL_depths_4d != nullptr)
        dL_depth_4d = dL_depths_4d[pix_id];
      if (dL_depths_3d != nullptr)
        dL_depth_3d = dL_depths_3d[pix_id];
      dL_mask = dL_masks[pix_id];
    }

    // Read separate gradients for 4D and 3D per pixel
    if (valid_pixel && idx < BLOCK_SIZE) {
      if (dL_dpix_4d != nullptr) {
        for (int ch = 0; ch < C; ++ch) {
          dL_dpixel_4d[ch] = dL_dpix_4d[ch * H * W + pix_id];
        }
      }
      if (dL_dpix_3d != nullptr) {
        for (int ch = 0; ch < C; ++ch) {
          dL_dpixel_3d[ch] = dL_dpix_3d[ch * H * W + pix_id];
        }
      }
      if (dL_depths_4d != nullptr)
        dL_depth_4d = dL_depths_4d[pix_id];
      if (dL_depths_3d != nullptr)
        dL_depth_3d = dL_depths_3d[pix_id];
    }

    // do work
    if (valid_splat && valid_pixel && 0 <= idx && idx < BLOCK_SIZE) {
      if (W <= pix.x || H <= pix.y)
        continue;

      if (splat_idx_in_tile >= last_contributor)
        continue;

      // compute blending values
      const float2 d = {xy.x - pixf.x, xy.y - pixf.y};
      const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) -
                          con_o.y * d.x * d.y;
      if (power > 0.0f)
        continue;
      const float G = exp(power);
      const float alpha = min(0.99f, con_o.w * G);
      if (alpha < 1.0f / 255.0f)
        continue;
      const float weight = alpha * T;

      // add the gradient contribution of this pixel's colour to the gaussian
      // Use separate gradients for 4D and 3D Gaussians
      float bg_dot_dpixel = 0.0f;
      float dL_dalpha = 0.0f;
      const bool is_4d = (gaussian_idx < P);
      for (int ch = 0; ch < C; ++ch) {
        ar[ch] += weight * c[ch];
        // Use combined gradient for main output, separate gradients for 4D/3D
        const float &dL_dchannel = dL_dpixel[ch];
        const float &dL_dchannel_4d =
            (is_4d && dL_dpix_4d != nullptr) ? dL_dpixel_4d[ch] : 0.0f;
        const float &dL_dchannel_3d =
            (!is_4d && dL_dpix_3d != nullptr) ? dL_dpixel_3d[ch] : 0.0f;

        // Accumulate gradients: combined + separate
        const float dL_dchannel_total =
            dL_dchannel + dL_dchannel_4d + dL_dchannel_3d;
        Register_dL_dcolors[ch] += weight * dL_dchannel_total;
        dL_dalpha += ((c[ch] * T) - (1.0f / (1.0f - alpha)) * (-ar[ch])) *
                     dL_dchannel_total;

        bg_dot_dpixel += bg_color[ch] * dL_dpixel[ch];
      }

      // // add the gradient contribution of this pixel's depth to the gaussian
      ard += weight * invd;
      Register_dL_dinvdepths += weight * dL_invdepth;

      // Gradient of loss w.r.t. depth of the Gaussian
      // dL/dz = dL/dD * dD/dz = dL_depth * alpha * T
      // We store this in the .z component of dL_dmean2D, which is treated as
      // dL/d(1/z) in computeCov2DCUDA. Conversion: dL/d(1/z) = dL/dz *
      // dz/d(1/z) = dL/dz * (-z^2) So we add: (dL_depth * alpha * T) * (-depth
      // * depth) Note: depth = 1.0f / invd Conversion: dL/d(1/z) = dL/dz *
      // dz/d(1/z) = dL/dz * (-z^2) So we add: (dL_depth * alpha * T) * (-depth
      // * depth) Note: depth = 1.0f / invd
      float depth_val = 1.0f / (invd + 1e-7f);
      float total_dL_depth = dL_depth + (is_4d ? dL_depth_4d : dL_depth_3d);
      Register_dL_dinvdepths +=
          (weight * total_dL_depth) * (-depth_val * depth_val);

      dL_dalpha +=
          ((invd * T) - (1.0f / (1.0f - alpha)) * (-ard)) * dL_invdepth;
      // Add approximate opacity gradient from depth (ignoring future
      // accumulated depth term)
      dL_dalpha += (depth_val * T) * total_dL_depth;

      // Account for last sample for colour
      dL_dalpha += (-T_final / (1.0f - alpha)) * bg_dot_dpixel;

      // Account for mask gradient
      dL_dalpha += (T_final / (1.0f - alpha)) * dL_mask;

      T *= (1.0f - alpha);

      // Helpful reusable temporary variables
      const float dL_dG = con_o.w * dL_dalpha;
      const float gdx = G * d.x;
      const float gdy = G * d.y;
      const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
      const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

      // accumulate the gradients
      const float tmp_x = dL_dG * dG_ddelx * ddelx_dx;
      Register_dL_dmean2D_x += tmp_x;
      const float tmp_y = dL_dG * dG_ddely * ddely_dy;
      Register_dL_dmean2D_y += tmp_y;

      Register_dL_dconic2D_x += -0.5f * gdx * d.x * dL_dG;
      Register_dL_dconic2D_y += -0.5f * gdx * d.y * dL_dG;
      Register_dL_dconic2D_w += -0.5f * gdy * d.y * dL_dG;
      Register_dL_dopacity += G * dL_dalpha;
    }
  }

  // finally add the gradients using atomics
  if (valid_splat) {
    if (gaussian_idx < P) {
      atomicAdd(&dL_dmean2D[gaussian_idx].x, Register_dL_dmean2D_x);
      atomicAdd(&dL_dmean2D[gaussian_idx].y, Register_dL_dmean2D_y);
      atomicAdd(&dL_dmean2D[gaussian_idx].z, Register_dL_dinvdepths);
      atomicAdd(&dL_dconic2D[gaussian_idx].x, Register_dL_dconic2D_x);
      atomicAdd(&dL_dconic2D[gaussian_idx].y, Register_dL_dconic2D_y);
      atomicAdd(&dL_dconic2D[gaussian_idx].w, Register_dL_dconic2D_w);
      atomicAdd(&dL_dopacity[gaussian_idx], Register_dL_dopacity);
      for (int ch = 0; ch < C; ++ch) {
        atomicAdd(&dL_dcolors[gaussian_idx * C + ch], Register_dL_dcolors[ch]);
      }
    } else {

      atomicAdd(&dL_dmean2D_static[gaussian_idx - P].x, Register_dL_dmean2D_x);
      atomicAdd(&dL_dmean2D_static[gaussian_idx - P].y, Register_dL_dmean2D_y);
      atomicAdd(&dL_dmean2D_static[gaussian_idx - P].z, Register_dL_dinvdepths);
      atomicAdd(&dL_dconic2D_static[gaussian_idx - P].x,
                Register_dL_dconic2D_x);
      atomicAdd(&dL_dconic2D_static[gaussian_idx - P].y,
                Register_dL_dconic2D_y);
      atomicAdd(&dL_dconic2D_static[gaussian_idx - P].w,
                Register_dL_dconic2D_w);
      atomicAdd(&dL_dopacity_static[gaussian_idx - P], Register_dL_dopacity);
      for (int ch = 0; ch < C; ++ch) {
        atomicAdd(&dL_dcolors_static[(gaussian_idx - P) * C + ch],
                  Register_dL_dcolors[ch]);
      }
    }

    // atomicAdd(&dL_dinvdepths[gaussian_idx], Register_dL_dinvdepths);
  }
}

void BACKWARD::preprocess(
    int P, int D, int D_t, int M, const float3 *means3D, const int *radii,
    const float *shs, const float *ts, const float *opacities,
    const bool *clamped, const uint32_t *tiles_touched, const glm::vec3 *scales,
    const float *scales_t, const glm::vec4 *rotations,
    const glm::vec4 *rotations_r, const float scale_modifier,
    const float *cov3Ds, const float *viewmatrix, const float *projmatrix,
    const float focal_x, float focal_y, const float tan_fovx, float tan_fovy,
    const glm::vec3 *campos, const float timestamp, const float time_duration,
    const bool rot_4d, const int gaussian_dim, const bool force_sh_3d,
    const float3 *dL_dmean2D, const float *dL_dconic, glm::vec3 *dL_dmean3D,
    float *dL_dcolor, float *dL_dcov3D, float *dL_dsh, float *dL_dts,
    glm::vec3 *dL_dscale, float *dL_dscale_t, glm::vec4 *dL_drot,
    glm::vec4 *dL_drot_r, float *dL_dopacity) {
  // Propagate gradients for the path of 2D conic matrix computation.
  // Somewhat long, thus it is its own kernel rather than being part of
  // "preprocess". When done, loss gradient w.r.t. 3D means has been
  // modified and gradient w.r.t. 3D covariance matrix has been computed.
  computeCov2DCUDA<<<(P + 255) / 256, 256>>>(
      P, means3D, radii, cov3Ds, focal_x, focal_y, tan_fovx, tan_fovy,
      viewmatrix, dL_dconic, dL_dmean2D, (float3 *)dL_dmean3D, dL_dcov3D);

  // Propagate gradients for remaining steps: finish 3D mean gradients,
  // propagate color gradients to SH (if desireD), propagate 3D covariance
  // matrix gradients to scale and rotation.
  preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
      P, D, D_t, M, (float3 *)means3D, radii, shs, ts, opacities, clamped,
      tiles_touched, (glm::vec3 *)scales, scales_t, (glm::vec4 *)rotations,
      (glm::vec4 *)rotations_r, scale_modifier, projmatrix, campos, timestamp,
      time_duration, rot_4d, gaussian_dim, force_sh_3d, (float3 *)dL_dmean2D,
      (glm::vec3 *)dL_dmean3D, dL_dcolor, dL_dcov3D, dL_dsh, dL_dts, dL_dscale,
      dL_dscale_t, dL_drot, dL_drot_r, dL_dopacity);
}

void BACKWARD::render(
    int P, const dim3 grid, const dim3 block, const uint2 *ranges,
    const uint32_t *point_list, int W, int H, int R, int B,
    const uint32_t *per_bucket_tile_offset, const uint32_t *bucket_to_tile,
    const float *sampled_T, const float *sampled_ar, const float *sampled_ard,
    const float *bg_color, const float2 *means2D, const float4 *conic_opacity,
    const float4 *conic_opacity_static, const float *colors,
    const float *colors_static, const float *depths, const float *flows_2d,
    const float *final_Ts, const uint32_t *n_contrib,
    const uint32_t *max_contrib, const float *pixel_colors,
    const float *pixel_invdepths, const float *dL_dpixels,
    const float *dL_invdepths, const float *dL_depths,
    const float *dL_depths_4d, const float *dL_depths_3d, const float *dL_masks,
    const float *dL_dpix_flow, const float *dL_dpix_4d, const float *dL_dpix_3d,
    float3 *dL_dmean2D, float4 *dL_dconic2D, float *dL_dopacity,
    float *dL_dcolors, float *dL_dflows, float3 *dL_dmean2D_static,
    float4 *dL_dconic2D_static, float *dL_dopacity_static,
    float *dL_dcolors_static) {

  const int THREADS = 32;
  PerGaussianRenderCUDA<NUM_CHANNELS>
      <<<((B * 32) + THREADS - 1) / THREADS, THREADS>>>(
          P, ranges, point_list, W, H, B, per_bucket_tile_offset,
          bucket_to_tile, sampled_T, sampled_ar, sampled_ard, bg_color, means2D,
          conic_opacity, conic_opacity_static, colors, colors_static, depths,
          flows_2d, final_Ts, n_contrib, max_contrib, pixel_colors,
          pixel_invdepths, dL_dpixels, dL_invdepths, dL_depths, dL_depths_4d,
          dL_depths_3d, dL_masks, dL_dpix_flow, dL_dpix_4d, dL_dpix_3d,
          dL_dmean2D, dL_dconic2D, dL_dopacity, dL_dcolors, dL_dflows,
          dL_dmean2D_static, dL_dconic2D_static, dL_dopacity_static,
          dL_dcolors_static);
}

void BACKWARD::preprocess_static(
    int P, int P_static, int D, int M, const float3 *means3D_static,
    const int *radii_static, const float *shs_static,
    const float *opacities_static, const bool *clamped,
    const uint32_t *tiles_touched, const glm::vec3 *scales_static,
    const glm::vec4 *rotations_static, const float scale_modifier,
    const float *cov3Ds_static, const float *viewmatrix,
    const float *projmatrix, const float focal_x, float focal_y,
    const float tan_fovx, float tan_fovy, const glm::vec3 *campos,
    const float3 *dL_dmean2D_static, const float *dL_dconic_static,
    glm::vec3 *dL_dmean3D_static, float *dL_dcolor_static,
    float *dL_dcov3D_static, float *dL_dsh_static, glm::vec3 *dL_dscale_static,
    glm::vec4 *dL_drot_static, float *dL_dopacity_static) {
  // Propagate gradients for the path of 2D conic matrix computation.
  // Somewhat long, thus it is its own kernel rather than being part of
  // "preprocess". When done, loss gradient w.r.t. 3D means has been
  // modified and gradient w.r.t. 3D covariance matrix has been computed.
  computeCov2DCUDA<<<(P_static + 255) / 256, 256>>>(
      P_static, means3D_static, radii_static, cov3Ds_static, focal_x, focal_y,
      tan_fovx, tan_fovy, viewmatrix, dL_dconic_static, dL_dmean2D_static,
      (float3 *)dL_dmean3D_static, dL_dcov3D_static);

  // Propagate gradients for remaining steps: finish 3D mean gradients,
  // propagate color gradients to SH (if desireD), propagate 3D covariance
  // matrix gradients to scale and rotation.
  preprocessCUDA_static<NUM_CHANNELS><<<(P_static + 255) / 256, 256>>>(
      P, P_static, D, M, (float3 *)means3D_static, radii_static, shs_static,
      opacities_static, clamped, tiles_touched, (glm::vec3 *)scales_static,
      (glm::vec4 *)rotations_static, scale_modifier, projmatrix, campos,
      (float3 *)dL_dmean2D_static, (glm::vec3 *)dL_dmean3D_static,
      dL_dcolor_static, dL_dcov3D_static, dL_dsh_static, dL_dscale_static,
      dL_drot_static, dL_dopacity_static);
}