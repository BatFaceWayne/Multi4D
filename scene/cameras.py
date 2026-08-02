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
from torch import nn
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCV

class Camera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, image,
                 image_name,
                 time = 0,
                 cxr=0.0, cyr=0.0, image_path=None
                 ):
        super(Camera, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        # Full source-image path (dynerf layout .../camXX/images/YYYY.png); used by the
        # TRASE semantic stage to locate the matching SAM mask masks/camXX_YYYY.pt.
        self.image_path = image_path
        self.time = time
        # Principal-point offset (Technicolor COLMAP intrinsics); 0.0 for dynerf (centered)
        self.cxr = cxr
        self.cyr = cyr

        self.original_image = image

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # dynerf keeps zfar=100 (validated); Technicolor's x100 scene rescale needs a farther plane
        self.zfar = 1000.0 if (cxr != 0.0 or cyr != 0.0) else 100.0
        self.znear = 0.01

        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1)
        # .cuda()
        if self.cxr == 0.0 and self.cyr == 0.0:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        else:
            self.projection_matrix = getProjectionMatrixCV(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=self.cxr, cy=self.cyr).transpose(0,1)
        # .cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

