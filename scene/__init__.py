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

import os
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
# NOTE: GaussianModel_dynamic is a RE-EXPORT for train.py / train_semantic.py
# (`from scene import ...`); it is not used in this file itself. Do not remove
# as "unused".
from scene.gaussian_model import GaussianModel, GaussianModel_dynamic
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import SH2RGB
import torch
import numpy as np
from scene.dataset import FourDGSdataset
from arguments import ModelParams
class Scene2gs_mixed:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None,
                 gaussians_second=None, gaussians_transient=None):
        """Container for the FG/BG/TR Gaussian branches and the train/test/video camera datasets of one scene."""
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.gaussians_second =gaussians_second
        self.gaussians_transient = gaussians_transient

        if load_iteration:
            self.loaded_iter = searchForMaxIteration(os.path.join(load_iteration, "point_cloud"))
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, getattr(args, 'use_dynerf_sky_pc', False))
        elif os.path.exists(os.path.join(args.source_path, "sparse", "0")):
            scene_info = sceneLoadTypeCallbacks["technicolor"](args.source_path)
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path)
        else:
            assert False, "Could not recognize scene type (expected poses_bounds.npy [dynerf], sparse/0 [technicolor], or dataset.json [nerfies/NeRF-DS])"
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras)

        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        self.gaussians._deformation.deformation_net.set_aabb(xyz_max, xyz_min)



        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(load_iteration,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(load_iteration,
                                                   "point_cloud",
                                                   "iteration_" + str(self.loaded_iter),
                                                   ))
            self.gaussians_second.load_ply(os.path.join(load_iteration,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud_second.ply"))
            # Load transient_gaussian from checkpoint if it exists
            if self.gaussians_transient is not None:
                transient_ply_path = os.path.join(load_iteration,
                                                   "point_cloud",
                                                   "iteration_" + str(self.loaded_iter),
                                                   "point_cloud_transient.ply")
                if os.path.exists(transient_ply_path):
                    self.gaussians_transient.load_ply(transient_ply_path)
                else:
                    # If checkpoint doesn't exist, initialize from foreground_gaussians
                    xyz = self.gaussians.get_xyz.detach().cpu().numpy()[:1]
                    sh_dc = self.gaussians._features_dc.squeeze(1).detach().cpu().numpy()[:1]
                    colors = SH2RGB(torch.tensor(sh_dc)).numpy()
                    normals = np.zeros_like(xyz)
                    pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)
                    self.gaussians_transient.create_from_pcd(pcd, self.cameras_extent)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians_second.create_from_pcd(scene_info.point_cloud_second, self.cameras_extent)
            # Initialize transient_gaussian from the same point cloud as foreground_gaussians
            if self.gaussians_transient is not None:
                # Extract point cloud from foreground_gaussians
                xyz = self.gaussians.get_xyz.detach().cpu().numpy()[:1]
                # Get colors from SH DC component (features_dc is [N, 1, 3])
                sh_dc = self.gaussians._features_dc.squeeze(1).detach().cpu().numpy()[:1]  # [N, 3]
                colors = SH2RGB(torch.tensor(sh_dc)).numpy()
                normals = np.zeros_like(xyz)
                pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)
                self.gaussians_transient.create_from_pcd(pcd, self.cameras_extent)

    def save(self, iteration, stage):
        # iteration == "best" -> a single dir that is overwritten on each new best.
        # Deliberately NOT "iteration_best": searchForMaxIteration int()-parses the
        # trailing token of every point_cloud/* entry.
        if iteration == "best":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/best")
        elif stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians_second.save_ply(os.path.join(point_cloud_path, "point_cloud_second.ply"))

        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if self.gaussians_transient is not None:
            self.gaussians_transient.save_ply(os.path.join(point_cloud_path, "point_cloud_transient.ply"))

        try:
            self.gaussians_second.save_deformation(point_cloud_path)
        except:
            pass
        try:
            self.gaussians.save_deformation(point_cloud_path)
        except:
            pass
    def getTrainCameras(self):
        return self.train_camera

    def getTestCameras(self):
        return self.test_camera
