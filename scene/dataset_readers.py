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
# -----------------------------------------------------------------------------
# Dataset Readers and Scene Builders
# -----------------------------------------------------------------------------
# Utility functions for loading Colmap/Blender/HyperNeRF cameras and constructing
# SceneInfo objects. This is a high-level IO layer and should remain free of any
# heavy math or rendering code.

import os
import sys
from typing import NamedTuple
import torch
import natsort
from PIL import Image
from utils.graphics_utils import getWorld2View2, focal2fov
import numpy as np
from plyfile import PlyData
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, \
    read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from scene.hyper_loader import Load_hyper_data, format_hyper_data
from tqdm import tqdm


class CameraInfo(NamedTuple):
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time: float
    cxr: float = 0.0
    cyr: float = 0.0


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    point_cloud_second: BasicPointCloud = None


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    return {"translate": translate, "radius": radius, "cam_centers": cam_centers}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def format_infos(dataset):
    # loading
    cameras = []
    image = dataset[0][0]
    for idx in tqdm(range(len(dataset))):
        image_path = None
        image_name = f"{idx}"
        time = dataset.image_times[idx]
        R, T = dataset.load_pose(idx)
        # image is CHW: shape[2] = W (FovX), shape[1] = H (FovY) — matches scene/dataset.py
        FovX = focal2fov(dataset.focal[0], image.shape[2])
        FovY = focal2fov(dataset.focal[0], image.shape[1])
        cameras.append(CameraInfo(R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                  image_path=image_path, image_name=image_name, width=image.shape[2],
                                  height=image.shape[1],
                                  time=time))

    return cameras


def readdynerfInfo(datadir, use_sky_pc_arg=False):
    # loading all the data follow hexplane format
    ply_path = os.path.join(datadir, "fused.ply")
    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
        datadir,
        "train",
        1.0,
        eval_index=0,
    )
    test_dataset = Neural3D_NDC_Dataset(
        datadir,
        "test",
        1.0,
        eval_index=0,
    )

    train_cam_infos = format_infos(train_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    pcd = fetchPly(ply_path)
    use_sky_pc = use_sky_pc_arg   # gated by ModelParams.use_dynerf_sky_pc (default False = current)
    sky_pc = None
    xyz_copy_ori = np.array(pcd.points)
    xyz = xyz_copy_ori
    if use_sky_pc:
        center = -nerf_normalization['translate']  # Center of the scene
        sphere_radius =  30 * nerf_normalization['radius'] #np.sqrt(((xyz_copy_ori - center) ** 2).sum(axis=1)).max() * 2

        # Radius of the scene
        num_points = 50000  # Number of points for the sphere
        phi = np.random.uniform(0, 2 * np.pi, num_points)  # Azimuthal angle
        cos_theta = np.random.uniform(-1, 1, num_points)  # Uniform distribution for theta
        theta = np.arccos(cos_theta)
        x = sphere_radius * np.sin(theta) * np.cos(phi) + center[0]
        y = sphere_radius * np.sin(theta) * np.sin(phi) + center[1]
        z = sphere_radius * np.cos(theta) + center[2]

        # Combine into a point cloud
        sky_pc = np.vstack((x, y, z)).T

    if sky_pc is not None:
        xyz = np.vstack((xyz_copy_ori, sky_pc))

    # KNOWN ARTIFACT — do NOT "fix" without re-validating hard-scene numbers:
    # fetchPly already divided colors by 255, so this second /255 double-divides.
    # Real BG points therefore initialize near-black, while sky points (use_sky_pc)
    # get random [0,1) DC. The e308_sky_deform golden-config behavior depends on this.
    shs_base = pcd.colors / 255
    if not use_sky_pc:
        shs = shs_base
    else:
        shs = np.random.random((xyz.shape[0], 3))  # / 255.0
        shs[:(len(shs_base))] = shs_base

    pcd = BasicPointCloud(points=xyz, colors=(shs), normals=np.zeros(np.shape(xyz)))



    # FG init: random points inside camera-bounded scene extent
    all_cam_center = np.hstack(nerf_normalization['cam_centers']).T

    camera_max = all_cam_center.max(axis=0) + 5 * nerf_normalization['radius']
    camera_min = all_cam_center.min(axis=0) - 5 * nerf_normalization['radius']

    pcd_max = (np.vstack((camera_max, xyz_copy_ori.max(axis=0)))).min(axis=0)
    pcd_min = (np.vstack((camera_min, xyz_copy_ori.min(axis=0)))).max(axis=0)

    num_pts = 10000
    print(f"Generating random point cloud ({num_pts})...")

    x = np.random.random((num_pts, 1)) * (pcd_max[0] - pcd_min[0]) + pcd_min[0]
    y = np.random.random((num_pts, 1)) * (pcd_max[1] - pcd_min[1]) + pcd_min[1]
    z = np.random.random((num_pts, 1)) * (pcd_max[2] - pcd_min[2]) + pcd_min[2]
    xyz = np.hstack([x, y, z])

    shs = np.random.random((xyz.shape[0], 3))
    pcd_2 = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((len(xyz), 3)))

    scene_info = SceneInfo(point_cloud=pcd_2,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           nerf_normalization=nerf_normalization,
                           point_cloud_second=pcd,
                           )

    return scene_info


def readColmapCamerasTechnicolor(cam_extrinsics, cam_intrinsics, images_folder, startime, duration):
    cam_infos = []
    keys_list = natsort.natsorted(list(cam_extrinsics.keys()))

    for t in range(duration):
        current_time = startime + t
        # Per-timestep image folder: ".../colmap_<startime>/images" -> ".../colmap_<current_time>/images"
        current_images_folder = images_folder.replace(f"colmap_{startime}", f"colmap_{current_time}")

        if not os.path.exists(current_images_folder):
            print(f"Warning: {current_images_folder} does not exist")
            continue

        for idx, key in enumerate(keys_list):
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]

            height = intr.height
            width = intr.width

            R = np.transpose(qvec2rotmat(extr.qvec))

            test_upscale = 100
            T = np.array(extr.tvec) * test_upscale  # scene rescale for numerical stability

            if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model == "OPENCV":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert False, "Colmap camera model not handled"

            image_path = os.path.join(current_images_folder, os.path.basename(extr.name))
            try:
                image_name = os.path.basename(image_path).split(".")[0] + "_" + str(current_time)
            except:
                image_name = os.path.basename(image_path) + "_" + str(current_time)

            image = Image.open(image_path)
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1)

            cxr = (intr.params[2] / width - 0.5)
            cyr = (intr.params[3] / height - 0.5)

            cam_info = CameraInfo(R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                  image_path=image_path, image_name=image_name, width=width, height=height,
                                  time=float(t)/duration, cxr=cxr, cyr=cyr)
            cam_infos.append(cam_info)
            sys.stdout.write('\r')
            sys.stdout.write("Reading camera {}/{}".format(idx + 1 + t*len(keys_list), len(keys_list)*duration))
            sys.stdout.flush()

    sys.stdout.write('\n')
    return cam_infos


def readColmapSceneInfoTechnicolor(path, duration=50):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images"

    starttime = os.path.basename(path).split("_")[1]  # colmap_0 -> "0"
    assert starttime.isdigit(), "Colmap folder name must be colmap_<startime>_<duration>!"
    starttime = int(starttime)

    cam_infos_unsorted = readColmapCamerasTechnicolor(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), startime=starttime, duration=duration)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    # Technicolor protocol: hold out cam10 as the test view
    train_cam_infos = [_ for _ in cam_infos if "cam10" not in _.image_name]
    test_cam_infos = [_ for _ in cam_infos if "cam10" in _.image_name]
    if len(test_cam_infos) > 0:
        uniquecheck = []
        for cam_info in test_cam_infos:
            if cam_info.image_name not in uniquecheck:
                uniquecheck.append(cam_info.image_name)
        sanitycheck = []
        for cam_info in train_cam_infos:
            if cam_info.image_name not in sanitycheck:
                sanitycheck.append(cam_info.image_name)
        for testname in uniquecheck:
            assert testname not in sanitycheck
    else:
        first_cam = cam_infos[0].image_name
        print("do custom loader training, select first cam as test frame: ", first_cam)
        cam_infos = natsort.natsorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
        train_cam_infos = [_ for _ in cam_infos if first_cam not in _.image_name]
        test_cam_infos = [_ for _ in cam_infos if first_cam in _.image_name]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    totalply_path = os.path.join(path, "fused.ply")

    try:
        pcd = fetchPly(totalply_path)
    except Exception as e:
        raise RuntimeError(
            f"Could not load point cloud from {totalply_path} ({e}). "
            f"The Technicolor reader requires a fused.ply in the scene root.") from e
    shs = np.array(pcd.colors)
    test_upscale = 100
    xyz = np.array(pcd.points) * test_upscale  # match the x100 camera rescale
    pcd = BasicPointCloud(points=xyz, colors=(shs), normals=np.zeros(np.shape(xyz)))
    print("Loaded point cloud from: ", totalply_path)

    xyz_copy_ori = np.array(pcd.points)
    pcd_max = xyz_copy_ori.max(axis=0)
    pcd_min = xyz_copy_ori.min(axis=0)
    num_pts = 10000
    print(f"Generating random point cloud ({num_pts})...")
    x = np.random.random((num_pts, 1)) * (pcd_max[0] - pcd_min[0]) + pcd_min[0]
    y = np.random.random((num_pts, 1)) * (pcd_max[1] - pcd_min[1]) + pcd_min[1]
    z = np.random.random((num_pts, 1)) * (pcd_max[2] - pcd_min[2]) + pcd_min[2]
    xyz = np.hstack([x, y, z])
    shs = np.random.random((xyz.shape[0], 3))
    pcd_2 = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((len(xyz), 3)))

    scene_info = SceneInfo(point_cloud=pcd_2,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           point_cloud_second=pcd,
                           )
    return scene_info


def readHyperDataInfos(datadir):
    # NeRF-DS / nerfies-format monocular dynamic scenes, always read at full
    # resolution. The scale used to be `1.0 if 'nerf-ds' in datadir else 0.5`,
    # i.e. chosen from a substring of the source path, so a scene stored outside a
    # folder literally named "nerf-ds" silently trained on rgb/2x. Every scale
    # ships with the data (1x/2x/4x/8x/16x), so it raised nothing -- it just
    # halved the resolution and produced numbers that did not match.
    train_cam_infos = Load_hyper_data(datadir, split="train")
    test_cam_infos = Load_hyper_data(datadir, split="test")
    print("load finished")
    train_cam = format_hyper_data(train_cam_infos, "train")
    print("format finished")

    nerf_normalization = getNerfppNorm(train_cam)

    # Point cloud: nerfies points.npy for BG init, random 10k inside bounds for FG init.
    print("Generating point cloud from nerfies...")
    xyz = np.load(os.path.join(datadir, "points.npy"))
    num_pts = xyz.shape[0]
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    pcd_max = pcd.points.max(axis=0)
    pcd_min = pcd.points.min(axis=0)
    num_pts = 10000
    x = np.random.random((num_pts, 1)) * (pcd_max[0] - pcd_min[0]) + pcd_min[0]
    y = np.random.random((num_pts, 1)) * (pcd_max[1] - pcd_min[1]) + pcd_min[1]
    z = np.random.random((num_pts, 1)) * (pcd_max[2] - pcd_min[2]) + pcd_min[2]
    xyz = np.hstack([x, y, z])
    shs = np.random.random((xyz.shape[0], 3))
    pcd_2 = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((len(xyz), 3)))

    scene_info = SceneInfo(point_cloud=pcd_2,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           point_cloud_second=pcd)
    return scene_info


sceneLoadTypeCallbacks = {
    "dynerf": readdynerfInfo,
    "technicolor": readColmapSceneInfoTechnicolor,
    "nerfies": readHyperDataInfos,
}
