import warnings

warnings.filterwarnings("ignore")

import json
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scene.utils import Camera
from typing import NamedTuple
from torch.utils.data import Dataset
from utils.graphics_utils import focal2fov


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


class Load_hyper_data(Dataset):
    def __init__(self,
                 datadir,
                 split="train"
                 ):

        datadir = os.path.expanduser(datadir)
        with open(f'{datadir}/metadata.json', 'r') as f:
            meta_json = json.load(f)
        with open(f'{datadir}/dataset.json', 'r') as f:
            dataset_json = json.load(f)

        self.all_img = dataset_json['ids']
        self.val_id = dataset_json['val_ids']
        self.split = split
        if len(self.val_id) == 0:
            self.i_train = np.array([i for i in np.arange(len(self.all_img)) if
                                     (i % 4 == 0)])
            self.i_test = self.i_train + 2
            self.i_test = self.i_test[:-1, ]
        else:
            self.train_id = dataset_json['train_ids']
            self.i_test = []
            self.i_train = []
            for i in range(len(self.all_img)):
                id = self.all_img[i]
                if id in self.val_id:
                    self.i_test.append(i)
                if id in self.train_id:
                    self.i_train.append(i)

        self.all_time = [meta_json[i]['time_id'] for i in self.all_img]
        max_time = max(self.all_time)
        self.all_time = [meta_json[i]['time_id'] / max_time for i in self.all_img]
        self.all_cam_params = []
        for im in self.all_img:
            camera = Camera.from_json(f'{datadir}/camera/{im}.json')
            self.all_cam_params.append(camera)

        # Always full resolution. The scale used to come from a `ratio` argument
        # that the caller derived from a substring of the source path, so a scene
        # stored outside a folder named "nerf-ds" silently trained on rgb/2x.
        self.all_img = [f'{datadir}/rgb/1x/{i}.png' for i in self.all_img]

        self.h, self.w = self.all_cam_params[0].image_shape
        self.map = {}

    def __getitem__(self, index):
        if self.split == "train":
            return self.load_raw(self.i_train[index])

        elif self.split == "test":
            return self.load_raw(self.i_test[index])

    def __len__(self):
        if self.split == "train":
            return len(self.i_train)
        elif self.split == "test":
            return len(self.i_test)

    def load_raw(self, idx):
        if idx in self.map.keys():
            return self.map[idx]
        camera = self.all_cam_params[idx]
        image = Image.open(self.all_img[idx])
        w = image.size[0]
        h = image.size[1]
        image = torch.from_numpy(np.asarray(image)).permute(2, 0, 1)

        image = image.to(torch.float32)[:3, :, :]
        time = self.all_time[idx]
        R = camera.orientation.T
        T = - camera.position @ R
        FovY = focal2fov(camera.focal_length, self.h)
        FovX = focal2fov(camera.focal_length, self.w)
        image_path = "/".join(self.all_img[idx].split("/")[:-1])
        image_name = self.all_img[idx].split("/")[-1]

        caminfo = CameraInfo(R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                             image_path=image_path, image_name=image_name, width=w, height=h, time=time
                             )
        self.map[idx] = caminfo
        return caminfo


def format_hyper_data(data_class, split):
    if split == "train":
        data_idx = data_class.i_train
    elif split == "test":
        data_idx = data_class.i_test
    cam_infos = []
    for index in tqdm(data_idx):
        camera = data_class.all_cam_params[index]
        time = data_class.all_time[index]
        R = camera.orientation.T
        T = - camera.position @ R

        FovY = focal2fov(camera.focal_length, data_class.h)
        FovX = focal2fov(camera.focal_length, data_class.w)
        image_path = "/".join(data_class.all_img[index].split("/")[:-1])
        image_name = data_class.all_img[index].split("/")[-1]

        cam_info = CameraInfo(R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                              image_path=image_path, image_name=image_name, width=int(data_class.w),
                              height=int(data_class.h), time=time
                              )
        cam_infos.append(cam_info)
    return cam_infos
