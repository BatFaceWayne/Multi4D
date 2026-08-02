import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class Neural3D_NDC_Dataset(Dataset):
    def __init__(
            self,
            datadir,
            split="train",
            downsample=1.0,
            eval_index=0,
    ):
        self.img_wh = (
            int(1352 / downsample),
            int(1014 / downsample),
        )  # According to the neural 3D paper, the default resolution is 1024x768
        self.root_dir = datadir
        self.split = split
        self.downsample = 2704 / self.img_wh[0]

        self.eval_index = eval_index


        self.load_meta()
        print(f"meta data loaded, total image:{len(self)}")

    def load_meta(self):
        """
        Load meta data from the dataset.
        """
        # Read poses and video file paths.
        poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        self.near_fars = poses_arr[:, -2:]
        videos = glob.glob(os.path.join(self.root_dir, "cam*.mp4"))
        videos = sorted(videos)
        assert len(videos) == poses_arr.shape[0]

        _, _, focal = poses[0, :, -1]
        focal = focal / self.downsample
        self.focal = [focal, focal]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)

        self.poses_all = poses
        self.image_paths, self.image_poses, self.image_times = self.load_images_path(videos, self.split)


    def load_images_path(self, videos, split):
        image_paths = []
        image_poses = []
        image_times = []
        countss = 300
        for index, video_path in enumerate(videos):

            if index == self.eval_index:
                if split == "train":
                    continue
            else:
                if split == "test":
                    continue
            count = 0
            video_images_path = video_path.split('.')[0]
            image_path = os.path.join(video_images_path, "images")
            video_frames = cv2.VideoCapture(video_path)

            if not os.path.exists(image_path):
                print(f"no images saved in {image_path}, extract images from video.")
                os.makedirs(image_path)
                this_count = 0
                while video_frames.isOpened():
                    ret, video_frame = video_frames.read()
                    if this_count >= countss: break
                    if ret:
                        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                        video_frame = Image.fromarray(video_frame)
                        if self.downsample != 1.0:
                            img = video_frame.resize(self.img_wh, Image.LANCZOS)
                        else:
                            img = video_frame
                        img.save(os.path.join(image_path, "%04d.png" % count))

                        count += 1
                        this_count += 1
                    else:
                        break

            images_path = os.listdir(image_path)
            images_path.sort()
            this_count = 0
            for idx, path in enumerate(images_path):
                if this_count >= countss: break
                image_paths.append(os.path.join(image_path, path))
                pose = np.array(self.poses_all[index])
                R = pose[:3, :3]
                R = -R
                R[:, 0] = -R[:, 0]
                T = -pose[:3, 3].dot(R)
                image_times.append(idx / countss)
                image_poses.append((R, T))
                this_count += 1

            if this_count != countss:
                raise ValueError(
                    f"Expected {countss} frames in {image_path} but found {this_count}. "
                    f"A partially extracted images folder corrupts time normalization "
                    f"(timestamps are idx/{countss}); delete the folder to re-extract from the video.")

        return image_paths, image_poses, image_times

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = img.resize(self.img_wh, Image.LANCZOS)

        img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)
        return img, self.image_poses[index], self.image_times[index]

    def load_pose(self, index):
        return self.image_poses[index]

