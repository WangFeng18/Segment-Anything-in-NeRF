import json
import os
import typing
from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
import torch
import glob
from einops import rearrange


class FeatureDataloader(ABC):
    def __init__(
        self,
        device: torch.device,
        npy_paths,
        image_shape: typing.Tuple[int, int],
        patch_size: int = 1,
    ):
        print("====================Image Shape====================")
        print(image_shape)
        print("====================Image Shape====================")
        self.device = device
        self.npy_path = npy_paths
        self.image_shape = image_shape
        self.patch_size = patch_size

        print(npy_paths)
        if npy_paths[0].endswith(".npy"):
            features = []
            for npy in npy_paths:
                feature = np.load(npy)
                feature = rearrange(feature, "c h w -> h w c")
                features.append(feature)
            features = np.stack(features, axis=0)  # n h w c
            self.features = torch.from_numpy(features).to(device)
        else:
            assert npy_paths[0].endswith(".pt")
            features = []
            for npy in npy_paths:
                feature = torch.load(npy)
                features.append(feature)
            features = torch.stack(features, dim=0)
            self.features = features.to(device)

    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y)
        img_scale = (
            self.features.shape[1] / self.image_shape[0],
            self.features.shape[2] / self.image_shape[1],
        )
        x_ind, y_ind = (img_points[:, 1] * img_scale[0]).long(), (img_points[:, 2] * img_scale[1]).long()
        return (self.features[img_points[:, 0].long(), x_ind, y_ind]).to(self.device)
