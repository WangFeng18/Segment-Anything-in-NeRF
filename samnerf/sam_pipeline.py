from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline
from nerfstudio.utils import profiler
from dataclasses import dataclass, field
import sys
import numpy as np
import os
from typing import Any, Dict, List, Mapping, Optional, Type, Union, cast
import cv2
import imageio
import torch
from tqdm import tqdm
from nerfstudio.cameras.cameras import Cameras, CameraType

first_time = True
debug = False

def save_img(tensor, filename):
    # tensor: [h, w, c]
    # print(tensor.shape)
    img = tensor.detach().cpu().numpy()
    img = (img * 255.).astype(np.uint8)
    cv2.imwrite(filename, img)

def get_rz(c2w):
    trans = c2w[:3, 3].cpu().numpy()
    return np.sqrt(trans[0] * trans[0] + trans[1] * trans[1]), trans[2]

def get_c2w_from_yz(y, z, tra):
    x = np.cross(y, z)
    rot = np.array([x, y, z]).T
    return np.concatenate([rot, tra[..., None]], axis=-1)

def make_z(t):
    return np.array([np.cos(t), np.sin(t), 0])

def get_c2w_t(c2w, t):
    r, zz = get_rz(c2w)
    z = make_z(t)
    tra = np.array([r * np.cos(t), r * np.sin(t), zz])
    return get_c2w_from_yz(np.array([0.,0.,1.]), z, tra)

@dataclass
class SamPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: SamPipeline)
    """target class to instantiate"""


class SamPipeline(VanillaPipeline):
    def __init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        # TODO remove it
        if batch["image"].shape[1] == 1297:
            batch["image"] = batch["image"][:, :-1, ...]
        print(camera_ray_bundle.shape)
        # print(batch.keys())
        # print(batch["image_idx"].shape)
        print(batch["image"].shape)

        # global first_time
        # if first_time:
        #     points = np.array([[100, 100], [200, 200]])
        #     first_time = False
        # else:
        #     points = np.array([[100, 100], [200, 200], [300, 300]])
        if debug:
            if step < 300:
                points = np.array([[648, 400]])
                intrin = np.zeros([3, 3])
                cam = self.datamanager.eval_dataset.cameras
                intrin[0, 0] = cam.fx[0, 0].item()
                intrin[1, 1] = cam.fy[0, 0].item()
                intrin[0, 2] = cam.cx[0, 0].item()
                intrin[1, 2] = cam.cy[0, 0].item()

                c2w = self.datamanager.eval_dataset.cameras.camera_to_worlds[image_idx]
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, points=points, intrin=intrin, c2w=c2w, fast=False)
                save_img(outputs["masked_rgb"], "test.png")
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
                assert "image_idx" not in metrics_dict
                metrics_dict["image_idx"] = image_idx
                assert "num_rays" not in metrics_dict
                metrics_dict["num_rays"] = len(camera_ray_bundle)
                self.train()
                return metrics_dict, {}
            
            print("\n\ndebug\n\n")
            points = np.array([[648, 400]])
            intrin = np.zeros([3, 3])
            cam = self.datamanager.eval_dataset.cameras
            intrin[0, 0] = cam.fx[0, 0].item()
            intrin[1, 1] = cam.fy[0, 0].item()
            intrin[0, 2] = cam.cx[0, 0].item()
            intrin[1, 2] = cam.cy[0, 0].item()

            c2w = self.datamanager.eval_dataset.cameras.camera_to_worlds[image_idx]

            n_t = 30
            frames = torch.zeros([30, 840, 1296, 3], dtype=torch.uint8)
            ts = np.linspace(0, np.pi, 30)

            camera_type = CameraType.PERSPECTIVE
            for i, t in tqdm(enumerate(ts)):
                print(t)
                c2wt = torch.from_numpy(get_c2w_t(c2w, t)).to(torch.float32)
                print(c2wt)

                camera = Cameras(
                    fx=intrin[0, 0],
                    fy=intrin[1, 1],
                    cx=intrin[0, 2],
                    cy=intrin[1, 2],
                    camera_type=camera_type,
                    camera_to_worlds=c2wt[None, ...],
                ).to("cuda")

                camera_ray_bundle = camera.generate_rays(camera_indices=0)
                
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, points, intrin, c2wt, fast=False)
                frames[i] = (outputs["masked_rgb"] * 255.).cpu().to(torch.uint8)

            
            imageio.mimwrite("figs/demo.mp4", frames, fps=10, quality=10)
            exit(0)
        
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict
