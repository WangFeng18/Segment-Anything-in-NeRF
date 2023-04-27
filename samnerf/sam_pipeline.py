from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline
from nerfstudio.utils import profiler
from dataclasses import dataclass, field
import sys
import numpy as np
import os
from typing import Any, Dict, List, Mapping, Optional, Type, Union, cast


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
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, fast=False)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict
