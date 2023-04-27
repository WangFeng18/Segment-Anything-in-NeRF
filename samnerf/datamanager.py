from __future__ import annotations

import os.path as osp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
import os
import torch
import yaml
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager
from rich.progress import Console
from samnerf.data.feature_loader import FeatureDataloader

CONSOLE = Console(width=120)


@dataclass
class SAMDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: SAMDataManager)
    sam_feature_path: str = "Path to the features extracted by SAM"
    use_dino_feature: bool = False
    use_clipseg_feature: bool = False
    dino_feature_path: str = "Path to the features extracted by DINO"


class SAMDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    config: SAMDataManagerConfig

    def __init__(
        self,
        config: SAMDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        self.includes_time = False

        sam_feature_filenames = [
            os.path.join(
                os.path.dirname(os.path.dirname(name)), "sam_features", os.path.basename(name).split(".")[0] + ".npy"
            )
            for name in self.train_dataparser_outputs.image_filenames
        ]
        self.sam_loader = FeatureDataloader(
            device,
            npy_paths=sam_feature_filenames,
            image_shape=list(self.train_dataset[0]["image"].shape[:2]),
            patch_size=self.config.patch_size,
        )

        if self.config.use_dino_feature:
            dino_feature_filenames = [
                os.path.join(
                    os.path.dirname(os.path.dirname(name)),
                    "dino_features",
                    os.path.basename(name).split(".")[0] + ".pt",
                )
                for name in self.train_dataparser_outputs.image_filenames
            ]
            self.dino_loader = FeatureDataloader(
                device,
                npy_paths=dino_feature_filenames,
                image_shape=list(self.train_dataset[0]["image"].shape[:2]),
                patch_size=1,
            )
        if self.config.use_clipseg_feature:
            clipseg_feature_filenames = [
                os.path.join(
                    os.path.dirname(os.path.dirname(name)),
                    "clipseg_features",
                    os.path.basename(name).split(".")[0] + ".pt",
                )
                for name in self.train_dataparser_outputs.image_filenames
            ]
            self.clipseg_loader = FeatureDataloader(
                device,
                npy_paths=clipseg_feature_filenames,
                image_shape=list(self.train_dataset[0]["image"].shape[:2]),
                patch_size=1,
                get_feature=lambda x: torch.cat(x["activations"], dim=-1)
                .squeeze()[1:, ...]
                .reshape(512 // 16, 512 // 16, -1),
            )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]

        # requires to calculate the center indices
        center_indices = ray_indices.reshape(-1, self.config.patch_size, self.config.patch_size, 3)[
            :, self.config.patch_size // 2, self.config.patch_size // 2, :
        ]
        ray_bundle = self.train_ray_generator(ray_indices)
        batch["sam"] = self.sam_loader(center_indices)
        if self.config.use_dino_feature:
            batch["dino"] = self.dino_loader(ray_indices)
        if self.config.use_clipseg_feature:
            batch["clipseg"] = self.clipseg_loader(ray_indices)
        # assume all cameras have the same focal length and image width
        return ray_bundle, batch
