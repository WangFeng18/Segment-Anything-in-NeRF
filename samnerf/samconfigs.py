from __future__ import annotations

from typing import Dict

import tyro
from pathlib import Path
from nerfacc import ContractionType

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.datamanagers.depth_datamanager import DepthDataManagerConfig
from nerfstudio.data.datamanagers.semantic_datamanager import SemanticDataManagerConfig
from nerfstudio.data.datamanagers.variable_res_datamanager import (
    VariableResDataManagerConfig,
)
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.dycheck_dataparser import DycheckDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import (
    InstantNGPDataParserConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import (
    PhototourismDataParserConfig,
)
from nerfstudio.data.dataparsers.sitcoms3d_dataparser import Sitcoms3DDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.models.depth_nerfacto import DepthNerfactoModelConfig
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.mipnerf import MipNerfModel
from nerfstudio.models.nerfacto import NerfactoModelConfig
from nerfstudio.models.nerfplayer_nerfacto import NerfplayerNerfactoModelConfig
from nerfstudio.models.nerfplayer_ngp import NerfplayerNGPModelConfig
from nerfstudio.models.semantic_nerfw import SemanticNerfWModelConfig
from nerfstudio.models.tensorf import TensoRFModelConfig
from nerfstudio.models.vanilla_nerf import NeRFModel, VanillaModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.pipelines.dynamic_batch import DynamicBatchPipelineConfig
from nerfstudio.plugins.registry import discover_methods
from samnerf.datamanager import SAMDataManagerConfig
from samnerf.sam_model import SAMModelConfig
from samnerf.sam_pipeline import SamPipelineConfig

method_configs: Dict[str, TrainerConfig] = {}
descriptions = {}

method_configs["samnerf_no_distill"] = TrainerConfig(
    method_name="samnberf_no_distill",
    steps_per_eval_batch=50000,
    steps_per_eval_image=10000000,
    steps_per_save=2000,
    max_num_iterations=30000,
    mixed_precision=True,
    pipeline=SamPipelineConfig(
        datamanager=SAMDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(
                # downscale_factor
                scale_factor=1.0,
                train_val_json_split=True,
                data=Path("/data/mipnerf360/room/"),
            ),
            use_dino_feature=False,
            train_num_rays_per_batch=4096 * 4,
            eval_num_rays_per_batch=4096 * 4,
            camera_optimizer=CameraOptimizerConfig(
                mode="off",
            ),
            patch_size=1,
            distill_sam=False,
        ),
        model=SAMModelConfig(
            distill_sam=False,
            kernel_size=3,
            use_clipseg_feature=False,
            eval_num_rays_per_chunk=1 << 15,
            use_appearance_embedding=False,
            hidden_layers=1,
            patch_size=1,
            sam_loss_weight=1.0,
            num_proposal_iterations=1,
            num_proposal_samples_per_ray=(64,),
            num_sam_samples=3,
            num_nerf_samples_per_ray=32,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0005, max_steps=30000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0005, max_steps=30000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer+wandb",
)

method_configs["samnerf_distill"] = TrainerConfig(
    method_name="samnerf_distill",
    steps_per_eval_batch=5000000,
    steps_per_eval_image=10000000,
    steps_per_save=2000,
    max_num_iterations=10000,
    mixed_precision=True,
    pipeline=SamPipelineConfig(
        datamanager=SAMDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(
                # downscale_factor
                scale_factor=1.0,
                train_val_json_split=True,
                data=Path("/data/machine/data/mipnerf360/room/"),
            ),
            use_dino_feature=False,
            train_num_rays_per_batch=4096 * 4,
            eval_num_rays_per_batch=4096 * 4,
            camera_optimizer=CameraOptimizerConfig(
                mode="off",
            ),
            patch_size=4,
            distill_sam=True,
            use_clipseg_feature=True,
        ),
        model=SAMModelConfig(
            distill_sam=True,
            kernel_size=3,
            use_clipseg_feature=True,
            eval_num_rays_per_chunk=1 << 15,
            use_appearance_embedding=False,
            hidden_layers=1,
            patch_size=4,
            sam_loss_weight=1.0,
            num_proposal_iterations=1,
            num_proposal_samples_per_ray=(64,),
            num_sam_samples=16,
            num_nerf_samples_per_ray=32,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0005, max_steps=10000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0005, max_steps=10000),
        },
        "conv": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),
        },
        "sam_field": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=10000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer+wandb",
)

for key in method_configs.keys():
    method_configs[key].wandb_name = key

external_methods, external_descriptions = discover_methods()
method_configs.update(external_methods)
descriptions.update(external_descriptions)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
