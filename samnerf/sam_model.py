from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type


import numpy as np
import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.ray_samplers import PDFSampler
from nerfstudio.model_components.renderers import DepthRenderer
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from torch.nn import Parameter
from torchtyping import TensorType
from samnerf.segment_anything import sam_model_registry, SamPredictor

from samnerf.sam_field import SAMField
from samnerf.sam_utils import generate_masked_img, get_feature_size
from samnerf.clipseg.models.clipseg import CLIPDensePredT
from samnerf.langsam import LanguageSAM
import torch.nn as nn
import time

from nerfstudio.utils import colormaps
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    RGBRenderer,
)
from einops import rearrange


class MeanRenderer(nn.Module):
    """Calculate average of embeddings along ray."""

    @classmethod
    def forward(
        cls,
        embeds: TensorType["bs":..., "num_samples", "num_classes"],
        weights: TensorType["bs":..., "num_samples", 1],
    ) -> TensorType["bs":..., "num_classes"]:
        """Calculate semantics along the ray."""
        output = torch.sum(weights * embeds, dim=-2)
        return output


@dataclass
class SAMModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: SAMModel)
    sam_loss_weight: float = 1.0
    use_dino_feature: bool = False
    use_clipseg_feature: bool = False
    dino_loss_weight: float = 1.0
    clipseg_loss_weight: float = 1.0
    n_scales: int = 30
    max_scale: float = 1.5
    """maximum scale used to compute relevancy with"""
    num_sam_samples: int = 24
    hidden_layers: int = 2
    hashgrid_layers: Tuple[int] = (12, 12)
    hashgrid_resolutions: Tuple[Tuple[int]] = ((16, 128), (128, 512))
    hashgrid_sizes: Tuple[int] = (19, 19)
    patch_size: int = 1
    kernel_size: int = 3

    distill_sam: bool = False
    sam_checkpoint: str = "/data/machine/nerfstudio/segment-anything/sam_vit_h_4b8939.pth"
    sharpening_temperature: float = 10.0


class Residual(nn.Module):
    def __init__(self, conv1, bn1, conv2, bn2):
        super().__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv2
        self.bn2 = bn2

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(x + out)


class SAMModel(NerfactoModel):
    config: SAMModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.renderer_mean = MeanRenderer()

        if self.config.distill_sam:
            self.sam_field = SAMField(
                self.config.hashgrid_layers,
                self.config.hashgrid_sizes,
                self.config.hashgrid_resolutions,
                hidden_layers=self.config.hidden_layers,
                use_dino_features=self.config.use_dino_feature,
                use_clipseg_features=self.config.use_clipseg_feature,
            )

            self.conv_head = nn.Sequential(
                nn.Conv2d(256, 256, self.config.kernel_size, stride=1, padding=(self.config.kernel_size - 1) // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, self.config.kernel_size, stride=1, padding=(self.config.kernel_size - 1) // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, self.config.kernel_size, stride=1, padding=(self.config.kernel_size - 1) // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, self.config.kernel_size, stride=1, padding=(self.config.kernel_size - 1) // 2),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(256, 256, self.config.kernel_size, stride=1, padding=(self.config.kernel_size - 1) // 2),
            )

            sam_checkpoint = self.config.sam_checkpoint
            model_type = "vit_h"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device="cuda")
            self.predictor = SamPredictor(sam)
            if self.config.use_clipseg_feature:
                model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64)
                model.eval()
                model.load_state_dict(
                    torch.load("samnerf/clipseg/weights/rd64-uni.pth", map_location=torch.device("cpu")), strict=False
                )
                model.cuda()
                self.clipseg = model
        else:
            self.lang_sam = LanguageSAM()

    def get_outputs(self, ray_bundle: RayBundle, get_rgbsigma=True, get_feature=["sam", "dino", "clipseg"], fast=False):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples_list.append(ray_samples)

        nerfacto_field_outputs, outputs, weights = self._get_outputs_nerfacto(ray_samples, fast=fast)
        weights_list.append(weights)

        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if not fast:
            for i in range(self.config.num_proposal_iterations):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(
                    weights=weights_list[i], ray_samples=ray_samples_list[i]
                )

        if self.config.distill_sam and len(get_feature) > 0:
            sam_weights, best_ids = torch.topk(weights, self.config.num_sam_samples, dim=-2, sorted=False)

            sam_weights = sam_weights**self.config.sharpening_temperature
            # print(sam_weights.shape)
            sam_weights = sam_weights / sam_weights.sum(dim=-2, keepdim=True)
            # sam_weights = torch.nn.functional.softmax(sam_weights * T, dim=-2)

            def gather_fn(tens):
                return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

            dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
            sam_samples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)

            sam_field_outputs = None
            if "sam" in get_feature:
                sam_field_outputs = self.sam_field.get_outputs(sam_samples, get_feautre=get_feature)
                if self.config.patch_size > 1:
                    feat_out = self.renderer_mean(embeds=sam_field_outputs["sam"], weights=sam_weights.detach())
                    feat_out = feat_out.reshape(-1, self.config.patch_size, self.config.patch_size, feat_out.shape[-1])
                    feat_out = feat_out.permute(0, 3, 1, 2)
                    feat_out = self.conv_head(feat_out).mean(dim=[2, 3])
                    outputs["sam"] = feat_out
                else:
                    outputs["sam"] = self.renderer_mean(embeds=sam_field_outputs["sam"], weights=sam_weights.detach())
            if "dino" in get_feature and self.config.use_dino_feature:
                if sam_field_outputs is None:
                    sam_field_outputs = self.sam_field.get_outputs(sam_samples, get_feautre=get_feature)
                outputs["dino"] = self.renderer_mean(embeds=sam_field_outputs["dino"], weights=sam_weights.detach())
            if "clipseg" in get_feature and self.config.use_clipseg_feature:
                if sam_field_outputs is None:
                    sam_field_outputs = self.sam_field.get_outputs(sam_samples, get_feautre=get_feature)
                outputs["clipseg"] = self.renderer_mean(
                    embeds=sam_field_outputs["clipseg"], weights=sam_weights.detach()
                )

        return outputs

    def _get_outputs_nerfacto(self, ray_samples: RaySamples, fast=False):
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        if not fast:
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
            accumulation = self.renderer_accumulation(weights=weights)

            outputs = {
                "rgb": rgb,
                "accumulation": accumulation,
                "depth": depth,
            }
        else:
            outputs = {
                "rgb": rgb,
            }

        return field_outputs, outputs, weights

    def forward(self, ray_bundle: RayBundle, **kwargs):
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, **kwargs)

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training and self.config.distill_sam:
            unreduced_sam = torch.nn.functional.mse_loss(outputs["sam"], batch["sam"], reduction="none")
            loss_dict["sam_loss"] = self.config.sam_loss_weight * unreduced_sam.mean(dim=-1).nanmean()

            if self.config.use_dino_feature:
                unreduced_dino = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"], reduction="none")
                loss_dict["dino_loss"] = self.config.dino_loss_weight * unreduced_dino.mean(dim=-1).nanmean()
            if self.config.use_clipseg_feature:
                unreduced_clipseg = torch.nn.functional.mse_loss(outputs["clipseg"], batch["clipseg"], reduction="none")
                loss_dict["clipseg_loss"] = self.config.clipseg_loss_weight * unreduced_clipseg.mean(dim=-1).nanmean()
        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        if self.config.distill_sam:
            param_groups["sam_field"] = list(self.sam_field.parameters())
            param_groups["conv"] = list(self.conv_head.parameters())
        return param_groups

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self,
        camera_ray_bundle: RayBundle,
        points=None,
        text_prompt=None,
        topk: int = 5,
        thresh: float = 0.5,
        fast=True,
    ):
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle, get_feature=[], fast=fast)
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        sz = camera_ray_bundle.shape

        # for sam distillation
        if self.config.distill_sam:
            feature_h, feature_w = get_feature_size(image_height, image_width)
            if sz[0] >= feature_h * self.config.patch_size and sz[1] >= feature_w * self.config.patch_size:
                h_indices = torch.linspace(0, sz[0] - 1, feature_h * self.config.patch_size, dtype=torch.long)
                w_indices = torch.linspace(0, sz[1] - 1, feature_w * self.config.patch_size, dtype=torch.long)
                hind, wind = torch.meshgrid(h_indices, w_indices)
                feature_camera_ray_bundle = camera_ray_bundle[hind.flatten(), wind.flatten()]
                feature_camera_ray_bundle = feature_camera_ray_bundle.reshape(
                    (feature_h, self.config.patch_size, feature_w, self.config.patch_size)
                )
                feature_camera_ray_bundle = feature_camera_ray_bundle._apply_fn_to_fields(lambda x: x.transpose(1, 2))

                num_rays = len(feature_camera_ray_bundle)
                for i in range(0, num_rays, num_rays_per_chunk):
                    start_idx = i
                    end_idx = i + num_rays_per_chunk
                    ray_bundle = feature_camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                    outputs = self.forward(ray_bundle=ray_bundle, get_feature=["sam"])
                    outputs_lists["sam"].append(outputs["sam"])

            if self.config.use_clipseg_feature:
                feature_h_clipseg, feature_w_clipseg = (
                    32,
                    32,
                )  # get_feature_size(image_height, image_width, largesize=32)
                h_indices = torch.linspace(0, sz[0] - 1, feature_h_clipseg, dtype=torch.long)
                w_indices = torch.linspace(0, sz[1] - 1, feature_w_clipseg, dtype=torch.long)
                hind, wind = torch.meshgrid(h_indices, w_indices)
                feature_camera_ray_bundle = camera_ray_bundle[hind.flatten(), wind.flatten()]
                feature_camera_ray_bundle = feature_camera_ray_bundle.reshape((feature_h_clipseg, feature_w_clipseg))
                num_rays = len(feature_camera_ray_bundle)

                for i in range(0, num_rays, num_rays_per_chunk):
                    start_idx = i
                    end_idx = i + num_rays_per_chunk
                    ray_bundle = feature_camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                    outputs = self.forward(ray_bundle=ray_bundle, get_feature=["clipseg"])
                    outputs_lists["clipseg"].append(outputs["clipseg"])

        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            if output_name == "sam":
                outputs[output_name] = torch.cat(outputs_list).view(feature_h, feature_w, -1)
            elif output_name == "clipseg":
                outputs[output_name] = torch.cat(outputs_list).view(feature_h_clipseg, feature_w_clipseg, -1)
            else:
                outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore

        # calculate SAM relevant
        if text_prompt is None:
            prompt = "a man is cooking"
        else:
            prompt = text_prompt
        input_points = points

        if self.config.distill_sam and "sam" in outputs:
            self.predictor.set_feature(outputs["sam"].permute(2, 0, 1), original_image_size=(image_height, image_width))
            if self.config.use_clipseg_feature:
                acts = []
                for _i in range(3):
                    _clipseg = outputs["clipseg"][..., 64 * _i : 64 * (_i + 1)].reshape(-1, 64).unsqueeze(dim=1)
                    _clipseg = torch.cat([_clipseg.mean(dim=0, keepdim=True), _clipseg], dim=0)
                    acts.append(_clipseg)
                inp_feature = {
                    "activations": acts,
                    "visual_q": None,
                    "transformed_image_size": (feature_h_clipseg, feature_w_clipseg),
                }
                clip_feature = self.clipseg(None, inp_feature=inp_feature, conditional=prompt)[0][0][0]
                clip_feature = clip_feature.sigmoid()
                outputs["clipseg_feature"] = clip_feature.unsqueeze(dim=-1)
                clip_feature = rearrange(clip_feature, "(h p1) (w p2) -> h w (p1 p2)", p1=16, p2=16).mean(dim=-1)
                _fh, _fw = clip_feature.shape
                amax = clip_feature.flatten().topk(k=1000)[1]

                amax_w = (amax % _fw).long()
                amax_h = (amax // _fw).long()

                _mask = clip_feature[amax_h, amax_w] > 0.7
                if _mask.any():
                    clip_points = torch.stack([amax_w, amax_h], dim=1)[_mask].cpu().numpy().astype(np.float32)
                    clip_points[..., 0] = clip_points[..., 0] / _fw * image_width
                    clip_points[..., 1] = clip_points[..., 1] / _fh * image_height
                    if input_points is not None:
                        input_points = np.concatenate([input_points, clip_points], axis=0)

            if input_points is not None:
                input_label = np.array(
                    [
                        1,
                    ]
                    * len(input_points)
                )
                outputs["masked_rgb"] = generate_masked_img(self.predictor, input_points, input_label, outputs["rgb"])
        elif not self.config.distill_sam:
            msk_img = self.lang_sam.set_and_segment(
                (outputs["rgb"].cpu().numpy() * 255).astype(np.uint8),
                prompt,
                pts=topk,
                thres=thresh,
                points=input_points,
                output_format="tensor",
            ).to(outputs["rgb"])
            outputs["masked_rgb"] = msk_img
            outputs["clipseg_feature"] = self.lang_sam.clipseg_feature
        return outputs

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        masked_rgb = outputs["masked_rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        if "sam" in outputs:
            sam_feature = outputs["sam"]
            sam_feature = sam_feature.cpu().numpy()
            np.save("res.npy", sam_feature)
        if "clipseg" in outputs:
            clipseg_feature = outputs["clipseg"].cpu().numpy()
            np.save("clipseg.npy", clipseg_feature)

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)
        # masked_rgb = torch.cat([masked_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
            "masked_rgb": masked_rgb,
        }
        if "clipseg_feature" in outputs:
            images_dict.update({"clipseg_feature": outputs["clipseg_feature"]})

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
