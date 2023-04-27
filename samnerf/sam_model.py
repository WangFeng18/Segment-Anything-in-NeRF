from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import cv2
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
from torchvision.utils import draw_keypoints

EPS = 1e-4
TOR = 1e-2

def save_img(tensor, filename):
    # tensor: [h, w, c]
    # print(tensor.shape)
    img = tensor.detach().cpu().numpy()
    img = (img * 255.).astype(np.uint8)
    cv2.imwrite(filename, img)

SIZE = 4 # for h = 840

def show_prompts(prompts, depth, intrin, c2w, img, prompts_3d, h=None):
    if len(prompts) == 0:
        return img 
    
    fx = intrin[0, 0]
    fy = intrin[1, 1]
    cx = intrin[0, 2]
    cy = intrin[1, 2]

    prompts = torch.from_numpy(prompts).to(torch.long)
    # print(prompts)

    coords = prompts - torch.tensor([[cx, cy]]).to(prompts.device)
    coords /= torch.tensor([[fx, -fy]])

    padding = -torch.ones_like(coords[..., :1])
    coords = torch.cat([coords, padding], dim=-1)[..., None, :]

    rotation = c2w[:3, :3]
    # print(rotation.shape)
    # print(coords.shape)
    rays_d = torch.sum(coords * rotation, dim=-1)
    rays_d /= rays_d.norm(dim=-1, keepdim=True)
    rays_o = c2w[:3, 3].unsqueeze(0).repeat(coords.shape[0], 1)
    ts = ((prompts_3d - rays_o) / rays_d).mean(dim=-1)
    # print(((prompts_3d - rays_o) / rays_d))
    # print(ts.shape)

    visible = ts < (depth[prompts[..., 1], prompts[..., 0]].to(ts.device).squeeze() + EPS)
    # print(visible)
    # print(prompts)
    prompts = prompts[visible]

    r = SIZE
    if h is not None:
        r = int(r * h / 840)
        r = max(r, 1)
    img = (255 * img).to(torch.uint8).moveaxis(-1, 0)
    img = draw_keypoints(img, prompts[None, ...], radius=r, colors="red").to(torch.float32) / 255.0
    img = img.moveaxis(0, -1)

    return img


def project(intrin, c2w, points):
    fx = intrin[0, 0]
    fy = intrin[1, 1]
    cx = intrin[0, 2]
    cy = intrin[1, 2]

    if c2w.shape[0] == 3:
        padding = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).to(c2w)
        c2w = torch.cat([c2w, padding], dim=0)
        # [4, 4]

    if points.shape[-1] == 3:
        padding = torch.tensor([[1.0]] * points.shape[0]).to(points)
        points = torch.cat([points, padding], dim=-1)
        # [n, 4]

    w2c = torch.inverse(c2w)[:3]

    img_coords = torch.einsum("ij,bj->bi", w2c, points)
    img_coords = -img_coords / img_coords[..., -1:]
    img_coords = img_coords[..., :2]
    # make img_z == -1

    img_coords[..., 0] *= fx
    img_coords[..., 0] += cx
    img_coords[..., 1] *= -fy
    img_coords[..., 1] += cy

    return img_coords.to(torch.int32)


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
    dino_loss_weight: float = 1.0
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

    sam_checkpoint: str = "/data/machine/nerfstudio/segment-anything/sam_vit_h_4b8939.pth"


class SAMModel(NerfactoModel):
    config: SAMModelConfig

    def populate_modules(self):
        super().populate_modules()

        use_sam = True
        self.sam_capable = True
        self.text_prompt_capable = False
        self.prompts = None

        self.renderer_mean = MeanRenderer()

        self.sam_field = SAMField(
            self.config.hashgrid_layers,
            self.config.hashgrid_sizes,
            self.config.hashgrid_resolutions,
            hidden_layers=self.config.hidden_layers,
            use_dino_features=self.config.use_dino_feature,
        )

        self.conv_head = nn.Sequential(
            nn.Conv2d(256, 256, self.config.kernel_size, stride=1, padding=(self.config.kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Conv2d(256, 256, self.config.kernel_size, stride=1, padding=(self.config.kernel_size - 1) // 2),
        )

        sam_checkpoint = self.config.sam_checkpoint
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device="cuda")
        self.predictor = SamPredictor(sam)

    def get_outputs(self, ray_bundle: RayBundle, get_rgbsigma=True, get_feature=True):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples_list.append(ray_samples)

        nerfacto_field_outputs, outputs, weights = self._get_outputs_nerfacto(ray_samples)
        weights_list.append(weights)

        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        sam_weights, best_ids = torch.topk(weights, self.config.num_sam_samples, dim=-2, sorted=False)
        T = 10
        sam_weights = sam_weights**T
        # print(sam_weights.shape)
        sam_weights = sam_weights / sam_weights.sum(dim=-2, keepdim=True)
        # sam_weights = torch.nn.functional.softmax(sam_weights * T, dim=-2)

        def gather_fn(tens):
            return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

        dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
        sam_samples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)

        outputs["sam_depth"] = self.renderer_depth(weights=sam_weights, ray_samples=ray_samples)

        if get_feature:
            sam_weights, best_ids = torch.topk(weights, self.config.num_sam_samples, dim=-2, sorted=False)
            T = 10
            sam_weights = sam_weights**T
            # print(sam_weights.shape)
            sam_weights = sam_weights / sam_weights.sum(dim=-2, keepdim=True)
            # sam_weights = torch.nn.functional.softmax(sam_weights * T, dim=-2)

            def gather_fn(tens):
                return torch.gather(tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1]))

            dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)
            sam_samples = ray_samples._apply_fn_to_fields(gather_fn, dataclass_fn)

            sam_field_outputs = self.sam_field.get_outputs(sam_samples)
            if self.config.patch_size > 1:
                feat_out = self.renderer_mean(embeds=sam_field_outputs["sam"], weights=sam_weights.detach())
                feat_out = feat_out.reshape(-1, self.config.patch_size, self.config.patch_size, feat_out.shape[-1])
                feat_out = feat_out.permute(0, 3, 1, 2)
                feat_out = self.conv_head(feat_out).mean(dim=[2, 3])
                outputs["sam"] = feat_out
                # breakpoint()
                # outputs["sam_depth"] = sam_samples.frustums.get_positions().mean(dim=-1, keepdim=True)
                # outputs["sam_depth"] = self.renderer_depth(weights=sam_weights, ray_samples=sam_samples)
                # print(outputs["sam_depth"].shape)
                # print(outputs["rgb"].shape)
            else:
                outputs["sam"] = self.renderer_mean(embeds=sam_field_outputs["sam"], weights=sam_weights.detach())
                # outputs["sam_depth"] = sam_samples.frustums.get_positions().mean(dim=-1, keepdim=True)
            if self.config.use_dino_feature:
                outputs["dino"] = self.renderer_mean(embeds=sam_field_outputs["dino"], weights=sam_weights.detach())

        return outputs

    def _get_outputs_nerfacto(self, ray_samples: RaySamples):
        field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
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
        if self.training:
            unreduced_sam = torch.nn.functional.mse_loss(outputs["sam"], batch["sam"], reduction="none")
            loss_dict["sam_loss"] = self.config.sam_loss_weight * unreduced_sam.mean(dim=-1).nanmean()

            if self.config.use_dino_feature:
                unreduced_dino = torch.nn.functional.mse_loss(outputs["dino"], batch["dino"], reduction="none")
                loss_dict["dino_loss"] = self.config.dino_loss_weight * unreduced_dino.mean(dim=-1).nanmean()
        return loss_dict

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["sam_field"] = list(self.sam_field.parameters())
        param_groups["conv"] = list(self.conv_head.parameters())
        return param_groups

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle, points=None, intrin=None, c2w=None):
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        _t1 = time.time()
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle, get_feature=False)
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        print(f"forwarding took {time.time() - _t1} seconds")
        sz = camera_ray_bundle.shape
        feature_h, feature_w = get_feature_size(image_height, image_width)
        print("=" * 20)
        print(feature_h, feature_w)
        print(sz)
        print("=" * 20)
        if sz[0] >= feature_h * self.config.patch_size and sz[1] >= feature_w * self.config.patch_size or True:
            _t1 = time.time()
            # Note the sz[0] and sz[1] should be closed to the multiples of 48*patch_size and 64*patch_size, or the mask may be misalgned

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
                outputs = self.forward(ray_bundle=ray_bundle, get_feature=True)
                outputs_lists["sam"].append(outputs["sam"])
            print(f"forwarding feature took {time.time() - _t1} seconds")
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            if output_name == "sam":
                outputs[output_name] = torch.cat(outputs_list).view(feature_h, feature_w, -1)
            else:
                outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore

        # print(outputs.keys())
        if "sam" in outputs:
            # print("here")
            self.predictor.set_feature(outputs["sam"].permute(2, 0, 1), original_image_size=(image_height, image_width))
            if points is None:
                self.prompts = None
                input_point = np.array([[int(500 / 1024 * image_width), int(375 / 768 * image_height)]])
                input_label = np.array([1])
                outputs["masked_rgb"] = generate_masked_img(self.predictor, input_point, input_label, outputs["rgb"])
            else:
                if points is not None:
                    print("points:", points)
                    assert c2w is not None
                    assert intrin is not None

                    fx = intrin[0, 0]
                    fy = intrin[1, 1]
                    cx = intrin[0, 2]
                    cy = intrin[1, 2]

                    if len(points) > 0:
                        breakpoint()
                        perform = False
                        if self.prompts is not None:
                            if len(points) > self.prompts.size(0):
                                points = points[self.prompts.size(0):]
                                perform = True
                        else:
                            perform = True

                        if perform:
                            points = torch.from_numpy(points).to(c2w.device).to(torch.long)
                            h, w = outputs["rgb"].shape[:2]
                            # img_x, img_y = (points[..., 0] * w).to(torch.long), (points[..., 1] * h).to(torch.long)
                            img_x = points[..., 0]
                            img_y = points[..., 1]
                            t = outputs["depth"][img_y, img_x] - TOR
                            # print(t)
                            x = (points[..., 0] - cx) / fx
                            y = -(points[..., 1] - cy) / fy
                            padding = -torch.ones_like(x)

                            coords = torch.stack([x, y, padding], dim=-1)[..., None, :]
                            rotation = c2w[:3, :3].unsqueeze(0).repeat(coords.shape[0], 1, 1)

                            # TODO: check here
                            direction = torch.sum(coords * rotation, dim=-1)
                            direction /= torch.norm(direction, dim=-1, keepdim=True)
                            new_point = c2w[:3, 3] + t.to(direction) * direction
                            if self.prompts is None:
                                self.prompts = new_point
                            else:
                                self.prompts = torch.cat([self.prompts, new_point], dim=0)
                    else:
                        self.prompts = None

                if self.prompts is not None:
                    h, w = outputs["rgb"].shape[:2]
                    prompts = project(intrin, c2w, self.prompts)
                    print("prompts", prompts)

                    bounds = torch.tensor([[w, h]]).to(prompts)
                    legal = torch.logical_and(prompts >= 0, prompts < bounds).all(dim=-1)

                    prompts = prompts[legal]
                    

                    # print(prompts)
                    # print(w)
                    # print(h)
                    # exit(0)
                    prompts = prompts.cpu().clone().numpy()

                    input_points = prompts
                    input_label = np.array(
                        [
                            1,
                        ]
                        * len(prompts)
                    )
                    outputs["masked_rgb"] = generate_masked_img(
                        self.predictor, input_points, input_label, outputs["rgb"]
                    )
                    outputs["masked_rgb"] = show_prompts(
                        prompts, outputs["depth"], intrin, c2w, outputs["masked_rgb"], self.prompts[legal], h
                    )
                    # print("here")
                    # save_img(outputs["masked_rgb"], "test.jpg")

                else:
                    outputs["masked_rgb"] = outputs["rgb"]
        else:
            outputs["masked_rgb"] = outputs["rgb"]

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
        sam_feature = outputs["sam"]
        # h, w, c = sam_feature.shape
        # if sam_feature.shape[0] % 16 != 0:
        #     sam_feature = torch.cat(
        #         [sam_feature, torch.zeros(16 - sam_feature.shape[0] % 16, *sam_feature.shape[1:])], dim=0
        #     )
        # if sam_feature.shape[1] % 16 != 0:
        #     sam_feature = torch.cat(
        #         [sam_feature, torch.zeros(sam_feature.shape[0], 16 - sam_feature.shape[1] % 16, c)], dim=0
        #     )

        # sam_feature = (
        #     rearrange(sam_feature, "(h p1) (w p2) c -> c h w (p1 p2)", p1=16, p2=16).mean(dim=-1).cpu().numpy()
        # )
        sam_feature = sam_feature.cpu().numpy()
        np.save("res.npy", sam_feature)

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

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
