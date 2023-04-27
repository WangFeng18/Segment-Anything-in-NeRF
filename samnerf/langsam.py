from samnerf.clipseg.models.clipseg import CLIPDensePredT
from samnerf.segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from einops import rearrange
import gradio as gr


class LanguageSAM:
    def __init__(self, device="cuda", sam_model="vit_b"):
        clipseg_model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64)
        clipseg_model.eval()
        clipseg_model.load_state_dict(
            torch.load("samnerf/clipseg/weights/rd64-uni.pth", map_location=torch.device("cpu")), strict=False
        )
        clipseg_model.to(device)
        self.clipseg_model = clipseg_model

        sam_checkpoint = {
            "vit_b": "/data/machine/nerfstudio/segment-anything/sam_vit_b_01ec64.pth",
            "vit_h": "/data/machine/nerfstudio/segment-anything/sam_vit_h_4b8939.pth",
        }
        sam = sam_model_registry[sam_model](checkpoint=sam_checkpoint[sam_model])
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.image = None
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize((512, 512)),
            ]
        )
        self.tensor_transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize((512, 512)),
            ]
        )
        self.device = device

    def show_mask_tensor(self, mask, random_color=False):
        if random_color:
            color = torch.cat([torch.rand(3, device=mask.device), torch.tensor([0.6], device=mask.device)], dim=0)
        else:
            color = torch.tensor([30 / 255, 144 / 255, 255 / 255, 0.6], device=mask.device)
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        return mask_image

    def get_masked_image(self, mask, image):
        mask_img = self.show_mask_tensor(mask, random_color=True)
        # mask_img = torch.from_numpy(mask_img).to(image.device)
        # print(f"from numpy time: {time.time() - _time1}")
        mask_p_img = mask_img[..., :3] * mask_img[..., 3:] + image * (1 - mask_img[..., 3:])
        return mask_p_img

    def generate_masked_img(self, points, labels):
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False,
            return_torch=True,
        )
        masks = masks.squeeze(dim=0)
        if isinstance(self.image, np.ndarray):
            mask_img = self.get_masked_image(masks[0], torch.from_numpy(self.image).to(self.device) / 255.0)
        else:
            mask_img = self.get_masked_image(masks[0], self.image.to(self.device) / 255.0)
        return mask_img

    def set_image(self, img_path="/data/machine/nerfstudio/test.jpg"):
        if isinstance(img_path, str):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = img_path
            # if self.image == image:
            #     print("same image")
            #     return

        self.image = image
        if isinstance(self.image, np.ndarray):
            image_pil = Image.fromarray(self.image)
            self.image_clipseg = self.transform(image_pil).unsqueeze(0)
        else:
            self.image_clipseg = self.image.permute(2, 0, 1)
            self.image_clipseg = self.tensor_transform(self.image_clipseg).unsqueeze(0)
            self.image = (self.image * 255).byte()
        self.predictor.set_image(self.image)

    def get_mask_by_prompt(
        self, prompt=["a man is cooking"], point_num=5, threshold=0.5, points=None, output_format="numpy"
    ):
        feat = self.clipseg_model(self.image_clipseg.to(self.device), conditional=prompt)[0][0, 0].sigmoid()
        feat = rearrange(feat, "(h p1) (w p2) -> h w (p1 p2)", p1=16, p2=16).mean(dim=-1)
        # inds = torch.nonzero(feat == feat.max())
        inds = feat.flatten().topk(k=point_num)[1]
        inds = torch.stack([torch.div(inds, feat.shape[1], rounding_mode="trunc"), inds % feat.shape[1]], dim=-1)
        inds = inds[feat[inds[..., 0], inds[..., 1]] > threshold]

        inds[..., 0] = inds[..., 0] / feat.shape[0] * self.image.shape[0]
        inds[..., 1] = inds[..., 1] / feat.shape[1] * self.image.shape[1]
        inds = inds.cpu().numpy()[..., ::-1]

        # points should be w,h order
        if points is not None:
            inds = np.concatenate([inds, points], axis=0)

        inp_labels = np.ones(inds.shape[0])
        if output_format == "numpy":
            masked_image = self.generate_masked_img(inds, inp_labels).cpu().numpy()
        else:
            masked_image = self.generate_masked_img(inds, inp_labels)
        return masked_image

    def set_and_segment(self, image, pmt, pts=5, thres=0.5, points=None, output_format="numpy"):
        self.set_image(image)
        return self.get_mask_by_prompt(
            prompt=[pmt], point_num=pts, threshold=thres, points=points, output_format=output_format
        )

    def launch_gradio(self):
        if not hasattr(self, "demo"):
            print("creating gradio interface")
            self.demo = gr.Interface(fn=self.set_and_segment, inputs=[gr.Image(type="numpy"), "text"], outputs="image")
        self.demo.launch(share=True)

    def close_gradio(self):
        self.demo.close()


# LangSAM = LanguageSAM()
