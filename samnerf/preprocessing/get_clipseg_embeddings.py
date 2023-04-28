import numpy as np
import glob
import torch
import argparse
import math
import os

# import matplotlib.pyplot as plt
import cv2
from clipseg.models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms


def init():
    # load model
    model = CLIPDensePredT(version="ViT-B/16", reduce_dim=64)
    model.eval()

    # non-strict, because we only stored decoder weights (not CLIP weights)
    model.load_state_dict(torch.load("clipseg/weights/rd64-uni.pth", map_location=torch.device("cpu")), strict=False)
    model.cuda()
    return model


def get_embeddings(image_path, model, transformed_size=(512, 512)):
    input_image = Image.open(image_path)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(transformed_size),
        ]
    )
    img = transform(input_image).unsqueeze(0).cuda()

    with torch.no_grad():
        preds = model(img, None, ["placeholder"], return_clip_feature=True)

    return preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/data/machine/data/mipnerf360/room/images/")
    parser.add_argument("--save_path", type=str, default="/data/machine/data/mipnerf360/room/clipseg_features/")

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model = init()
    print("Predictor initialized")
    img_paths = []
    for img_path in glob.glob(os.path.join(args.image_path, "*")):
        if img_path.endswith(".png") or img_path.endswith(".jpg") or img_path.endswith(".JPG"):
            img_paths.append(img_path)

    img_paths = sorted(img_paths)
    print(img_paths)
    for img_path in img_paths:
        feature = get_embeddings(img_path, model, transformed_size=(512, 512))
        base_name = os.path.basename(img_path).split(".")[0] + ".pt"
        save_path = os.path.join(args.save_path, base_name)
        torch.save(feature, save_path)
        print(f"saving {img_path} at {save_path}")
