import numpy as np
import glob
import torch
import argparse
import math
import os

# import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor


def init():
    sam_checkpoint = "samnerf/segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def get_embeddings(image_path, predictor):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    predictor.set_image(image)
    feature = predictor.features
    if h < w:
        H = int(math.ceil((h / w) * feature.shape[-1]))
        feature = feature[:, :, :H, :]
    elif h > w:
        W = int(math.ceil((w / h) * feature.shape[-1]))
        feature = feature[:, :, :, :W]
    return feature


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/data/machine/data/mipnerf360/room/images/")
    parser.add_argument("--save_path", type=str, default="/data/machine/data/mipnerf360/room/sam_features/")

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    predictor = init()
    print("Predictor initialized")
    img_paths = []
    for img_path in glob.glob(os.path.join(args.image_path, "*")):
        if img_path.endswith(".png") or img_path.endswith(".jpg") or img_path.endswith(".JPG"):
            img_paths.append(img_path)

    img_paths = sorted(img_paths)
    print(img_paths)
    for img_path in img_paths:
        feature = get_embeddings(img_path, predictor)
        base_name = os.path.basename(img_path).split(".")[0] + ".npy"
        save_path = os.path.join(args.save_path, base_name)
        np.save(save_path, feature.squeeze().cpu().numpy())
        print(f"saving {img_path} at {save_path}")
