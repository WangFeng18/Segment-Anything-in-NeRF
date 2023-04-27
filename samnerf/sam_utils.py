import numpy as np
import torch
import math


def get_feature_size(h, w):
    if h < w:
        _h = int(math.ceil((h / w) * 64))
        _w = 64
    elif h > w:
        _w = int(math.ceil((w / h) * 64))
        _h = 64
    return _h, _w


def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image


def get_masked_image(mask, image):
    mask_img = show_mask(mask, random_color=True)
    mask_img = torch.from_numpy(mask_img).to(image.device)
    mask_p_img = mask_img[..., :3] * mask_img[..., 3:] + image * (1 - mask_img[..., 3:])
    return mask_p_img


def generate_masked_img(predictor, points, labels, image):
    masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=False,
    )
    mask_img = get_masked_image(masks[0], image)
    return mask_img
