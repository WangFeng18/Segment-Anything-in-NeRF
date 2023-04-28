## SAM-NeRF: A Simple Baseline for Segmenting Anything in NeRF with Language Prompts.


### SAM with Language prompts:
https://user-images.githubusercontent.com/43294876/235158631-c2384188-81cb-40f9-a7e4-e635c9912c8c.mp4

### Promptable Segmentation for NeRF
https://user-images.githubusercontent.com/43294876/235242529-16e224c2-6219-432c-8537-f821408f0533.mp4

## Intro
This repo provides a support of language prompts for SAM, by combinning ClipSeg and SAM. We also provides a simple baseline for connecting SAM with NeRF, as well as a simple SAM feature distillation method. Specifically, we provide:

> - A simple extension of Segmenta Anything for incorporating language prompt, by combining ClipSeg with SAM, the inference of ClipSeg features are very fast, costing only 0.04s per image which brings neglectable overheads.
> - A simple implmentation of combining segment-anything with NeRF, providing the function of locking 3D objects for different views, and segmentation in 3D by language and point prompts.
>  - An implementation of distilling SAM and ClipSeg features into 3D fields, in this pipeline, the image encoders of SAM and ClipSeg are replaced by a volumetric rendering process which reduces the time of image encoding (mainly coming from the much lower rendering resolutions, we believe the image encoder will give much acceleration and results through reducing the input image size). Besides, we implement a patch-based rendering and aggregate neighbouring features to make up the loss of inner-interactions of patches, which improve the mask qualities.

### Getting Started

#### 1. SAM with Language prompts

We provide the usage of language promptable SAM in samclip.ipynb

#### 2. Segment Anything in NeRF

Without 3D feature distillation
```python
python -m samnerf.train samnerf.train samnerf_no_distill --vis viewer+wandb --viewer.websocket-port 7007
```

With 3D feature distillation, this method will distill the feature of SAM encoder into a 3D feature fields. The image encoding process is replaced by a volumetric rendering.
```python
python -m samnerf.train samnerf.train samnerf_distill --vis viewer+wandb --viewer.websocket-port 7007
```

### Acknowledgement
Our codes are based on [Segment-Anything](https://github.com/facebookresearch/segment-anything), [ClipSeg](https://github.com/timojl/clipseg), [nerfstudio](https://github.com/nerfstudio-project/nerfstudio), and [LERF](https://github.com/kerrj/lerf).