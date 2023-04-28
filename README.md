## SAM-NeRF: A Simple Baseline for Segmenting Anything in NeRF with Language Prompts.


https://user-images.githubusercontent.com/43294876/235251968-4b9103ee-a6ca-4406-b8b0-7c0d95e097be.mp4


https://user-images.githubusercontent.com/43294876/235252013-b6977241-d9af-4dfd-bb2d-d21f30391057.mp4

## Intro
This repo provides a support of language prompts for SAM, by combinning ClipSeg and SAM. We also provides a simple baseline for connecting SAM with NeRF, as well as a simple SAM feature distillation method. Specifically, we provide:

> - A simple extension of Segmenta Anything for incorporating language prompt, by combining ClipSeg with SAM, the inference of ClipSeg features are very fast, costing only 0.04s per image which brings neglectable overheads.
> - A simple implmentation of combining segment-anything with NeRF, providing the function of locking 3D objects for different views, and segmentation in 3D by language and point prompts.
>  - An implementation of distilling SAM and ClipSeg features into 3D fields, in this pipeline, the image encoders of SAM and ClipSeg are replaced by a volumetric rendering process which reduces the time of image encoding (mainly coming from the much lower rendering resolutions, we believe the image encoder will give much acceleration and results through reducing the input image size). Besides, we implement a patch-based rendering and aggregate neighbouring features to make up the loss of inner-interactions of patches, which improve the mask qualities.

### Getting Started

#### SAM with Language prompts

We provide the usage of language promptable SAM in samclip.ipynb

#### Segment Anything in NeRF

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
