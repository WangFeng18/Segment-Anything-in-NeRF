## SAM-NeRF: A Simple Baseline for Segmenting Anything in NeRF with Language Prompts.


https://user-images.githubusercontent.com/43294876/235251968-4b9103ee-a6ca-4406-b8b0-7c0d95e097be.mp4


https://user-images.githubusercontent.com/43294876/235252013-b6977241-d9af-4dfd-bb2d-d21f30391057.mp4


This repo provides a support of language prompts for SAM, by combinning ClipSeg and SAM. We also provides a simple baseline for connecting SAM with NeRF, as well as a 2D-to-3D SAM feature distillation method. Specifically, we provide:

> - An extension of Segmenta Anything for incorporating language prompt, by combining ClipSeg with SAM, the inference of ClipSeg features are very fast, costing only 0.04s per image which brings neglectable overheads.
> - An implmentation of combining segment-anything with NeRF, providing the function of locking 3D objects for different views, and segmentation in 3D by language and point prompts.
>  - An implementation of distilling SAM and ClipSeg features into 3D fields, in this pipeline, the image encoders of SAM and ClipSeg are replaced by a volumetric rendering process which reduces the time of image encoding (mainly coming from the much lower rendering resolutions, we believe the image encoder will give much acceleration and results through reducing the input image size). Besides, we implement a patch-based rendering and aggregate neighbouring features to make up the loss of inner-interactions of patches, which improve the mask qualities.

### Getting Started

#### 1. SAM with Language prompts

We provide the usage of language promptable SAM in samclip.ipynb, and provide a gradio program for interactively segmenting objects with language prompts.

#### 2. Segment Anything in NeRF
```
# data pre-processing, get the json files for training nerf in nerfstudio
bash samnerf/preprocessing/mipnerf350.sh json
```

Without 3D feature distillation
```python
python -m samnerf.train samnerf.train samnerf_no_distill --vis viewer+wandb --viewer.websocket-port 7007
```

With 3D feature distillation, this method will distill the feature of SAM encoder into a 3D feature fields. The image encoding process is replaced by a volumetric rendering.
```bash
# first extract the features of SAM encoder and ClipSeg features
bash samnerf/preprocessing/mipnerf350.sh feature
# training nerf
python -m samnerf.train samnerf.train samnerf_distill --vis viewer+wandb --viewer.websocket-port 7007
```

### Acknowledgement
Our codes are based on 
[Segment-Anything](https://github.com/facebookresearch/segment-anything),
```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

[ClipSeg](https://github.com/timojl/clipseg),
```
@InProceedings{lueddecke22_cvpr,
    author    = {L\"uddecke, Timo and Ecker, Alexander},
    title     = {Image Segmentation Using Text and Image Prompts},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {7086-7096}
}
```

 [nerfstudio](https://github.com/nerfstudio-project/nerfstudio),
 ```
@article{nerfstudio,
    author = {Tancik, Matthew and Weber, Ethan and Ng, Evonne and Li, Ruilong and Yi,
            Brent and Kerr, Justin and Wang, Terrance and Kristoffersen, Alexander and Austin,
            Jake and Salahi, Kamyar and Ahuja, Abhik and McAllister, David and Kanazawa, Angjoo},
    title = {Nerfstudio: A Modular Framework for Neural Radiance Field Development},
    journal = {arXiv preprint arXiv:2302.04264},
    year = {2023},
}
```
and [LERF](https://github.com/kerrj/lerf),
```
@article{kerr2023lerf,
  title={LERF: Language Embedded Radiance Fields},
  author={Kerr, Justin and Kim, Chung Min and Goldberg, Ken and Kanazawa, Angjoo and Tancik, Matthew},
  journal={arXiv preprint arXiv:2303.09553},
  year={2023}
}
```

### Citation
If you find the project is useful, please consider citing:

``` 
@misc{sam-nerf,
    Author = {Feng Wang and Zilong Chen},
    Year = {2023},
    Note = {https://github.com/WangFeng18/Explore-Sam-in-NeRF/tree/main},
    Title = {SamNeRF: A Simple Baseline for Segmenting Anything in NeRF with Language Prompts}
}
```