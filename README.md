## SAM-NeRF: A Simple Baseline for Segmenting Anything in NeRF with Language Prompts.



https://user-images.githubusercontent.com/43294876/235264157-32e2fe1d-08dd-4823-bbac-e3675587358f.mp4



https://user-images.githubusercontent.com/43294876/235252013-b6977241-d9af-4dfd-bb2d-d21f30391057.mp4


This repository provides language prompt support for SAM through a combination of ClipSeg and SAM. Additionally, we offer a simple baseline for connecting SAM with NeRF and a 2D-to-3D SAM feature distillation method. Specifically, this project contains:

> - An extension of Segment Anything for incorporating language prompts by combining ClipSeg with SAM. The inference of ClipSeg features is fast, taking only 0.04s per image and causing negligible overhead.
> - An implementation of combining Segment Anything with NeRF, allowing for locking 3D objects for different views and segmentation in 3D by language and point prompts.
> - An implementation of distilling SAM and ClipSeg features into 3D fields. In this pipeline, the image encoders of SAM and ClipSeg are replaced by a volumetric rendering process, significantly reducing the time of image encoding. The acceleration mainly comes from the much lower rendering resolutions, we believe the image encoder will give more acceleration and better results through reducing the input image size. We use a patch-based rendering and aggregate neighboring features to make up for the loss of inner-interactions of patches, improving the mask qualities.
> - A viewer for visualizing the trained SAM-NeRF. This viewer allows users to lock onto a certain 3D object via clicking or providing text prompt. For language prompts, the viewer can also search objects by text and provide a heatmap indicating pixel-level relevance.


## Install
### 1. Install required packages
```bash
git clone https://github.com/WangFeng18/Explore-Sam-in-NeRF.git
# or if ssh is available
git clone git@github.com:WangFeng18/Explore-Sam-in-NeRF.git
cd Explore-Sam-in-NeRF
pip install -r requirements.txt
```

### 2. Download pretrained models

#### Download with script

```bash
bash download.sh
```


#### Or download manually (Same as using script)

For pretrained CLIPseg model:

```bash
cd samnerf/clipseg
wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip
unzip -d weights -j weights.zip
```

For pertained SAM model:

```bash
cd samnerf/segment-anything
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```


### 3. Host your own viewer

Install `viser` via

```bash
pip install viser
```

and make sure `node.js` and `yarn` are available on your machine. Refer to [this](https://www.digitalocean.com/community/tutorials/how-to-install-node-js-on-ubuntu-20-04) if you are not clear about how to install.



Host on your machine via

```bash
cd nerfstudio/viewer/app
yarn
yarn start
```

After compiling, the viewer is available on `<your_machine_ip>:<port>/?websocket_url=ws://localhost:<ws_port>`. By default, `<port>` will be set to 4000, you can change the `PORT` variable to what you need in `nerfstudio/viewer/app/.env.development`. `<ws_port> `is set through `--viewer.websocket_port <ws_port>` in the command line with your NeRF training.

**For a more complete viewer instruction, checkout [here](./nerfstudio/viewer/intructions.md) :hear_no_evil: .**

NOTE: The viewer is currently work in progress, and there may exist some bugs. Please let us know if you encounter something unexpected, thanks in advance for you help :smiling_face_with_three_hearts: . 



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
