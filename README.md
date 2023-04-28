# SAM-NeRF: A Very Simple Baseline for Segmenting Anything in NeRF with Language Prompts.


## SAM with Language prompts:
https://user-images.githubusercontent.com/43294876/235158631-c2384188-81cb-40f9-a7e4-e635c9912c8c.mp4

### Promptable Segmentation for NeRF
https://user-images.githubusercontent.com/43294876/235242529-16e224c2-6219-432c-8537-f821408f0533.mp4

## Intro
This repo provides:
 - A simple extension of Segmenta Anything for incorporating language prompt, by combining ClipSeg with SAM, the inference of ClipSeg features are very fast, costing only 0.04s per image which brings neglectable overheads.
 - A simple implmentation of combining segment-anything with NeRF, providing the function of locking 3D objects for different views, and segmentation in 3D by language and point prompts.
 - An implementation of distilling SAM and ClipSeg features into 3D fields, in this pipeline, the image encoders of SAM and ClipSeg are replaced by a volumetric rendering process which reduces the time of image encoding (mainly coming from the much lower rendering resolutions, we believe the image encoder will give much acceleration and results through reducing the input image size). Besides, we implement a patch-based rendering and aggregate neighbouring features to make up the loss of inner-interactions of patches, which improve the mask qualities.



## Install

```bash
git clone https://github.com/WangFeng18/Explore-Sam-in-NeRF.git
# or if ssh is available
git clone git@github.com:WangFeng18/Explore-Sam-in-NeRF.git
cd Explore-Sam-in-NeRF
pip install -r requirements.txt
```



### Download pretrained models

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



### Host your own viewer

Install `viser` via

```bash
pip install viser
```

and make sure `node.js` and `yarn` are available on your machine. Refer to [this](https://www.digitalocean.com/community/tutorials/how-to-install-node-js-on-ubuntu-20-04) if you do not how to install.



Host on your machine via

```bash
cd nerfstudio/viewer/app
yarn
yarn start
```

After compiling, the viewer is available on `<your_machine_ip>:<port>/?websocket_url=ws://localhost:<ws_port>`. By default, `<port>` will be set to 4000, you can change the `PORT` variable to what you need in `nerfstudio/viewer/app/.env.development`. `<ws_port> `is set through `--viewer.websocket_port <ws_port>` in the command line with your NeRF training.



NOTE: The viewer is currently work in progress, and there may exist tons of bugs. Please let us know if you encounter something unexpected, thanks in advance for you help :smiling_face_with_three_hearts:. 
