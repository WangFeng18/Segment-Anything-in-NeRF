
## Intro
This repo provides:
1) A simple extension of Segmenta Anything for incorporating language prompt, by combining ClipSeg with SAM, the inference of ClipSeg features are very fast, costing only 0.04s per image which brings neglectable overheads.
2) A simple implmentation of combining segment-anything with NeRF, providing the function of locking 3D objects for different views, and segmentation in 3D by language and point prompts.
3) An implementation of distilling SAM and ClipSeg features into 3D fields, in this pipeline, the image encoders of SAM and ClipSeg are replaced by a volumetric rendering process which reduces the time of image encoding (mainly coming from the much lower rendering resolutions, we believe the image encoder will give much acceleration and results through reducing the input image size). Besides, we implement a patch-based rendering and aggregate neighbouring features to make up the loss of inner-interactions of patches, which improve the mask qualities.

## Demo

https://user-images.githubusercontent.com/43294876/235158631-c2384188-81cb-40f9-a7e4-e635c9912c8c.mp4

### Promptable Segmentation for NeRF

https://user-images.githubusercontent.com/43294876/235242529-16e224c2-6219-432c-8537-f821408f0533.mp4

