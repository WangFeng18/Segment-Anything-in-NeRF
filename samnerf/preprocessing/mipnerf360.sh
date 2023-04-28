SCENE=$1
TYPE=$2

workdir=$(cd $(dirname $0); pwd)
echo $workdir
cd $workdir
cd ../

if [[ $TYPE == *"feature"* ]]; then
    # you should modify the data dir before $SCENE
    python -m preprocessing.get_image_embeddings --image_path /data/machine/data/mipnerf360/$SCENE/images/ --save_path /data/machine/data/mipnerf360/$SCENE/sam_features/
    python -m preprocessing.get_clipseg_embeddings --image_path /data/machine/data/mipnerf360/$SCENE/images/ --save_path /data/machine/data/mipnerf360/$SCENE/clipseg_features/
fi

if [[ $TYPE == *"json"* ]]; then
    # generate transform.json
    python preprocessing/llff2nerf.py  /data/machine/data/mipnerf360/$SCENE --images images_4 --downscale 4 --hold 60
fi

