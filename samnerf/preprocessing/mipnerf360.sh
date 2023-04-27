SCENE=$1

workdir=$(cd $(dirname $0); pwd)
echo $workdir
cd $workdir
cd ../

python -m preprocessing.get_image_embeddings --image_path /data/machine/data/mipnerf360/$SCENE/images/ --save_path /data/machine/data/mipnerf360/$SCENE/sam_features/

# generate transform.json
python preprocessing/llff2nerf.py  /data/machine/data/mipnerf360/$SCENE --images images_4 --downscale 4 --hold 60