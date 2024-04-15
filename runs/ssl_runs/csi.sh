seed=$1
ds=$2
epochs=$3
savedir=$4
gpu_id=$5

CUDA_VISIBLE_DEVICES=$gpu_id python main_ssl.py \
    default_cfg=./configs_ssl/csi.yaml \
    DATASET.name=$ds \
    TRAIN.epochs=$epochs \
    DEFAULT.seed=$seed \
    DEFAULT.savedir=$savedir

