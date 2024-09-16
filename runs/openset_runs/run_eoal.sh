seed=$1
ds=$2
ood_ratio=$3
id_ratio=$4
cost=$5
cycle=$6
epochs=$7
savedir=$8
gpu_id=$9

let n_end=$cost*$cycle

CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    default_cfg=./configs/default_setting.yaml \
    openset_cfg=./configs/openset_al/eoal.yaml \
    DEFAULT.exp_name=ood$ood_ratio-id$id_ratio \
    DATASET.name=$ds \
    TRAIN.epochs=$epochs \
    AL.ood_ratio=$ood_ratio \
    AL.id_ratio=$id_ratio \
    AL.n_start=$cost \
    AL.n_query=$cost \
    AL.n_end=$n_end \
    DEFAULT.seed=$seed \
    DEFAULT.savedir=$savedir
