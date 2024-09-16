strategies='least_confidence learning_loss'
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

for s in $strategies
do

    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
        default_cfg=./configs/default_setting.yaml \
        strategy_cfg=./configs/standard_al/$s.yaml \
        openset_cfg=./configs/openset_al/clipnal.yaml \
        DEFAULT.exp_name=ood$ood_ratio-id$id_ratio-use_clipn \
        DATASET.name=$ds \
        TRAIN.epochs=$epochs \
        AL.ood_ratio=$ood_ratio \
        AL.id_ratio=$id_ratio \
        AL.n_start=$cost \
        AL.n_query=$cost \
        AL.n_end=$n_end \
        DEFAULT.seed=$seed \
        DEFAULT.savedir=$savedir

    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
        default_cfg=./configs/default_setting.yaml \
        strategy_cfg=./configs/standard_al/$s.yaml \
        openset_cfg=./configs/openset_al/clipnal.yaml \
        DEFAULT.exp_name=ood$ood_ratio-id$id_ratio-use_sim \
        DATASET.name=$ds \
        TRAIN.epochs=$epochs \
        AL.ood_ratio=$ood_ratio \
        AL.id_ratio=$id_ratio \
        AL.n_start=$cost \
        AL.n_query=$cost \
        AL.n_end=$n_end \
        AL.openset_params.use_sim=True \
        DEFAULT.seed=$seed \
        DEFAULT.savedir=$savedir

done
