strategies='least_confidence learning_loss'
seed=$1
ds=$2
ood_ratio=$3
savedir=$4
gpu_id=$5

for s in $strategies
do

    if [[ $ds == 'CIFAR10' ]];then
        id_class=4
    elif [[ $ds == 'CIFAR100' ]];then
        id_class=40
    elif [[ $ds == 'Tiny_ImageNet_200' ]];then
        id_class=80
    fi

    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
        default_cfg=./configs/default_setting.yaml \
        strategy_cfg=./configs/$s.yaml \
        openset_cfg=./configs_openset/clipnal.yaml \
        DEFAULT.exp_name=ood$ood_ratio-use_clipn \
        DATASET.name=$ds \
        AL.ood_ratio=$ood_ratio \
        AL.nb_id_class=$id_class \
        DEFAULT.seed=$seed \
        DEFAULT.savedir=$savedir

    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
        default_cfg=./configs/default_setting.yaml \
        strategy_cfg=./configs/$s.yaml \
        openset_cfg=./configs_openset/clipnal.yaml \
        DEFAULT.exp_name=ood$ood_ratio-use_sim \
        DATASET.name=$ds \
        AL.ood_ratio=$ood_ratio \
        AL.nb_id_class=$id_class \
        AL.openset_params.use_sim=True \
        DEFAULT.seed=$seed \
        DEFAULT.savedir=$savedir

done
