strategies='random_sampling learning_loss least_confidence kcenter_greedy'
seed=$1
ds=$2
ood_ratio=$3
savedir=$4

for s in $strategies
do
    if [[ $ds == 'CIFAR10' ]];then
        id_class=4
    elif [[ $ds == 'CIFAR100' ]];then
        id_class=40
    fi

    python main.py \
        default_cfg=./configs/default_setting.yaml \
        strategy_cfg=./configs/$s.yaml \
        DEFAULT.exp_name=ood$ood_ratio \
        DATASET.name=$ds \
        AL.ood_ratio=$ood_ratio \
        AL.nb_id_class=$id_class \
        DEFAULT.seed=$seed \
        DEFAULT.savedir=$savedir

done

