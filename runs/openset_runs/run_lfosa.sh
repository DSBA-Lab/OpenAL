seed=$1
ds=$2
ood_ratio=$3
savedir=$4

if [[ $ds == 'CIFAR10' ]];then
    id_class=4
elif [[ $ds == 'CIFAR100' ]];then
    id_class=40
fi

python main.py \
    default_cfg=./configs/default_setting.yaml \
    openset_cfg=./configs_openset/lfosa.yaml \
    DEFAULT.exp_name=ood$ood_ratio \
    DATASET.name=$ds \
    AL.ood_ratio=$ood_ratio \
    AL.nb_id_class=$id_class \
    DEFAULT.seed=$seed \
    DEFAULT.savedir=$savedir
