seed_list="0 1 2 3 4"
datasets="CIFAR10 CIFAR100 Tiny_ImageNet_200"
ood_ratio="0.1"
id_ratio_list="0.2 0.3 0.4"
savedir=./results
gpu_id=0

for seed in $seed_list
do
    for ds in $datasets
    do
        for id_ratio in $id_ratio_list
        do
            CUDA_VISIBLE_DEVICES=$gpu_id python main_zeroshot.py \
                default_cfg=./configs/default_setting.yaml \
                strategy_cfg=./configs/standard_al/least_confidence.yaml \
                openset_cfg=./configs/openset_al/clipnal.yaml \
                DEFAULT.exp_name=ood$ood_ratio-id$id_ratio \
                DATASET.name=$ds \
                AL.ood_ratio=$ood_ratio \
                AL.id_ratio=$id_ratio \
                DEFAULT.seed=$seed \
                DEFAULT.savedir=$savedir \
                DEFAULT.zeroshot=true \
                MODEL.name=CLIPN
        done
    done
done


seed_list="0 1 2 3 4"
datasets="CIFAR10 CIFAR100"
ood_ratio="0.6"
id_ratio_list="0.2 0.3 0.4"
savedir=./results
gpu_id=0

for seed in $seed_list
do
    for ds in $datasets
    do
        for id_ratio in $id_ratio_list
        do
            CUDA_VISIBLE_DEVICES=$gpu_id python main_zeroshot.py \
                default_cfg=./configs/default_setting.yaml \
                strategy_cfg=./configs/standard_al/least_confidence.yaml \
                openset_cfg=./configs/openset_al/clipnal.yaml \
                DEFAULT.exp_name=ood$ood_ratio-id$id_ratio \
                DATASET.name=$ds \
                AL.ood_ratio=$ood_ratio \
                AL.id_ratio=$id_ratio \
                DEFAULT.seed=$seed \
                DEFAULT.savedir=$savedir \
                AL.n_start=500 \
                AL.n_query=500 \
                AL.n_end=2000 \
                TRAIN.epochs=10 \
                TRAIN.params.steps_per_epoch=50 \
                OPTIMIZER.lr=0.01 \
                DEFAULT.zeroshot=false \
                MODEL.name=CLIPN
        done
    done
done


seed_list="0 1 2 3 4"
datasets="Tiny_ImageNet_200"
ood_ratio="0.6"
id_ratio_list="0.2 0.3 0.4"
savedir=./results
gpu_id=0

for seed in $seed_list
do
    for ds in $datasets
    do
        for id_ratio in $id_ratio_list
        do
            CUDA_VISIBLE_DEVICES=$gpu_id python main_zeroshot.py \
                default_cfg=./configs/default_setting.yaml \
                strategy_cfg=./configs/standard_al/least_confidence.yaml \
                openset_cfg=./configs/openset_al/clipnal.yaml \
                DEFAULT.exp_name=ood$ood_ratio-id$id_ratio \
                DATASET.name=$ds \
                AL.ood_ratio=$ood_ratio \
                AL.id_ratio=$id_ratio \
                DEFAULT.seed=$seed \
                DEFAULT.savedir=$savedir \
                AL.n_start=1000 \
                AL.n_query=1000 \
                AL.n_end=4000 \
                TRAIN.epochs=10 \
                TRAIN.params.steps_per_epoch=100 \
                OPTIMIZER.lr=0.01 \
                DEFAULT.zeroshot=false \
                MODEL.name=CLIPN
        done
    done
done