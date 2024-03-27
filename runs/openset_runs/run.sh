seed_list="0 1 2 3 4"
dataset=$1
ood_ratio=$2
savedir=$3

for seed in $seed_list
do
    echo "[STANDARD AL] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio"
    bash runs/openset_runs/run_standard_al.sh $seed $dataset $ood_ratio $savedir

    echo "[OPEN-SET AL - CLIPNAL] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio"
    bash runs/openset_runs/run_clipn.sh $seed $dataset $ood_ratio $savedir

    echo "[OPEN-SET AL - LfOSA] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio"
    bash runs/openset_runs/run_lfosa.sh $seed $dataset $ood_ratio $savedir

    echo "[OPEN-SET AL - EOAL] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio"
    bash runs/openset_runs/run_eoal.sh $seed $dataset $ood_ratio $savedir

    echo "[OPEN-SET AL - MQNet] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio"
    bash runs/openset_runs/run_mqnet.sh $seed $dataset $ood_ratio $savedir
done