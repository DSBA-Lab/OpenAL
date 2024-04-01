# seed_list="0 1 2 3 4"
seed_list="0"
dataset=$1
ood_ratio=$2
savedir=$3
gpu_id=$4

for seed in $seed_list
do
    echo "[STANDARD AL] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio"
    bash runs/openset_runs/run_standard_al.sh $seed $dataset $ood_ratio $savedir $gpu_id random_sampling

    echo "[STANDARD AL] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio"
    bash runs/openset_runs/run_standard_al.sh $seed $dataset $ood_ratio $savedir $gpu_id learning_loss

    echo "[OPEN-SET AL - LfOSA] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio"
    bash runs/openset_runs/run_lfosa.sh $seed $dataset $ood_ratio $savedir $gpu_id

    echo "[OPEN-SET AL - EOAL] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio"
    bash runs/openset_runs/run_eoal.sh $seed $dataset $ood_ratio $savedir $gpu_id

    echo "[STANDARD AL] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio"
    bash runs/openset_runs/run_standard_al.sh $seed $dataset $ood_ratio $savedir $gpu_id least_confidence

    echo "[STANDARD AL] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio"
    bash runs/openset_runs/run_standard_al.sh $seed $dataset $ood_ratio $savedir $gpu_id kcenter_greedy

    echo "[OPEN-SET AL - CLIPNAL] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio"
    bash runs/openset_runs/run_clipn.sh $seed $dataset $ood_ratio $savedir $gpu_id

    echo "[OPEN-SET AL - MQNet] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio"
    bash runs/openset_runs/run_mqnet.sh $seed $dataset $ood_ratio $savedir $gpu_id
done