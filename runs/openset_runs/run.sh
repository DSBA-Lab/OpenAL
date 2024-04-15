# seed_list="0 1 2 3 4"
seed_list="0"
dataset=$1
ood_ratio=$2
id_ratio=$3
cost=$4
cycle=$5
epochs=$6
savedir=$7
gpu_id=$8

for seed in $seed_list
do
    echo "[STANDARD AL] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio, ID Ratio: $id_ratio"
    bash runs/openset_runs/run_standard_al.sh $seed $dataset $ood_ratio $id_ratio $cost $cycle $epochs $savedir $gpu_id random_sampling

    echo "[OPEN-SET AL - LfOSA] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio, ID Ratio: $id_ratio"
    bash runs/openset_runs/run_lfosa.sh $seed $dataset $ood_ratio $id_ratio $cost $cycle $epochs $savedir $gpu_id

    echo "[OPEN-SET AL - EOAL] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio, ID Ratio: $id_ratio"
    bash runs/openset_runs/run_eoal.sh $seed $dataset $ood_ratio $id_ratio $cost $cycle $epochs $savedir $gpu_id

    echo "[STANDARD AL] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio, ID Ratio: $id_ratio"
    bash runs/openset_runs/run_standard_al.sh $seed $dataset $ood_ratio $id_ratio $cost $cycle $epochs $savedir $gpu_id learning_loss

    echo "[STANDARD AL] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio, ID Ratio: $id_ratio"
    bash runs/openset_runs/run_standard_al.sh $seed $dataset $ood_ratio $id_ratio $cost $cycle $epochs $savedir $gpu_id least_confidence

    echo "[STANDARD AL] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio, ID Ratio: $id_ratio"
    bash runs/openset_runs/run_standard_al.sh $seed $dataset $ood_ratio $id_ratio $cost $cycle $epochs $savedir $gpu_id kcenter_greedy

    echo "[OPEN-SET AL - CLIPNAL] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio, ID Ratio: $id_ratio "
    bash runs/openset_runs/run_clipn.sh $seed $dataset $ood_ratio $id_ratio $cost $cycle $epochs $savedir $gpu_id

    # echo "[OPEN-SET AL - MQNet] SEED: $seed, DATASET: $dataset, OOD RATIO: $ood_ratio, ID Ratio: $id_ratio"
    # bash runs/openset_runs/run_mqnet.sh $seed $dataset $ood_ratio $id_ratio $cost $cycle $epochs $savedir $gpu_id
done