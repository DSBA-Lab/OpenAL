dataset=CIFAR100
cost=500
epochs=200
id_ratio=$1

bash runs/openset_runs/run_parallel.sh $dataset $id_ratio $cost $epochs