dataset=Tiny_ImageNet_200
cost=1000
epochs=300
id_ratio=$1

bash runs/openset_runs/run_parallel.sh $dataset $id_ratio $cost $epochs