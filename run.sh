datasets='CIFAR10 CIFAR100'
strategy='learning_loss least_confidence margin_sampling entropy_sampling random_sampling'
seed_list='0 1 2 3 4'

for seed in $seed_list
do
    for d in $datasets
    do
        for s in $strategy
        do
            python main_benchmark.py --yaml_config ./configs/benchmark/$d/${s}.yaml --seed $seed
        done
    done
done