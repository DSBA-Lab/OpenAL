datasets='CIFAR10 CIFAR100'
strategy='random_sampling least_confidence margin_sampling entropy_sampling'

for d in $datasets
do
    for s in $strategy
    do
        python main_benchmark.py --yaml_config ./configs/$d/$s.yaml
    done
done