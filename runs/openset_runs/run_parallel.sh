export CUDA_VISIBLE_DEVICES=0 &
bash runs/openset_runs/run.sh CIFAR10 0.1 ./results_openset &

export CUDA_VISIBLE_DEVICES=1 &
bash runs/openset_runs/run.sh CIFAR10 0.2 ./results_openset &

export CUDA_VISIBLE_DEVICES=2 &
bash runs/openset_runs/run.sh CIFAR10 0.4 ./results_openset &

export CUDA_VISIBLE_DEVICES=3 &
bash runs/openset_runs/run.sh CIFAR10 0.6 ./results_openset &