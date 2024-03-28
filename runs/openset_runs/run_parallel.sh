# export CUDA_VISIBLE_DEVICES=0 &
bash runs/openset_runs/run.sh CIFAR10 0.1 ./results_openset-test &
# export CUDA_VISIBLE_DEVICES=1 &
tmux split-window -v bash runs/openset_runs/run.sh CIFAR10 0.2 ./results_openset-test &
# export CUDA_VISIBLE_DEVICES=2 &
tmux split-window -v bash runs/openset_runs/run.sh CIFAR10 0.4 ./results_openset-test &
# export CUDA_VISIBLE_DEVICES=3 &
tmux split-window -v bash runs/openset_runs/run.sh CIFAR10 0.6 ./results_openset-test &
tmux select-layout even-vertical &
