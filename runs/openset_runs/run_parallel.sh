bash runs/openset_runs/run.sh CIFAR10 0.1 ./results_openset-test 0 &
tmux split-window -v 'bash runs/openset_runs/run.sh CIFAR10 0.2 ./results_openset-test 1' &
tmux split-window -v 'bash runs/openset_runs/run.sh CIFAR10 0.4 ./results_openset-test 2' &
tmux split-window -v 'bash runs/openset_runs/run.sh CIFAR10 0.6 ./results_openset-test 3' &

tmux select-layout even-vertical &
