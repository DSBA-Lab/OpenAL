
bash runs/openset_runs/run_lfosa.sh 0 CIFAR100 0.1 &
tmux split-window -v 'bash runs/openset_runs/run_lfosa.sh 0 CIFAR100 0.2' &
tmux split-window -v 'bash runs/openset_runs/run_lfosa.sh 0 CIFAR100 0.4' &
tmux split-window -v 'bash runs/openset_runs/run_lfosa.sh 0 CIFAR100 0.6' &

tmux select-layout even-vertical &