bash runs/ssl_runs/csi.sh 0 CIFAR10 1000 ./checkpoints 0 &
tmux split-window -v "bash runs/ssl_runs/csi.sh 0 CIFAR100 1000 ./checkpoints 0" &

tmux select-layout even-vertical &
wait

bash runs/ssl_runs/simclr.sh 0 CIFAR10 1000 ./checkpoints 0 &
tmux split-window -v "bash runs/ssl_runs/simclr.sh 0 CIFAR100 1000 ./checkpoints 0" &

tmux select-layout even-vertical &
wait