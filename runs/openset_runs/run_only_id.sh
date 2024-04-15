# default setting
cycle=10
savedir="./results_openset-test"

# option
dataset=CIFAR10
cost=500
epochs=200

bash runs/openset_runs/run.sh $dataset 0.0 0.2 $cost $cycle $epochs $savedir 0 &
tmux split-window -v "bash runs/openset_runs/run.sh $dataset 0.0 0.3 $cost $cycle $epochs $savedir 1" &
tmux split-window -v "bash runs/openset_runs/run.sh $dataset 0.0 0.4 $cost $cycle $epochs $savedir 2" &

tmux select-layout even-vertical &
wait

# option
dataset=CIFAR100
cost=500
epochs=200

bash runs/openset_runs/run.sh $dataset 0.0 0.2 $cost $cycle $epochs $savedir 0 &
tmux split-window -v "bash runs/openset_runs/run.sh $dataset 0.0 0.3 $cost $cycle $epochs $savedir 1" &
tmux split-window -v "bash runs/openset_runs/run.sh $dataset 0.0 0.4 $cost $cycle $epochs $savedir 2" &

tmux select-layout even-vertical &
wait

# option
dataset=Tiny_ImageNet_200
cost=1000
epochs=300

bash runs/openset_runs/run.sh $dataset 0.0 0.2 $cost $cycle $epochs $savedir 0 &
tmux split-window -v "bash runs/openset_runs/run.sh $dataset 0.0 0.3 $cost $cycle $epochs $savedir 1" &
tmux split-window -v "bash runs/openset_runs/run.sh $dataset 0.0 0.4 $cost $cycle $epochs $savedir 2" &

tmux select-layout even-vertical &
wait