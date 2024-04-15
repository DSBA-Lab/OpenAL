# default setting
cycle=10
savedir="./results_openset-test"

# option
dataset=$1
id_ratio=$2
cost=$3
epochs=$4

bash runs/openset_runs/run.sh $dataset 0.1 $id_ratio $cost $cycle $epochs $savedir 0 &
tmux split-window -v "bash runs/openset_runs/run.sh $dataset 0.2 $id_ratio $cost $cycle $epochs $savedir 1" &
tmux split-window -v "bash runs/openset_runs/run.sh $dataset 0.4 $id_ratio $cost $cycle $epochs $savedir 2" &
tmux split-window -v "bash runs/openset_runs/run.sh $dataset 0.6 $id_ratio $cost $cycle $epochs $savedir 3" &

tmux select-layout even-vertical &
wait