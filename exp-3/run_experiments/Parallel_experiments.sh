#!/bin/bash
logs_fold=""
script_fold=""
ns=( 100 500 1000 2000 5000 10000 20000 60000)
ratios=( 0.15 0.50 1. ) 
# This will start 24 processes of heavy computations. It should be run on a powerful machine.
for cur_n in ${ns[@]}
do
    for ratio in ${ratios[@]}
    do
        python $script_fold/MNIST_MMC_exp.py --ratio $ratio --n_choosen $cur_n --type_exp "incomplete" > $logs_fold/logs/exp_inc_B"$ratio"_n"$cur_n".log &
    done
    python $script_fold/MNIST_MMC_exp.py --n_choosen $cur_n --type_exp "complete" > $logs_fold/logs/exp_comp_n"$cur_n".log &
done
wait

