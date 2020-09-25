#!/bin/bash
# bash task3.sh

inst="../instances/i-"
ext=".txt"

cust_fun(){
    for (( seed=0; seed<=49; seed++ ))
    do
        python3 bandit.py --instance ${inst}$1${ext} --algorithm epsilon-greedy --epsilon $2 --randomSeed $seed --horizon 102400 >> T3/$2_$1_epsilon_greedy.txt 
        echo --epsilon $2 --instanceName $1 --seed $seed
    done
    python3 calculate_mean.py T3/$2_$1_epsilon_greedy.txt >> T3/fin/instance_$1_final.txt
    
}

for instanceName in 1 2 3
do 
    rm ../expt/${instanceName}
    rm ../expt/results/instance_${instanceName}_final.txt
    for ep in 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01
    do   
        cust_fun $instanceName $ep &
    done
done

wait