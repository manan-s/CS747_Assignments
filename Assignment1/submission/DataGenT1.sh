#!/bin/bash

inst="../instances/i-"
ext=".txt"
finalFile="OutputDataT1.txt"

rm $finalFile
cust_fun(){
    for horizon in 100 400 1600 6400 25600 102400
    do 
        for (( seed=0; seed<=49; seed++ ))
        do
            python3 bandit.py --instance ${inst}$1${ext} --algorithm $2 --epsilon 0.02 --randomSeed $seed --horizon $horizon >> $finalFile
        done
    done
}

for instanceName in 1 2 3
do 
    for algo in "epsilon-greedy" "ucb" "kl-ucb" "thompson-sampling"
    do   
        cust_fun $instanceName $algo &
    done
done

wait