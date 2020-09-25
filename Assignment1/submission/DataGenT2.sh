#!/bin/bash

inst="../instances/i-"
ext=".txt"
finalFile="finalT2.txt"

rm $finalFile
cust_fun(){ 
    for algo in "thompson-sampling" "thompson-sampling-with-hint"
    do
        for horizon in 100 400 1600 6400 25600 102400
        do
            python3 bandit.py --instance ${inst}$1${ext} --algorithm $algo --epsilon 0.02 --randomSeed $2 --horizon $horizon >> $finalFile
            echo --instance $instanceName --algorithm $algo --randomSeed $2
        done
    done
}

for instanceName in 1 2 3
do
    for (( seed=0; seed<=49; seed++ ))
    do 
        cust_fun $instanceName $seed &
    done
done
wait