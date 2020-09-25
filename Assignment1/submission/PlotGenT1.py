import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.ticker as ticker

# result_file = "../expt/results/task4_1.txt"
result_file = "outputDataT1.txt"
instance_list = ["../instances/i-1.txt","../instances/i-2.txt","../instances/i-3.txt"]
# instance_list = ["i-1.txt","i-2.txt","i-3.txt"]
algorithms = ["kl-ucb","ucb","epsilon-greedy","thompson-sampling"]
final_dict = {"../instances/i-1.txt":{}, "../instances/i-2.txt":{}, "../instances/i-3.txt":{}}
horizons =  [100, 400, 1600, 6400, 25600, 102400]
# final_dict = {"i-3.txt":{}, "i-1.txt":{}, "i-2.txt":{}}
with open(result_file,'r') as f:
    lines = f.readlines()
    for line in lines:
        x = line.rstrip().split(', ')
        if x[1] in final_dict[x[0]].keys():
            final_dict[x[0]][x[1]][int(np.log2(int(x[4])/100)/2)] += float(x[5])/50.0
        else:
            final_dict[x[0]][x[1]] = [0]*6
            final_dict[x[0]][x[1]][int(np.log2(int(x[4])/100)/2)] += float(x[5])/50.0

for i,instance in enumerate(instance_list):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.xaxis.set_ticks(horizons)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    kl_ucb = final_dict[instance]["kl-ucb"]
    plt.plot(horizons,kl_ucb,label="KL-UCB")
    ucb = final_dict[instance]["ucb"]
    plt.plot(horizons,ucb,label="UCB")
    ts = final_dict[instance]["thompson-sampling"]
    plt.plot(horizons,ts,label="Thompson-Sampling")
    eg = final_dict[instance]["epsilon-greedy"]
    plt.plot(horizons,eg,label="Epsilon-greedy")
    plt.xlabel("Horizon (Logarithmic Scale, Base 2)")
    plt.ylabel("Average Regret")
    plt.legend()

    pltTitle = instance.replace('../instances/', '')
    pltTitle = pltTitle.replace('.txt', '')
    pltTitle = pltTitle.replace('i-', 'Instance ')
    # plt.plot(x_axis,kl_ucb,x_axis,ts,x_axis,ucb,x_axis,eg)
    plt.title("{}".format(pltTitle))
    plt.savefig("T1_instance{}.png".format(i+1))