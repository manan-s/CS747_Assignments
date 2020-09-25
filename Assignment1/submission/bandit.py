import os
import sys
import numpy as np

from epsilonGreedy import epsilon_greedy
from UCB import UCB
from KL_UCB import KL_UCB
from ThompsonSampling import ThomsonSampling
from ThompsonHint import ThomsonHint

def parseargs():
    opts = [opt for opt in sys.argv[1:] if opt.startswith("--")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

    try:
        instance_path = args[opts.index('--instance')]
        algorithm = args[opts.index('--algorithm')]
        randomSeed = int(args[opts.index('--randomSeed')])
        epsilon = float(args[opts.index('--epsilon')])
        horizon = int(args[opts.index('--horizon')])
    
    except ValueError:
        #print(sys.argv)
        print("Atleast one of the arguments isn't provided")
        sys.exit(f"Usage1: {sys.argv[0]} (--instance | --algorithm | --epsilon | --randomSeed | --horizon) <arguments>...")
    
    except:
        print("Please check the arguments")
        sys.exit(f"Usage2: {sys.argv[0]} (--instance | --algorithm | --epsilon | --randomSeed | --horizon) <arguments>...")
    
    #Storing the instance file to list
    instance = []
    with open(instance_path, "r") as f:
        for line in f:
            instance.append(float(line))
    
    return instance_path, instance, algorithm, randomSeed, epsilon, horizon

def main():
    filename, instance, algorithm, seed, epsilon, horizon = parseargs()

    if (algorithm == 'epsilon-greedy'):
        #for horizon in [100, 400, 1600, 6400, 25600, 102400]:
        #for seed in range(50):
        reward = epsilon_greedy(instance, epsilon, horizon, seed)
        regret = horizon*max(instance) - reward
        print('{}, {}, {}, {}, {}, {}'.format(filename, algorithm, seed, epsilon, horizon, regret))
    
    elif (algorithm == 'ucb'):
        #for horizon in [100, 400, 1600, 6400, 25600, 102400]:
        #for seed in range(50):
        reward = UCB(instance, horizon, seed)
        regret = horizon*max(instance) - reward
        print('{}, {}, {}, {}, {}, {}'.format(filename, algorithm, seed, epsilon, horizon, regret))
    
    elif (algorithm == 'kl-ucb'):
        #for horizon in [100, 400, 1600, 6400, 25600, 102400]:
        #for seed in range(50):
        reward = KL_UCB(instance, horizon, seed)
        regret = horizon*max(instance) - reward
        print('{}, {}, {}, {}, {}, {}'.format(filename, algorithm, seed, epsilon, horizon, regret))

    elif (algorithm == 'thompson-sampling'):
        #for horizon in [100, 400, 1600, 6400, 25600, 102400]:
        #for seed in range(50):
        reward = ThomsonSampling(instance, horizon, seed)
        regret = horizon*max(instance) - reward
        print('{}, {}, {}, {}, {}, {}'.format(filename, algorithm, seed, epsilon, horizon, regret))
    
    elif (algorithm == 'thompson-sampling-with-hint'):
        #for horizon in [100, 400, 1600, 6400, 25600, 102400]:
        #for seed in range(50):
        reward = ThomsonHint(instance, horizon, seed)
        regret = horizon*max(instance) - reward
        print('{}, {}, {}, {}, {}, {}'.format(filename, algorithm, seed, epsilon, horizon, regret))

    else:
        print("Invalid Algorithm")
        print(algorithm)
        sys.exit(f"Usage: {sys.argv[0]} (--instance | --algorithm | --epsilon | --randomSeed | --horizon) <arguments>...")

main()


