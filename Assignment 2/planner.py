import os
import sys
import numpy as np
from LP import LPsolver
from VI import VIsolver
from HPI import HPIsolver

def parseargs():
    opts = [opt for opt in sys.argv[1:] if opt.startswith("--")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]

    try:
        mdp_path = args[opts.index('--mdp')]
        algorithm = args[opts.index('--algorithm')]
    
    except ValueError:
        print("Atleast one of the arguments isn't provided")
        sys.exit(f"Usage1: {sys.argv[0]} (--mdp | --algorithm) <arguments>...")
    
    except:
        print("Please check the arguments")
        sys.exit(f"Usage2: {sys.argv[0]} (--mdp | --algorithm) <arguments>...")
    
    file_obj = open(mdp_path, "r")
    file_lines = file_obj.readlines()

    num_states = int((file_lines[0].split())[-1])
    num_actions = int((file_lines[1].split())[-1])
    start_state = int((file_lines[2].split())[-1])
    end_states = [int(i) for i in (file_lines[3].split())[1:]]
    rewards = np.zeros((num_states, num_actions, num_states), dtype=np.float128)
    transitions = np.zeros((num_states, num_actions, num_states), dtype=np.float128)
    
    i = 4
    while (file_lines[i].split())[0] == "transition":
        SAS = tuple([int(j) for j in (file_lines[i].split())[1:4]])
        RP = [float(j) for j in (file_lines[i].split())[4:]]
        rewards[SAS] = RP[0]
        transitions[SAS] = RP[1]
        i+=1

    mdp_type = (file_lines[i].split())[-1]
    if mdp_path == "continuing":
        end_states = []
    discount = float((file_lines[i+1].split())[-1])

    MDP = {}
    MDP["S"] = num_states
    MDP["A"] = num_actions
    MDP["T"] = transitions
    MDP["R"] = rewards
    MDP["gamma"] = discount
    MDP["type"] = mdp_type
    MDP["start"] = start_state
    MDP["end"] = end_states

    return MDP, algorithm

def main():
    MDP, algorithm = parseargs()

    if algorithm == "vi":
        optimal_policy = VIsolver(MDP)
    
    elif algorithm == "lp":
        optimal_policy = LPsolver(MDP)
    
    elif algorithm == "hpi":
        optimal_policy = HPIsolver(MDP)

    else:
        print("Invalid Algorithm")
        sys.exit(f"Usage: {sys.argv[0]} .... --algorithm <vi | lp | hpi>")

if __name__ == '__main__':
    main()


