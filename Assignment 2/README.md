# Assignment 2

## Runing Instructions:
1) For value function and optimal policy estimation for an underlying MDP:  

    `python planner.py --mdp <path> --algorithm <algo>`
    
    `<path>` is path to the MDP file  
    `<algo>` is one of `vi`, `lp` or `hpi`

2) For solving maze via formulating the task as MDPs:

    `python encoder.py --grid <gridfile> > mdpfile`  
    `python planner.py --mdp mdpfile --algorithm <algo> > value_and_policy_file`  
    `python decoder.py --grid <gridfile> --value_policy value_and_policy_file > pathfile`  
    
    `<gridfile>` is path to the grid file 
    
3) Accessory functions (provided):

- To generate an MDP of `s` states, `a` actions, `df` discount factor:

    `python generateMDP.py --S <s> --A <a> --gamma <df> --mdptype continuing --rseed 0`

- To visualize maze and the solution

    `python visualize.py <gridfile> <pathfile>`  
    
    `<pathfile>` is the (path to) output of the decoder, and is an optional argument in case the solution is to be visualized too
  
 
<br>  

The detailed format is available on the assignment [webpage](https://www.cse.iitb.ac.in/~shivaram/teaching/cs747-a2020/pa-2/programming-assignment-2.html).
