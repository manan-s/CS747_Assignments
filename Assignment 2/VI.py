import numpy as np
from operator import add, mul

def VIsolver(MDP):
    Q = np.zeros((MDP["S"], MDP["A"]), dtype=np.float128)
    
    precision = 1e-8

    V = np.zeros(MDP["S"], dtype=np.float128)
    V0 = np.zeros(MDP["S"], dtype=np.float128)
    value_func = np.zeros(MDP["S"], dtype=np.float128)
    CV = np.multiply(MDP["R"], MDP["T"]).sum(axis=2)

    while True:
        
        V0 = np.copy(V)
        action_list = MDP["gamma"]*(np.multiply(V, MDP["T"])).sum(axis=2) + CV
        V = np.max(action_list, axis=1)
    
        if np.max(abs(V-V0)) < precision:
            value_func = V
            break
    
    '''
    for s in range(0, MDP["S"]):
        for a in range(0, MDP["A"]):
            Q[s, a] = sum( (MDP["T"][s, a, s_dash]*( MDP["R"][s, a, s_dash] + MDP["gamma"]*value_func[s_dash])) for s_dash in range(0, MDP["S"]) )
    '''
    Pi_optimal = np.argmax(action_list, axis=1)
    
    for s in range(0, MDP["S"]):
        print (str(format(value_func[s], '.6f')) + ' ' + str(Pi_optimal[s]))
    
    return Pi_optimal