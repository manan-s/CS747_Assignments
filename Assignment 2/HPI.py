import numpy as np
from operator import add, mul

def HPIsolver(MDP):
    Q= np.zeros((MDP["S"], MDP["A"]), dtype=np.float128)
    policy = np.zeros(MDP["S"], dtype=np.int)
    converged = 0

    while not converged:
        
        V = np.zeros(MDP["S"], dtype=np.float128)
        V0 = np.zeros(MDP["S"], dtype=np.float128)
        
        while True:
            for s in range(MDP["S"]):
                V[s] = np.sum(MDP["T"][s, policy[s], :]*MDP["R"][s, policy[s], :] + MDP["gamma"]*MDP["T"][s, policy[s], :]*V0)

            if np.allclose(V, V0, rtol=1e-13, atol=1e-15):
                break
            
            else:
                V0 = np.copy(V)
        
        for s in range(MDP["S"]):
            Q[s] = np.sum(MDP["T"][s] * MDP["R"][s] + MDP["gamma"] * MDP["T"][s] * V, axis=1)

        improvable_states = []

        for s in range(MDP["S"]):
            if (Q[s, policy[s]] < np.amax(Q[s, :])):
                improvable_states.append(s)
                policy[s] = np.argmax(Q[s, :])
        
        if(len(improvable_states)==0):
            converged = 1
    
    for s in range(0, MDP["S"]):
        print (str(format(V[s], '.6f')) + ' ' + str(policy[s]))

    return policy