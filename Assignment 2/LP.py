import numpy as np
import pulp

def LPsolver(MDP):

    LPproblem = pulp.LpProblem("V*", pulp.LpMinimize)
    states = range(0, MDP["S"])

    V = np.zeros(MDP["S"], dtype=np.float128)
    Q= np.zeros((MDP["S"], MDP["A"]), dtype=np.float128)

    state_variables = pulp.LpVariable.dicts('vpi', states, cat='Continuous')
    LPproblem += pulp.lpSum([state_variables[s] for s in states])
    
    for s in range(0, MDP["S"]):
        for a in range(0, MDP["A"]):
            LPproblem += state_variables[s] >= pulp.lpSum( (MDP["T"][s, a, s_dash]*( MDP["R"][s, a, s_dash] + MDP["gamma"] * state_variables[s_dash] ) ) for s_dash in range(0, MDP["S"]))

    status = pulp.PULP_CBC_CMD(msg = 0, gapRel = 1e-10).solve(LPproblem)

    for x in range(0,MDP["S"]):
        V[x] = pulp.value(state_variables[x])
    
    action_list = MDP["gamma"]*(np.multiply(V, MDP["T"])).sum(axis=2) + np.multiply(MDP["R"], MDP["T"]).sum(axis=2)
    Pi_optimal = np.argmax(action_list, axis=1)
    
    for s in range(0, MDP["S"]):
        print (str(format(V[s], '.6f')) + ' ' + str(Pi_optimal[s]))
    
    return Pi_optimal