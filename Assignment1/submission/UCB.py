import numpy as np

def pull_arm(armID, instance):
    arm = instance[armID]
    return np.sum(np.random.binomial(size = 1, n = 1, p = arm))

def UCB_function(t, nArms, empirical_means, nPulls):
    '''
    Returns arm ID with highest value of UCB function
    '''
    if t < nArms:
    	return t
    
    else:
        ucb_values = empirical_means + np.sqrt(2 * np.log(t) * np.divide(np.ones([1, nArms]), np.array(nPulls)))
        max_ucb = np.amax(ucb_values)
        indices = np.where(ucb_values == max_ucb)
        return np.amax(indices)

def UCB(instance, horizon, seed):
    np.random.seed(seed)
    nArms = len(instance)
    empirical_means = [0.0]*nArms
    nPulls = [0]*nArms
    nOnes = [0]*nArms
    reward = 0

    for t in range(horizon):
        empirical_means = np.array([i/float(j) if j!=0 else 0 for i,j in zip(nOnes, nPulls)])
        
        armID = UCB_function(t, nArms, empirical_means, nPulls)
        temp_reward = pull_arm(armID, instance)

        nOnes[armID] += temp_reward
        nPulls[armID] += 1
        reward += temp_reward
    
    empirical_means = np.array([i/float(j) if j!=0 else 0 for i,j in zip(nOnes, nPulls)])
    
    return reward