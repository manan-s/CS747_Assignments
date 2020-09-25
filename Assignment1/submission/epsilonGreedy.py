import numpy as np 

def pull_arm(armID, instance):
    arm = instance[armID]
    return np.sum(np.random.binomial(size = 1, n = 1, p = arm))

def epsilon_greedy(instance, epsilon, horizon, seed):
    np.random.seed(seed)
    nArms = len(instance)
    empirical_means = [0.0]*nArms
    nPulls = [0]*nArms
    nOnes = [0]*nArms
    reward = 0
    
    for t in range(horizon):
        
        if np.sum(np.random.binomial(size = 1, n = 1, p = epsilon)) == 1:
    
            armID = np.random.randint(0, nArms)
            nPulls[armID] += 1
            temp_reward = pull_arm(armID, instance)
            reward += temp_reward
            nOnes[armID] += temp_reward
            empirical_means[armID] = (nOnes[armID])/(1.0*nPulls[armID])
        
        else:
        
            max_mean = max(empirical_means)
            armID = empirical_means.index(max_mean)
            nPulls[armID] += 1
            temp_reward = pull_arm(armID, instance)
            reward += temp_reward
            nOnes[armID] += temp_reward
            empirical_means[armID] = (nOnes[armID])/(1.0*nPulls[armID])
    
    return reward
