import numpy as np

def pull_arm(armID, instance):
    arm = instance[armID]
    return np.sum(np.random.binomial(size = 1, n = 1, p = arm))

def KL(p, q):
    if p == 0:
        return (1-p)*np.log((1-p)/(1-q))

    elif p == 1:
        return p*np.log(p/q)

    else:
        return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))


def find_q(lhs, rhs, value, mean):
    '''
    Finds the maximum value 'q' between [lhs, rhs] such 
    that KL(mean, q) <= value through binary search
    '''
    mid = 0.5*(lhs+rhs)

    if (rhs - lhs) < 1e-3:
        return lhs
    
    if 0 <= (value - KL(mean, mid)) <= 1e-3:
        return mid
    
    elif KL(mean, mid) > value:
        return find_q(lhs, mid, value, mean)
    
    elif KL(mean, mid) < value:
        return find_q(mid, rhs, value, mean)


def KL_UCB_function(t, nArms, empirical_means, nPulls):
    '''
    Returns arm ID with highest value of UCB function
    '''
    if t < nArms:
    	return t

    kl_ucb_values = np.zeros(nArms, dtype=float)

    for i in range(nArms):
        value = (np.log(t) + 3*np.log(np.log(t)))/nPulls[i]
        kl_ucb_values[i] = find_q(empirical_means[i], 1, value, empirical_means[i])

    max_ucb = np.amax(kl_ucb_values)
    indices = np.where(kl_ucb_values == max_ucb)
    return np.amax(indices)

def KL_UCB(instance, horizon, seed):
    np.random.seed(seed)
    nArms = len(instance)
    empirical_means = [0.0]*nArms
    nPulls = [0]*nArms
    nOnes = [0]*nArms
    reward = 0

    for t in range(horizon):
        #print(t)
        empirical_means = np.array([i/float(j) if j!=0 else 0 for i,j in zip(nOnes, nPulls)])
        
        armID = KL_UCB_function(t, nArms, empirical_means, nPulls)
        temp_reward = pull_arm(armID, instance)

        nOnes[armID] += temp_reward
        nPulls[armID] += 1
        reward += temp_reward

    empirical_means = np.array([i/float(j) if j!=0 else 0 for i,j in zip(nOnes, nPulls)])
    
    return reward