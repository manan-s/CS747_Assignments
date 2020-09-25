import numpy as np 

def pull_arm(armID, instance):
    arm = instance[armID]
    return np.sum(np.random.binomial(size = 1, n = 1, p = arm))

def BetaSampler (nOnes, nPulls, sorted_TrueMean, empirical_means, horizon):
    '''
    Returns ID of the arm, from distribution of which, the sample of largest value is obtained
    '''
    samples = [0]*len(nOnes)
    sorted_EmpMean = sorted(empirical_means, reverse=True)

    for i in range(len(nOnes)):
        rank_in_sorted_EMP = sorted_EmpMean.index(empirical_means[i])
        true_guess = sorted_TrueMean[rank_in_sorted_EMP]
        if nPulls[i] == 0:
            weight = 0
        else:
            weight = horizon*np.exp(-1*abs(sorted_TrueMean[rank_in_sorted_EMP] - empirical_means[i]))
        samples[i] = np.random.beta(nOnes[i] + (1.0/(nPulls[i]+1))*weight*true_guess + 1, nPulls[i] - nOnes[i] + weight - weight*true_guess + 1)

    max_sample = max(samples)
    armID = samples.index(max_sample)

    return armID

def ThomsonHint (instance, horizon, seed):
    np.random.seed(seed)
    nArms = len(instance)
    nPulls = [0]*nArms
    nOnes = [0]*nArms
    reward = 0

    sorted_TrueMean = sorted(instance, reverse=True)

    for t in range(0, horizon):
        empirical_means = np.array([i/float(j) if j!=0 else 0 for i,j in zip(nOnes, nPulls)])
        armID = BetaSampler(nOnes, nPulls, sorted_TrueMean, empirical_means, horizon)
        temp_reward = pull_arm(armID, instance)

        nOnes[armID] += temp_reward
        nPulls[armID] += 1
        reward += temp_reward
    
    empirical_means = np.array([i/float(j) if j!=0 else 0 for i,j in zip(nOnes, nPulls)])

    return reward
    