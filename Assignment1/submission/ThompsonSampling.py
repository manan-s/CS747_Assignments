import numpy as np

def pull_arm(armID, instance):
    arm = instance[armID]
    return np.sum(np.random.binomial(size = 1, n = 1, p = arm))

def BetaSampler (nOnes, nPulls):
    '''
    Returns ID of the arm, from distribution of which, the sample of largest value is obtained
    '''
    samples = [0]*len(nOnes)

    for i in range(len(nOnes)):
        samples[i] = np.random.beta(nOnes[i] + 1, nPulls[i] - nOnes[i] +1)

    max_sample = max(samples)
    armID = samples.index(max_sample)

    return armID

def ThomsonSampling(instance, horizon, seed):
    np.random.seed(seed)
    nArms = len(instance)
    nPulls = [0]*nArms
    nOnes = [0]*nArms
    reward = 0

    for t in range(horizon):

        armID = BetaSampler(nOnes, nPulls)
        temp_reward = pull_arm(armID, instance)

        nOnes[armID] += temp_reward
        nPulls[armID] += 1
        reward += temp_reward
    
    empirical_means = np.array([i/float(j) if j!=0 else 0 for i,j in zip(nOnes, nPulls)])

    return reward
