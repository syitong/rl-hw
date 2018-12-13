import numpy as np

def _greedy(Q,s, w_noise, b_noise):
    qmax = np.max(Q(s, w_noise, b_noise))
    actions = []
    for i,q in enumerate(Q(s, w_noise, b_noise)):
        if q == qmax:
            actions.append(i)
    return actions

def greedy(Q,s, w_noise, b_noise):
    return np.random.choice(_greedy(Q, s, w_noise, b_noise))

def ep_greedy(Q,s,w_noise, b_noise,ep):
    # input(Q(s))
    if np.random.rand() < ep:
        return np.random.choice(len(Q(s, w_noise, b_noise)))
    else:
        return greedy(Q,s, w_noise, b_noise)
