# Generate the transition matrix and rewards for the nchain problem
# It supports both continuing and episodic versions
import numpy as np
BACKWARD = 0
FORWARD = 1

def nchain(nS=5, slip=0.2, small=2, large=10, episodic=False):
    nA = 2
    isd = np.ones(nS) / sum(np.ones(nS))
    P = {s : {a : [] for a in range(nA)} for s in range(nS)}
    for s,val1 in P.items():
        for a,val2 in val1.items():
            p = slip if a == 0 else 1 - slip
            val2.append((1-p, 0, small, episodic))
            if s < nS - 2:
                val2.append((p, s+1, 0, False))
            else:
                val2.append((p, nS-1, large, episodic))
    if episodic:
        P[0][0] = [(1, 0, 0, True)]
        P[0][1] = [(1, 0, 0, True)]
        P[nS-1][0] = [(1, nS-1, 0, True)]
        P[nS-1][1] = [(1, nS-1, 0, True)]

    return nS, nA, P, isd
