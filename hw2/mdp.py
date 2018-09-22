import numpy as np
import gym
#from lib.gridworld import GridworldEnv
#from lib.nchain import nchain

def reset(nA,nS):
    POLICY = {}
    pie = np.ones(nA) * 1/nA
    for s in range(nS):
        POLICY[s] = pie
    V_init = np.zeros(nS)
    return V_init, POLICY

def _onestep_q(V, s, a, trans_mat, gamma=1):
    q = 0
    for item in trans_mat[s][a]:
        p, s_next, rew, done = item
        q += p * (rew + gamma * V[s_next])
    return q

def _onestep_v(V, s, policy, trans_mat, gamma=1):
    v = 0
    for a,_ in trans_mat[s].items():
        q = _onestep_q(V, s, a, trans_mat, gamma)
        v += policy[s][a] * q
    delta = np.abs(v-V[s])
    return delta, v

def policy_eval(trans_mat, V_init, policy, theta, gamma=1, inplace=True):
    V = V_init
    U = np.zeros(np.shape(V_init))
    while True:
        delta = 0
        for s in range(len(trans_mat)):
            dd,v = _onestep_v(V, s, policy, trans_mat, gamma)
            delta = max(delta, dd)
            if inplace:
                V[s] = v
            else:
                U[s] = v
        print(delta)
        if not inplace:
            V = U
        if delta < theta:
            return V

def policy_improve(V, s, trans_mat, gamma=1):
    nA = len(trans_mat[s])
    q = np.ones(nA) * -np.inf
    for a,_ in trans_mat[s].items():
        q[a] = _onestep_q(V, s, a, trans_mat, gamma)
    aa = (q == max(q))
    aa = aa / sum(aa)
    return aa, q

def policy_iter(trans_mat, V_init, policy, theta, gamma=1, inplace=True):
    is_stable = False
    while not is_stable:
        is_stable = True
        V = policy_eval(trans_mat, V_init, policy, theta, gamma, inplace=inplace)
        for s in range(len(trans_mat)):
            aa, _ = policy_improve(V, s, trans_mat, gamma)
            if not np.array_equal(policy[s], aa):
                policy[s] = aa
                is_stable = False
    return V, policy

def value_iter(trans_mat, V_init, theta, gamma=1, inplace=True):
    V = V_init
    U = np.zeros(np.shape(V_init))
    next_iter = True
    policy = {}
    while next_iter:
        delta = 0
        for s in range(len(trans_mat)):
            aa, q = policy_improve(V, s, trans_mat, gamma)
            delta = max(delta, np.abs(V[s] - max(q)))
            if inplace:
                V[s] = max(q)
            else:
                U[s] = max(q)
        if not inplace:
            V = U
        if delta < theta:
            next_iter = False
    for s in range(len(trans_mat)):
        policy[s] = np.zeros(len(trans_mat[s]))
        aa, _ = policy_improve(V, s, trans_mat, gamma)
        policy[s] = aa
    return V, policy

def print_pol(policy,nrow,nA,a_list='nesw'):
    ncol = len(policy) // nrow
    for idx in range(nrow):
        for jdx in range(ncol):
            output = ''
            index = idx*nrow+jdx
            for kdx in range(nA):
                if policy[index][kdx] != 0:
                    output += a_list[kdx]
            print('{0:^{1}}'.format(output,len(a_list)+2),end='')
        print('\n')

