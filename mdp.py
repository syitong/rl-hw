import numpy as np
import gym
from lib.gridworld import GridworldEnv
from lib.nchain import nchain

def reset(nA):
    POLICY = {}
    pie = np.ones(nA) * 1/nA
    for s in range(len(trans_mat)):
        POLICY[s] = pie
    V_init = np.zeros(len(trans_mat))
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
    V[s] = v
    return delta

def policy_eval(trans_mat, V_init, policy, theta, gamma=1):
    V = V_init
    while True:
        delta = 0
        for s in range(len(trans_mat)):
            dd = _onestep_v(V, s, policy, trans_mat, gamma)
            delta = max(delta, dd)
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

def policy_iter(trans_mat, V_init, policy, theta, gamma=1):
    is_stable = False
    while not is_stable:
        is_stable = True
        V = policy_eval(trans_mat, V_init, policy, theta, gamma)
        for s in range(len(trans_mat)):
            aa, _ = policy_improve(V, s, trans_mat, gamma)
            if not np.array_equal(policy[s], aa):
                policy[s] = aa
                is_stable = False
    return V, policy

def value_iter(trans_mat, V_init, theta, gamma=1):
    V = V_init
    next_iter = True
    policy = {}
    while next_iter:
        delta = 0
        for s in range(len(trans_mat)):
            aa, q = policy_improve(V, s, trans_mat, gamma)
            delta = max(delta, np.abs(V[s] - max(q)))
            V[s] = max(q)
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

if __name__ == '__main__':
    env = GridworldEnv(slip=0.2, episodic=False)
    trans_mat = env.P
    nA = env.nA
    np.set_printoptions(precision=2)

    V_init, POLICY = reset(nA)
    V = policy_eval(trans_mat, V_init, POLICY, theta = 0.0001, gamma=0.9)
    print('uniformly random policy evaluation:')
    print(V.reshape(5,-1))

    V_init, POLICY = reset(nA)
    V, policy = policy_iter(trans_mat, V_init, POLICY, theta = 0.0001, gamma = 0.9)
    print('optimal value function after policy iteration')
    print(V.reshape(5,-1))
    print('optimal policy after policy iteration')
    print_pol(policy,5,4)

    V_init, POLICY = reset(nA)
    V, policy = value_iter(trans_mat, V_init, theta = 0.0001, gamma = 0.9)
    print('optimal value function after value iteration')
    print(V.reshape(5,-1))
    print('optimal policy after value iteration')
    print_pol(policy,5,4)

    env = GridworldEnv(slip=0.2, episodic=True)
    trans_mat = env.P
    V_init, POLICY = reset(nA)
    V, policy = value_iter(trans_mat, V_init, theta = 0.0001, gamma = 1.)
    print('optimal value function after value iteration')
    print(V.reshape(5,-1))
    V_init, POLICY = reset(nA)
    V, policy = value_iter(trans_mat, V_init, theta = 0.0001, gamma = 0.9)
    print('optimal value function after value iteration')
    print(V.reshape(5,-1))
    V_init, POLICY = reset(nA)
    V, policy = value_iter(trans_mat, V_init, theta = 0.0001, gamma = 0.8)
    print('optimal value function after value iteration')
    print(V.reshape(5,-1))
