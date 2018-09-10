import numpy as np
import gym
from lib.gridworld import GridworldEnv
from lib.nchain import nchain

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
    qq = -np.inf
    for a,_ in trans_mat[s].items():
        q = _onestep_q(V, s, a, trans_mat, gamma)
        if q > qq:
            aa = a
            qq = q
    return aa, qq

def policy_iter(trans_mat, V_init, policy, theta, gamma=1):
    is_stable = False
    while not is_stable:
        is_stable = True
        V = policy_eval(trans_mat, V_init, policy, theta, gamma)
        for s in range(len(trans_mat)):
            aa, _ = policy_improve(V, s, trans_mat, gamma)
            if policy[s][aa] < 1:
                policy[s] = np.zeros(len(trans_mat[s]))
                policy[s][aa] = 1
                is_stable = False
    return V, policy

def value_iter(trans_mat, V_init, theta, gamma=1):
    V = V_init
    next_iter = True
    policy = {}
    while next_iter:
        delta = 0
        for s in range(len(trans_mat)):
            aa, qq = policy_improve(V, s, trans_mat, gamma)
            delta = max(delta, np.abs(V[s] - qq))
            V[s] = qq
        if delta < theta:
            next_iter = False
    for s in range(len(trans_mat)):
        policy[s] = np.zeros(len(trans_mat[s]))
        aa, _ = policy_improve(V, s, trans_mat, gamma)
        policy[s][aa] = 1
    return V, policy

if __name__ == '__main__':
    # env = gym.make('FrozenLake-v0').unwrapped
    env = GridworldEnv(slip=0.2, episodic=True)
    trans_mat = env.P
    nA = env.nA
    # _, nA, trans_mat, _ = nchain(episodic=True)
    POLICY = {}
    pie = np.ones(nA) * 1/nA
    for s in range(len(trans_mat)):
        POLICY[s] = pie
    V_init = np.zeros(len(trans_mat))

    # V = policy_eval(trans_mat, V_init, POLICY, theta = 0.0001, gamma=0.9)
    V, policy = policy_iter(trans_mat, V_init, POLICY, theta = 0.0001, gamma = 0.9)
    # V, policy = value_iter(trans_mat, V_init, theta = 0.0001, gamma = 0.8)
    np.set_printoptions(precision=2)
    print(V.reshape((5,-1)))
    print(np.array([np.argmax(p) for _,p in policy.items()]).reshape((5,-1)))
