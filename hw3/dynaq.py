import numpy as np
from numpy import array
import matplotlib as mpl
import copy, sys
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import gym
from mdp import reset,policy_eval
from randomwalk import RandomWalk
import mytaxi

def _greedy(Q,s):
    qmax = np.max(Q[s])
    actions = []
    for i,q in enumerate(Q[s]):
        if q == qmax:
            actions.append(i)
    return actions

def greedy(Q,s):
    return np.random.choice(_greedy(Q,s))

def ep_greedy(Q,s,ep):
    if np.random.rand() < ep:
        return np.random.choice(len(Q[s]))
    else:
        return greedy(Q,s)

def testshow(env,policy):
    s = env.reset()
    done = False
    cum_rew = 0
    while not done:
        env.render()
        a = action(policy,s)
        ss, r, done, _ = env.step(a)
        s = ss
        cum_rew += r
    env.render()
    print(cum_rew)

def _onestep_q(s, a, ss, r, Q, gamma=1, alpha=0.9):
    Q[s,a] = Q[s,a] + alpha * (r + gamma * np.max(Q[ss]) - Q[s,a])

def dynaq(env,n=10,gamma=1,alpha=0.9,ep=0.05,episodes=1000):
    nS = env.nS
    nA = env.nA
    rew_alloc = []
    P = {} # {s:{a:(tot_num_visit,[{ss:num_visit},],exp_rew),},}
    Q = np.zeros((nS,nA))
    rew_list = np.zeros(episodes)
    for idx in range(episodes):
        s = env.reset()
        done = False
        counter = 0
        cum_rew = 0
        while not done:
            a = ep_greedy(Q,s,ep)
            s, r, done, _ = env.step(a)
            s_onestep_q(s, a, ss, r, Q, gamma, alpha)
            s = ss
            cum_rew = cum_rew * gamma + r
            counter += 1
        print(counter,idx)
        rew_list[idx] = cum_rew
    rew_alloc.append(rew_list)
    rew_list = np.mean(np.array(rew_alloc),axis=0)
    return Q, rew_list
