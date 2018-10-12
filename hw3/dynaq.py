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

def model(P,s,a):
    p , keys = [], []
    tot = P[(s,a)][0]
    for key,val in P[(s,a)][1].items():
        p.append(val / tot)
        keys.append(key)
    return np.random.choice(keys,p=p)

def dynaq(env,n=10,gamma=1,alpha=0.9,ep=0.05,episodes=1000):
    nS = env.nS
    nA = env.nA
    rew_alloc = []
    P = {} # {(s,a):(tot_num_visit,{ss:num_visit},exp_rew)}
    Q = np.zeros((nS,nA))
    tot_steps = 0
    rew_list = np.zeros(episodes)
    for idx in range(episodes):
        s = env.reset()
        done = False
        counter = 0
        cum_rew = 0
        while not done:
            a = ep_greedy(Q,s,ep)
            ss, r, done, _ = env.step(a)
            _onestep_q(s, a, ss, r, Q, gamma, alpha)
            # update the model
            P.setdefault((s,a),(0,{},0))
            P[(s,a)][0] += 1
            P[(s,a)][1][ss] = P[(s,a)][1].setdefault(ss,0) + 1
            P[(s,a)][2] = (P[(s,a)][2] * (P[(s,a)][0] - 1) + r) / P[(s,a)][0]
            cum_rew +=  gamma**counter * r
            s = ss
            counter += 1
            tot_steps += 1
            pairs = list(P.keys())
            # planning step
            for jdx in range(n):
                s,a = pairs(np.random.choice(len(pairs)))
                ss = model(P,s,a)
                r = P[(s,a)][2]
                _onestep_q(s,a,ss,r,Q)
        print(counter,idx)
        rew_list[idx] = cum_rew
    rew_alloc.append(rew_list)
    rew_list = np.mean(np.array(rew_alloc),axis=0)
    return Q, rew_list, tot_steps
