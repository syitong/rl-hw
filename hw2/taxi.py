import numpy as np
from numpy import array
import matplotlib as mpl
import copy
mpl.use('Agg')
import matplotlib.pyplot as plt
import gym
from mdp import reset,policy_eval

env = gym.make('Taxi-v2').unwrapped
nS = env.observation_space.n
nA = env.action_space.n
np.random.seed(3)
env.seed(5)

def rms(a,b):
    return np.sqrt(np.mean((a-b)**2))

def td0(policy,baseline,nS=nS,gamma=1,alpha=0.9,episodes=1000):
    V = np.zeros(nS)
    RMS = []
    for idx in range(episodes):
        s = env.reset()
        done = False
        counter = 0
        RMS_ep = 0
        while not done:
            a = action(policy,s)
            ss, r, done, _ = env.step(a)
            V[s] = V[s] + alpha * (r + gamma * V[ss] - V[s])
            s = ss
            RMS_ep += rms(baseline,V)
            counter += 1
        print(counter,idx)
        RMS.append(RMS_ep / counter)
    return np.array(RMS), V

def plot_td0(baseline,runs,policy,
        nS=nS,gamma=1,alpha=0.9,episodes=1000):
    RMS = np.zeros(episodes)
    for idx in range(runs):
        RMS_run, _ = td0(policy,baseline,nS=nS,
            gamma=1,alpha=alpha,episodes=episodes)
        RMS += RMS_run
        print('run ',idx)
    RMS /= runs
    plt.plot(RMS)
    plt.savefig('td0.eps')

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

def policy_gen(Q,ep):
    policy = {}
    for s in range(nS):
        row = np.zeros(nA)
        actions = _greedy(Q,s)
        row[actions] = (1-ep) / len(actions)
        row = row + np.ones(nA) * ep / nA
        policy[s] = row
    policy[nS] = np.ones(nA) / nA
    return policy

def action(policy,s):
    p = policy[s] / sum(policy[s])
    return np.random.choice(nA,p=p)

def qlearn(nS=nS,nA=nA,gamma=1,alpha=0.9,ep=0.05,episodes=1000):
    Q = np.zeros((nS,nA))
    rew_list = np.zeros(episodes)
    for idx in range(episodes):
        s = env.reset()
        done = False
        counter = 0
        cum_rew = 0
        while not done:
            a = ep_greedy(Q,s,ep)
            ss, r, done, _ = env.step(a)
            Q[s,a] = Q[s,a] + alpha * (r + gamma * np.max(Q[ss]) - Q[s,a])
            s = ss
            cum_rew += r
            counter += 1
        print(counter,idx)
        rew_list[idx] = cum_rew
    return Q, rew_list

def testshow(Q,env=env):
    s = env.reset()
    done = False
    cum_rew = 0
    while not done:
        env.render()
        a = greedy(Q,s)
        ss, r, done, _ = env.step(a)
        s = ss
        cum_rew += r
    print(cum_rew)

if __name__ == '__main__':

    # trans_mat = {}
    # for s,pi in env.P.items():
    #     trans_mat[s] = {}
    #     for a,p in pi.items():
    #         trans_mat[s][a] = []
    #         for row in p:
    #             if row[3] == True:
    #                 lrow = list(row)
    #                 lrow[1] = 500
    #                 row = tuple(lrow)
    #             trans_mat[s][a].append(row)

    # trans_mat[500] = {0: [(1.0,500,0,True)]}

    # V_init,_ = reset(nA,nS+1)
    # Q, rew_list = qlearn(gamma=1,alpha=0.9,ep=0.1,episodes=1000)
    # policy = policy_gen(Q,0.1)
    # params = {
    #     'trans_mat': trans_mat,
    #     'V_init': V_init,
    #     'policy': policy,
    #     'theta': 0.01,
    #     'inplace': False
    #     }
    # baseline = policy_eval(**params)
    # np.save('qbase',baseline)
    # with open('qpolicy','w') as fp:
    #     fp.write(str(policy))
    baseline = np.load('qbase.npy')
    with open('qpolicy','r') as fp:
        policy = eval(fp.read())
    plot_td0(policy=policy,baseline=baseline[:-1],alpha=0.01,runs=1,episodes=1000)
    # plt.plot(rew_list)
    # plt.savefig('qlearn.eps')
    # testshow(Q,env)
