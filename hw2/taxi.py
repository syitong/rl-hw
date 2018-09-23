import numpy as np
from numpy import array
import matplotlib as mpl
import copy
mpl.use('Agg')
import matplotlib.pyplot as plt
import gym
from mdp import reset,policy_eval
from randomwalk import RandomWalk

# env = gym.make('Taxi-v2').unwrapped
# nS = env.observation_space.n
# nA = env.action_space.n
env = RandomWalk()
nS = 7
nA = 2
np.random.seed(3)
# env.seed(5)

def rms(a,b):
    return np.sqrt(np.mean((a-b)**2))

def td0(V_init,policy,baseline,nS=nS,gamma=1,alpha=0.9,episodes=1000):
    V = V_init.copy()
    RMS = []
    for idx in range(episodes):
        s = env.reset()
        done = False
        counter = 0
        while not done:
            a = action(policy,s)
            ss, r, done, _ = env.step(a)
            V[s] = V[s] + alpha * (r + gamma * V[ss] - V[s])
            s = ss
            counter += 1
        print(counter,idx)
        RMS_ep = rms(baseline,V)
        RMS.append(RMS_ep)
    return np.array(RMS), V

def plot_td0(baseline,runs,V_init,policy,
        nS=nS,gamma=1,alpha=0.9,episodes=1000):
    RMS = np.zeros(episodes)
    V = np.zeros(nS)
    for idx in range(runs):
        RMS_run,V_run = td0(V_init,policy,baseline,nS=nS,
            gamma=1,alpha=alpha,episodes=episodes)
        RMS += RMS_run
        V += V_run
        print('run ',idx)
    RMS /= runs
    V /= runs
    fig = plt.figure()
    plt.plot(RMS)
    plt.savefig('td0-error.eps')
    plt.close(fig)
    fig = plt.figure()
    plt.plot(V,marker='o',linestyle='None',label='td')
    plt.plot(baseline,marker='x',linestyle='None',label='base')
    plt.legend(loc=1)
    plt.savefig('td0-qplot.eps')
    plt.close(fig)

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

def testshow(policy,env=env):
    s = env.reset()
    done = False
    cum_rew = 0
    while not done:
        env.render()
        a = action(policy,s)
        ss, r, done, _ = env.step(a)
        s = ss
        cum_rew += r
    print(cum_rew)

if __name__ == '__main__':
    baseline = np.arange(7) / 6
    baseline[-1] = 0
    V_init, policy = reset(nA,nS)
    V_init[1:6] = 0.5
    plot_td0(V_init=V_init,policy=policy,baseline=baseline,gamma=1,alpha=0.15,runs=100,episodes=100)
    # trans_mat = {}
    # for s,pi in env.P.items():
    #     trans_mat[s] = {}
    #     for a,p in pi.items():
    #         trans_mat[s][a] = []
    #         for row in p:
    #             if row[3] == True:
    #                 lrow = list(row)
    #                 lrow[1] = nS
    #                 row = tuple(lrow)
    #             trans_mat[s][a].append(row)

    # trans_mat[nS] = {0: [(1.0,nS,0,True)]}

    # V_init,policy = reset(nA,nS+1)
    # Q, rew_list = qlearn(gamma=1,alpha=0.9,ep=0.1,episodes=100000)
    # np.save('Qtable',Q)
    # policy = policy_gen(Q,0.1)
    # with open('qpolicy','w') as fp:
    #     fp.write(str(policy))
    # params = {
    #     'trans_mat': trans_mat,
    #     'V_init': V_init,
    #     'policy': policy,
    #     'theta': 0.01,
    #     'inplace': False,
    #     'gamma':0.6
    #     }
    # baseline = policy_eval(**params)
    # np.save('qbase',baseline)
    # baseline = np.load('qbase.npy')
    # with open('qpolicy','r') as fp:
    #     policy = eval(fp.read())
    # plot_td0(policy=policy,baseline=baseline[:-1],gamma=0.6,alpha=0.01,runs=1,episodes=10000)
    # plt.plot(rew_list)
    # plt.savefig('qlearn.eps')
    # testshow(policy,env)
