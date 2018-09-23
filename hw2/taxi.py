import numpy as np
from numpy import array
import matplotlib as mpl
import copy
# mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import gym
from mdp import reset,policy_eval
from randomwalk import RandomWalk
import mytaxi

def rms(a,b):
    return np.sqrt(np.mean((a-b)**2))

def td0(env,V_init,policy,baseline,gamma=1,alpha=0.9,episodes=1000):
    np.random.seed(3)
    env.seed(5)
    V = V_init.copy()
    nS = env.nS
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

def plot_td0(env,baseline,runs,V_init,policy,
        gamma=1,alpha=0.9,episodes=1000):
    RMS = np.zeros(episodes)
    V = np.zeros(nS)
    for idx in range(runs):
        RMS_run,V_run = td0(env,V_init,policy,baseline,
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
    plt.legend(loc=3)
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
    if sum(policy[s]) != 1:
        p = policy[s] / sum(policy[s])
    else:
        p = policy[s]
    return np.random.choice(nA,p=p)

def qlearn(env,gamma=1,alpha=0.9,ep=0.05,runs=1,episodes=1000):
    np.random.seed(3)
    env.seed(5)
    nS = env.nS
    nA = env.nA
    rew_alloc = []
    for run in range(runs):
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
        rew_alloc.append(rew_list)
    rew_list = np.mean(np.array(rew_alloc),axis=0)
    return Q, rew_list

def QtoV(Q):
    V = np.zeros(len(Q))
    for idx,row in enumerate(Q):
        V[idx] = np.max(row)
    return V

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
    print(cum_rew)

if __name__ == '__main__':
    # Example 6.2 Random Walk
    # env = RandomWalk()
    # baseline = np.arange(7) / 6
    # baseline[-1] = 0
    # V_init, policy = reset(nA,nS)
    # V_init[1:6] = 0.5
    # plot_td0(env=env,V_init=V_init,policy=policy,
    #     baseline=baseline,gamma=1,alpha=0.15,runs=100,episodes=100)

    # Homework 2
    env = gym.make('Taxi-v3').unwrapped
    nA = env.nA
    nS = env.nS
    V_init,policy = reset(nA,nS)
    Q, rew_list = qlearn(env=env,gamma=1,alpha=0.9,ep=0.1,episodes=10000)
    policy = policy_gen(Q,0.1)
    with open('policy','w') as fp:
        fp.write(str(policy))
    params = {
        'trans_mat': env.P,
        'V_init': V_init,
        'policy': policy,
        'theta': 0.01,
        'inplace': False,
        'gamma':1
        }
    baseline = policy_eval(**params)
    np.save('baseline',baseline)
    baseline = np.load('baseline.npy')
    with open('policy','r') as fp:
        policy = eval(fp.read())
    V_init,_ = reset(nA,nS)
    plot_td0(env=env,V_init=V_init,policy=policy,
        baseline=baseline,gamma=1,alpha=0.1,runs=1,episodes=50000)
    Q, rew_list = qlearn(env=env,gamma=1,alpha=0.9,ep=0.1,runs=20,episodes=500)
    np.save('avg_rew',rew_list)
    fig = plt.figure()
    plt.plot(rew_list)
    plt.savefig('qlearn-interim.eps')
    plt.close(fig)
    Q, rew_list = qlearn(env=env,gamma=1,alpha=0.9,ep=0.1,runs=1,episodes=100000)
    np.save('Qtable',Q)
    Vq = QtoV(Q)
    fig = plt.figure()
    plt.plot(baseline,marker='x',linestyle='None',label='baseline')
    plt.plot(Vq,marker='o',linestyle='None',label='Vq')
    plt.savefig('optv.eps')
    plt.close(fig)
    fig = plt.figure()
    plt.plot(rew_list[-100:])
    plt.savefig('qlearn-asym.eps')
    plt.close(fig)
    # testshow(policy,env)
