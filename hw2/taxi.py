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

def rms(a,b):
    return np.sqrt(np.mean((a-b)**2))

def td0(env,V_init,policy,baseline,gamma=1,alpha=0.9,episodes=1000,runs=1):
    nS = env.nS
    np.random.seed(4)
    env.seed(5)
    RMS = np.zeros((runs,episodes))
    for run in range(runs):
        V = V_init.copy()
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
            RMS[run,idx] = RMS_ep
    RMS = np.mean(RMS,axis=0)
    return RMS, V

def plot_rms(rms_dict):
    fig = plt.figure()
    for key,val in rms_dict.items():
        plt.plot(val,label=key)
    plt.savefig('rms-error.eps')
    plt.close(fig)

def plot_value(baseline,V,label):
    fig = plt.figure()
    plt.plot(V,marker='o',linestyle='None',label=label)
    plt.plot(baseline,marker='x',linestyle='None',label='base')
    plt.legend()
    plt.savefig('value-plot.eps')
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
                cum_rew = cum_rew * gamma + r
                counter += 1
            print(counter,idx)
            rew_list[idx] = cum_rew
        rew_alloc.append(rew_list)
    rew_list = np.mean(np.array(rew_alloc),axis=0)
    return Q, rew_list

def mc_control(env,gamma=1,ep=0.1,runs=1,episodes=1000,T=0):
    np.random.seed(3)
    env.seed(5)
    nA = env.nA
    nS = env.nS
    rew_alloc = []
    for run in range(runs):
        Q = np.zeros((nS,nA))
        Q1 = np.zeros((nS,nA))
        C = np.zeros((nS,nA))
        rew_list = np.zeros(episodes)
        for idx in range(episodes):
            s = env.reset()
            done = False
            counter = 0
            cum_rew = 0
            G = np.zeros((nS,nA))
            qset = set()
            while not done:
                if T and counter > T:
                    break
                a = ep_greedy(Q1,s,ep)
                if (s,a) not in qset:
                    G[s,a] = 0
                    qset.add((s,a))
                # if do not terminate immature episode, we consider
                # immediate reward in Q table to alter the agent's
                # bad behaviour.
                elif not T:
                    Q1[s,a] = (C[s,a] * Q[s,a] + G[s,a]) / (C[s,a]+1)
                ss, r, done, _ = env.step(a)
                s = ss
                G = gamma * G + r
                cum_rew = cum_rew * gamma + r
                # input('{},{}'.format(a,cum_rew))
                counter += 1
            print('\repisode {} steps {}     '.format(idx,counter),end='')
            sys.stdout.flush()
            for (s,a) in qset:
                Q[s,a] = (C[s,a] * Q[s,a] + G[s,a]) / (C[s,a]+1)
                C[s,a] += 1
            Q1 = copy.deepcopy(Q)
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
    env.render()
    print(cum_rew)

if __name__ == '__main__':
    # Example 6.2 Random Walk
    # env = RandomWalk()
    # nA = env.nA
    # nS = env.nS
    # baseline = np.arange(7) / 6
    # baseline[-1] = 0
    # V_init, policy = reset(nA,nS)
    # V_init[1:6] = 0.5
    # rms,V = td0(env=env,V_init=V_init,policy=policy,
    #     baseline=baseline,gamma=1,alpha=0.15,episodes=100,runs=100)
    # rms_dict = {'td0':rms}
    # plot_rms(rms_dict)
    # plot_value(baseline,V,'td0')

    # Homework 2
    env = gym.make('Taxi-v3').unwrapped
    nA = env.nA
    nS = env.nS
    # V_init,policy = reset(nA,nS)
    # Q, rew_list = qlearn(env=env,gamma=1,alpha=0.9,ep=0.1,episodes=10000)
    # policy = policy_gen(Q,0.1)
    # with open('policy','w') as fp:
    #     fp.write(str(policy))
    # params = {
    #     'trans_mat': env.P,
    #     'V_init': V_init,
    #     'policy': policy,
    #     'theta': 0.01,
    #     'inplace': False,
    #     'gamma':1
    #     }
    # baseline = policy_eval(**params)
    # np.save('baseline',baseline)
    # baseline = np.load('baseline.npy')
    policy = np.load('policy.npy')
    # V_init,_ = reset(nA,nS)
    # plot_td0(env=env,V_init=V_init,policy=policy,
    #     baseline=baseline,gamma=1,alpha=0.1,runs=1,episodes=50000)
    # Q, rew_list = qlearn(env=env,gamma=1,alpha=0.9,ep=0.1,runs=20,episodes=500)
    # np.save('avg_rew',rew_list)
    # fig = plt.figure()
    # plt.plot(rew_list)
    # plt.savefig('qlearn-interim.eps')
    # plt.close(fig)
    # Q, rew_list = qlearn(env=env,gamma=1,alpha=0.9,ep=0.1,runs=1,episodes=100000)
    # np.save('Qtable',Q)
    # Vq = QtoV(Q)
    # fig = plt.figure()
    # plt.plot(baseline,marker='x',linestyle='None',label='baseline')
    # plt.plot(Vq,marker='o',linestyle='None',label='Vq')
    # plt.savefig('optv.eps')
    # plt.close(fig)
    # fig = plt.figure()
    # plt.plot(rew_list[-100:])
    # plt.savefig('qlearn-asym.eps')
    # plt.close(fig)
    testshow(env,policy)

    # mc control
    # env = gym.make('Taxi-v3').unwrapped
    # nA = env.nA
    # nS = env.nS
    # Q, rew_list = mc_control(env=env,gamma=1,
    #     ep=0.1,runs=10,episodes=1000)
    # np.save('rew_list',rew_list)
    # fig = plt.figure()
    # plt.plot([rew for idx,rew in enumerate(rew_list) if idx % 50 == 0])
    # plt.savefig('mc_control.eps')
    # plt.close(fig)
