import numpy as np
from numpy import array
import matplotlib as mpl
import copy, sys
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import gym
import mytaxi
from maze import Maze

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

def policy_gen(Q,ep,nS,nA):
    policy = {}
    for s in range(nS):
        row = np.zeros(nA)
        actions = _greedy(Q,s)
        row[actions] = (1-ep) / len(actions)
        row = row + np.ones(nA) * ep / nA
        policy[s] = row
    policy[nS] = np.ones(nA) / nA
    return policy

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
        input()
    env.render()
    print(cum_rew)

def _onestep_q(s, a, ss, r, Q, gamma=1, alpha=0.9):
    Q[s,a] = Q[s,a] + alpha * (r + gamma * np.max(Q[ss]) - Q[s,a])

class plan_model(dict):
    def __init__(self,max_mem=10):
        self._max_mem = max_mem
        super().__init__()

    def feed(self,s,a,ss,r):
        # update the model
        action = self.setdefault(s,{})
        entry = action.setdefault(a,[[],[]])
        # this is for deterministic environment, for stochastic case, comment
        # out the if clause 
        # if ss not in entry[1].keys():
        #     entry[0] = 1
        #     entry[1] = {}
        entry[0].append(ss)
        if len(entry[0]) > self._max_mem:
            entry[0].pop(0)
        entry[1].append(r)
        if len(entry[1]) > self._max_mem:
            entry[1].pop(0)
    
    def sample(self):
        state = np.random.choice(list(self.keys()))
        action = np.random.choice(list(self[state].keys()))
        entry = self[state][action]
        idx = np.random.choice(len(entry[0]))
        ss = entry[0][idx]
        r = entry[1][idx]
        return state, action, ss, r

class plan_model_stationary(dict):
    def __init__(self):
        super().__init__()

    def feed(self,s,a,ss,r):
        # update the model
        action = self.setdefault(s,{})
        entry = action.setdefault(a,[0,{},0])
        # this is for deterministic environment, for stochastic case, comment
        # out the if clause 
        # if ss not in entry[1].keys():
        #     entry[0] = 0
        #     entry[1] = {}
        entry[0] += 1
        entry[1][ss] = entry[1].setdefault(ss,0) + 1
        entry[2] = (entry[2] * (entry[0] - 1) + r) / entry[0]

    def sample(self):
        state = np.random.choice(list(self.keys()))
        action = np.random.choice(list(self[state].keys()))
        entry = self[state][action]
        p, ss_list = [],[]
        for key,val in entry[1].items():
            ss_list.append(key)
            p.append(val/entry[0])
        ss = np.random.choice(ss_list,p=p)
        r = entry[2]
        return state, action, ss, r

def dynaq_step(env1,env2,n=10,gamma=1.,alpha=1.,ep=0.1,
        max_steps=100,switch_time=-1,max_mem=10):
    env = env1
    nS = env.nS
    nA = env.nA
    if max_mem > 0:
        model = plan_model(max_mem)
    else:
        model = plan_model_stationary()
    Q = np.zeros((nS,nA))
    tot_steps = 0
    rew_list = np.zeros(max_steps)
    switched = False
    while tot_steps < max_steps:
        if not switched and tot_steps > switch_time > 0:
           env = env2
           switched = True
        s = env.reset()
        done = False
        counter = 0
        while not done:
            if tot_steps == max_steps:
                break
            a = ep_greedy(Q,s,ep)
            ss, r, done, _ = env.step(a)
            _onestep_q(s, a, ss, r, Q, gamma, alpha)
            model.feed(s, a, ss, r)
            s = ss
            counter += 1
            rew_list[tot_steps] = r
            # planning step
            for jdx in range(n):
                s_, a_, ss_, r_ = model.sample()
                _onestep_q(s_, a_, ss_, r_, Q, gamma, alpha)
            print('\rtotal steps {}, episode length {}    '.format(tot_steps,
                counter),end='')
            sys.stdout.flush()
            tot_steps += 1
    print('')
    return Q, rew_list, tot_steps

def dynaq(env1,env2,n=10,gamma=1.,alpha=1.,ep=0.1,
        episodes=500,switch_time=-1,max_mem=10):
    env = env1
    nS = env.nS
    nA = env.nA
    if max_mem > 0:
        model = plan_model(max_mem)
    else:
        model = plan_model_stationary()
    Q = np.zeros((nS,nA))
    tot_steps = 0
    rew_list = np.zeros(episodes)
    steps_list = np.zeros(episodes)
    switched = False
    episode = 0
    while episode < episodes:
        if not switched and episode > switch_time > 0:
           env = env2
           switched = True
        s = env.reset()
        done = False
        counter = 0
        rew = 0
        while not done:
            a = ep_greedy(Q,s,ep)
            ss, r, done, _ = env.step(a)
            _onestep_q(s, a, ss, r, Q, gamma, alpha)
            model.feed(s, a, ss, r)
            s = ss
            counter += 1
            rew += r
            # planning step
            for jdx in range(n):
                s_, a_, ss_, r_ = model.sample()
                _onestep_q(s_, a_, ss_, r_, Q, gamma, alpha)
            print('\repisode {}, episode length {}    '.format(episode,
                counter),end='')
            sys.stdout.flush()
        rew_list[episode] = rew
        steps_list[episode] = counter
        episode += 1
    print('')
    return Q, rew_list, steps_list

def QtoV(Q):
    V = np.zeros(len(Q))
    for idx,row in enumerate(Q):
        V[idx] = np.max(row)
    return V

def action(policy,s):
    if sum(policy[s]) != 1:
        p = policy[s] / sum(policy[s])
    else:
        p = policy[s]
    return np.random.choice(len(policy[0]),p=p)

def plot_dynaq_maze():
    maze1 = Maze()
    maze2 = Maze()
    maze1.START_STATE = [5, 3]
    maze1.GOAL_STATES = [[0, 8]]
    maze2.START_STATE = [5, 3]
    maze2.GOAL_STATES = [[0, 8]]
    old_obstacles = [[3, i] for i in range(0, 8)]
    maze1.obstacles = old_obstacles

    # new obstalces will block the optimal path
    new_obstacles = [[3, i] for i in range(1, 9)]
    maze2.obstacles = new_obstacles

    # obstacles will change after 1000 steps
    # the exact step for changing will be different
    # However given that 1000 steps is long enough for both algorithms to converge,
    # the difference is guaranteed to be very small
    obstacle_switch_time = 1000
    max_steps = 3000
    runs = 50

    avg_rew1 = []
    avg_rew2 = []
    for run in range(runs):
        print('run ',run)
        Q1, rew_list1, tot_steps1 = \
            dynaq_step(maze1, maze2, n=10, alpha=1., gamma=0.95,
                    max_steps=max_steps, switch_time = obstacle_switch_time,
                    max_mem=1)
        avg_rew1.append(rew_list1)
        Q2, rew_list2, tot_steps2 = \
            dynaq_step(maze1, maze2, n=10, alpha=1., gamma=0.95,
                    max_steps=max_steps, switch_time = obstacle_switch_time,
                    max_mem=-1)
        avg_rew2.append(rew_list2)
    avg_rew1 = np.mean(np.array(avg_rew1),axis=0)
    cum_rew1 = [sum(avg_rew1[:i+1]) for i in range(len(avg_rew1))]
    avg_rew2 = np.mean(np.array(avg_rew2),axis=0)
    cum_rew2 = [sum(avg_rew2[:i+1]) for i in range(len(avg_rew2))]
    fig = plt.figure()
    plt.plot(cum_rew1,label='mem-10')
    plt.plot(cum_rew2,label='mem-inf')
    plt.legend()
    plt.title("Cumulative Reward")
    plt.savefig('dynaq_maze.png')
    plt.close(fig)

def plot_q_vs_dynaq():
    env1 = gym.make('Taxi-v4')
    env2 = gym.make('Taxi-v5')
    episodes = 100
    switch_time = -1
    runs = 5
    avg_rew1,avg_rew2 = [], []
    avg_steps1,avg_steps2 = [], []
    for run in range(runs):
        Q1, rew_list1, steps_list1 = \
            dynaq(env1, env2, n=10, alpha=1., gamma=1.0,
                    episodes=episodes,switch_time=switch_time)
        avg_rew1.append(rew_list1)
        avg_steps1.append(steps_list1)
        Q2, rew_list2, steps_list2 = \
            dynaq(env1, env2, n=0, alpha=1., gamma=1.0,
                    episodes=episodes,switch_time=switch_time)
        avg_rew2.append(rew_list2)
        avg_steps2.append(steps_list2)
    avg_rew1 = np.mean(np.array(avg_rew1),axis=0)
    avg_rew2 = np.mean(np.array(avg_rew2),axis=0)
    avg_steps1 = np.mean(np.array(avg_steps1),axis=0)
    avg_steps2 = np.mean(np.array(avg_steps2),axis=0)
    cum_steps1 = [sum(avg_steps1[:idx]) for idx in range(episodes)]
    cum_steps2 = [sum(avg_steps2[:idx]) for idx in range(episodes)]
    fig = plt.figure()
    plt.plot(avg_rew1,label='dyna-q')
    plt.plot(avg_rew2,label='q')
    plt.legend()
    plt.title("Reward Per Episode")
    plt.savefig('dynaq_taxi_rewards.png')
    plt.close(fig)
    fig = plt.figure()
    plt.plot(cum_steps1,label='dyna-q')
    plt.plot(cum_steps2,label='q')
    plt.legend()
    plt.title("Steps Per Episode")
    plt.savefig('dynaq_taxi_steps.png')
    plt.close(fig)
    fig = plt.figure()
    V1 = QtoV(Q1)
    V2 = QtoV(Q2)
    plt.plot(V1,marker='o',linestyle='None',label='dyna-q')
    plt.plot(V2,marker='x',linestyle='None',label='q')
    plt.legend()
    plt.title("Value Function")
    plt.savefig('dynaq_taxi_value.png')
    plt.close(fig)

def plot_non_stationary():
    env1 = gym.make('Taxi-v4')
    env2 = gym.make('Taxi-v5')
    episodes = 300
    switch_time = 100
    runs = 5
    avg_rew1,avg_rew2 = [], []
    avg_steps1,avg_steps2 = [], []
    for run in range(runs):
        Q2, rew_list2, steps_list2 = \
            dynaq(env1, env2, n=10, alpha=1., gamma=1.0,
                    episodes=episodes,switch_time=switch_time,max_mem=-1)
        avg_rew2.append(rew_list2)
        avg_steps2.append(steps_list2)
        Q1, rew_list1, steps_list1 = \
            dynaq(env1, env2, n=10, alpha=1., gamma=1.0,
                    episodes=episodes,switch_time=switch_time,max_mem=10)
        avg_rew1.append(rew_list1)
        avg_steps1.append(steps_list1)
    avg_rew1 = np.mean(np.array(avg_rew1),axis=0)
    avg_rew2 = np.mean(np.array(avg_rew2),axis=0)
    avg_steps1 = np.mean(np.array(avg_steps1),axis=0)
    avg_steps2 = np.mean(np.array(avg_steps2),axis=0)
    cum_steps1 = [sum(avg_steps1[:idx]) for idx in range(episodes)]
    cum_steps2 = [sum(avg_steps2[:idx]) for idx in range(episodes)]
    fig = plt.figure()
    plt.plot(avg_rew1[10:],label='mem-10')
    plt.plot(avg_rew2[10:],label='mem-inf')
    plt.legend()
    plt.title("Reward Per Episode")
    plt.savefig('dynaq_non_stationary_rewards.png')
    plt.close(fig)
    fig = plt.figure()
    plt.plot(cum_steps1,label='mem-10')
    plt.plot(cum_steps2,label='mem-inf')
    plt.legend()
    plt.title("Steps Per Episode")
    plt.savefig('dynaq_non_stationary_steps.png')
    plt.close(fig)
    fig = plt.figure()
    V1 = QtoV(Q1)
    V2 = QtoV(Q2)
    plt.plot(V1,marker='o',linestyle='None',label='mem-10')
    plt.plot(V2,marker='x',linestyle='None',label='mem-inf')
    plt.legend()
    plt.title("Value Function")
    plt.savefig('dynaq_non_stationary_value.png')
    plt.close(fig)
    # policy = policy_gen(Q1,0,env2.nS,env1.nA)
    # testshow(env2,policy)
    
if __name__ == '__main__':
    plot_non_stationary()
