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

class plan_model(dict):
    def __init__(self):
        self._max_mem = 10
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

def dynaq_step(env1,env2,n=10,gamma=1.,alpha=1.,ep=0.1,max_steps=100,switch_time=-1):
    env = env1
    nS = env.nS
    nA = env.nA
    model = plan_model()
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

def dynaq(env1,env2,n=10,gamma=1.,alpha=1.,ep=0.1,episodes=500,switch_time=-1):
    env = env1
    nS = env.nS
    nA = env.nA
    model = plan_model()
    Q = np.zeros((nS,nA))
    tot_steps = 0
    rew_list = np.zeros(episodes)
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
        episode += 1
    print('')
    return Q, rew_list

def QtoV(Q):
    V = np.zeros(len(Q))
    for idx,row in enumerate(Q):
        V[idx] = np.max(row)
    return V

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
    runs = 20

    avg_rew = []
    for run in range(runs):
        Q, rew_list, tot_steps = \
            dynaq_step(maze1, maze2, n=10, alpha=1., gamma=0.95,
                    max_steps=max_steps, switch_time = obstacle_switch_time)
        avg_rew.append(rew_list)
    avg_rew = np.mean(np.array(avg_rew),axis=0)
    cum_rew = [sum(avg_rew[:i+1]) for i in range(len(avg_rew))]
    fig = plt.figure()
    plt.plot(cum_rew)
    plt.title("Cumulative Reward")
    plt.savefig('dynaq_maze.png')
    plt.close(fig)

if __name__ == '__main__':
    env1 = gym.make('Taxi-v4')
    env2 = gym.make('Taxi-v5')
    episodes = 1000
    switch_time = 950
    runs = 1
    avg_rew1,avg_rew2 = [], []
    for run in range(runs):
        Q1, rew_list1 = \
            dynaq(env1, env2, n=10, alpha=1., gamma=1.0,
                    episodes=episodes,switch_time=switch_time)
        avg_rew1.append(rew_list1)
        Q2, rew_list2 = \
            dynaq(env1, env2, n=0, alpha=1., gamma=1.0,
                    episodes=episodes,switch_time=switch_time)
        avg_rew2.append(rew_list2)
    avg_rew1 = np.mean(np.array(avg_rew1),axis=0)
    avg_rew2 = np.mean(np.array(avg_rew2),axis=0)
    fig = plt.figure()
    plt.plot(avg_rew1,label='dyna-q')
    plt.plot(avg_rew2,label='q')
    plt.legend()
    plt.title("Reward Per Episode")
    plt.savefig('dynaq_taxi.png')
    plt.close(fig)
    # V1 = QtoV(Q1)
    # V2 = QtoV(Q2)
    # fig = plt.figure()
    # plt.plot(V1,marker='o',linestyle='None',label='dyna-q')
    # plt.plot(V2,marker='x',linestyle='None',label='q')
    # plt.legend()
    # plt.title("Value Function")
    # plt.savefig('dynaq_taxi_Q.png')
    # plt.close(fig)
