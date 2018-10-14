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
        super().__init__()

    def feed(self,s,a,ss,r):
        # update the model
        action = self.setdefault(s,{})
        entry = action.setdefault(a,[0,{},0])
        entry[0] += 1
        # this is for deterministic environment, for stochastic case, comment
        # out the if clause 
        if ss not in entry[1].keys():
            entry[0] = 1
            entry[1] = {}
        entry[1][ss] = entry[1].setdefault(ss,0) + 1
        entry[2] = (entry[2] * (entry[0] - 1) + r) / entry[0]
    
    def sample(self):
        state = np.random.choice(list(self.keys()))
        action = np.random.choice(list(self[state].keys()))
        entry = self[state][action]
        tot = entry[0]
        p, next_s = [], []
        for ss,val in entry[1].items():
            p.append(val / tot)
            next_s.append(ss)
        return state, action, np.random.choice(next_s,p=p), entry[2]

def dynaq(env,n=10,gamma=1.,alpha=1.,ep=0.1,max_steps=100,switch_time=-1):
    nS = env.nS
    nA = env.nA
    model = plan_model()
    Q = np.zeros((nS,nA))
    tot_steps = 0
    rew_list = np.zeros(max_steps)
    env.obstacles = env.old_obstacles
    while tot_steps < max_steps:
        if tot_steps > switch_time > 0:
           env.obstacles = env.new_obstacles
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

if __name__ == '__main__':
    blocking_maze = Maze()
    blocking_maze.START_STATE = [5, 3]
    blocking_maze.GOAL_STATES = [[0, 8]]
    old_obstacles = [[3, i] for i in range(0, 8)]
    blocking_maze.old_obstacles = old_obstacles

    # new obstalces will block the optimal path
    new_obstacles = [[3, i] for i in range(1, 9)]
    blocking_maze.new_obstacles = new_obstacles

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
            dynaq(blocking_maze, n=10, alpha=1., gamma=0.95,
                    max_steps=max_steps, switch_time = obstacle_switch_time)
        avg_rew.append(rew_list)
    avg_rew = np.mean(np.array(avg_rew),axis=0)
    cum_rew = [sum(avg_rew[:i+1]) for i in range(len(avg_rew))]
    fig = plt.figure()
    plt.plot(cum_rew)
    plt.title("Cumulative Reward")
    plt.savefig('dynaq_maze.png')
    plt.close(fig)
