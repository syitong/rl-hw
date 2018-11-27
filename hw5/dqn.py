import sys,time
# from tiles3 import IHT, tiles
import gym
from gym.envs.registration import register
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from model import nn_model
from utils import ep_greedy

class memory(list):
    def __init__(self,length):
        self.length = length
        super().__init__()
    def add(self,x):
        if len(self) < self.length:
            super().append(x)
        elif len(self) == self.length:
            super().pop(0)
            super().append(x)
    def sample(self,size):
        output = []
        for i in range(size):
            output += [np.random.choice(self)]
        return output

def Qtable(Q, a_list):
    def proxy(s):
        Qtable = []
        for a in a_list:
            Qtable += [Q(s,a)]
        return Qtable
    return proxy

def dqn(N, num_episodes, env, ep, batch_size, gamma, a_list, C, lrate, T=5000):
    nA = len(a_list)
    D = memory(N)
    dS = 2
    model = nn_model(dS, a_list) # implement two networks in one model with an update method.
    Q = model.Q
    Qhat = model.Qhat
    num_steps = []
    for episode in range(num_episodes):
        s = env.reset()
        for t in range(T):
            a = ep_greedy(Qtable(Q, [0,1,2]), s, ep)
            ss, r, done, _ = env.step(a)
            D.add({'s':s,'a':a,'r':r,'ss':ss,'done':done})
            batch = D.sample(batch_size)
            y = np.empty(batch_size)
            s_batch = []
            a_batch = []
            for idx in range(batch_size):
                if batch[idx]['done']:
                    y[idx] = batch[idx]['r']
                else:
                    y[idx] = batch[idx]['r'] + \
                        gamma * max([Qhat(batch[idx]['ss'], a) for a in a_list])
                s_batch += [list(batch[idx]['s'])]
                a_batch += [batch[idx]['a']]
            model.fit(np.array(s_batch), np.array(a_batch), y, lrate)
            if t % C == 0:
                model.update()
            if done:
                steps = t
                break
            print('\repisode: {}, # of steps: {}'.format(episode, t),end='')
            sys.stdout.flush()
        num_steps += [steps]
    return num_steps

def plot_dqn(num_steps):
    fig = plt.figure()
    plt.plot(np.array(num_steps))
    plt.title('Performance of DQN on Mountain Car')
    plt.xlabel('episode')
    plt.ylabel('# of steps in log scale')
    plt.savefig('dqn_perform.eps')
    plt.close()

if __name__ == '__main__':
    N = 50
    num_episodes=50
    register(
        id='MountainCar-v1',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        max_episode_steps=2000,
        reward_threshold=-110.0,
    )
    env = gym.make('MountainCar-v1')
    ep = 0.1
    batch_size = 5
    gamma = 1
    a_list = [0,1,2]
    C = 10
    lrate = 0.1

    num_steps = dqn(N, num_episodes, env, ep, batch_size, gamma, a_list, C, lrate)
    plot_dqn(num_steps)
