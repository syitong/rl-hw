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

# def Qtable(Q, a_list):
#     def proxy(s):
#         Qtable = []
#         for a in a_list:
#             Qtable += [Q(s,a)]
#         return Qtable
#     return proxy

def dqn(N, num_episodes, env,
        ep_start, batch_size,
        gamma, a_list, C, lrate, lambda_, T=5000):
    nA = len(a_list)
    D = memory(N)
    dS = 2
    model = nn_model(dS, a_list, 'test1', lambda_, lrate) # implement two networks in one model with an update method.
    Q = model.Q
    Qhat = model.Qhat
    num_steps = []
    success = 0
    iter = 0
    for episode in range(num_episodes):
        s = env.reset()
        ep = 0.1 + (ep_start-0.1) / (success + 1)
        for t in range(T):
            a = ep_greedy(Q, s, ep)
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
                        gamma * max(Qhat(batch[idx]['ss']))
                s_batch += [batch[idx]['s']]
                a_batch += [batch[idx]['a']]
            model.fit(np.array(s_batch), np.array(a_batch), y)
            s = ss.copy()
            if iter % 100 == 0:
                loss = model.get_loss(np.array(s_batch), np.array(a_batch), y)
            if t % C == C - 1:
                model.update()
            iter += 1
            print('\repisode: {}, # of steps: {:<8}, loss: {:<4}'.format(episode, t, loss),end='')
            if done:
                break
            sys.stdout.flush()
        if t < 4000:
            success += 1
        num_steps += [t]
    model.save()
    print('\n')
    return num_steps

def eval_perform(agent, env, rounds):
    avg_score = 0
    for idx in range(rounds):
        score = 0
        done = False
        s = env.reset()
        while not done:
            a = ep_greedy(agent.Q, s, 0.)
            ss, r, done, _ = env.step(a)
            score += r
            s = ss.copy()
            print('\r', score ,a, '            ', end='')
        sys.stdout.flush()
        avg_score = (avg_score * idx + score) / (idx + 1)
    print('\n')
    return avg_score

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
        max_episode_steps=5000,
        reward_threshold=-110.0,
    )
    env = gym.make('MountainCar-v1')
    ep_start = 0.5
    batch_size = 5
    gamma = 1
    a_list = [0,1,2]
    C = 50
    lrate = 0.00002
    lambda_ = 0.

    num_steps = dqn(N, num_episodes, env,
        ep_start, batch_size, gamma, a_list, C, lrate, lambda_)
    plot_dqn(num_steps)
    agent = nn_model(2, a_list, 'test1', lambda_, lrate, load=True) # implement two networks in one model with an update method.
    rounds = 10
    print(eval_perform(agent, env, rounds))
