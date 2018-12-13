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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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
        output = np.random.choice(self,size)
        return output

def dqn(N, env, ep_start, ep_end, ep_rate, batch_size,
        gamma, a_list, C, lrate, lambda_, criteria, test_episodes=10,
        learn_starts=5):
    nA = len(a_list)
    D = memory(N)
    nS = env.observation_space.shape[0]
    model = nn_model(nS, a_list, 'test1', lrate) # implement two networks in one model with an update method.
    Q = model.Q
    Qhat = model.Qhat
    score_list = []
    iter = 0
    episode = 0
    while True:
        s = env.reset()
        done = False
        ep = ep_start - min(ep_rate * (max(episode - learn_starts, 0)), ep_start - ep_end)
        w_noise = np.random.randn(64,len(a_list)) * ep
        b_noise = np.random.randn(len(a_list)) * ep
        while not done:
            a = ep_greedy(Q, s, w_noise, b_noise, 0)
            ss, r, done, _ = env.step(a)
            D.add({'s':s,'a':a,'r':r,'ss':ss,'done':done})
            s = ss
            if episode < learn_starts:
                iter += 1
                continue
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
            model.fit(np.array(s_batch), np.array(a_batch), y, w_noise, b_noise)
            if iter % 100 == 0:
                loss = model.get_loss(np.array(s_batch), np.array(a_batch), y,
                        w_noise, b_noise)
                print('\rtotal iter: {}, episode: {}, loss: {:<.4}     '.format(iter, episode, loss),end='')
                sys.stdout.flush()
            if iter % C == C - 1:
                model.update()
            iter += 1
        if episode % 50 == 0:
            print('')
            score = eval_perform(model, env, test_episodes, a_list)
            score_list += [score]
            if score > criteria:
                break
        episode += 1
    model.save()
    return score_list, model

def eval_perform(agent, env, episodes, a_list):
    avg_score = 0
    w_noise = np.zeros((64,len(a_list)))
    b_noise = np.zeros(len(a_list))
    for idx in range(episodes):
        score = 0
        done = False
        s = env.reset()
        while not done:
            env.render()
            a = ep_greedy(agent.Q, s, w_noise, b_noise, 0.)
            ss, r, done, _ = env.step(a)
            score += r
            s = ss
        avg_score = (avg_score * idx + score) / (idx + 1)
    print('avg_score: {}'.format(avg_score))
    return avg_score

def plot_dqn(num_steps, name):
    fig = plt.figure()
    plt.plot(np.array(num_steps))
    plt.title('Performance of DQN on Mountain Car')
    plt.xlabel('per 100 episodes')
    plt.ylabel('score in testing')
    plt.savefig(name+'.eps')
    plt.close()

if __name__ == '__main__':
    np.random.seed(3)
    name = 'MountainCar-v0'
    env = gym.make(name)
    N = 10000
    criteria = -110
    ep_start = 1.
    ep_end = 0.1
    ep_rate = 0.005
    batch_size = 32
    gamma = 1.
    a_list = [0,1,2]
    C = 500
    lrate = 0.001
    lambda_ = 0.

    t1 = time.process_time()
    score_list, agent = dqn(N, env,
        ep_start, ep_end, ep_rate, batch_size, gamma, a_list, C, lrate, lambda_, criteria)
    t2 = time.process_time()
    print('training time:', t2-t1)
    plot_dqn(score_list, name)
    # agent = nn_model(2, a_list, 'test1', lambda_, lrate, load=True) # implement two networks in one model with an update method.
