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
        output = np.random.choice(self,size)
        return output

def dqn(N, env, ep_start, ep_end, ep_rate, batch_size,
        gamma, a_list, C, lrate, lambda_, criteria, test_episodes=100):
    nA = len(a_list)
    D = memory(N)
    nS = env.observation_space.shape[0]
    model = nn_model(nS, a_list, 'test1', lrate) # implement two networks in one model with an update method.
    Q = model.Q
    Qhat = model.Qhat
    score_list = []
    iter = 0
    episode = learn_starts = 0
    while True:
        s = env.reset()
        done = False
        ep = ep_start - min(ep_rate * (episode - learn_starts), ep_start - ep_end)
        while not done:
            a = ep_greedy(Q, s, ep)
            ss, r, done, _ = env.step(a)
            D.add({'s':s,'a':a,'r':r,'ss':ss,'done':done})
            s = ss
            if iter < batch_size:
                iter += 1
                learn_starts = episode
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
            model.fit(np.array(s_batch), np.array(a_batch), y)
            if iter % 100 == 0:
                loss = model.get_loss(np.array(s_batch), np.array(a_batch), y)
                print('\rtotal iter: {}, episode: {}, loss: {:<.4}     '.format(iter, episode, loss),end='')
                sys.stdout.flush()
            if iter % C == C - 1:
                model.update()
            iter += 1
        if episode % 100 == 0:
            print('')
            score = eval_perform(model, env, test_episodes)
            score_list += [score]
            if score > criteria:
                break
        episode += 1
    model.save()
    return score_list, model

def eval_perform(agent, env, episodes):
    avg_score = 0
    for idx in range(episodes):
        score = 0
        done = False
        s = env.reset()
        while not done:
            # env.render()
            a = ep_greedy(agent.Q, s, 0.)
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
    name = 'CartPole-v0'
    env = gym.make(name)
    N = 10000
    # env = gym.make('MountainCar-v0')
    criteria = 195
    ep_start = 0.5
    ep_end = 0.01
    ep_rate = 0.0001
    batch_size = 32
    gamma = 1.
    a_list = [0,1]
    C = 1
    lrate = 0.0001
    lambda_ = 0.

    t1 = time.process_time()
    score_list, agent = dqn(N, env,
        ep_start, ep_end, ep_rate, batch_size, gamma, a_list, C, lrate, lambda_, criteria)
    t2 = time.process_time()
    print('training time:', t2-t1)
    plot_dqn(score_list, name)
    # agent = nn_model(2, a_list, 'test1', lambda_, lrate, load=True) # implement two networks in one model with an update method.
