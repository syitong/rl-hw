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

# def Qtable(Q, a_list):
#     def proxy(s):
#         Qtable = []
#         for a in a_list:
#             Qtable += [Q(s,a)]
#         return Qtable
#     return proxy

def dqn(N, num_episodes, env,
        ep_start, batch_size,
        gamma, a_list, C, lrate, lambda_, test_rounds=10,
        learning_starts=2000, T=200):
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
        if iter > learning_starts:
            ep = ep_start - min(0.01 * (episode - start_episode), ep_start - 0.1)
        else:
            ep = 1.
        for t in range(T):
            if iter == learning_starts:
                start_episode = episode
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
            if iter > learning_starts:
                model.fit(np.array(s_batch), np.array(a_batch), y)
            s = ss.copy()
            if iter % 100 == 0:
                loss = model.get_loss(np.array(s_batch), np.array(a_batch), y)
            if iter > learning_starts and t % C == C - 1:
                model.update()
            iter += 1
            # print('\repisode: {}, # of steps: {:<5}, loss: {:<.4}, # of success: {}     '.format(
                # episode, t, loss, success),end='')
            # sys.stdout.flush()
            if done:
                break
        if t < 199:
            success += 1
        if episode % 50 == 0:
            num_steps += [eval_perform(model, env, test_rounds)]
    model.save()
    print('\n')
    return num_steps, model

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
            # print('\r', score ,a, '            ', end='')
        # sys.stdout.flush()
        avg_score = (avg_score * idx + score) / (idx + 1)
    print('\n')
    return avg_score

def plot_dqn(num_steps, name):
    fig = plt.figure()
    plt.plot(np.array(num_steps))
    plt.title('Performance of DQN on Mountain Car')
    plt.xlabel('per 50 episodes')
    plt.ylabel('avg # of steps')
    plt.savefig(name+'.eps')
    plt.close()

if __name__ == '__main__':
    prefix = sys.argv[1]
    # np.random.seed(3)
    N = 50000
    num_episodes= 501
    register(
        id='MountainCar-v1',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        max_episode_steps=1001,
        reward_threshold=-110.0,
    )
    env = gym.make('MountainCar-v1')
    ep_start = 1.
    batch_size = 32
    gamma = 1
    a_list = [0,1,2]
    C = 500
    lrate = 0.001
    lambda_ = 0.

    t1 = time.process_time()
    num_steps, agent = dqn(N, num_episodes, env,
        ep_start, batch_size, gamma, a_list, C, lrate, lambda_)
    t2 = time.process_time()
    print('training time:', t2-t1)
    plot_dqn(num_steps, 'test-'+str(prefix))
    # agent = nn_model(2, a_list, 'test1', lambda_, lrate, load=True) # implement two networks in one model with an update method.
