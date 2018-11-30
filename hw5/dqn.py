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

def dqn(N, max_iter, env,
        ep_start, batch_size,
        gamma, a_list, C, lrate, lambda_, test_rounds=10,
        learning_starts=1000, T=300, epsilon_rate=0.1):
    nA = len(a_list)
    D = memory(N)
    dS = 4
    model = nn_model(dS, a_list, 'test1', lambda_, lrate) # implement two networks in one model with an update method.
    model.update()
    Q = model.Q
    Qhat = model.Qhat
    num_steps = []
    success = 0
    learning_starts = False
    iter = 0
    while iter < max_iter:
        s = env.reset()
        done = False
        total_r = 0
        t = 0
        if iter > learning_starts:
            ep = ep_start - min((ep_start - 0.02) /
                0.1 * (max_iter - learning_starts) * iter, ep_start - 0.02)
        else:
            ep = 1.
        while not done:
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
            if iter > learning_starts and iter % C == C - 1:
                model.update()
            total_r += r
            t += 1
            print('\rtotal iter: {}, # of steps: {:<5}, loss: {:<.4}, # of success: {}     '.format(
                iter, t, loss, success),end='')
            sys.stdout.flush()
            if iter % 1000 == 0:
                print('')
                num_steps += [eval_perform(model, env, test_rounds)]
                env.reset()
            iter += 1
            if done:
                break
        if total_r > 199:
            success += 1
        if success > 100:
            break
    model.save()
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
            print('\r', score ,a, '            ', end='')
        sys.stdout.flush()
        avg_score = (avg_score * idx + score) / (idx + 1)
    print('')
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
    max_iter = 100000
    register(
        id='MountainCar-v1',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        max_episode_steps=20000,
        reward_threshold=-110.0,
    )
    env = gym.make('CartPole-v0')
    ep_start = 1.
    batch_size = 32
    gamma = 0.9
    a_list = [0,1]
    C = 1
    lrate = 0.0001
    lambda_ = 0.

    t1 = time.process_time()
    num_steps, agent = dqn(N, max_iter, env,
        ep_start, batch_size, gamma, a_list, C, lrate, lambda_)
    t2 = time.process_time()
    print('training time:', t2-t1)
    plot_dqn(num_steps, 'test-'+str(prefix))
    # agent = nn_model(2, a_list, 'test1', lambda_, lrate, load=True) # implement two networks in one model with an update method.
