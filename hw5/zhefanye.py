'''
Project: proj5
Created Date: Wednesday, December 12th 2018
Author: Zhefan Ye <zhefanye@gmail.com>
-----
Copyright (c) 2018 TBD
Do whatever you want
'''

import math
import random

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym


class DQN(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.policy_net = Net(self.state_space_dim, self.action_space_dim)
        self.target_net = Net(self.state_space_dim, self.action_space_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = []
        self.steps = 0

    def choose_action(self, state):
        self.steps += 1
        epsilon = self.epsilon_low + (self.epsilon_high-self.epsilon_low) * \
            (math.exp(-1.0 * self.steps/self.decay))
        if random.random() < epsilon:
            action = random.randrange(self.action_space_dim)
        else:
            state = torch.tensor(state, dtype=torch.float).view(1, -1)
            action = torch.argmax(self.policy_net(state)).item()
        return action

    def push(self, *transition):
        if len(self.memory) == self.capacity:
            self.memory.pop(0)
        self.memory.append(transition)

    def learn(self, episode):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state = zip(*batch)
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long).view(
            self.batch_size, -1)
        reward = torch.tensor(reward, dtype=torch.float).view(
            self.batch_size, -1)
        next_state = torch.tensor(next_state, dtype=torch.float)

        expected_q_values = reward + self.gamma * \
            torch.max(self.target_net(next_state).detach(), dim=1)[
                0].view(self.batch_size, -1)
        q_value = self.policy_net(state).gather(1, action)

        loss = nn.MSELoss()(q_value, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if episode % self.delay_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super(Net, self).__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x


def performance_curve(episode, score):
    """train 100 episodes, test 100 episodes using trained greedy policy,
    take the average of total reward received within one episode
    """
    plt.figure(1)
    plt.clf()
    plt.title('Performance')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(episode, score)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.savefig('output/f2.png')


def q1_a():
    env = gym.make('CartPole-v0')
    params = {
        'gamma': 0.8,
        'epsilon_high': 0.9,
        'epsilon_low': 0.05,
        'decay': 200,
        'delay_update': 1,
        'lr': 0.001,
        'capacity': 10000,
        'batch_size': 32,
        'state_space_dim': env.observation_space.shape[0],
        'action_space_dim': env.action_space.n
    }
    dqn = DQN(**params)

    score = []
    episodes = []

    for episode in range(1000):
        state = env.reset()
        episode_reward = 0
        while True:
            env.render()
            action = dqn.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            if done:
                reward = -1

            dqn.push(state, action, reward, next_state)

            if done:
                print('episode: {}, score: {}'.format(
                    episode, episode_reward))
                break

            episode_reward += reward
            state = next_state
            dqn.learn(episode)

        if episode % 100 == 0:
            score.append(episode_reward)
            episodes.append(episode)
            for i in range(100):
                episode_reward += episode_reward
            episode_reward /= 100
            performance_curve(episodes, score)


def q1_b():
    env = gym.make('MountainCar-v0')
    params = {
        'gamma': 0.8,
        'epsilon_high': 0.9,
        'epsilon_low': 0.05,
        'decay': 200,
        'delay_update': 1,
        'lr': 0.001,
        'capacity': 10000,
        'batch_size': 128,
        'state_space_dim': env.observation_space.shape[0],
        'action_space_dim': env.action_space.n
    }
    dqn = DQN(**params)

    score = []
    episodes = []

    for episode in range(1000):
        state = env.reset()
        episode_reward = 0
        while True:
            env.render()
            action = dqn.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            if done:
                reward = -1

            dqn.push(state, action, reward, next_state)

            if done:
                print('episode: {}, score: {}'.format(
                    episode, episode_reward))
                break

            episode_reward += reward
            state = next_state
            dqn.learn(episode)

        if episode % 100 == 0:
            score.append(episode_reward)
            episodes.append(episode)
            for i in range(100):
                episode_reward += episode_reward
            episode_reward /= 100
            performance_curve(episodes, score)


if __name__ == "__main__":
    # q1_a()
    q1_b()
