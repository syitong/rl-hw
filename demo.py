import gym
from lib.gridworld import GridworldEnv

env = GridworldEnv()
env.reset()
for _ in range(1000):
    env._render()
    env.step(env.action_space.sample()) # take a random action
    input('next?')
