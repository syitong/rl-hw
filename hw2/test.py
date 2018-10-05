import gym
import mytaxi
import numpy as np

env = gym.make('Taxi-v3').unwrapped
episodes = 100

for _ in range(episodes):
    env.reset()
    done = False
    while not done:
        env.render()
        env.step(env.action_space.sample())
        input('next?')

