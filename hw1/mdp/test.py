import numpy as np
import gym
from lib.gridworld import GridworldEnv
from mdp import reset,print_pol
from lukeross import policy_eval, policy_iter, value_iter

env = GridworldEnv(slip=0.2, episodic=False)
trans_mat = env.P
nA = env.nA
nS = env.nS
np.set_printoptions(precision=2)
inplace = False

POLICY = np.ones((nS,nA)) / nA
V = policy_eval(trans_mat, POLICY, theta = 0.0001, gamma=0.9)
print('uniformly random policy evaluation:')
print(V.reshape(5,-1))

POLICY = np.ones((nS,nA)) / nA
policy,V = policy_iter(trans_mat, theta = 0.0001, gamma = 0.9)
print('optimal value function after policy iteration')
print(V.reshape(5,-1))
print('optimal policy after policy iteration')
print_pol(policy,5,4)

POLICY = np.ones((nS,nA)) / nA
policy,V = value_iter(trans_mat, theta = 0.0001, gamma = 0.9)
print('optimal value function after value iteration')
print(V.reshape(5,-1))
print('optimal policy after value iteration')
print_pol(policy,5,4)
