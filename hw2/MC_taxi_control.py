import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import sys
import copy

matplotlib.style.use('ggplot')
# import RidiculusTaxi
import mytaxi

def my_argmax(Q_s):
    qmax = np.max(Q_s)
    actions = []
    for i,q in enumerate(Q_s):
        if q == qmax:
            actions.append(i)
    return actions

def eps_policy(Q_s, epsilon=0.1, nA=nA):
    A = np.ones(nA, dtype=float) * epsilon / nA
    actions = my_argmax(Q_s)
    A[actions] += (1.0 - epsilon) / len(actions)
    probs = A / sum(A)
    action = np.random.choice(nA, p=probs)
    return action

def mc_control_epsilon_greedy(env, runs, num_episodes, discount_factor=1.0, epsilon=0.1, alpha=0.9):

    nA = env.nA
    nS = env.nS

    rew_alloc = []
    for run in range(runs):
        Q = np.zeros((nS,nA))
        rew_list = np.zeros(num_episodes)
        returns_sum = defaultdict(float)
        returns_count = defaultdict(float)
        for i_episode in range(num_episodes):
            # Generate an episode.
            # An episode is an array of (state, action, reward) tuples
            episode = []
            state = env.reset()
            done = False
            counter = 0
            while not done:
    #        for t in range(100):
                counter += 1
                if i_episode % 100 == 0:
                    print('\rEpisode {}/{} Step {}     '.format(i_episode,num_episodes,counter), end='')
                sys.stdout.flush()
                action = eps_policy(Q[state], nA=nA)
                next_state, reward, done, _ = env.step(action)
                rew_list[i_episode] += reward
                episode.append((state, action, reward))
                if done:
                    break
                state = next_state

            # Find all (state, action) pairs we've visited in this episode
            sa_in_episode = set([(x[0], x[1]) for x in episode])
            for state, action in sa_in_episode:
                sa_pair = (state, action)
                # Find the first occurance of the (state, action) pair in the episode
                first_occurence_idx = next(i for i,x in enumerate(episode)
                                           if x[0] == state and x[1] == action)
                # Sum up all rewards since the first occurance
                G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
                # Calculate average return for this state over all sampled episodes
                returns_sum[sa_pair] = alpha*returns_sum[sa_pair] + G
                returns_count[sa_pair] += 1.0
                # The policy is improved implicitly by changing the Q dictionary
                Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
        rew_alloc.append(rew_list)
    rew_list = np.mean(np.array(rew_alloc),axis=0)
    fig = plt.figure()
    plt.plot(rew_list)
    plt.savefig('mc_control-interim.eps')
    plt.close(fig)
    return Q, rew_list


# In[5]:


def plot_value_function(V, baseline, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    V_ordered = OrderedDict(sorted(V.items()))

    print('\n')
    print(len(V.keys()))

    v_s = np.zeros(len(V.keys()))
    idx = 0
    for key, val in V_ordered.items():
        v_s[idx] = val
        idx +=1

    # print(np.sort(np.asarray(V.keys())))

#     plt.plot(np.asarray(v_s), marker='o',linewidth=2)
    plt.plot(v_s,marker='o',linestyle='None',label='mc')
    plt.plot(baseline,marker='x',linestyle='None',label='base')
    plt.legend(["MC", "Baseline"])
    plt.title(title)
    plt.xlabel("State", fontsize=20)
    plt.savefig("MC_control.png")
