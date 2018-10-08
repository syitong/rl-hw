#######################################
# Q4 - Luke Ross (lukeross@umich.edu) #
#######################################

import numpy as np
import matplotlib.pyplot as plt

class MultiArmBandit():
    def __init__(self, means, distr='bernoulli'):
        if distr == 'bernoulli':
            self.p = means
            self.N = len(self.p)
        else:
            raise ValueError("Unrecognized distribution \'" + str(distr) + "\'")

    def pullArm(self, a):
        return np.random.binomial(1, self.p[a])

class UCB():
    def __init__(self, N, c, bestMean=None, bestIdx=None, numSteps=1000):
        self.N = N
        self.c = c
        self.estimates = np.zeros(N)
        self.numPulls = np.ones(N)
        self.bestMean = bestMean
        self.bestIdx = bestIdx
        self.iterNum = 0
        self.rewards = np.zeros(numSteps)

        if self.bestMean is not None:
            self.regret = np.zeros(numSteps)

        if self.bestIdx is not None:
            self.numBestAction = 0
            self.percentBestAction = np.zeros(numSteps)

    def randArgmax(self, arr):
        return np.random.choice(np.flatnonzero(arr == arr.max()))

    def step(self, bandit):
        actions = self.estimates + self.c * np.sqrt(np.log(self.iterNum + 1) / self.numPulls)
        action = self.randArgmax(actions)
        self.numPulls[action] += 1

        reward = bandit.pullArm(action)
        self.rewards[self.iterNum] = reward

        if self.bestMean is not None:
            self.regret[self.iterNum] = self.regret[max(0, self.iterNum-1)] + self.bestMean - self.rewards[self.iterNum]
        if self.bestIdx is not None:
            if action == self.bestIdx:
                self.numBestAction += 1
            self.percentBestAction[self.iterNum] = self.numBestAction / (self.iterNum + 1)
        self.estimates[action] += (reward - self.estimates[action]) / (self.numPulls[action]) # Treat initial estimate as valid measurement or no? N(a) + 0 ==> yes
        self.iterNum += 1

    def clearHistory(self):
        self.estimates = np.zeros_like(self.estimates)
        self.rewards = np.zeros_like(self.rewards)
        self.numPulls = np.ones_like(self.numPulls)
        if self.bestMean is not None:
            self.regret = np.zeros_like(self.regret)
        if self.bestIdx is not None:
            self.numBestAction = 0
            self.percentBestAction = np.zeros_like(self.percentBestAction)
        self.iterNum = 0

class EpsilonGreedy():
    def __init__(self, N, epsilon=0, estimates=None, bestMean=None, bestIdx=None, numSteps=1000):
        self.N = N
        self.epsilon = epsilon
        self.initEstimates = estimates
        self.numPulls = np.ones(N)
        self.bestMean = bestMean
        self.bestIdx = bestIdx
        self.iterNum = 0
        self.rewards = np.zeros(numSteps)

        if self.initEstimates is None: #Default initialize to zero
            self.initEstimates = np.zeros(N)
        self.estimates = np.copy(self.initEstimates)

        if self.bestMean is not None:
            self.regret = np.zeros(numSteps)

        if self.bestIdx is not None:
            self.numBestAction = 0
            self.percentBestAction = np.zeros(numSteps)

    def randArgmax(self, arr):
        return np.random.choice(np.flatnonzero(arr == arr.max()))

    def step(self, bandit):
        if np.random.binomial(1, self.epsilon): #Explore w/ probability epsilon
            action = np.random.randint(0, self.N, dtype='int')
        else:
            action = self.randArgmax(self.estimates)

        reward = bandit.pullArm(action)
        self.rewards[self.iterNum] = reward
        self.numPulls[action] += 1

        if self.bestMean is not None:
            self.regret[self.iterNum] = self.regret[max(0, self.iterNum-1)] + self.bestMean - self.rewards[self.iterNum]
        if self.bestIdx is not None:
            if action == self.bestIdx:
                self.numBestAction += 1
            self.percentBestAction[self.iterNum] = self.numBestAction / (self.iterNum + 1)
        self.estimates[action] += (reward - self.estimates[action]) / (self.numPulls[action]) # Treat initial estimate as valid measurement or no? N(a) + 0 ==> yes
        self.iterNum += 1

    def clearHistory(self):
        self.estimates = np.copy(self.initEstimates)
        self.rewards = np.zeros_like(self.rewards)
        self.numPulls = np.ones_like(self.numPulls)
        if self.bestMean is not None:
            self.regret = np.zeros_like(self.regret)
        if self.bestIdx is not None:
            self.numBestAction = 0
            self.percentBestAction = np.zeros_like(self.percentBestAction)
        self.iterNum = 0

    def plotReward(self):
        plt.figure()
        plt.plot(self.rewards)
        plt.show()

def runAlgos(bandit, agents, rewards, regrets=None, percents=None, numRuns=2000, numSteps=1000):
    for run in range(numRuns):
        for agentNo, agent in enumerate(agents):
            seed = 23498745 + run*len(agents) + agentNo
            np.random.seed(seed)
            for step in range(numSteps):
                # print(agent.estimates)
                agent.step(bandit)
        for i in range(len(agents)):
            rewards[i] = updateMean(rewards[i], agents[i].rewards, run)
        if regrets is not None:
            for i in range(len(agents)):
                regrets[i] = updateMean(regrets[i], agents[i].regret, run)
        if percents is not None:
            for i in range(len(agents)):
                percents[i] = updateMean(percents[i], agents[i].percentBestAction, run)
        for agent in agents:
            agent.clearHistory()
        if int(run % int(round(numRuns/10))) == 0:
            print('Progress: ' + str(round(100*run/numRuns)) + '%')
    print('Progress: 100%')
    return (rewards, regrets, percents)

def updateMean(old, new, n):
    return old + (new - old) / (n + 1)

def subplot(x):
    f, axarr = plt.subplots(len(x), sharex=True)
    f.suptitle('Epsilon Greedy Methods')
    for i in range(len(x)):
        axarr[i].plot(x[i])
        axarr[i].set_ylim(0,1)
    plt.show()

def sameplot(x, legendLabels, yLabel, ylim0, ylim1):
    for i in range(len(x)):
        plt.plot(x[i], label=legendLabels[i])
    plt.ylim(ylim0, ylim1)
    plt.ylabel(yLabel)
    plt.xlabel('Time Step')
    plt.legend(loc='best')



#######################################
# Q5 - Luke Ross (lukeross@umich.edu) #
#######################################

import numpy as np
import matplotlib.pyplot as plt

def lin2grid(n, gridShape=(5,5)):
    return np.unravel_index(n, gridShape)

def grid2lin(x, y, gridShape=(5,5)):
    return np.ravel_multi_index((x, y), gridShape)

def uniform_policy(dim=5, a=4):
    return np.ones((dim*dim, a)) / a

def visualize_policy(policy, dim=5):
    actions = {0: 'n', 1: 'e', 2: 's', 3: 'w'}
    vis = [['' for i in range(dim)] for j in range(dim)]
    for row in range(dim):
        for col in range(dim):
            a = np.argmax(policy[grid2lin(row, col)])
            print(actions[a] + ' ', end='')
        print()

def print_value(V):
    print(np.around(V,2))

def transition(state, action, dim=5, A=(0,1), Ap=(4,1), B=(0,3), Bp=(2,3), slip_prob=0.2):
    currentRow, currentCol = lin2grid(state)
    nextRow, nextCol = currentRow, currentCol
    reward = 0

    if (currentRow, currentCol) == A:
        reward = 10
        nextRow, nextCol = Ap
        return [(1, grid2lin(nextRow, nextCol), reward)]

    if (currentRow, currentCol) == B:
        reward = 5
        nextRow, nextCol = Bp
        return [(1, grid2lin(nextRow, nextCol), reward)]

    if action == 0: #north
        nextRow -= 1
    elif action == 1: #east
        nextCol += 1
    elif action == 2: #south
        nextRow += 1
    elif action == 3: #west
        nextCol -= 1

    if nextCol >= dim:
        reward = -1
        nextCol = dim-1
    if nextRow >= dim:
        reward = -1
        nextRow = dim-1
    if nextCol < 0:
        reward = -1
        nextCol = 0
    if nextRow < 0:
        reward = -1
        nextRow = 0

    return [
        (1-slip_prob, grid2lin(nextRow, nextCol), reward),
        (slip_prob, grid2lin(nextRow, nextCol), 0)
    ]

def transition_ep(state, action, dim=5, A=(0,1), Ap=(4,1), B=(0,3), Bp=(2,3), slip_prob=0.2):
    currentRow, currentCol = lin2grid(state)
    nextRow, nextCol = currentRow, currentCol
    reward = 0

    if (currentRow, currentCol) == A:
        reward = 10
        nextRow, nextCol = Ap
        return [(1, grid2lin(nextRow, nextCol), reward)]

    if (currentRow, currentCol) == B:
        reward = 5
        nextRow, nextCol = Bp
        return [(1, grid2lin(nextRow, nextCol), reward)]

    if (currentRow, currentCol) == Ap:
        reward = 0
        nextRow, nextCol = Ap
        return [(1, grid2lin(nextRow, nextCol), reward)]

    if (currentRow, currentCol) == Bp:
        reward = 0
        nextRow, nextCol = Bp
        return [(1, grid2lin(nextRow, nextCol), reward)]

    if action == 0: #north
        nextRow -= 1
    elif action == 1: #east
        nextCol += 1
    elif action == 2: #south
        nextRow += 1
    elif action == 3: #west
        nextCol -= 1

    if nextCol >= dim:
        reward = -1
        nextCol = dim-1
    if nextRow >= dim:
        reward = -1
        nextRow = dim-1
    if nextCol < 0:
        reward = -1
        nextCol = 0
    if nextRow < 0:
        reward = -1
        nextRow = 0

    return [
        (1-slip_prob, grid2lin(nextRow, nextCol), reward),
        (slip_prob, grid2lin(nextRow, nextCol), 0)
    ]

'''
P = {
    s1: {a1: [(p(s'_1|s1,a1), s'_1, reward(s'_1,s1,a1)),
                (p(s'_2|s1,a1), s'_2, reward(s'_2,s1,a1)),
                ...
                ],
            a2: ...,
            ...
        },
    s2: ...,
    ...
    }
'''
def gridworld(slip_prob=0.2):
    P = {}
    for s in range(5*5):
        stateDict = {}
        for a in range(4):
            stateDict[a] = transition(s, a, slip_prob=slip_prob)
        P[s] = stateDict
    return P

def gridworld_ep(slip_prob=0.2):
    P = {}
    for s in range(5*5):
        stateDict = {}
        for a in range(4):
            stateDict[a] = transition_ep(s, a, slip_prob=slip_prob)
        P[s] = stateDict
    return P

def policy_eval(P, policy=uniform_policy(), theta=0.0001, gamma=0.9):
    V = np.zeros((5,5))
    delta = theta + 1 # Initially larger than theta by any arbitrary amount
    while delta > theta:
        delta = 0
        nextV = np.zeros_like(V)
        for s in range(5*5): # For each s in S
            sum = 0
            for a in range(4): # Sum over actions
                for spIdx in range(len(P[s][a])): # Sum over next states s'
                    sp = P[s][a][spIdx][1]
                    sum += policy[s,a] * P[s][a][spIdx][0] * (P[s][a][spIdx][2] + gamma*V[lin2grid(sp)])
            nextV[lin2grid(s)] = sum
            delta = max(delta, np.abs(V[lin2grid(s)] - nextV[lin2grid(s)]))
        V = nextV
    return V

def policy_improvement(policy, P, V, gamma=0.9):
    policy_stable = True
    for s in range(5*5):
        old_action = np.argmax(policy[s])
        sum = np.zeros_like(policy[s])
        for a in range(4):
            for spIdx in range(len(P[s][a])):
                sp = P[s][a][spIdx][1]
                sum[a] += P[s][a][spIdx][0] * (P[s][a][spIdx][2] + gamma*V[lin2grid(sp)])
        best_action = np.argmax(sum)
        if old_action != best_action:
            policy_stable = False
        new_actions = np.zeros(4)
        new_actions[best_action] = 1
        policy[s] = new_actions
    return policy, policy_stable

def policy_iter(P, theta=0.0001, gamma=0.9):
    policy = uniform_policy()
    policy_stable = False
    while not policy_stable:
        V = policy_eval(P, policy=policy)
        policy, policy_stable = policy_improvement(policy, P, V)
    return policy, V

def value_iter(P, theta=0.0001, gamma=0.9):
    # Compute V*
    V = np.zeros((5,5))
    delta = theta + 1 #Arbitrary value > theta
    while delta > theta:
        delta = 0
        nextV = np.zeros_like(V)
        for s in range(5*5):
            sum = np.zeros(4)
            for a in range(4):
                for spIdx in range(len(P[s][a])):
                    sp = P[s][a][spIdx][1]
                    sum[a] += P[s][a][spIdx][0] * (P[s][a][spIdx][2] + gamma*V[lin2grid(sp)])
            bestIdx = np.argmax(sum)
            nextV[lin2grid(s)] = sum[bestIdx]
            delta = max(delta, np.abs(V[lin2grid(s)] - nextV[lin2grid(s)]))
        V = nextV

    # Compute deterministic policy
    policy = np.zeros((25, 4))
    for s in range(5*5):
        actions = np.zeros(4)
        for a in range(4):
            for spIdx in range(len(P[s][a])):
                sp = P[s][a][spIdx][1]
                actions[a] += P[s][a][spIdx][0] * (P[s][a][spIdx][2] + gamma*V[lin2grid(sp)])
        policy[s][np.argmax(actions)] = 1

    return policy, V

if __name__ == '__main__':
    N = 5
    numSteps = 1000
    numRuns = 2000
    means = np.linspace(0.1, 0.8, N)
    bandit = MultiArmBandit(means)

    greedy01 = EpsilonGreedy(N, epsilon=0.01, bestMean=np.max(means), bestIdx=np.argmax(means), numSteps=numSteps)
    greedy1 = EpsilonGreedy(N, epsilon=0.1, bestMean=np.max(means), bestIdx=np.argmax(means), numSteps=numSteps)
    greedy3 = EpsilonGreedy(N, epsilon=0.3, bestMean=np.max(means), bestIdx=np.argmax(means), numSteps=numSteps)

    opt1 = EpsilonGreedy(N, estimates=np.ones(N), bestMean=np.max(means), bestIdx=np.argmax(means), numSteps=numSteps)
    opt5 = EpsilonGreedy(N, estimates=np.ones(N)*5, bestMean=np.max(means), bestIdx=np.argmax(means), numSteps=numSteps)
    opt50 = EpsilonGreedy(N, estimates=np.ones(N)*50, bestMean=np.max(means), bestIdx=np.argmax(means), numSteps=numSteps)

    ucb02 = UCB(N, c=0.2, bestMean=np.max(means), bestIdx=np.argmax(means), numSteps=numSteps)
    ucb1 = UCB(N, c=1, bestMean=np.max(means), bestIdx=np.argmax(means), numSteps=numSteps)
    ucb2 = UCB(N, c=2, bestMean=np.max(means), bestIdx=np.argmax(means), numSteps=numSteps)

    agents = [greedy01, greedy1, greedy3,
              opt1, opt5, opt50,
              ucb02, ucb1, ucb2]
    rewards = [greedy01.rewards, greedy1.rewards, greedy3.rewards,
               opt1.rewards, opt5.rewards, opt50.rewards,
               ucb02.rewards, ucb1.rewards, ucb2.rewards]
    regrets = [greedy01.regret, greedy1.regret, greedy3.regret,
              opt1.regret, opt5.regret, opt50.regret,
              ucb02.regret, ucb1.regret, ucb2.regret]
    percents = [greedy01.percentBestAction, greedy1.percentBestAction, greedy3.percentBestAction,
                opt1.percentBestAction, opt5.percentBestAction, opt50.percentBestAction,
                ucb02.percentBestAction, ucb1.percentBestAction, ucb2.percentBestAction]
    labels = ['ε = 0.01', 'ε = 0.1', 'ε = 0.3',
              'Q1 = 1', 'Q1 = 5', 'Q1 = 50',
              'c = 0.2', 'c = 1', 'c = 2']

    rewards, regrets, percents = runAlgos(bandit, agents, rewards, regrets=regrets, percents=percents, numRuns=numRuns, numSteps=numSteps)

    plt.subplot(331)
    sameplot(regrets[0:3], labels[0:3], 'Cumulative Regret', 0, 200)
    plt.subplot(332)
    sameplot(rewards[0:3], labels[0:3], 'Averaged Reward', 0, 1)
    plt.subplot(333)
    sameplot(percents[0:3], labels[0:3], '% Optimal Action', 0, 1)

    plt.subplot(334)
    sameplot(regrets[3:6], labels[3:6], 'Cumulative Regret', 0, 200)
    plt.subplot(335)
    sameplot(rewards[3:6], labels[3:6], 'Averaged Reward', 0, 1)
    plt.subplot(336)
    sameplot(percents[3:6], labels[3:6], '% Optimal Action', 0, 1)

    plt.subplot(337)
    sameplot(regrets[6:9], labels[6:9], 'Cumulative Regret', 0, 200)
    plt.subplot(338)
    sameplot(rewards[6:9], labels[6:9], 'Averaged Reward', 0, 1)
    plt.subplot(339)
    sameplot(percents[6:9], labels[6:9], '% Optimal Action', 0, 1)

    plt.show()

    #######################################
    #                End Q4               #
    #######################################
    P = gridworld_ep()
    policy, V = value_iter(P, gamma=0.8) #Critial value in [0.840896415, 0.840896416]
    print_value(V)
    visualize_policy(policy)

#######################################
#                End Q5               #
#######################################
