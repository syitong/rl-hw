from __future__ import division
import time
import numpy as np

class BernoulliBandit():

    def __init__(self, num_arms, probas=None):
        # np.random.seed(rng)
        assert probas is None or len(probas) == num_arms
        self.num_arms = num_arms
        if probas is None:
            self.probas = [np.random.random() for _ in range(self.n)]
        else:
            self.probas = probas

        self.best_proba = max(self.probas)
        self.best_arm = np.argmax(self.probas)
        # print('best arm:{}, mean reward:{}'.format(self.best_arm, self.best_proba))

    def generate_reward(self, i):
        if np.random.random() <= self.probas[i]:
            return 1
        else:
            return 0


class Solver(object):
    def __init__(self, bandit):
        assert isinstance(bandit, BernoulliBandit)

        self.bandit = bandit

        self.counts = np.zeros(self.bandit.num_arms)
        self.actions = [] 
        self.reward = []
        self.optimal_action = []
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.

    def update_regret(self, r):
        self.regret += self.bandit.best_proba - r
        self.regrets.append(self.regret)

    def update_opt_action_frac(self, i):
        if i == self.bandit.best_arm:
            self.optimal_action.append(1)
        else:
            self.optimal_action.append(0)

    def run_one_step(self):
        raise NotImplementedError

    def run(self, num_steps):
        assert self.bandit is not None
        for _ in range(num_steps):
            i, r = self.run_one_step()

            self.counts[i] += 1
            self.reward.append(r)
            self.actions.append(i)
            self.update_regret(r)
            self.update_opt_action_frac(i)


class EpsilonGreedy(Solver):
    def __init__(self, bandit, eps, init_proba=0.0):

        # np.random.seed(rng)
        super(EpsilonGreedy, self).__init__(bandit)

        assert 0. <= eps <= 1.0
        self.eps = eps
 
        self.estimates = init_proba * np.ones(self.bandit.num_arms)

    # def my_argmax():
    def run_one_step(self):
        if np.random.random() < self.eps:
            i = np.random.randint(0, self.bandit.num_arms)
        else:
            i = np.random.choice(np.flatnonzero(self.estimates == max(self.estimates)))

        r = self.bandit.generate_reward(i)
        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i, r


class UCB(Solver):
    def __init__(self, bandit, c=1.0, init_proba=0.0):
        # np.random.seed(rng)
        super(UCB, self).__init__(bandit)
        self.t = 0
        self.estimates = init_proba * np.ones(self.bandit.num_arms)
        self.c = c


    def run_one_step(self):
        self.t += 1
        
        if len(np.flatnonzero(self.counts == 0)) >= 1:
            i = np.random.choice(np.flatnonzero(self.counts == 0))
        else:
            Q_temp = self.estimates + self.c * np.sqrt(np.log(self.t) / np.float32(self.counts))
            i = np.random.choice(np.flatnonzero(Q_temp == max(Q_temp)))

        r = self.bandit.generate_reward(i)

        self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])

        return i, r

