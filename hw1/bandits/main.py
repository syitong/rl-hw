import matplotlib  
import seaborn as sns
sns.set()

import matplotlib.pyplot as plt
import numpy as np
import collections
import time

from bandits import BernoulliBandit, Solver, EpsilonGreedy, UCB
from utils import plot_results


def experiment(N, T, num_runs, expr_id):
    # Arms' mean
    arms_mean = np.linspace(0.1, 0.8, N)

    b = BernoulliBandit(N, probas=arms_mean)

    ##################
    ## Epsilon Greedy
    names = ['0.01','0.1','0.3']
    regret = collections.OrderedDict([(names[0], []), 
                                      (names[1], []), 
                                      (names[2], [])])
    reward = collections.OrderedDict([(names[0], []), 
                                      (names[1], []), 
                                      (names[2], [])])
    opt_act = collections.OrderedDict([(names[0], []), 
                                      (names[1], []), 
                                      (names[2], [])])
    for trial_id in range(num_runs):
        # choose different random seed for each run
        seed = trial_id + expr_id * 100
        np.random.seed(seed)
        test_solvers = [
            EpsilonGreedy(b, eps=0.01, init_proba=0),
            EpsilonGreedy(b, eps=0.1, init_proba=0),
            EpsilonGreedy(b, eps=0.3, init_proba=0),
        ] 
        for idx, s in enumerate(test_solvers):
            s.run(T)
            regret[names[idx]].append(s.regrets)
            reward[names[idx]].append(s.reward)
            opt_act[names[idx]].append(s.optimal_action)

    figname = "EpsilonGreedy_randseed_{}.png".format(expr_id)
    plot_results(regret, reward, opt_act, figname, "Epsilon Greedy")

    ###########################
    ## Optimistic initial value
    names = ['1','5','50']
    regret = collections.OrderedDict([(names[0], []), 
                                      (names[1], []), 
                                      (names[2], [])])
    reward = collections.OrderedDict([(names[0], []), 
                                      (names[1], []), 
                                      (names[2], [])])
    opt_act = collections.OrderedDict([(names[0], []), 
                                      (names[1], []), 
                                      (names[2], [])])
    for trial_id in range(num_runs):
        seed = trial_id + expr_id * 100
        np.random.seed(seed)
        b = BernoulliBandit(N, probas=arms_mean)
        test_solvers = [
            EpsilonGreedy(b, eps=0, init_proba=1.0),
            EpsilonGreedy(b, eps=0, init_proba=5.0),
            EpsilonGreedy(b, eps=0, init_proba=50.0)
        ] 
        for idx, s in enumerate(test_solvers):
            s.run(T)
            regret[names[idx]].append(s.regrets)
            reward[names[idx]].append(s.reward)
            opt_act[names[idx]].append(s.optimal_action)

    figname = "OptInit_randseed_{}.png".format(expr_id)
    plot_results(regret, reward, opt_act, figname, 'Optimistic Initial Value')


    ## UCB
    names = ['0.2','1','2']
    regret = collections.OrderedDict([(names[0], []), 
                                      (names[1], []), 
                                      (names[2], [])])
    reward = collections.OrderedDict([(names[0], []), 
                                      (names[1], []), 
                                      (names[2], [])])
    opt_act = collections.OrderedDict([(names[0], []), 
                                      (names[1], []), 
                                      (names[2], [])])
    for trial_id in range(num_runs):
        # choose different random seed for each run
        seed = trial_id + expr_id * 100
        np.random.seed(seed)
        b = BernoulliBandit(N, probas=arms_mean)
        test_solvers = [
            UCB(b, c=0.2),
            UCB(b, c=1.0),
            UCB(b, c=2.0)
        ] 
        for idx, s in enumerate(test_solvers):
            s.run(T)
            regret[names[idx]].append(s.regrets)
            reward[names[idx]].append(s.reward)
            opt_act[names[idx]].append(s.optimal_action)

    figname = "UCB_randseed_{}.png".format(expr_id)
    plot_results(regret, reward, opt_act, figname, "UCB")


        
if __name__ == '__main__':

    experiment(5, 1000, 2000, 0)



