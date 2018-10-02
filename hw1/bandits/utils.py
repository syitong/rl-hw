import matplotlib  
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set()


def plot_results(regret, reward, opt_action_frac, figname, method):
    
    fig = plt.figure(figsize=(14, 4))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    colors = ['r', 'b', 'g']
    idx = 0
    for key, item in regret.items():
        item = np.asarray(item).T
        print('plot_res:{}'.format(item.shape))
        rmean = np.mean(item, axis=1)
        ax1.plot(rmean, color=colors[idx], linewidth=2, label = key)
        idx +=1

    ax1.set_xlabel('Time step', fontsize=16)
    ax1.set_ylabel('Cumulative regret', fontsize=16)
    ax1.set_ylim([0.0, 600])
    props = dict(boxstyle='round', facecolor='lavender', alpha=0.5)
    ax1.text(0.1, 1.25, method, fontsize=20, rotation='horizontal', transform=ax1.transAxes, 
               verticalalignment='top', bbox=props)
    ax1.legend(loc='upper center', bbox_to_anchor=(1.82, 1.35), ncol=3, fontsize=20)
    ax1.grid('k', ls='--', alpha=0.3)

    idx = 0
    for key, item in reward.items():
        item = np.asarray(item).T
        print('plot_res:{}'.format(item.shape))
        rmean = np.mean(item, axis=1)
        ax2.plot(rmean, color=colors[idx], linewidth=2)
        idx +=1

    ax2.set_xlabel('Time step', fontsize=16)
    ax2.set_ylabel('Averaged reward', fontsize=16)
    ax2.set_ylim([0.2, 1.0])


    idx = 0
    for key, item in opt_action_frac.items():
        item = np.asarray(item).T
        print('plot_res:{}'.format(item.shape))
        rmean = np.mean(item, axis=1)
        ax3.plot(rmean, color=colors[idx], linewidth=2)
        idx +=1

    ax3.set_xlabel('Time step', fontsize=16)
    ax3.set_ylabel('Optimal action frac', fontsize=16)
    ax3.set_ylim([0.0, 1.0])
    
    plt.subplots_adjust(top=0.75, bottom=0.12, wspace=0.20, left=0.08, right=0.96)
    plt.savefig(figname)

