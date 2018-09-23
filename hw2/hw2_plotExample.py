import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


fig, axes = plt.subplots(1, 4, figsize=(16, 4))

titles=["MC prediction", "MC prediction", "TD0", "TD0"]
xlabels=["Episodes", "State", "Episodes", "State"]
ylabels = ["RMS", r"$V_{\pi}(s)$", "RMS", r"$V_{\pi}(s)$"]
for i in range(4):
    ax = axes[i]
    ax.plot()

    ax.set_title(titles[i], fontsize=20)
    ax.set_xlabel(xlabels[i], fontsize=20)
    ax.set_ylabel(ylabels[i], fontsize=20)
    # ax.grid(False)
    # ax.axis('off')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

plt.subplots_adjust(top=0.88, bottom=0.15, wspace=0.35, left=0.06, right=0.96)
plt.savefig('hw2_plotExamp.png')


fig = plt.figure()
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

titles=["MC Control", "Q learning"]
xlabels=["Episodes", "Episodes"]
ylabels = ["Sum of rewards received \n within each episode", "Sum of rewards received \n within each episode"]
for i in range(2):
    ax = axes[i]
    ax.plot()

    ax.set_title(titles[i], fontsize=20)
    ax.set_xlabel(xlabels[i], fontsize=20)
    ax.set_ylabel(ylabels[i], fontsize=20)
    # ax.grid(False)
    # ax.axis('off')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

plt.subplots_adjust(top=0.88, bottom=0.15, wspace=0.35, left=0.09, right=0.96)
plt.savefig('hw2_plotExamp2.png')








