import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import sys

suffix = sys.argv[1]

score = []
with open('DQN.o32342426-'+suffix,'r') as f:
    for line in f.readlines():
        if line[:3] == 'avg':
            score += [float(line[11:-1])]

fig = plt.figure()
plt.plot(score)
plt.title('Performance of DQN on MountainCar-v0')
plt.xlabel('per 50 episodes')
plt.ylabel('score in testing')
plt.savefig('MountainCar'+'-'+suffix+'.eps')
plt.close()
