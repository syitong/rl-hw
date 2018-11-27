import numpy as np
from .model import nn_model
from .utils import ep_greedy

class memory(list):
    def __init__(self,length):
        self.length = length
        super().__init__()
    def add(self,x):
        if len(self) < self.length:
            super().append(x)
        elif len(self) == self.length:
            super().pop(0)
            super().append(x)
    def sample(self,size):
        output = []
        for i in range(size):
            output += [np.random.choice(self)]
        return output

def Qtable(Q, s, a_list):
    Qtable = []
    for a in a_list:
        Q += [Q(s,a)]
    return Qtable

def dqn(N, num_episodes, env, T, ep, batch_size, gamma, a_list, C):
    nA = len(a_list)
    D = memory(N)
    model = nn_model # implement two networks in one model with an update method.
    Q = model.Q_eval
    Qhat = model.Qhat_eval
    s = env.reset()
    dS = len(s)
    for episode in range(num_episodes):
        for t in range(T):
            a = ep_greedy(Qtable(Q, s, [0,1,2]),ep)
            ss, r, done, _ = env.step(a)
            D.add({'s':s,'a':a,'r':r,'ss':ss,'done':done})
            batch = D.sample(batch_size)
            y = np.empty(batch_size)
            x = np.empty((batch_size,dS+1))
            for idx in range(batch_size):
                if batch[idx]['done']:
                    y[idx] = batch[idx]['r']
                else:
                    y[idx] = batch[idx]['r'] + \
                        gamma * max([Qhat(batch[idx]['ss'],a)
                            for a in a_list])
                x[idx,:] = list(batch[idx]['s'])
            Q.fit(x,y)
            if t % C == 0:
                model.update()
