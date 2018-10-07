import numpy as np
import matplotlib.pyplot as plt

##########-----Q4-----##########
def ep_greedy(mu,ep,T):
    N = mu.size
    Q = np.zeros(N)
    numselect = np.zeros(N)
    mumax = np.argmax(mu)
    cumrwd = 0
    cumreg = np.zeros(T)
    avgrwd = np.zeros(T)
    optact = np.zeros(T)
    for t in range(T):
        a = getgreedy(Q,ep)
        R = 1 if np.random.rand() < mu[a] else 0
        numselect[a] += 1
        Q[a] = Q[a]+(R-Q[a])/numselect[a]
        cumrwd += R
        cumreg[t] = (t+1)*mu[mumax] - cumrwd
        #avgrwd[t] = cumrwd/(t+1)
        avgrwd[t] = R
        if a == mumax:
            optact[t] = 100
    return (cumreg,avgrwd,optact)

def getgreedy(Q,ep):
    if np.random.rand() < ep:
        return np.random.choice(Q.size)
    else:
        maxbandit = np.argwhere(Q == np.amax(Q)).flatten()
        return np.random.choice(maxbandit)

def optimistic(mu,q1,T):
    N = mu.size
    Q = np.ones(N)*q1
    numselect = np.ones(N)
    mumax = np.argmax(mu)
    cumrwd = 0
    cumreg = np.zeros(T)
    avgrwd = np.zeros(T)
    optact = np.zeros(T)
    for t in range(T):
        a = getgreedy(Q,0)
        R = 1 if np.random.rand() < mu[a] else 0
        numselect[a] += 1
        Q[a] = Q[a]+(R-Q[a])/numselect[a]
        cumrwd += R
        cumreg[t] = (t+1)*mu[mumax] - cumrwd
        #avgrwd[t] = cumrwd/(t+1)
        avgrwd[t] = R
        if a == mumax:
            optact[t] = 100
    return (cumreg,avgrwd,optact)

def ucb(mu,c,T):
    N = mu.size
    Q = np.zeros(N)
    numselect = np.zeros(N)
    mumax = np.argmax(mu)
    cumrwd = 0
    cumreg = np.zeros(T)
    avgrwd = np.zeros(T)
    optact = np.zeros(T)
    for t in range(T):
        bnd = np.zeros(N)
        for i in range(N):
            if numselect[i] == 0:
                bnd[i] = c*10000
            else:
                bnd[i] = c*np.sqrt(np.log(t+1)/numselect[i])
        a = getgreedy(Q+bnd,0)
        R = 1 if np.random.rand() < mu[a] else 0
        numselect[a] += 1
        Q[a] = Q[a]+(R-Q[a])/numselect[a]
        cumrwd += R
        cumreg[t] = (t+1)*mu[mumax] - cumrwd
        #avgrwd[t] = cumrwd/(t+1)
        avgrwd[t] = R
        if a == mumax:
            optact[t] = 100
    return (cumreg,avgrwd,optact)

def saveAllFigures(filename,T,eps,grd_data,q1s,opt_data,cs,ucb_data):
    ylbl = ['Cumulative regret','Averaged reward','% Optimal action']

    cummax = max(max([np.max(grd_data[j][0]) for j in range(3)]),
                max([np.max(opt_data[j][0]) for j in range(3)]),
                max([np.max(ucb_data[j][0]) for j in range(3)]))*1.1
    avgmax = max(max([np.max(grd_data[j][1]) for j in range(3)]),
                max([np.max(opt_data[j][1]) for j in range(3)]),
                max([np.max(ucb_data[j][1]) for j in range(3)]))*1.1
    optmax = 100
    foomax = [cummax,avgmax,optmax]
    cummin = min(min([np.min(grd_data[j][0]) for j in range(3)]),
                min([np.min(opt_data[j][0]) for j in range(3)]),
                min([np.min(ucb_data[j][0]) for j in range(3)]),0)*1.1
    avgmin = min(min([np.min(grd_data[j][1]) for j in range(3)]),
                min([np.min(opt_data[j][1]) for j in range(3)]),
                min([np.min(ucb_data[j][1]) for j in range(3)]))*0.9
    optmin = 0
    foomin = [cummin,avgmin,optmin]


    fig, ax = plt.subplots(1,3)
    for i in range(3):
        for j in range(3):
            ax[i].plot(range(T),grd_data[j][i],label=r'$\epsilon=$'+str(eps[j]))
        ax[i].set_xlabel('time step')
        ax[i].set_ylabel(ylbl[i])
        ax[i].legend(loc='upper left')
        ax[i].set_ylim(foomin[i],foomax[i])
    fig.set_size_inches(4*3,4)
    fig.tight_layout()
    fig.savefig(filename + '_ep_greedy.pdf',orientation="landscape",bbox_inches="tight")
    fig.clear()
    plt.close(fig)

    fig, ax = plt.subplots(1,3)
    for i in range(3):
        for j in range(3):
            ax[i].plot(range(T),opt_data[j][i],label=r'$Q_1=$'+str(q1s[j]))
        ax[i].set_xlabel('time step')
        ax[i].set_ylabel(ylbl[i])
        ax[i].legend(loc='upper left')
        ax[i].set_ylim(foomin[i],foomax[i])
    fig.set_size_inches(4*3,4)
    fig.tight_layout()
    fig.savefig(filename + '_optimistic.pdf',orientation="landscape",bbox_inches="tight")
    fig.clear()
    plt.close(fig)

    fig, ax = plt.subplots(1,3)
    for i in range(3):
        for j in range(3):
            ax[i].plot(range(T),ucb_data[j][i],label=r'$c=$'+str(cs[j]))
        ax[i].set_xlabel('time step')
        ax[i].set_ylabel(ylbl[i])
        ax[i].legend(loc='upper left')
        ax[i].set_ylim(foomin[i],foomax[i])
    fig.set_size_inches(4*3,4)
    fig.tight_layout()
    fig.savefig(filename + '_ucb.pdf',orientation="landscape",bbox_inches="tight")
    fig.clear()
    plt.close(fig)


def sim_ep_greedy(mu,ep,T):
    print('starting ep_greedy: ep='+str(ep))
    avgcumreg = np.zeros(T)
    avgavgrwd = np.zeros(T)
    avgoptact = np.zeros(T)
    for s in range(sim):
        if s % 20 == 0:
            print('scenario '+ str(s))
        (cumreg,avgrwd,optact) = ep_greedy(mu,ep,T)
        avgcumreg += (cumreg-avgcumreg)/(s+1)
        avgavgrwd += (avgrwd-avgavgrwd)/(s+1)
        avgoptact += (optact-avgoptact)/(s+1)
    return (avgcumreg,avgavgrwd,avgoptact)

def sim_optimistic(mu,q1,T):
    print('starting optimistic: q1='+str(q1))
    avgcumreg = np.zeros(T)
    avgavgrwd = np.zeros(T)
    avgoptact = np.zeros(T)
    for s in range(sim):
        if s % 20 == 0:
            print('scenario '+ str(s))
        (cumreg,avgrwd,optact) = optimistic(mu,q1,T)
        avgcumreg += (cumreg-avgcumreg)/(s+1)
        avgavgrwd += (avgrwd-avgavgrwd)/(s+1)
        avgoptact += (optact-avgoptact)/(s+1)
    return (avgcumreg,avgavgrwd,avgoptact)

def sim_ucb(mu,c,T):
    print('starting ucb: c='+str(c))
    avgcumreg = np.zeros(T)
    avgavgrwd = np.zeros(T)
    avgoptact = np.zeros(T)
    for s in range(sim):
        if s % 20 == 0:
            print('scenario '+ str(s))
        (cumreg,avgrwd,optact) = ucb(mu,c,T)
        avgcumreg += (cumreg-avgcumreg)/(s+1)
        avgavgrwd += (avgrwd-avgavgrwd)/(s+1)
        avgoptact += (optact-avgoptact)/(s+1)
    return (avgcumreg,avgavgrwd,avgoptact)


if __name__ == "__main__":
    sim = 2000
    T   = 1000
    Ns  = [5]
    eps = [0.01,0.1,0.3]
    q1s = [1,5,50]
    cs  = [0.2,1,2]
    for N in Ns:
        print('with setting N='+str(N))
        mu = np.linspace(0.1,0.8,num=N)
        grd_data = []
        opt_data = []
        ucb_data = []
        for ep in eps:
            grd_data.append(sim_ep_greedy(mu,ep,T))

        for q1 in q1s:
            opt_data.append(sim_optimistic(mu,q1,T))

        for c in cs:
            ucb_data.append(sim_ucb(mu,c,T))

        saveAllFigures('HW1/N_'+str(N),T,eps,grd_data,q1s,opt_data,cs,ucb_data)
