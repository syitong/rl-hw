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


# if __name__ == "__main__":
#     sim = 2000
#     T   = 1000
#     Ns  = [5]
#     eps = [0.01,0.1,0.3]
#     q1s = [1,5,50]
#     cs  = [0.2,1,2]
#     for N in Ns:
#         print('with setting N='+str(N))
#         mu = np.linspace(0.1,0.8,num=N)
#         grd_data = []
#         opt_data = []
#         ucb_data = []
#         for ep in eps:
#             grd_data.append(sim_ep_greedy(mu,ep,T))
#
#         for q1 in q1s:
#             opt_data.append(sim_optimistic(mu,q1,T))
#
#         for c in cs:
#             ucb_data.append(sim_ucb(mu,c,T))
#
#         saveAllFigures('HW1/N_'+str(N),T,eps,grd_data,q1s,opt_data,cs,ucb_data)


##########-----Q5-----##########

def gridworld(slip_prob=0.2):
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
    # slip_prob is the probability the agent slips.

    # dictionary of row,col -> state index
    statedict = {}
    ss = 0
    for row in range(5):
        for col in range(5):
            statedict[row,col] = ss
            ss += 1

    # First, generate plain grid
    P = {}
    for row in range(5):
        for col in range(5):
            P_s = {}

            north = []
            if row-1 < 0:
                north.append((1.0,statedict[row,col],-1.0))
            else:
                north.append((slip_prob,statedict[row,col],0.0))
                north.append((1.0-slip_prob,statedict[row-1,col],0.0))

            east = []
            if col+1 > 4:
                east.append((1.0,statedict[row,col],-1.0))
            else:
                east.append((slip_prob,statedict[row,col],0.0))
                east.append((1.0-slip_prob,statedict[row,col+1],0.0))

            south = []
            if row+1 > 4:
                south.append((1.0,statedict[row,col],-1.0))
            else:
                south.append((slip_prob,statedict[row,col],0.0))
                south.append((1.0-slip_prob,statedict[row+1,col],0.0))

            west = []
            if col-1 < 0:
                west.append((1.0,statedict[row,col],-1.0))
            else:
                west.append((slip_prob,statedict[row,col],0.0))
                west.append((1.0-slip_prob,statedict[row,col-1],0.0))

            P_s[0] = north
            P_s[1] = east
            P_s[2] = south
            P_s[3] = west

            P[statedict[row,col]] = P_s

    # now deal with exceptions and overwrite
    # A (0,1) -> A' (4,1)
    # B (0,3) -> B' (2,3)
    for a in range(4):
        P[statedict[0,1]][a] = [(1.0,statedict[4,1],10.0)]
        P[statedict[0,3]][a] = [(1.0,statedict[2,3], 5.0)]

    return P

def policy_eval(P, V = np.zeros((5,5)), policy=np.ones((25,4))*0.25, theta=0.0001, gamma=0.9):
    '''
        P: as returned by your gridworld(slip=0.2).
        policy: probability distribution over actions for each states.
            Default to uniform policy.
        theta: stopping condition.
        gamma: the discount factor.
        V: 5 by 5 numpy array where each entry is the value of the
            corresponding location. Initialize V with zeros.
    '''
    while True:
        Delta = 0
        for s in range(25):
            row = s // 5
            col = s % 5
            v = V[row,col]
            newV = 0 # updated value of V(s)
            for a in range(4):
                pi = policy[s,a]
                T = P[s][a]
                for i in range(len(T)):
                    rowp = T[i][1] // 5
                    colp = T[i][1] % 5
                    newV += pi * T[i][0] * (T[i][2] + gamma * V[rowp,colp])
            V[row,col] = newV
            Delta = max(Delta,abs(v-newV))
        if Delta < theta:
            break

    return V

def policy_iter(P, theta=0.0001, gamma=0.9):
    '''
        policy: 25 by 4 numpy array where each row is a probability
            distribution over moves for a state. If it is
            deterministic, then the probability will be a one hot vector.
            If there is a tie between two actions, break the tie with
            equal probabilities.
            Initialize the policy with zeros.
    '''
    policy = np.ones((25,4))*0.25
    V = np.zeros((5,5))
    while True:
        V = policy_eval(P,V=V,policy=policy,theta=theta,gamma=gamma)
        policy_stable = True
        for s in range(25):
            old_action = np.copy(policy[s])
            vals = np.zeros(4) # to take maximum later
            for a in range(4):
                T = P[s][a]
                for i in range(len(T)):
                    rowp = T[i][1] // 5
                    colp = T[i][1] % 5
                    vals[a] += T[i][0] * (T[i][2] + gamma * V[rowp,colp])
            # get all elements of argmax
            a = np.argwhere(vals == np.amax(vals)).flatten().tolist()
            policy[s] = np.zeros(4)
            policy[s,a] = 1/len(a)
            if np.sum(np.absolute(policy[s] - old_action))>1e-9:
                policy_stable = False
        if policy_stable:
            break
    return V, policy

def value_iter(P, theta=0.0001, gamma=0.9):
    V = np.zeros((5,5))
    while True:
        Delta = 0
        for s in range(25):
            row = s // 5
            col = s % 5
            v = V[row,col]
            newV = 0 # take maximum of foo
            for a in range(4):
                T = P[s][a]
                foo = 0 # expected cost to go for taking action a at s
                for i in range(len(T)):
                    rowp = T[i][1] // 5
                    colp = T[i][1] % 5
                    foo += T[i][0] * (T[i][2] + gamma * V[rowp,colp])
                if foo > newV:
                    newV = foo
            V[row,col] = newV
            Delta = max(Delta,abs(v-newV))
        if Delta < theta:
            break
    policy = np.zeros((25,4))
    for s in range(25):
        vals = np.zeros(4) # to take maximum later
        for a in range(4):
            T = P[s][a]
            for i in range(len(T)):
                rowp = T[i][1] // 5
                colp = T[i][1] % 5
                vals[a] += T[i][0] * (T[i][2] + gamma * V[rowp,colp])
        # get all elements of argmax
        a = np.argwhere(vals == np.amax(vals)).flatten().tolist()
        policy[s,a] = 1/len(a)
    return V, policy

def gridworld_ep(slip_prob=0.2):
    # slip_prob is the probability the agent slips.

    # dictionary of row,col -> state index
    statedict = {}
    ss = 0
    for row in range(5):
        for col in range(5):
            statedict[row,col] = ss
            ss += 1
    # load original gridworld
    P = gridworld(slip_prob=slip_prob)

    # Change the states
    # A' (4,1)
    # B' (2,3)
    # into sink states

    for a in range(4):
        P[statedict[4,1]][a] = [(1.0,statedict[4,1],0.0)]
        P[statedict[2,3]][a] = [(1.0,statedict[2,3],0.0)]

    return P

def visualizePolicy(policy):
    action2str = {0: 'n', 1: 'e', 2: 's', 3: 'w'}
    disp = []
    for row in range(5):
        foo = [] # row of disp
        for col in range(5):
            s = 5 * row + col
            prob = policy[s]
            string = ''
            for a in range(4):
                if prob[a]>0:
                    string += action2str[a]
            foo.append(string)
        disp.append(foo)
    for row in range(5):
        for col in range(5):
            print(disp[row][col]+"&\t",end="")
        print("")

def bisecsearch(P,tol=0.001,ub=1.0,lb=0.8):
    while True:
        g = (ub+lb)/2

        (V,policy) = value_iter(P,gamma=g)
        if policy[4,3]>0:
            lb = g
        else:
            ub = g

        if ub-lb < tol:
            break
    return (ub,lb)


if __name__ == "__main__":
    # (a)
    print('(a)')
    P = gridworld()

    #(b)
    print('(b)')
    V = policy_eval(P)
    print(np.round(V,decimals=2))

    #(c)
    print('(c)')
    (V,policy) = policy_iter(P)
    print(np.round(V,decimals=2))
    visualizePolicy(policy)

    #(d)
    print('(d)')
    (V,policy) = value_iter(P)
    print(np.round(V,decimals=2))
    visualizePolicy(policy)

    #(e)
    print('(e)')
    P = gridworld_ep()
    gammas = [1.0,0.8]
    for g in gammas:
        (V,policy) = value_iter(P,gamma=g)
        print('g='+str(g))
        print(np.round(V,decimals=2))
        visualizePolicy(policy)


    #(f)
    print('(f)')
    (ub,lb) = bisecsearch(P)
    gammas = [ub,lb]
    for g in gammas:
        (V,policy) = value_iter(P,gamma=g)
        print('g='+str(g))
        print(np.round(V,decimals=2))
        visualizePolicy(policy)
