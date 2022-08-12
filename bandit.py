import numpy as np

def madowJointInclusionMatrix(p):
    N=p.size
    P=np.zeros(N+1)
    for i in range(1,N+1):
        P[i]=P[i-1]+p[i]

    Pkl=np.zeros((N,N))
    for k in range(N):
        for l in range(k+1,N):
            Pkl[k,l]=p[k:l].sum()

    pkl = np.zeros((N,N))
    for k in range(N):
        for l in range(k+1,N):
            delta=np.modf(P[k,l])
            pkl[k,l]=min( max(0, p[k]- delta), p[l]) + min(p[k],max(0, delta+p[l]-1))
            pkl[l,k]=pkl[k,l]
        pkl[k,k]=p[k]
    
    return pkl

def _next_power_of_2(n):
    count = 0
    if n and not (n & (n - 1)):
        return n
    while n != 0:
        n >>= 1
        count += 1
    return 1 << count

def elementary_symmetric_polynomial(X, k):
    X_ = np.zeros(_next_power_of_2(len(X)), dtype=np.float64)
    X_[:len(X)] = X
    W = np.ones_like(X_, dtype=np.float64)
    X_ = np.vstack((W, X_)).T

    K = X_.shape[0]
    while K > 1:
        X_temp = []
        for i in range(0, K, 2):
            x, y = list(X_[i]), list(X_[i + 1])
            X_temp.append(np.polymul(x, y)[:k + 1])
        X_ = np.asarray(X_temp)
        K = K // 2

def madowSampling(N, p, k):
    assert len(p) == N
    S = []
    p = np.insert(p, 0, 0)
    cum_p = np.cumsum(np.asarray(p))
    x = np.random.uniform()
    for i in range(k):
        for j in range(1, len(cum_p)):
            if cum_p[j - 1] <= x + i < cum_p[j]:
                S.append(j - 1)
    return S

def sageLinBandit(seq,T,N,k,eta,gamma,pi=None):
    
    if pi==None:
        pi=np.ones(N)/N
    X_hat = np.zeros(N)
    total_reward=0
    files_seen=np.zeros(N)
    regret = np.zeros(T)
    X_ff = np.zeros(N)
    total_reward_ff=0
    regret_ff = np.zeros(T)
    
    for t in range(T):

        # update weight
        w = np.exp(eta*X_hat)

        # compute p_i
        e_k = elementary_symmetric_polynomial(w, k)
        p_dash = np.zeros(N)
        for i in range(N):
            w_i=np.delete(w,i)
            p_dash[i]= (w[i]*elementary_symmetric_polynomial(w_i,k-1)) / e_k
        p=(1-gamma)*p_dash + gamma*pi

        # sample set S
        S=madowSampling(N,p,k)
        S_ind=np.zeros(N)
        S_ind[S]=1
        S_ind=np.array(S_ind)

        # bandit feedback
        r=int(seq[t] in S)
        total_reward = total_reward + r
        
        # optimal 
        files_seen[seq[t]] += 1
        opt = files_seen[(-files_seen).argsort()[:k]].sum()
        regret[t] = (opt - total_reward)/ (t+1)

        # estimate reward
        Q=madowJointInclusionMatrix(p)
        x_hat = r*(np.linalg.inv(Q) @ S_ind)

        # update cumulative reward
        X_hat = X_hat + x_hat

        # full feedback
        w_ff = np.exp(eta*X_ff)

        e_k_ff=elementary_symmetric_polynomial(w, k)
        p_ff=np.zeros(N)
        for i in range(N):
            w_i_ff = np.delete(w_ff,i)
            p_ff[i] = (w_ff[i]*elementary_symmetric_polynomial(w_i_ff,k-1)) / e_k_ff
        
        S_ff = madowSampling(N,p_ff,k)
        S_ff_ind = np.zeros(N)
        S_ff_ind[S_ff]=1
        S_ff_ind=np.array(S_ff_ind)

        if seq[t] in S_ff:
            total_reward_ff += 1
        regret_ff[t] = (opt - total_reward_ff)/(t+1)

    return regret, regret_ff
        


