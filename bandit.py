import numpy as np
import numpy.linalg as la
from tqdm import tqdm
from numba import njit,jit
import sys

@njit
def _madowJointInclusionMatrix(p):
    N=p.size
    P=np.zeros(N)
    for i in range(1,N):
        P[i]=P[i-1]+p[i]

    Pkl=np.zeros((N,N))
    for k in range(N):
        for l in range(k+1,N):
            Pkl[k,l]=p[k:l].sum()

    pkl = np.zeros((N,N))
    for k in range(N):
        for l in range(k+1,N):
            delta= Pkl[k,l]-np.floor(Pkl[k,l]) #np.modf(Pkl[k,l])
            pkl[k,l]=min( max(0, p[k]- delta), p[l]) + min(p[k],max(0, delta+p[l]-1))
            pkl[l,k]=pkl[k,l]
        pkl[k,k]=p[k]
    
    return pkl

@njit
def _next_power_of_2(n):
    count = 0
    if n and not (n & (n - 1)):
        return n
    while n != 0:
        n >>= 1
        count += 1
    return 1 << count

# @njit
def _elementary_symmetric_polynomial(X, k):
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
    return X_.flatten()[k]

# @jit
def _madowSampling(N, p, k):
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

##############################################################
def bandit_ips_estimator(seq,T,N,k):
    X_hat = np.zeros(N)
    total_reward=0
    files_seen=np.zeros(N)
    regret = np.zeros(T)
    eta = np.sqrt(k*np.log(N*np.exp(1)/k)/(3*T*N))
    gamma=eta*N
    pi=np.ones(N)*k/N
    
    pbar = tqdm(range(T), dynamic_ncols=True, leave=True,position=0) 
    for t in range(T):
        # update weight
        w = np.exp(eta*X_hat)

        # compute p_i
        e_k = _elementary_symmetric_polynomial(w, k)
        
        p_dash = np.zeros(N)
        for i in range(N):
            w_i=np.delete(w,i)
            p_dash[i]= (w[i]*_elementary_symmetric_polynomial(w_i,k-1)) / e_k
        p=(1-gamma)*p_dash + gamma*pi

        # sample set S
        S=_madowSampling(N,p,k)
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

        x_hat=np.zeros(N)
        if r>0:
            for i in range(N):
                if i in S:
                    x_hat[i]=1.0/p[i]
            # estimate reward
            # Q=_madowJointInclusionMatrix(p)
            # x_hat = r*(np.linalg.inv(Q) @ S_ind)

            # update cumulative reward
        X_hat = X_hat + x_hat
        
        pbar.update(1)
        pbar.set_description(f"IPS: Time: {t + 1} | Reward: {total_reward} | OPT: {opt} | Regret: {regret[t]:4f}")
        
    return regret, total_reward

##############################################################
def bandit_ipsl_estimator(seq,T,N,k):
    X_hat = np.zeros(N)
    total_reward=0
    files_seen=np.zeros(N)
    regret = np.zeros(T)
    w=np.zeros(N)
    # might not be the best possible eta
    eta = np.sqrt(k*np.log(N*np.exp(1)/k)/(3*T*N))
    e_k=0
    # exploration parameter
    gamma=eta*N
    
    # uniform distribution
    pi=np.ones(N)*k/N
    
    # progress bar initialized
    pbar = tqdm(range(T), dynamic_ncols=True, leave=True,position=0)
    for t in range(T):
        # update weight
        # if t % 1000 == 0:
        #     print(f"t:{t}, \n X_hat: { X_hat}, \n w: {w},  \n e_k: {e_k}")
        try:
            m=0#np.floor((min(X_hat)+max(X_hat))/2)
            w = np.exp(eta*(X_hat-m))
            # if sum(w)>1 or sum(w)<1e-01:
            # w = w / sum(w)
        except FloatingPointError:
            print(f"X_hat: { X_hat}, w: {w}")
            sys.exit(0)
            
        # compute p_i
        e_k = _elementary_symmetric_polynomial(w, k)
        
        p_dash = np.zeros(N)
        for i in range(N):
            w_i=np.delete(w,i)
            try:
                a=_elementary_symmetric_polynomial(w_i,k-1)
                p_dash[i]= (w[i]*a) / e_k
            except FloatingPointError:
                print(f' a: {a}, e_k: {e_k}, w_i: {w_i}') # type: ignore
                sys.exit(0)
                    
        # explicit exploration
        p=(1-gamma)*p_dash + gamma*pi

        # sample set S
        S=_madowSampling(N,p,k)
        S_ind=np.zeros(N)
        S_ind[S]=1
        S_ind=np.array(S_ind)

        # bandit feedback
        r=int(seq[t] in S)
        total_reward = total_reward + r
        
        # optimal in hindsight
        files_seen[seq[t]] += 1
        opt = files_seen[(-files_seen).argsort()[:k]].sum()
        regret[t] = (opt - total_reward)/ (t+1)

        x_hat=np.zeros(N)
        for i in range(N):
            if i in S:
                x_hat[i] = 1 - (1.0-r)/p[i]
            else:
                if r == 0:
                    x_hat[i] = 1
                    
            # estimate reward
            # Q=_madowJointInclusionMatrix(p)
            # x_hat = r*(np.linalg.inv(Q) @ S_ind)

            # update cumulative reward
        X_hat = X_hat + x_hat
        
        pbar.update(1)
        pbar.set_description(f"IPSL: Time: {t + 1} | Reward: {total_reward} | OPT: {opt} | Regret: {regret[t]:4f}")
        
    return regret, total_reward

def bandit_ipsf_estimator(seq,T,N,k):
    X_hat = np.zeros(N)
    total_reward=0
    files_seen=np.zeros(N)
    regret = np.zeros(T)
    w=np.zeros(N)
    # might not be the best possible eta
    eta = np.sqrt(k*np.log(N*np.exp(1)/k)/(3*T*N))
    e_k=0
    # exploration parameter
    gamma=eta*N
    
    # uniform distribution
    pi=np.ones(N)*k/N
    
    # progress bar initialized
    pbar = tqdm(range(T), dynamic_ncols=True, leave=True,position=0)
    for t in range(T):
        # update weight
        # if t % 1000 == 0:
        #     print(f"t:{t}, \n X_hat: { X_hat}, \n w: {w},  \n e_k: {e_k}")
        try:
            m=0#np.floor((min(X_hat)+max(X_hat))/2)
            w = np.exp(eta*(X_hat-m))
            # if sum(w)>1 or sum(w)<1e-01:
            # w = w / sum(w)
        except FloatingPointError:
            print(f"X_hat: { X_hat}, w: {w}")
            sys.exit(0)
            
        # compute p_i
        e_k = _elementary_symmetric_polynomial(w, k)
        
        p_dash = np.zeros(N)
        for i in range(N):
            w_i=np.delete(w,i)
            try:
                a=_elementary_symmetric_polynomial(w_i,k-1)
                p_dash[i]= (w[i]*a) / e_k
            except FloatingPointError:
                print(f' a: {a}, e_k: {e_k}, w_i: {w_i}') # type: ignore
                sys.exit(0)
                    
        # explicit exploration
        p=(1-gamma)*p_dash + gamma*pi

        # sample set S
        S=_madowSampling(N,p,k)
        S_ind=np.zeros(N)
        S_ind[S]=1
        S_ind=np.array(S_ind)

        # bandit feedback
        r=int(seq[t] in S)
        total_reward = total_reward + r
        
        # optimal in hindsight
        files_seen[seq[t]] += 1
        opt = files_seen[(-files_seen).argsort()[:k]].sum()
        regret[t] = (opt - total_reward)/ (t+1)

        x_hat=np.zeros(N)
        for i in range(N):
            if i in S:
                x_hat[i] = r
            else:
                if r == 0:
                    x_hat[i] = 1
                    
            # estimate reward
            # Q=_madowJointInclusionMatrix(p)
            # x_hat = r*(np.linalg.inv(Q) @ S_ind)

            # update cumulative reward
        X_hat = X_hat + x_hat
        
        pbar.update(1)
        pbar.set_description(f"IPSF: Time: {t + 1} | Reward: {total_reward} | OPT: {opt} | Regret: {regret[t]:4f}")
        
    return regret, total_reward

##########################################################################

def hedge(seq,T,N,k):
    X = np.zeros(N)
    total_reward=0
    regret = np.zeros(T)
    eta = np.sqrt(k*np.log(N*np.exp(1)/k)/T)
    files_seen=np.zeros(N)

    pbar = tqdm(range(T), dynamic_ncols=True, leave=True,position=0)
    for t in  range(T):
        # FULL FEEDBACK
        w = np.exp(eta*X)

        e_k=_elementary_symmetric_polynomial(w, k)
        p=np.zeros(N)
        for i in range(N):
            w_i = np.delete(w,i)
            p[i] = (w[i]*_elementary_symmetric_polynomial(w_i,k-1)) / e_k
        
        S = _madowSampling(N,p,k)
        S_ind = np.zeros(N)
        S_ind[S]=1
        S_ind=np.array(S_ind)

        # optimal 
        files_seen[seq[t]] += 1
        opt = files_seen[(-files_seen).argsort()[:k]].sum()
        regret[t] = (opt - total_reward)/ (t+1)


        if seq[t] in S:
            total_reward += 1
        regret[t] = (opt - total_reward)/(t+1)

        X[seq[t]] += 1

        pbar.update(1)
        pbar.set_description(f"Hedge: Time: {t + 1} | Reward: {total_reward} | OPT: {opt} | Regret:{regret[t]:4f}")
        
    return regret, total_reward

###################################################################################

def bandit_least_sq_estimator(seq,T,N,k):
    eta = np.sqrt(k*np.log(N*np.exp(1)/k)/(3*T*N))

    pi=np.ones(N)*k/N
    gamma=eta*N
    
    X_hat = np.zeros(N)
    total_reward=0
    files_seen=np.zeros(N)
    regret = np.zeros(T)

    pbar = tqdm(range(T), dynamic_ncols=True, leave=True,position=0) 
    for t in range(T):
        # LEAST SQUARE EST
        # update weight
        w = np.exp(eta*X_hat)

        # compute p_i
        e_k = _elementary_symmetric_polynomial(w, k)
        
        p_dash = np.zeros(N)
        for i in range(N):
            w_i=np.delete(w,i)
            p_dash[i]= (w[i]*_elementary_symmetric_polynomial(w_i,k-1)) / e_k
        p=(1-gamma)*p_dash + gamma*pi

        # sample set S
        S=_madowSampling(N,p,k)
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
        Q=_madowJointInclusionMatrix(p)
        x_hat = r*(np.linalg.inv(Q) @ S_ind)

        # update cumulative reward
        X_hat = X_hat + x_hat
        pbar.update(1)
        pbar.set_description(f"Least Sq: Time: {t + 1} | Reward: {total_reward} | OPT: {opt} | Regret:{regret[t]:4f}")
        
    return regret, total_reward

#######################################################################

def bandit_sparse_estimator(seq,T,N,k):
    X_hat = np.zeros(N)
    total_reward =0
    regret = np.zeros(T)
    eta = np.sqrt(k*np.log(N*np.exp(1)/k)/(3*T*N))
    files_seen=np.zeros(N)
    pbar = tqdm(range(T), dynamic_ncols=True, leave=True,position=0) 
    for t in range(T):
        # update weight
        w = np.exp(eta*X_hat)
        
        # compute p_i
        e_k = _elementary_symmetric_polynomial(w, k)
        p = np.zeros(N)
        for i in range(N):
            w_i = np.delete(w,i)
            p[i]= (w[i]*_elementary_symmetric_polynomial(w_i,k-1))/e_k
        
        # sample set
        S=_madowSampling(N,p,k)
        S_ind=np.zeros(N)
        S_ind[S] = 1
        S_ind= np.array(S_ind)

        # optimal 
        files_seen[seq[t]] += 1
        opt = files_seen[(-files_seen).argsort()[:k]].sum()
        regret[t] = (opt - total_reward)/ (t+1)

        # bandit feedback
        r=int(seq[t] in S)
        total_reward += r
        regret[t]=(opt-total_reward)/(t+1)

        # estimate reward
        Q=_madowJointInclusionMatrix(p)
        temp1 = np.zeros(N)
        temp2=np.zeros(N)
        for i in range(N):
            temp2[i]=np.dot(S_ind,Q[:,i])/(la.norm(Q[:,i])**2)
            temp1[i]=la.norm(S_ind - temp2*Q[:,i])
        x_hat=np.zeros(N)
        temp1[S_ind == 0]=np.inf
        b=np.argmin(temp1)
        x_hat[b]=temp2[b]
        
        X_hat += x_hat
        pbar.update(1)
        pbar.set_description(f"Sparse: Time: {t + 1} | Reward: {total_reward} | OPT: {opt} | Regret:{regret[t]:4f}")
        
    return regret, total_reward