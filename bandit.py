import numpy as np
import numpy.linalg as la
from tqdm import tqdm

# numba compiles simple python function just-in-time for speedup.
# not very useful in our case, since the numpy functions used to evaluate
# ESPs are not implemented efficiently.
from numba import njit,jit

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
    
    # initializing
    X_hat = np.zeros(N)
    total_reward=0
    files_seen=np.zeros(N)
    regret = np.zeros(T)


    eta = np.sqrt(k*np.log(N*np.exp(1)/k)/(3*T*N))
    gamma=eta*N
    pi=np.ones(N)*k/N # uniform distribution 
    
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

        if r>0:
            x_hat=np.zeros(N)
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
        for i in range(N):
            if i in S:
                x_hat[i] = 1 - (1.0-r)/p[i]
            else:
                x_hat[i] = 1
            # estimate reward
            # Q=_madowJointInclusionMatrix(p)
            # x_hat = r*(np.linalg.inv(Q) @ S_ind)

            # update cumulative reward
            X_hat = X_hat + x_hat
        
        pbar.update(1)
        pbar.set_description(f"IPS: Time: {t + 1} | Reward: {total_reward} | OPT: {opt} | Regret: {regret[t]:4f}")
        
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


# def sageLinBandit(seq,T,N,k,gamma=None,pi=None):
    
#     if pi==None:
#         pi=np.ones(N)*k/N
    
#     eta = np.sqrt(k*np.log(N*np.exp(1)/k)/(3*T*N))
#     eta_ff = np.sqrt(k*np.log(N*np.exp(1)/k)/T)
#     if gamma==None:
#         gamma=eta*N

#     X_hat_sp = np.zeros(N)
#     total_reward_sp =0
#     regret_sp = np.zeros(T)

#     X_hat = np.zeros(N)
#     total_reward=0
#     files_seen=np.zeros(N)
#     regret = np.zeros(T)

#     X_ff = np.zeros(N)
#     total_reward_ff=0
#     regret_ff = np.zeros(T)
#     pbar = tqdm(range(T), dynamic_ncols=True, leave=True,position=0) 
#     for t in tqdm(range(T),position=0, leave=True):
        
#         # LEAST SQUARE EST
#         # update weight
#         w = np.exp(eta*X_hat)

#         # compute p_i
#         e_k = _elementary_symmetric_polynomial(w, k)
        
#         p_dash = np.zeros(N)
#         for i in range(N):
#             w_i=np.delete(w,i)
#             p_dash[i]= (w[i]*_elementary_symmetric_polynomial(w_i,k-1)) / e_k
#         p=(1-gamma)*p_dash + gamma*pi

#         # sample set S
#         S=_madowSampling(N,p,k)
#         S_ind=np.zeros(N)
#         S_ind[S]=1
#         S_ind=np.array(S_ind)

#         # bandit feedback
#         r=int(seq[t] in S)
#         total_reward = total_reward + r
        
#         # optimal 
#         files_seen[seq[t]] += 1
#         opt = files_seen[(-files_seen).argsort()[:k]].sum()
#         regret[t] = (opt - total_reward)/ (t+1)

#         # estimate reward
#         Q=madowJointInclusionMatrix(p)
#         x_hat = r*(np.linalg.inv(Q) @ S_ind)

#         # update cumulative reward
#         X_hat = X_hat + x_hat

#         ###############################################
#         # FULL FEEDBACK
#         w_ff = np.exp(eta_ff*X_ff)

#         e_k_ff=_elementary_symmetric_polynomial(w_ff, k)
#         p_ff=np.zeros(N)
#         for i in range(N):
#             w_i_ff = np.delete(w_ff,i)
#             p_ff[i] = (w_ff[i]*_elementary_symmetric_polynomial(w_i_ff,k-1)) / e_k_ff
        
#         S_ff = _madowSampling(N,p_ff,k)
#         S_ff_ind = np.zeros(N)
#         S_ff_ind[S_ff]=1
#         S_ff_ind=np.array(S_ff_ind)

#         if seq[t] in S_ff:
#             total_reward_ff += 1
#         regret_ff[t] = (opt - total_reward_ff)/(t+1)

#         X_ff[seq[t]] += 1


#         ##################################################
#         # SPARSE ESTIMATOR
        
#         # update weight
#         w_sp = np.exp(eta*X_hat_sp)
        
#         # compute p_i
#         e_k_sp = _elementary_symmetric_polynomial(w_sp, k)
#         p_sp = np.zeros(N)
#         for i in range(N):
#             w_i_sp = np.delete(w_sp,i)
#             p_sp[i]= (w_sp[i]*_elementary_symmetric_polynomial(w_i_sp,k-1))/e_k_sp
        
#         # sample set
#         S_sp=_madowSampling(N,p_sp,k)
#         S_ind_sp=np.zeros(N)
#         S_ind_sp[S_sp] = 1
#         S_ind_sp= np.array(S_ind_sp)

#         # bandit feedback
#         r_sp=int(seq[t] in S_sp)
#         total_reward_sp += r_sp
#         regret_sp[t]=(opt-total_reward_sp)/(t+1)

#         # estimate reward
#         Q_sp=madowJointInclusionMatrix(p_sp)
#         temp_sp = np.zeros(N)
#         for i in range(N):
#             t1=np.dot(S_ind_sp,Q_sp[:,i])/(la.norm(Q_sp[:,i])**2)
#             temp_sp[i]=la.norm(S_ind_sp - t1*Q_sp[:,i])
#         x_hat_sp=np.zeros(N)
#         x_hat_sp[np.argmin(temp_sp)]=2
        
#         X_hat_sp += x_hat_sp

#         pbar.update(1)
#         pbar.set_description(
#             f"Time: {t + 1} | Reward LinBandit: {total_reward} | Reward FF: {total_reward_ff}|  Reward sp: {total_reward_sp} | OPT: {opt} \n"
#             f"| LinBandit Regret:{regret[t]:4f} | FF Regret: {regret_ff[t]:4f} | Sp Regret: {regret_sp[t]:4f}"
#         )
#     return regret, regret_ff, regret_sp