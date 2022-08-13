import numpy as np

def generate_data_exp(N,T,beta):
    p:list[float]=[]
    for i in range(N):
        p.append(np.exp(-beta*i))
    p=np.array(p)/(sum(p))
    a=np.random.choice(N,T,p=p)
    np.save(f'data/data_exp_N{N}_T{T}_beta{beta}.npy',a)


def main():
    N=20
    T=10**5
    beta_list = [x/10 for x in range(1,11)]
    for beta in beta_list:
        generate_data_exp(N,T,beta)

if __name__ == "__main__":
    main()
