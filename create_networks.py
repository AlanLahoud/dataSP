import numpy as np

def get_bin_M(M=20, sparsity=4, seed_n=42):
    np.random.seed(seed_n)
    exp_m = np.zeros((M,M))
    bin_m = np.zeros((M,M))
    for i in range(0, M):
        for j in range(0, M):
            if j>i:
                exp_m[i, j] = np.exp(-sparsity*(j-i)/(M))
    bin_m = np.random.binomial(1, exp_m, exp_m.shape)
    for i in range(0, M):
        for j in range(0, M):
            if j==i+1:
                bin_m[i, j] = 1
    return bin_m

def get_adj_cost_M(bin_m, seed_n=42):
    np.random.seed(seed_n)
    M = bin_m.shape[0]
    cost_m = np.zeros((M,M))
    for i in range(0, M):
        for j in range(0, M):
            if bin_m[i, j] == 1:
                cost_m[i, j] = np.clip(10*(j-i) + 4*np.random.normal(), 1, None)
            else:
                cost_m[i, j] = 100*M            
    return cost_m