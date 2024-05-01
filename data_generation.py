import numpy as np
import random
import torch

class ArtificialDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        return

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_i = self.X[idx]
        y_i = self.y[:,idx,:]
        return X_i, y_i


def gen_intermediate(num, n_assets, x1, x2, x3):      
        factor = num * 2/(n_assets)
        return x1**factor + x2**factor + x3**factor
    
    
def gen_data(N, n_assets, nl, seed_number=42, samples_dist=1):
    np.random.seed(seed_number)
    x1 = np.random.normal(1, 1, size = N).clip(0)
    x2 = np.random.normal(1, 1, size = N).clip(0)
    x3 = np.random.normal(1, 1, size = N).clip(0)
    X = np.vstack((x1, x2, x3)).T
    
    def geny(x1, x2, x3, N, n_assets, sn):
        np.random.seed(sn)
        Y = np.zeros((N, n_assets))
        for i in range(1, n_assets + 1):
            interm = gen_intermediate(i, n_assets, x1, x2, x3)
            Y[:,i-1] = 0.3*(np.sin(interm) - np.sin(interm).mean())

            sz = Y[:,0].shape[0]
            szh = (Y[:,0].shape[0])//2
        
            np.random.seed(sn+10*i)
            noise = np.hstack(
                (np.random.normal(-4, 4, szh), 
                 np.random.normal(4, 1, sz - szh))
            )            
            np.random.seed(sn+10*i)
            np.random.shuffle(noise)                        
            Y[:,i-1] = Y[:,i-1] + nl*(noise)                   
        return 30*Y
        
    Y = geny(x1, x2, x3, N, n_assets, seed_number)
        
    Y_dist = np.zeros((samples_dist, N, n_assets))
    for i in range(0, samples_dist):
        Y_dist[i, :, :] = geny(x1, x2, x3, N, n_assets, seed_number+i)
        
    return X, Y, Y_dist


def gen_cond_dist(N, n_assets, n_samples, nl, seed_number=420):
    np.random.seed(seed_number)
    Y_dist = np.zeros((n_samples, N, n_assets))
    for i in range(0, n_samples):
        Y_dist[i, :, :] = gen_data(N, n_assets, nl, seed_number=np.random.randint(0,999999))[1]
    return Y_dist