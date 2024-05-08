import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import datasp
import models
import data_utils
import utils

import argparse
import pickle

from tqdm import tqdm

import time



##################################################################################### 
#################################PARAMETERS########################################## 
##################################################################################### 


def parse_arguments():
    parser = argparse.ArgumentParser(description='Set parameters for the program.')

    parser.add_argument('--Vs', type=float, default=40, 
                        help='Nr sampling nodes')
    parser.add_argument('--N_EPOCHS', type=int, default=100, 
                        help='N EPOCHS train')
    parser.add_argument('--seed_n', type=int, default=0, 
                        help='Seed number')
    
    parser.add_argument('--beta', type=float, default=30., 
                        help='Beta Smooth')
    parser.add_argument('--lr', type=float, default=0.0001, 
                        help='Learning Rate')
    parser.add_argument('--N_batches', type=int, default=100, 
                        help='N Batches in one Epoch')
    parser.add_argument('--bs_X', type=int, default=16, 
                        help='How many floyd warshalls in a batch')
    parser.add_argument('--ps_f', type=float, default=0.01, 
                        help='How many paths in one floyd warshall (factor)')
    
    parser.add_argument('--load_model', type=int, default=0, 
                        help='Load previous model?')  
    
    parser.add_argument('--dev', type=str, default='cpu', 
                        help='Device to use')
    
    
    return parser.parse_args()


# Parsing arguments
args = parse_arguments()

N_EPOCHS = args.N_EPOCHS
seed_n = args.seed_n

beta_smooth = args.beta
lr = args.lr
N_batches = args.N_batches
bs_X = args.bs_X
ps_f = args.ps_f

load_model = args.load_model

dev = args.dev

path_data = './cabspotting_preprocessing/'

d_large = torch.tensor(2500.)

epochs_wait = 4


##################################################################################### 
##################################################################################### 
#####################################################################################


df_features = pd.read_csv(f'{path_data}features_per_trip_useful.csv')

df_trips = pd.read_csv(f'{path_data}full_useful_trips.csv')

df_edges = pd.read_csv(f'{path_data}graph_0010_080.csv')

df_nodes = pd.read_csv(f'{path_data}nodes_0010_080.csv')
df_nodes['node_sorted'] = df_nodes['node_id_new']

# We want to train in part of the drivers
unique_drivers = df_trips['driver'].drop_duplicates()
selected_drivers = unique_drivers.sample(frac=0.7, random_state=seed_n)
df_trips = df_trips[df_trips['driver'].isin(selected_drivers)]

df_trips = df_trips[df_trips.groupby('trip_id_new').node_id.transform('nunique')>1]
df_trips = df_trips.sort_values(by=['driver','trip_id_new','date_time'])
df_features['day_of_Week'] = df_features['day_of_Week'].astype(int).map({
    0: 0, 1: 0, 2: 0, 3: 0, 4: 1,
    5: 2, 
    6: 3 })

df_features = pd.get_dummies(df_features, columns=['day_of_Week'])
df_features['time_start'] = (df_features['time_start'] - df_features['time_start'].min()) / (df_features['time_start'].max() - df_features['time_start'].min())

indices_trips = df_trips[['trip_id','driver','trip_id_new']].drop_duplicates()
df_features = indices_trips.merge(df_features, on=['trip_id','driver'], how='left')
df_features.iloc[:,-4:] = df_features.iloc[:,-4:].astype(int)

feats = ['day_of_Week_0','day_of_Week_1','day_of_Week_2','day_of_Week_3',
         'is_Holiday','time_start']
n_features = len(feats)

n_trips = len(df_features)

prior_M, edges_prior, M_indices = data_utils.get_prior_and_M_indices(
    df_nodes, df_edges)

assert (df_trips.trip_id_new.unique() == df_features.trip_id_new.unique()).all()

trip_ids = df_trips.trip_id_new.unique()

V = M_indices.max()+1
X_np = np.array(df_features[feats])
node_idx_sequence_trips = df_trips.groupby('trip_id_new')['node_id'].apply(list)
edges_seq_original = node_idx_sequence_trips.apply(
    lambda x: np.column_stack([x[:-1], x[1:]]))
start_nodes_original = node_idx_sequence_trips.apply(
    lambda x: x[0])
end_nodes_original = node_idx_sequence_trips.apply(
    lambda x: x[-1])

edges_idx_on_original = np.zeros((len(edges_seq_original), 
                                  len(M_indices)), dtype=int)
edges_seq_original_np = np.array(edges_seq_original)

N_train = len(edges_seq_original)

print('Processing Data')
for i in tqdm(range(len(edges_seq_original))):
    matching_indices = []
    for row in edges_seq_original_np[i]:
        idx = np.where(np.isin(M_indices[:,0], row[0])\
                       *np.isin(M_indices[:,1], row[1]))[0].item()
        edges_idx_on_original[i, idx] = 1

edges_seq_original = list(edges_seq_original)
node_idx_sequence_trips = list(node_idx_sequence_trips)

end_to_end_nodes_original = np.vstack((
    np.array(start_nodes_original), 
    np.array(end_nodes_original))).T

edges_idx_on_original_tensor = torch.tensor(
    edges_idx_on_original, dtype=torch.float32)


X = torch.tensor(X_np, dtype=torch.float32)

ps_in_batch = int(ps_f*X.shape[0])

# Should we use nodes sampling during training?
Vs = int(args.Vs)
bool_scale = False
if Vs < V and Vs > 0:
    bool_scale = True
else:
    # Todo: treat this case
    print('Use Vs<V!!!')
    exit()

##################################################################################### 
######################### MODEL LOAD OR CREATE ###################################### 
#####################################################################################

print('----- Model Load or Create -----')


inp_s_model = X.shape[-1]
model = models.ANN(
    input_size=inp_s_model, output_size=len(M_indices), hl_sizes=[1024, 1024])    
model = model.to(dev)


criterion = torch.nn.KLDivLoss(reduction='none')
def cross_entropy_cont(target, prediction):
    return criterion(torch.log(prediction + 0.00001), target).sum(-1)

model_path = f'saved_models/cabspot_{Vs}_{seed_n}_{ps_f}.pkl'
model_path_inter = f'saved_models/cabspot_inter_{Vs}_{seed_n}_{ps_f}'

if load_model:
    try:
        model = models.ANN(
            input_size=inp_s_model, 
            output_size=len(M_indices), 
            hl_sizes=[1024, 1024])
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device(dev)))
        model = model.to(dev)
        print('MODEL LOADED')
    except:
        print('FAILED TO LOAD')
        pass
else:
    print('MODEL CREATED')
    pass

model = model.to(dev)
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=10e-5)
print('MODEL ON ', next(model.parameters()).device)

print('----- Model Load or Create Finished -----')


##################################################################################### 
##################################################################################### 
#####################################################################################


prior_dY = 0.0*prior_M[M_indices[:,0], M_indices[:,1]].to(dev)
prior_M = prior_M.to(dev)
M_indices = M_indices.to(dev)

elements, frequencies = utils.get_nodes_and_freqs(node_idx_sequence_trips)

not_best_count_accum = 0
loss_batch_avg_best = torch.inf


for epochs in range(0,N_EPOCHS):
    
    loss_batch_avg = 0
    
    for batch in range(0, N_batches): 
        
        selected_indexes, selected_trips, nodes_selected, nodes_excluded =\
        utils.selected_trips_and_idx(
                node_idx_sequence_trips, M_indices, 
            elements, frequencies, Vs, V)
        if selected_indexes == None:
            continue
        X_selected = X[selected_indexes]
        
        idcs_batch = data_utils.generate_n_combinations(
            X_selected, ps_in_batch-1, bs_X)
              
        start_time = time.time() 

        loss_batch = torch.tensor(0.)
        
        X_batch = X_selected[idcs_batch[:,0]].to(dev)
        dY = model(X_batch)                
        M_Y_pred = data_utils.costs_to_matrix(prior_M, M_indices, dY)

        M_Y_pred_selected, M_indices_selected_mapped =\
        utils.select_Ms_from_selected_idx_and_trips(
        M_Y_pred, Vs, M_indices, nodes_excluded, nodes_selected, 
            torch.tensor(beta_smooth), dev) 
        
        M_Y_pred_selected_shuf, M_indices_selected_mapped_shuf, selected_trips_shuf =\
        utils.shuffle_nodes_order(
            Vs, M_Y_pred_selected, M_indices_selected_mapped, selected_trips)

        # Inference: Compute P
        probs_pred = datasp.datasp(
            M_Y_pred_selected_shuf,        
            M_indices_selected_mapped_shuf,
            dev, 
            torch.tensor(beta_smooth), 
            d_large)
                
        # Groundtruth: Compute F
        mib = data_utils.get_m_inter_batch(selected_trips_shuf, idcs_batch, Vs, Vs)
        mib = torch.tensor(mib, dtype=torch.float32).to(dev)
        m_inter_total = mib.sum(1)/mib.sum(1).sum(-1).unsqueeze(-1)

        mask = ~torch.isnan(m_inter_total)
        true_paths_dist = m_inter_total[mask].reshape(-1, Vs)
        pred_paths_dist = probs_pred[mask].reshape(-1, Vs)
        loss_main = cross_entropy_cont(true_paths_dist, pred_paths_dist).mean()
               
        reg = (dY - prior_dY).pow(2).mean()

        loss_total = loss_main + 0.00001*reg
        
        opt.zero_grad()
        loss_total.backward()
        opt.step()
        
        print('Batch', batch, round(loss_main.item(), 3), round(reg.item(), 3), 
              '\tTime: ', round(time.time() - start_time, 3))
                
        loss_batch_avg += (loss_main/N_batches).detach() 
        
        
    if loss_batch_avg>=loss_batch_avg_best:
        not_best_count_accum = not_best_count_accum + 1
        print('Did not improve results nr ', not_best_count_accum)
    else:
        _ = data_utils.check_or_create_folder("saved_models")
        torch.save(model.state_dict(), model_path_inter + f'_{epochs}.pkl') 
        loss_batch_avg_best = loss_batch_avg
        not_best_count_accum = 0

    _ = data_utils.check_or_create_folder("saved_models")
    torch.save(model.state_dict(), model_path)            
    print('Batches AVG:', loss_batch_avg.item())

    if not_best_count_accum >= epochs_wait:
        print('Converged, exiting')
        break