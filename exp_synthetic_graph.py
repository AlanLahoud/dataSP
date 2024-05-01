import torch
import torch.nn as nn
import numpy as np
import datasp
import models
import create_networks
import data_generation
import itertools
import data_utils
import utils
import time
import os
import argparse
import pickle
from tqdm import tqdm


##################################################################################### 
#################################PARAMETERS########################################## 
##################################################################################### 
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Set parameters')
    
    parser.add_argument('--M', type=int, default=30, 
                        help='Number of nodes')
    
    parser.add_argument('--sparsity', type=float, default=1.5, 
                        help='Sparsity on adj. matrix')
    
    parser.add_argument('--N_train', type=int, default=5000, 
                        help='Number of training samples')
    
    parser.add_argument('--N_val', type=int, default=800, 
                        help='Number of validation samples')
    
    parser.add_argument('--N_test', type=int, default=2000, 
                        help='Number of test samples')
    
    parser.add_argument('--noise_data', type=float, default=0.01, 
                        help='Noise in data')
    
    parser.add_argument('--perc_end_nodes_seen', type=float, default=0.5, 
                        help='Percentage of end nodes seen')

    parser.add_argument('--N_EPOCHS', type=int, default=100, 
                        help='N EPOCHS train')
    
    parser.add_argument('--seed_n', type=int, default=0, 
                        help='Seed number')
    
    parser.add_argument('--beta', type=float, default=1., 
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
        
    parser.add_argument('--Vs', type=float, default=-1, 
                        help='Nr sampling nodes')

    return parser.parse_args()



# Parsing arguments
args = parse_arguments()

# Assigning arguments to variables
M = args.M
V = M

sparsity = args.sparsity
N_train = args.N_train
N_val = args.N_val
N_test = args.N_test
noise_data = args.noise_data
perc_end_nodes_seen = args.perc_end_nodes_seen
N_EPOCHS = args.N_EPOCHS
seed_n = args.seed_n
beta_smooth = args.beta
lr = args.lr
N_batches = args.N_batches
bs_X = args.bs_X
ps_f = args.ps_f
load_model = args.load_model
ps_in_batch = int(ps_f*N_train)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Should we use nodes sampling during training?
Vs = int(args.Vs)
bool_scale = False
if Vs < V and Vs > 0:
    bool_scale = True
else:
    Vs = V
    
# Stop if results do not improve
epochs_wait = 4

# Consider this value infinite distance
d_large = torch.tensor(2500.)

print(f'RUNNING WITH {dev}, V={M}, Vs={Vs}')

##################################################################################### 
##################################################################################### 
#####################################################################################


##################################################################################### 
######################GENERATE AND PROCESS DATA###################################### 
#####################################################################################

print('----- Start Generating and Processing Data -----')

np.random.seed(seed_n)
torch.manual_seed(seed_n)

bin_M, prior_M, E, M_indices = data_utils.create_graph(M, sparsity, seed_n)

X, X_val, X_test, dY, dY_val, dY_test, M_Y, M_Y_val, M_Y_test = \
data_utils.generate_synthetic_data(
    N_train, N_val, N_test, noise_data, E, M_indices, prior_M, seed_n)

seen_permutations, unseen_permutations = \
data_utils.source_end_nodes_permutation(M, perc_end_nodes_seen)

end_to_end_nodes_train = \
data_utils.gen_source_end_nodes_train(seen_permutations, N_train)

paths_demonstration_train = \
data_utils.gen_paths(end_to_end_nodes_train, N_train, M_Y, BBB=50)

end_to_end_nodes_val = \
data_utils.gen_source_end_nodes_train(seen_permutations+unseen_permutations, N_val)

paths_demonstration_val = \
data_utils.gen_paths(end_to_end_nodes_val, N_val, M_Y_val, BBB=50)


nodes_in_cluster_sorted = np.arange(0, M)
data_processed = data_utils.process_paths(
    paths_demonstration_train, nodes_in_cluster_sorted, M_indices,
    seed_n, M, sparsity, noise_data, perc_end_nodes_seen)
node_idx_sequence_trips = data_processed["node_idx_sequence_trips"]
edges_seq_original = data_processed["edges_seq_original"]
edges_idx_on_original = data_processed["edges_idx_on_original"]
start_nodes_original = data_processed["start_nodes_original"]
end_nodes_original = data_processed["end_nodes_original"]


data_processed_val = data_utils.process_paths(
    paths_demonstration_val, nodes_in_cluster_sorted, M_indices,
    seed_n, M, sparsity, noise_data, perc_end_nodes_seen, False)
node_idx_sequence_trips_val = data_processed_val["node_idx_sequence_trips"]
edges_seq_original_val = data_processed_val["edges_seq_original"]
edges_idx_on_original_val = data_processed_val["edges_idx_on_original"]
start_nodes_original_val = data_processed_val["start_nodes_original"]
end_nodes_original_val = data_processed_val["end_nodes_original"]


print('----- Finish Generating and Processing Data -----')

##################################################################################### 
##################################################################################### 
#####################################################################################



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


model_path = f'saved_models/{M}_{Vs}_{seed_n}_{sparsity}_{noise_data}.pkl'
if load_model:
    try:
        model = models.ANN(input_size=inp_s_model, 
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

# Heuristics to sample nodes (graph sampling)
elements, frequencies = utils.get_nodes_and_freqs(node_idx_sequence_trips)

not_best_count_accum = 0
loss_batch_avg_best = torch.inf
loss_val_best = torch.inf


for epochs in range(0,N_EPOCHS):
    
    loss_batch_avg = 0
    
    #for batch in tqdm(range(0, N_batches)):
    for batch in range(0, N_batches):
    
        if bool_scale:
            selected_indexes, selected_trips, nodes_selected, nodes_excluded =\
            utils.selected_trips_and_idx(
                node_idx_sequence_trips, M_indices, elements, frequencies, Vs, V)
            if selected_indexes == None:
                continue
            X_selected = X[selected_indexes]
        else:
            selected_trips = node_idx_sequence_trips
            X_selected = X

        idcs_batch = data_utils.generate_n_combinations(
            X_selected, ps_in_batch-1, bs_X)
        start_time = time.time() 

        X_batch = X_selected[idcs_batch[:,0]].to(dev)
        dY = model(X_batch)                
        M_Y_pred = data_utils.costs_to_matrix(prior_M, M_indices, dY)

        if bool_scale:        
            M_Y_pred_selected, M_indices_selected_mapped =\
            utils.select_Ms_from_selected_idx_and_trips(
                M_Y_pred, Vs, M_indices, 
                nodes_excluded, nodes_selected, 
                torch.tensor(beta_smooth), dev)            
        else:
            M_Y_pred_selected = M_Y_pred
            M_indices_selected_mapped = M_indices

        M_Y_pred_selected_shuf, M_indices_selected_mapped_shuf, selected_trips_shuf =\
        utils.shuffle_nodes_order(
            Vs, M_Y_pred_selected, M_indices_selected_mapped, selected_trips)

        #import pdb
        #pdb.set_trace()
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
        
        
        
    with torch.no_grad():
        dY_val = model(X_val.to(dev)) 
        M_Y_pred_val = data_utils.costs_to_matrix(prior_M, M_indices, dY_val)

        probs_pred_val = datasp.datasp(
            M_Y_pred_val,
            M_indices,
            dev,
            torch.tensor(beta_smooth),
            d_large
        ) 

        mib_val = data_utils.get_m_inter_batch(
            node_idx_sequence_trips_val, 
            np.expand_dims(np.arange(0,N_val), 1), M, M)
        mib_val = torch.tensor(mib_val, dtype=torch.float32).to(dev)
        m_inter_total_val = mib_val.sum(1)/mib_val.sum(1).sum(-1).unsqueeze(-1)

        mask_val = ~torch.isnan(m_inter_total_val)
        true_paths_dist_val = m_inter_total_val[mask_val].reshape(-1, M)
        pred_paths_dist_val = probs_pred_val[mask_val].reshape(-1, M)
        loss_mse_val = cross_entropy_cont(true_paths_dist_val, 
                                          pred_paths_dist_val).mean()


    if loss_mse_val>=loss_val_best:
        not_best_count_accum = not_best_count_accum + 1
        print('Did not improve results nr ', not_best_count_accum)
    else:
        loss_val_best = loss_mse_val
        not_best_count_accum = 0
        _ = data_utils.check_or_create_folder("saved_models")
        torch.save(model.state_dict(), model_path)

    print('Batches AVG:', loss_batch_avg.item(), '\t Val Loss:', 
          loss_mse_val.item())

    if not_best_count_accum >= epochs_wait:
        print('Converged, exiting')
        break
        

# Test results without noise
_, _, _, _, _, dY_test_mean, _, _, M_Y_mean_test =\
data_utils.generate_synthetic_data(
        1, 1, N_test, 0.0, E, M_indices, prior_M, seed_n)        
with torch.no_grad():        
    end_to_end_nodes_test = \
    data_utils.gen_source_end_nodes_train(
        seen_permutations+unseen_permutations, N_test)
    model = models.ANN(
        input_size=X.shape[-1], 
        output_size=len(M_indices), 
        hl_sizes=[1024, 1024])
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device('cpu')))
    dY_test_pred = model(X_test)

    M_opt_test_mean = data_utils.costs_to_matrix(
        prior_M, M_indices, dY_test_mean)
    M_opt_pred = data_utils.costs_to_matrix(
        prior_M, M_indices, dY_test_pred)
    
    paths_demonstration_test_mean = utils.batch_dijkstra(
        M_opt_test_mean, end_to_end_nodes_test)
    paths_demonstration_test_mean_M = utils.paths_to_M_indices(
        paths_demonstration_test_mean, M_indices)
    
    paths_pred_test = utils.batch_dijkstra(
        M_opt_pred.detach(), end_to_end_nodes_test)
    paths_pred_test_M = utils.paths_to_M_indices(
        paths_pred_test, M_indices)
    
    match_metric = utils.compute_metrics_batch(
        paths_demonstration_test_mean_M, paths_pred_test_M)[:,-1].mean()

    jaccard_index = utils.compute_jaccard_similarity(
         paths_demonstration_test_mean_M, paths_pred_test_M).mean()
    
    print("Results test data:")    
    print('Match:', match_metric)
    print('Jaccard:', jaccard_index)