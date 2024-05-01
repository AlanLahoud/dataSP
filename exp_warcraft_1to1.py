import argparse
import torch
import numpy as np
import os
import utils_ww
import utils
import data_utils
import datasp
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import time

##################################################################################### 
#################################PARAMETERS########################################## 
##################################################################################### 
        
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Set parameters for the program.')
            
    parser.add_argument('--N', type=int, default=18, 
                        help='Grid size')
    parser.add_argument('--Vs', type=int, default=70, 
                        help='Nr sampling nodes')
    
    parser.add_argument('--N_train', type=int, default=10000, 
                        help='Nr sampling nodes')
    parser.add_argument('--N_EPOCHS', type=int, default=100, 
                        help='N EPOCHS train')
    
    parser.add_argument('--beta', type=float, default=100., 
                        help='Beta Smooth')
    parser.add_argument('--lr', type=float, default=0.0002, 
                        help='Learning Rate')
    parser.add_argument('--N_batches', type=int, default=30, 
                        help='N Batches in one Epoch')
    parser.add_argument('--bs_X', type=int, default=24, 
                        help='How many floyd warshalls in a batch')
    
    parser.add_argument('--seed_n', type=int, default=0)   
    parser.add_argument('--load_model', type=int, default=0, 
                        help='Load previous model?')
        
    return parser.parse_args()

# Parsing arguments
args = parse_arguments()

seed_n = args.seed_n
torch.manual_seed(seed_n)
np.random.seed(seed_n)

N = args.N
Vs = args.Vs

N_train = args.N_train
N_EPOCHS = args.N_EPOCHS

beta_smooth = args.beta
lr = args.lr
N_batches = args.N_batches
bs_X = args.bs_X

load_model = args.load_model
ps_in_batch = 1

# Should we use nodes sampling during training?
Vs = int(args.Vs)
bool_scale = False
if Vs < N**2 and Vs > 0:
    bool_scale = True
else:
    # Todo: treat this case
    print('Use Vs<N**2!!!')
    exit()
    
    
epochs_wait = 10

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'RUNNING WITH {dev}')

M_indices = utils_ww.get_M_indices(N)

data_dir = f'./data_warcraft/{N}x{N}/'

data_suffix = "maps"
train_prefix = "train"
val_prefix = "val"

train_inputs = np.load(os.path.join(data_dir, train_prefix + "_" + data_suffix + ".npy")).astype('float')
train_weights = np.load(os.path.join(data_dir, train_prefix + "_vertex_weights.npy"))
train_labels = np.load(os.path.join(data_dir, train_prefix + "_shortest_paths.npy"))

val_inputs = np.load(os.path.join(data_dir, val_prefix + "_" + data_suffix + ".npy")).astype('float')
val_weights = np.load(os.path.join(data_dir, val_prefix + "_vertex_weights.npy"))
val_labels = np.load(os.path.join(data_dir, val_prefix + "_shortest_paths.npy"))

train_inputs = train_inputs.transpose(0,3,1,2)
val_inputs = val_inputs.transpose(0,3,1,2)

mean, std = (
    np.mean(train_inputs, axis=(0, 2, 3), keepdims=True),
    np.std(train_inputs, axis=(0, 2, 3), keepdims=True),
)

del(train_inputs)

val_inputs -= mean
val_inputs /= std

true_paths_nodes = []
for i in tqdm(range(0, N_train)):
    true_paths_nodes.append(utils_ww.get_path_nodes(M_indices, train_labels[i]))

val_paths_nodes = []
for i in tqdm(range(0, 1000)):
    val_paths_nodes.append(utils_ww.get_path_nodes(M_indices, val_labels[i]))
    
    
print('----- Finish Generating and Processing Data -----')

##################################################################################### 
##################################################################################### 
#####################################################################################


##################################################################################### 
######################### MODEL LOAD OR CREATE ###################################### 
#####################################################################################

print('----- Model Load or Create -----')

model = utils_ww.CombRenset18(N**2, 3)
model = model.to(dev)

prior_M = torch.zeros((N**2,N**2))

criterion = torch.nn.KLDivLoss(reduction='none')
def cross_entropy_cont(target, prediction):
    return criterion(torch.log(prediction + 0.00001), target).sum(-1)

model_path = f'saved_models/safw_ww_{N}_{beta_smooth}_{bs_X}_{lr}_{seed_n}.pkl'
if load_model:
    try:
        model = utils_ww.CombRenset18(N**2, 3)
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
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=10e-4)
print('MODEL ON ', next(model.parameters()).device)

print('----- Model Load or Create Finished -----')


##################################################################################### 
##################################################################################### 
#####################################################################################

val_inputs_tensor = torch.from_numpy(val_inputs).float()

M_indices = M_indices.to(dev)

elements, frequencies = utils.get_nodes_and_freqs(true_paths_nodes)

not_best_count_accum = 0
loss_batch_avg_best = torch.inf
perc_correct_best = 0.

scheduler = lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.01, total_iters=50)

imgs_tot = np.load(f'./data_warcraft/{N}x{N}/train_maps.npy', mmap_mode='r').astype('int')


for epochs in range(0,N_EPOCHS):

    loss_batch_avg = 0
    
    for batch in range(0, N_batches): 
        
        start_time = time.time()
        
        selected_indexes, selected_trips, nodes_selected, nodes_excluded =\
        utils.selected_trips_and_idx(
            true_paths_nodes, M_indices, elements, frequencies, Vs, N**2)
        if selected_indexes == None:
            continue

        idcs_batch = torch.randint(0, len(selected_indexes), (bs_X,)).unsqueeze(1)
        X_batch = imgs_tot[selected_indexes][idcs_batch[:,0]].astype('float32') 
        
        X_batch = X_batch.transpose(0,3,1,2)        
        X_batch -= mean
        X_batch /= std
    
        X_batch = torch.from_numpy(X_batch).float()
      
        X_batch = X_batch.to(dev)
    
        nodes_pred_batch = model(X_batch).clip(0.001)
        
        M_pred_batch = utils_ww.nodes_to_M_batch(nodes_pred_batch)
        
        M_Y_pred_selected, M_indices_selected_mapped =\
        utils.select_Ms_from_selected_idx_and_trips(
            M_pred_batch, Vs, M_indices, 
            nodes_excluded, nodes_selected, 
            torch.tensor(beta_smooth), dev)            

        M_Y_pred_selected_shuf, M_indices_selected_mapped_shuf, selected_trips_shuf =\
        utils.shuffle_nodes_order(
            Vs, M_Y_pred_selected, M_indices_selected_mapped, selected_trips)
        
        probs_pred = datasp.datasp(
            M_Y_pred_selected_shuf, 
            M_indices_selected_mapped_shuf, 
            dev, beta_smooth)
        
        mib = data_utils.get_m_inter_batch(
            selected_trips_shuf, idcs_batch, Vs, Vs)
        mib = torch.tensor(mib, dtype=torch.float32).to(dev)
        m_inter_total = mib.sum(1)/mib.sum(1).sum(-1).unsqueeze(-1)
        
        mask = ~torch.isnan(m_inter_total)
        true_paths_dist = m_inter_total[mask].reshape(-1, Vs)
        pred_paths_dist = probs_pred[mask].reshape(-1, Vs)
        loss_main = cross_entropy_cont(true_paths_dist, pred_paths_dist).mean()
        
        opt.zero_grad()
        loss_main.backward()
        opt.step()
        
        print('Batch', batch, round(loss_main.item(), 3), 
              '\tTime: ', round(time.time() - start_time, 3))
        
        loss_batch_avg += (loss_main/N_batches).detach()
        
        
    with torch.no_grad():
        N_eval = 1000
        nodes_pred = model(val_inputs_tensor.to(dev)).clip(0.001)
        M_pred = utils_ww.nodes_to_M_batch(nodes_pred)

        path_pred = utils.batch_dijkstra(M_pred.detach().cpu().numpy(), 
                                                   np.repeat(np.array([[0,N**2-1]]), N_eval, 0))

        path_pred_map_all = torch.zeros((N_eval, N**2))
        for i in range(0, N_eval):
            path_pred_map = torch.zeros((N**2,))
            path_pred_map[path_pred[i]] = 1
            path_pred_map_all[i] = path_pred_map

        path_pred_map_all = path_pred_map_all.reshape(-1,N,N)

        cost_pred = (path_pred_map_all*val_weights[:N_eval]).sum(-1).sum(-1)

        cost_true = (val_labels.astype(float)[:N_eval]*val_weights[:N_eval]).sum(-1).sum(-1)

        perc_correct = (cost_pred - cost_true < 0.001).sum()/N_eval
        perc_correct_2 = (cost_pred - cost_true < 0.1).sum()/N_eval
        perc_correct_3 = (cost_pred - cost_true < 0.5).sum()/N_eval
    
    
        if perc_correct<=perc_correct_best:
            not_best_count_accum = not_best_count_accum + 1
            print('Did not improve results nr ', not_best_count_accum)
        else:
            perc_correct_best = perc_correct
            not_best_count_accum = 0
            _ = data_utils.check_or_create_folder("saved_models")
            torch.save(model.state_dict(), model_path)
            
        print(epochs, 
              ': Batches AVG:', round(loss_batch_avg.item(), 4), 
              '\t VAL perc:', round(perc_correct.item(), 4),
              '\t VAL perc <0.1:', round(perc_correct_2.item(), 4),
              '\t VAL perc <0.5:', round(perc_correct_3.item(), 4),
             )
        scheduler.step()
        
        if not_best_count_accum >= epochs_wait:
            print('Converged, exiting')
            break   