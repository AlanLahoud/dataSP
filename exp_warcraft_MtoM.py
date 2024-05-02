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
import multiprocessing



##################################################################################### 
#################################PARAMETERS########################################## 
##################################################################################### 
        
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Set parameters for the program.')
            
    parser.add_argument('--N', type=int, default=18, 
                        help='Grid size')
    parser.add_argument('--Vs', type=int, default=100, 
                        help='Nr sampling nodes')
    
    parser.add_argument('--N_train', type=int, default=350, 
                        help='Nr sampling nodes')
    parser.add_argument('--N_EPOCHS', type=int, default=100, 
                        help='N EPOCHS train')
    
    parser.add_argument('--beta', type=float, default=30., 
                        help='Beta Smooth')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning Rate')
    parser.add_argument('--N_batches', type=int, default=100, 
                        help='N Batches in one Epoch')
    parser.add_argument('--bs_X', type=int, default=8, 
                        help='How many floyd warshalls in a batch')
    
    parser.add_argument('--seed_n', type=int, default=0)   
    parser.add_argument('--load_model', type=int, default=0, 
                        help='Load previous model?')
    
    parser.add_argument('--suboptimals', type=int, default=0, 
                        help='noise in paths')
        
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

# Paths with (1) or without (0) noise
suboptimals = args.suboptimals

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

paths_per_img = 30

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'RUNNING WITH {dev}')

M_indices = utils_ww.get_M_indices(N)

N_val = 1000

data_dir = f'./data_warcraft/{N}x{N}/'

data_suffix = "maps"
train_prefix = "train"
val_prefix = "val"

train_inputs = np.load(os.path.join(
    data_dir, train_prefix + "_" + data_suffix + ".npy")).astype('float')[:N_train]
train_weights = np.load(
    os.path.join(data_dir, train_prefix + "_vertex_weights.npy"))[:N_train]

val_inputs = np.load(
    os.path.join(data_dir, val_prefix + "_" + data_suffix + ".npy")).astype('float')
val_weights = np.load(
    os.path.join(data_dir, val_prefix + "_vertex_weights.npy"))
val_labels = np.load(
    os.path.join(data_dir, val_prefix + "_shortest_paths.npy"))    

train_inputs = train_inputs.transpose(0,3,1,2)
val_inputs = val_inputs.transpose(0,3,1,2)

mean, std = (
    np.mean(train_inputs, axis=(0, 2, 3), keepdims=True),
    np.std(train_inputs, axis=(0, 2, 3), keepdims=True),
)
train_inputs -= mean
train_inputs /= std

val_inputs -= mean
val_inputs /= std


noise_w = 0.
suffix_noise=''
if suboptimals:
    noise_w = 0.5
    suffix_noise = '_noise_05'

train_weights_pert = utils_ww.perturb_weights(train_weights, noise=noise_w)
val_weights_pert = utils_ww.perturb_weights(val_weights, noise=noise_w)

s_t_nodes = []
for samp in range(N_train):
    s_t_inner = []
    for p in range(paths_per_img):
        s_t_inner.append(utils_ww.gen_s_t_nodes(N))
    s_t_nodes.append(s_t_inner)
s_t_nodes = np.array(s_t_nodes)


train_labels = utils_ww.sp_dij(
    train_weights_pert, s_t_nodes, paths_per_img=paths_per_img)


def process_path_train(i_p):
    i, p = i_p
    return utils_ww.get_path_nodes(
        M_indices, train_labels[i, p], 
        s_t_nodes[i, p, 0, 0] * N + s_t_nodes[i, p, 0, 1],
        s_t_nodes[i, p, 1, 0] * N + s_t_nodes[i, p, 1, 1]
    )

indices = [(i, p) for i in range(0, N_train) for p in range(paths_per_img)]
with multiprocessing.Pool() as pool:
    true_paths_nodes = list(tqdm(pool.imap(process_path_train, indices), 
                                 total=len(indices)))


#true_paths_nodes = []
#for im in range(N_train):
#    for p in range(paths_per_img):
#        true_paths_nodes.append(true_paths_nodes_np[im,p])


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


model_path =\
f'saved_models/MtoM_{N}_{suffix_noise}_{beta_smooth}_{bs_X}_{lr}_{seed_n}.pkl'
print('Model path:', model_path)
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

train_inputs_tensor = torch.from_numpy(train_inputs).float()
val_inputs_tensor = torch.from_numpy(val_inputs).float()

M_indices = M_indices.to(dev)

elements, frequencies = utils.get_nodes_and_freqs(true_paths_nodes)

not_best_count_accum = 0
loss_batch_avg_best = torch.inf
perc_correct_best = 0.

scheduler = lr_scheduler.LinearLR(opt, 
                                  start_factor=1.0, end_factor=0.01, total_iters=50)



for epochs in range(0,N_EPOCHS):

    loss_batch_avg = 0
    
    for batch in range(0, N_batches):  
     
        start_time = time.time()        

        with torch.no_grad():
            selected_indexes, selected_trips, nodes_selected, nodes_excluded =\
            utils.selected_trips_and_idx(
                true_paths_nodes, M_indices, 
                elements, frequencies, Vs, N**2)
            if selected_indexes == None:
                continue

            selected_indexes = np.array(selected_indexes)              
            selected_indexes_imgs_all = selected_indexes//paths_per_img              
            selected_indexes_imgs = list(set(list(selected_indexes_imgs_all)))            
            X_selected = train_inputs_tensor[selected_indexes_imgs]

            idcs_batch = torch.randint(
                0, X_selected.shape[0], (bs_X,)).unsqueeze(1)
            
            idcs_imgs = idcs_batch[:,0].sort().values
            mask = np.isin(selected_indexes_imgs_all, idcs_imgs)
            idcs_paths = selected_indexes[mask]
            
        
        X_batch = X_selected[idcs_imgs].to(dev)

        nodes_pred_batch = model(X_batch).clip(0.001)
        M_pred_batch = utils_ww.nodes_to_M_batch(nodes_pred_batch)
        
        M_Y_pred_selected, M_indices_selected_mapped =\
        utils.select_Ms_from_selected_idx_and_trips(
                M_pred_batch, Vs, M_indices, 
            nodes_excluded, nodes_selected, 
            torch.tensor(beta_smooth), dev)
        
        
        M_Y_pred_selected_shuf, M_indices_selected_mapped_shuf, selected_trips_shuf =\
        utils.shuffle_nodes_order(
            Vs, M_Y_pred_selected, M_indices_selected_mapped, 
            selected_trips)
        
        
        probs_pred = datasp.datasp(
            M_Y_pred_selected_shuf, 
            M_indices_selected_mapped_shuf, 
            dev, beta_smooth)
        
        idx_batch_paths = np.searchsorted(selected_indexes, idcs_paths)
        
        sel_imgs_idx = selected_indexes_imgs_all[idx_batch_paths]
        
        value_changes = np.diff(sel_imgs_idx, prepend=sel_imgs_idx[0]) != 0
        value_changes_int = value_changes.astype(int)
        sel_imgs_idx_sorted = np.cumsum(value_changes_int)
        
        m_inter_total = torch.zeros(bs_X, Vs, Vs, Vs)
        for i, p in zip(sel_imgs_idx_sorted, idx_batch_paths):  
            m_inter_total[i] += data_utils.get_m_inter(
                selected_trips_shuf[p], Vs, Vs)

        m_inter_total = (m_inter_total/m_inter_total.sum(-1).unsqueeze(-1)).to(dev)
        
        mask = ~torch.isnan(m_inter_total)
        true_paths_dist = m_inter_total[mask].reshape(-1, Vs)
        pred_paths_dist = probs_pred[mask].reshape(-1, Vs)
        loss_main = cross_entropy_cont(true_paths_dist, pred_paths_dist).mean()
        
        opt.zero_grad()
        loss_main.backward()
        opt.step()
        
        loss_batch_avg += (loss_main/N_batches).detach()
    
    
    with torch.no_grad():
        N_eval = 1000
        nodes_pred = model(val_inputs_tensor.to(dev)).clip(0.001)
        M_pred = utils_ww.nodes_to_M_batch(nodes_pred)

        path_pred = utils.batch_dijkstra(
            M_pred.detach().cpu().numpy(),
            np.repeat(np.array([[0,N**2-1]]),
                      N_eval, 0))

        path_pred_map_all = torch.zeros((N_eval, N**2))
        for i in range(0, N_eval):
            path_pred_map = torch.zeros((N**2,))
            path_pred_map[path_pred[i]] = 1
            path_pred_map_all[i] = path_pred_map

        path_pred_map_all = path_pred_map_all.reshape(-1,N,N)

        cost_pred = (path_pred_map_all*val_weights[:N_eval]).sum(-1).sum(-1)

        cost_true = (
            val_labels.astype(float)[:N_eval]*val_weights[:N_eval]).sum(-1).sum(-1)

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