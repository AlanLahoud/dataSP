import numpy as np
import torch
import create_networks
import data_generation
import itertools
import utils
import os
import pickle


def check_or_create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        return f"Folder '{folder_name}' created."
    else:
        return f"Folder '{folder_name}' already exists."


def create_graph(M, sparsity, seed_n):
    bin_M = torch.tensor(create_networks.get_bin_M(M=M, sparsity=sparsity))
    prior_M = torch.tensor(create_networks.get_adj_cost_M(bin_m = bin_M), dtype=torch.float32)
    E = int(bin_M.sum().detach().numpy())
    M_indices = np.zeros((E, 2))
    M_indices[:,0], M_indices[:,1] = np.where(bin_M.detach().numpy()==1)
    M_indices = torch.tensor(M_indices, dtype=torch.long)
    return bin_M, prior_M, E, M_indices 


def costs_to_matrix(prior, M_indices, dY):
    N = dY.shape[0] #Batch
    Mat = prior.unsqueeze(0).expand((N, prior.shape[0], prior.shape[1])).clone()
    for n, (i, j) in enumerate(zip(M_indices[:,0], M_indices[:,1])):
        Mat[:, int(i), int(j)] = (prior[int(i), int(j)]).unsqueeze(0) + dY[:,n]
    return Mat.clamp(0.001, None)


def generate_synthetic_data(N_train, N_val, N_test, 
                            noise_data, E, M_indices, 
                            prior_M, seed_n):
    
    X, dY, _ = data_generation.gen_data(
        N_train, E, nl=noise_data, seed_number=seed_n, samples_dist=1)
    
    X_val, dY_val, _ = data_generation.gen_data(
        N_val, E, nl=noise_data, seed_number=seed_n+100, samples_dist=1)
    
    X_test, dY_test, _ = data_generation.gen_data(
        N_test, E, nl=noise_data, seed_number=seed_n+200, samples_dist=1)

    X = torch.tensor(X, dtype=torch.float32)
    dY = torch.tensor(dY, dtype=torch.float32)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    dY_val = torch.tensor(dY_val, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    dY_test = torch.tensor(dY_test, dtype=torch.float32)
    
    M_Y = costs_to_matrix(prior_M, M_indices, dY)
    M_Y_val = costs_to_matrix(prior_M, M_indices, dY_val)
    M_Y_test = costs_to_matrix(prior_M, M_indices, dY_test)
    
    return X, X_val, X_test, dY, dY_val, dY_test, M_Y, M_Y_val, M_Y_test


def source_end_nodes_permutation(M, perc_end_nodes_seen):
    all_permutations = list(itertools.permutations(range(M), 2))
    filtered_permutations = [perm for perm in all_permutations if perm[0] < perm[1]]
    size_seen = int(perc_end_nodes_seen*len(filtered_permutations))
    seen_indices = np.random.choice(len(filtered_permutations), size_seen, replace=False)
    unseen_indices = np.array(list(set(np.arange(0,len(filtered_permutations))) - set(seen_indices)))
    seen_permutations = [filtered_permutations[i] for i in seen_indices]
    unseen_permutations = [filtered_permutations[i] for i in unseen_indices]
    return seen_permutations, unseen_permutations


def gen_source_end_nodes_train(seen_permutations, N_train):
    end_to_end_nodes_train = np.zeros((N_train, 2))
    for i in range(0, N_train):
        random_index = np.random.choice(len(seen_permutations))
        idx = seen_permutations[random_index]
        end_to_end_nodes_train[i, :] = idx
    end_to_end_nodes_train = end_to_end_nodes_train.astype(int)
    return end_to_end_nodes_train


def gen_paths(end_to_end_nodes_train, N_train, M_Y, BBB=50):
    paths_demonst_train = []
    for i in range(0, N_train//BBB):
        paths_demonst_train.append(
            utils.batch_dijkstra(
                M_Y[i*BBB:(i+1)*BBB], 
                end_to_end_nodes_train[i*BBB:(i+1)*BBB]))
    paths_demonst_tr = [it for subl in paths_demonst_train for it in subl]
    return paths_demonst_tr



def process_paths(paths_demonstration_train, 
                  nodes_in_cluster_sorted, M_indices,
                  seed_n, M, sparsity, noise_data, 
                  perc_end_nodes_seen, train=True):
    
    prefix_train = ''
    if not train:
        prefix_train='_val'
    
    file_data_process=\
    f'./data_synthetic_gen/{seed_n}_{M}_{sparsity}_{noise_data}_{perc_end_nodes_seen}_{prefix_train}.pkl'

    if not os.path.exists(file_data_process):

        node_idx_sequence_trips = []
        edges_seq_original = []
        edges_idx_on_original = []
        start_nodes_original = []
        end_nodes_original = []

        for idx in range(0, len(paths_demonstration_train)):
            node_sequence_trip = paths_demonstration_train[idx]
            node_idx_sequence_trip = np.searchsorted(
                nodes_in_cluster_sorted, node_sequence_trip)
            node_idx_sequence_trips.append(node_idx_sequence_trip)
            start_nodes_original.append(node_idx_sequence_trip[0])
            end_nodes_original.append(node_idx_sequence_trip[-1])
            edges_sequence_trip = np.column_stack(
                [node_idx_sequence_trip[:-1], 
                 node_idx_sequence_trip[1:]])
            edges_seq_original.append(edges_sequence_trip)
            edges_idx_sequence_trip = np.array(
                [1 if any(np.array_equal(edge, t) for t in edges_sequence_trip) \
                 else 0 for edge in M_indices])
            edges_idx_on_original.append(edges_idx_sequence_trip)  

        data = {
            "node_idx_sequence_trips": node_idx_sequence_trips,
            "edges_seq_original": edges_seq_original,
            "edges_idx_on_original": edges_idx_on_original,
            "start_nodes_original": start_nodes_original,
            "end_nodes_original": end_nodes_original
        }

        _ = check_or_create_folder('data_synthetic_gen')

        with open(file_data_process, 'wb') as file:
            pickle.dump(data, file)
        print("Data saved to", file_data_process)
    else:
        with open(file_data_process, 'rb') as file:
            data = pickle.load(file)
        
    return data



def combined_distance(sample, data):
    d1 = (data[:, 0] - sample[0]).abs()
    d2 = (data[:, 1] - sample[1]).abs()
    d3 = (data[:, 2] - sample[2]).abs()
    total_dist = (d1+d2+d3)/3
    return total_dist



def find_k_similar_indices(data, k):
    idx = torch.randint(0, len(data), (1,))
    sample = data[idx.item()]
    distances = combined_distance(sample, data)
    distances[idx] = float('inf')
    k_indices = distances.topk(k, largest=False)[1]
    all_indices = torch.cat((idx, k_indices))
    return all_indices



def generate_n_combinations(data, k, n):
    all_indices = [find_k_similar_indices(data, k) for _ in range(n)]
    return torch.stack(all_indices)



def get_m_inter(node_idx_sequence_trip, V, Vk):
    m_inter = np.zeros((V, V, Vk))
    
    subpaths = []
    
    for i in range(len(node_idx_sequence_trip)-1):
        for j in range(i+1, len(node_idx_sequence_trip)):
            if j-i == 1:
                subpath = [node_idx_sequence_trip[i], node_idx_sequence_trip[i], node_idx_sequence_trip[j]]
            else:
                max_node = max(node_idx_sequence_trip[i+1:j])
                subpath = [node_idx_sequence_trip[i], max_node, node_idx_sequence_trip[j]]
            subpaths.append(subpath)

    for subpath in subpaths:
        i, k, j = subpath
        m_inter[i, j, k] = 1.
        
    return m_inter


def get_m_inter_batch(node_idx_sequence_trips, idcs_batch, V, Vk):
    
    B1 = idcs_batch.shape[0]
    B2 = idcs_batch.shape[1]
    
    m_inter_batch = np.zeros((B1, B2, V, V, Vk))  
    
    for i in range(0, B1):
        for j in range(0, B2):
            id_trip = idcs_batch[i,j]
            m_inter_batch[i,j,:,:,:] = get_m_inter(node_idx_sequence_trips[id_trip], V, Vk)
            
    return m_inter_batch



def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) \
    * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def create_prior_distance_matrix(nodes_df, df_edges):
    nodes_df = nodes_df.sort_values(
        by='node_sorted').reset_index(drop=True)
    sorted_node_ids = np.sort(nodes_df['node_sorted'].values)

    n = len(sorted_node_ids)
    adj_matrix_sorted = np.zeros((n, n), dtype=int)

    for _, row in df_edges.iterrows():
        i = row['node_from']
        j = row['node_to']
        adj_matrix_sorted[i, j] = 1

    distance_matrix = np.full((n, n), 5000.)

    for i in range(n):
        for j in range(n):
            if adj_matrix_sorted[i, j] == 1:
                lat1, lon1 = nodes_df.loc[i, ['node_lat', 'node_lon']]
                lat2, lon2 = nodes_df.loc[j, ['node_lat', 'node_lon']]
                distance_matrix[i, j] = haversine(lat1, lon1, lat2, lon2)
                
    return adj_matrix_sorted, distance_matrix

def get_prior_and_M_indices(nodes, edges):
    bin_M, prior_M = create_prior_distance_matrix(nodes, edges)
    prior_M = torch.tensor(prior_M, dtype=torch.float32)
    E = int(bin_M.sum())
    M_indices = np.zeros((E, 2))
    M_indices[:,0], M_indices[:,1] = np.where(bin_M==1)
    M_indices = torch.tensor(M_indices, dtype=torch.long)
    edges_prior = prior_M[M_indices[:, 0], M_indices[:, 1]]
    return prior_M, edges_prior, M_indices