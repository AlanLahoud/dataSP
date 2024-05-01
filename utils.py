import numpy as np
from collections import Counter
from collections import deque
from multiprocessing import Pool, cpu_count
import torch
from scipy.sparse.csgraph import dijkstra as dij
from scipy.sparse import csr_matrix
import heapq
import datasp



def get_nodes_and_freqs(node_idx_sequence_trips):
    flattened_trips = [item for sublist in node_idx_sequence_trips for item in sublist]
    node_count_freq = Counter(flattened_trips)
    elements, frequencies = zip(*node_count_freq.items())
    return np.array(elements), np.array(frequencies)



def process_chunk(chunk_data, nodes_selected_set, map_nodes_selected):
    chunk, start_idx = chunk_data
    new_chunk = []
    indices = []
    for idx, inner_list in enumerate(chunk, start=start_idx):
        filtered_and_replaced_list = [
            map_nodes_selected[element] for element in inner_list \
            if element in nodes_selected_set]
        if len(filtered_and_replaced_list) >= 2:
            new_chunk.append(filtered_and_replaced_list)
            indices.append(idx)
    return new_chunk, indices



def find_close_nodes(edge_tensor, start_node, num_nodes):
    graph = {}
    for edge in edge_tensor:
        node_from, node_to = edge.tolist()
        if node_from not in graph:
            graph[node_from] = []
        if node_to not in graph:
            graph[node_to] = []
        graph[node_from].append(node_to)
        graph[node_to].append(node_from)

    # Breadth-first search
    visited = set()
    queue = deque([start_node])
    while queue and len(visited) < num_nodes:
        current_node = queue.popleft()
        if current_node not in visited:
            visited.add(current_node)
            if current_node in graph:
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

    return list(visited)



def selected_trips_and_idx(
    node_idx_sequence_trips, M_indices, elements, frequencies, Vs, V):

    shuffled_indices = torch.randperm(M_indices.size(0))
    M_indices_shuf = M_indices[shuffled_indices]
    nodes_group = find_close_nodes(
        M_indices_shuf, np.random.choice(elements, 1).item(), Vs//2)
    
    elements_rest = list(set(elements)- set(nodes_group))
    args_rest = np.array(
        [np.where(np.array(elements)==el)[0].squeeze().item() for el in elements_rest])
    
    nodes_selected = np.random.choice(
        elements[args_rest], size=Vs-len(nodes_group), 
        p=np.array(frequencies[args_rest])/sum(frequencies[args_rest]), replace=False)
    
    nodes_selected = np.array(list(set(list(nodes_group)).union(nodes_selected)))

    if len(nodes_selected)<Vs:
        print('Warning!!! Did not select enough nodes in this batch.')
        return None, None, None, None
    
    nodes_selected = torch.tensor(nodes_selected)

    nodes_excluded = torch.tensor(
        np.array(list(set(np.arange(0, V)) - set(nodes_selected.numpy()))))

    nodes_selected, _ = nodes_selected.sort()
    nodes_excluded, _ = nodes_excluded.sort()

    map_nodes_selected = dict(zip(nodes_selected.numpy(), np.arange(0, Vs)))

    nodes_selected_set = set(nodes_selected.numpy())

    chunk_size = (len(node_idx_sequence_trips) // 16) + 1
    chunks = [(node_idx_sequence_trips[i:i + chunk_size], i) for i in range(0, len(node_idx_sequence_trips), chunk_size)]

    with Pool() as pool:
        results = pool.starmap(
            process_chunk, [(chunk, nodes_selected_set, 
                             map_nodes_selected) for chunk in chunks])

    selected_trips = []
    selected_indexes = []
    for new_list, indices in results:
        selected_trips.extend(new_list)
        selected_indexes.extend(indices)
            
    return selected_indexes, selected_trips, nodes_selected, nodes_excluded



def select_Ms_from_selected_idx_and_trips(
    M_Y_pred, Vs, M_indices, 
    nodes_excluded, nodes_selected, 
    beta, dev, d_large=2000.):
    
    M_Y_pred_new = M_Y_pred.clone()

    for n in nodes_excluded:
        M_indices_selected_mapped = torch.argwhere((M_Y_pred_new<d_large).sum(0)>0)
        mask_ind = (M_indices_selected_mapped == n)

        n_to_node = M_indices_selected_mapped[mask_ind[:,1]][:,0]
        node_to_n = M_indices_selected_mapped[mask_ind[:,0]][:,1]

        M_indices_to_check = torch.cartesian_prod(n_to_node, node_to_n)      

        M_Y_pred_new = datasp.remove_node_and_adjust_vectorized(
            M_Y_pred_new, n, M_indices_to_check, beta)

    idx_combinations = torch.cartesian_prod(
        nodes_selected, nodes_selected, nodes_selected)
    idx_combinations_2 = torch.cartesian_prod(
        nodes_selected, nodes_selected)

    M_Y_pred_selected = M_Y_pred_new[:, idx_combinations_2[:,0], 
                                     idx_combinations_2[:,1]].clone()

    M_Y_pred_selected = M_Y_pred_selected.reshape(M_Y_pred.shape[0], Vs, Vs)

    M_indices_selected_mapped = torch.argwhere(
        (M_Y_pred_selected<d_large).sum(0)>0)
    M_indices_selected = M_indices[torch.isin(
        M_indices, nodes_selected.to(dev)).sum(1) == 2]

    return M_Y_pred_selected, M_indices_selected_mapped



def shuffle_nodes_order(
    Vs, M_Y_pred_selected, M_indices_selected_mapped, 
    selected_trips):
        k_nodes = torch.arange(Vs)
        k_nodes_shufled = k_nodes[torch.randperm(Vs)]
        shuffle_k_dict = {
            int(k_nodes_shufled[i]):int(k_nodes[i]) for i in range(Vs)} 

        # We want to remove bias of node ordering
        M_Y_pred_selected_shuf = M_Y_pred_selected[
            :,k_nodes_shufled][:, :, k_nodes_shufled]     
        M_indices_selected_mapped_shuf = M_indices_selected_mapped.clone()
        for key, value in shuffle_k_dict.items():
            M_indices_selected_mapped_shuf[
                M_indices_selected_mapped == key] = value           
        selected_trips_shuf = [[
            shuffle_k_dict[p] for p in sublist] for sublist in selected_trips] 
        
        return M_Y_pred_selected_shuf, M_indices_selected_mapped_shuf, selected_trips_shuf
    
    

def dijkstra(adj_matrix_and_indices):
    adjacency_matrix, start_node, end_node, matrix_index = adj_matrix_and_indices
    graph = csr_matrix(adjacency_matrix)
    distances, predecessors = dij(csgraph=graph, 
                                  directed=True, 
                                  indices=start_node, 
                                  return_predecessors=True)
    path = [end_node]    
    while path[-1] != start_node:
        path.append(predecessors[path[-1]])       
    path.reverse()    
    return matrix_index, path



def batch_dijkstra(adjacency_matrices, node_pairs):
    """
    :param adjacency_matrices: Batch of adjacency matrices of shape (B, V, V)
    :param node_pairs: Matrix of shape (B, 2) where first column is start node, second is end node
    """
    with Pool(cpu_count()) as pool:
        input_data = [(adjacency_matrices[i], node_pairs[i][0], node_pairs[i][1], i) for i in range(len(adjacency_matrices))]
        results = pool.map(dijkstra, input_data)    
    results.sort(key=lambda x: x[0])
    return [path for _, path in results]



def paths_to_M_indices(paths, M_indices):
    M_indices_np = M_indices.detach().numpy()
    paths_M = np.zeros((len(paths), M_indices_np.shape[0]))
    
    for n in range(0, len(paths)):
        p = paths[n]
        for i in range(0, len(p)-1):
            index_M_ind = np.where((M_indices_np[:, 0] == p[i]) & (M_indices_np[:, 1] == p[i+1]))[0].item()
            paths_M[n,index_M_ind] = 1
    return paths_M.astype(int)



def compute_metrics_percentage(actual, pred):
    if len(actual) != len(pred):
        raise ValueError("Both lists should have the same length")
    
    total_samples = len(actual)
    num_ones_actual_percentage = (sum(actual) / total_samples) * 100
    TP_percentage = (sum([1 for a, p in zip(actual, pred) if a == 1 and p == 1]) / total_samples) * 100
    TN_percentage = (sum([1 for a, p in zip(actual, pred) if a == 0 and p == 0]) / total_samples) * 100
    FP_percentage = (sum([1 for a, p in zip(actual, pred) if a == 0 and p == 1]) / total_samples) * 100
    FN_percentage = (sum([1 for a, p in zip(actual, pred) if a == 1 and p == 0]) / total_samples) * 100
    MATCH = np.all(actual == pred).astype(float)
    return [num_ones_actual_percentage, TP_percentage, TN_percentage, FP_percentage, FN_percentage, MATCH]



def compute_jaccard_similarity(array1, array2):
    jaccard_similarities = []

    for i in range(array1.shape[0]):
        intersection = np.sum(np.logical_and(array1[i], array2[i]))
        union = np.sum(np.logical_or(array1[i], array2[i]))
        similarity = intersection / union if union != 0 else 0
        jaccard_similarities.append(similarity)
        
    return np.array(jaccard_similarities)



def compute_metrics_batch(paths_M_actual, paths_M_pred):
    metrics = np.zeros((len(paths_M_pred), 6))
    for i in range(0,len(paths_M_pred)):
        metrics[i, :] = compute_metrics_percentage(paths_M_actual[i], paths_M_pred[i])
    return metrics
