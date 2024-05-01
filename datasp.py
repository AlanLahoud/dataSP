import torch
import torch.nn as nn
import numpy as np
import heapq
import utils

softmax_func = nn.Softmax(-1)

def min_smooth(x1, x2, beta: torch.Tensor = torch.tensor(1.0)):
    M = torch.minimum(x1*beta, x2*beta)
    return (1/beta)*M - (1/beta)*torch.log(torch.exp(-x1*beta + M) + torch.exp(-x2*beta + M))

def argmin_smooth(x1, x2, beta: torch.Tensor = torch.tensor(1.0)):
    return softmax_func(torch.stack((-beta*x1, -beta*x2), dim=-1))

def d_is_large(d, d_large):
    return d>d_large

def datasp(
    weight_matrix, 
    M_indices, 
    dev: str,
    beta_smooth: torch.Tensor = torch.tensor(1.0),
    d_large: torch.Tensor = torch.tensor(2000.)):
    
    bs = weight_matrix.shape[0]  # Batch size
    num_vertices = weight_matrix.shape[1]  # #Nodes

    shortest_distances = weight_matrix
    
    argmins_tensor = torch.zeros(
        (bs, num_vertices, num_vertices, num_vertices), 
        dtype=torch.float32).to(dev)
    
    # Initial path choices based on the graph structure (existing edges)
    argmins_tensor[:, M_indices[:, 0], M_indices[:, 1], M_indices[:, 0]] = 1.
    
    for k in range(num_vertices):
        # Broadcasting to create a matrix of shape (bs, num_vertices, num_vertices)
        shortcut = shortest_distances[:, :, k, None] + shortest_distances[:, k, None, :]
        direct = shortest_distances[:, :, :]
        
        # Avoiding operations where i == k, j == i, or j == k
        i_mask = torch.ones(num_vertices, dtype=torch.bool).to(weight_matrix.device)
        i_mask[k] = False  # i != k
        j_mask = torch.ones(num_vertices, dtype=torch.bool).to(weight_matrix.device)
        j_mask[k] = False  # j != k

        mask = i_mask[:, None] & j_mask[None, :]  # Combining masks for i and j      
        mask = mask[None, :, :].expand(bs, num_vertices, num_vertices) 
        
        large_mask = ~d_is_large(shortcut, d_large)
        mask = mask & large_mask

        argmins = argmin_smooth(direct, shortcut, beta=beta_smooth)
        
        argmins_tensor_clone = argmins_tensor.clone()
        
        argmins_tensor = torch.where(
            mask.unsqueeze(-1), 
            argmins_tensor_clone * argmins[:,:,:,0].unsqueeze(-1), 
            argmins_tensor)
        
        argmins_tensor[:, :, :, k] = torch.where(
            mask, argmins[:,:,:,1], argmins_tensor[:, :, :, k])
        
        new_shortest_distances = min_smooth(direct, shortcut, beta=beta_smooth)
        
        shortest_distances = torch.where(mask, new_shortest_distances, shortest_distances)
            
    return argmins_tensor




def normal_cdf(x):
    """Cumulative distribution function for the standard normal distribution."""
    return 0.5 * (1 + torch.erf(x / sqrt(2.0)))



def probabilistic_argmin(a, b, sa, sb):
    z = (a - b)/torch.sqrt(sa**2 + sb**2)
    argmins = 1 - normal_cdf(z)
    return argmins



def exclude_row_col(tensor, index):
    rows_excluded = torch.cat((tensor[:,:index], tensor[:,index + 1:]), dim=1)
    result = torch.cat((rows_excluded[:,:, :index], rows_excluded[:,:, index + 1:]), dim=2)
    return result



def remove_node_and_adjust_vectorized(
    adjacency_matrix, node, M_indices, 
    beta_smooth: torch.Tensor = torch.tensor(1.0)):
    
    mask = ~(M_indices[:, 0] == node) \
    & ~(M_indices[:, 1] == node) \
    & ~(M_indices[:, 0] == M_indices[:, 1])
    valid_edges = M_indices[mask]

    i_indices = valid_edges[:, 0]
    j_indices = valid_edges[:, 1]

    shortcut = adjacency_matrix[:, :, node][:, i_indices] \
    + adjacency_matrix[:, node, :][:, j_indices]
    direct = adjacency_matrix[:, i_indices, j_indices]

    argmins = argmin_smooth(
        direct, shortcut, beta=beta_smooth)
    
    adj_best = adjacency_matrix.clone()
    
    adj_best[:, i_indices, j_indices] = min_smooth(
        direct, shortcut, beta=beta_smooth)
    
    return adj_best