import torch
import torch.nn as nn
from collections import deque
import torchvision
from math import sqrt


def get_M_indices(N):
    M = torch.full((N**2, N**2), float('inf'))
    
    I_range = torch.arange(N**2)
    J_range = torch.arange(N**2)
    I, J = torch.meshgrid(I_range, J_range, indexing='ij')

    abs_diff = torch.abs(I - J)

    cond1 = (abs_diff == 1) | (abs_diff == N-1) | (abs_diff == N) | (abs_diff == N+1)
    cond2 = (J % N != N-1) | (I % N != 0)
    cond3 = (I % N != N-1) | (J % N != 0)

    mask = cond1 & cond2 & cond3
    
    M[mask] = 0.
    
    idx_edges = torch.where(M<1000.)
    n_edges = idx_edges[0].shape[0]
    M_indices = torch.zeros((n_edges, 2))
    M_indices[:,0] = idx_edges[0]
    M_indices[:,1] = idx_edges[1]
    M_indices = torch.tensor(M_indices, dtype=torch.long)
    return M_indices



class CombRenset18(nn.Module):

    def __init__(self, out_features, in_channels):
        super().__init__()
        self.resnet_model = torchvision.models.resnet18(pretrained=False, num_classes=out_features)
        del self.resnet_model.conv1
        self.resnet_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        output_shape = (int(sqrt(out_features)), int(sqrt(out_features)))
        self.pool = nn.AdaptiveMaxPool2d(output_shape)
        #self.last_conv = nn.Conv2d(128, 1, kernel_size=1,  stride=1)


    def forward(self, x):
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        x = self.resnet_model.layer1(x)
        #x = self.resnet_model.layer2(x)
        #x = self.resnet_model.layer3(x)
        #x = self.last_conv(x)
        x = self.pool(x)
        x = x.mean(dim=1)
        return x
    
    
    
def get_path_nodes(M_indices, grid, st=-1, en=-1):

    N = grid.shape[0] 
    
    if st==-1 and en==-1:
        st=0
        en=N**2-1

    valid_edges = []
    for edge in M_indices:
        if grid[torch.div(edge[0], N, rounding_mode='trunc'), edge[0] % N] == 1 and grid[torch.div(edge[1], N, rounding_mode='trunc'), edge[1] % N] == 1:
            valid_edges.append(edge)


    valid_edges = torch.stack(valid_edges)
    valid_edges

    def bfs(graph, start, end):
        queue = deque([start])
        visited = set()
        paths = {start: [start]}

        while queue:
            current_node = queue.popleft()

            if current_node == end:
                return paths[current_node]

            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    paths[neighbor] = paths[current_node] + [neighbor]
        return None

    graph = {i: [] for i in range(N**2)}
    for edge in valid_edges:
        graph[edge[0].item()].append(edge[1].item())

    path_indices = bfs(graph, st, en)

    return path_indices



def nodes_to_M_batch(nodes):
    batch_size, N, _ = nodes.shape
    M_batch = torch.full((batch_size, N**2, N**2), 10000., device=nodes.device)
    
    I_range = torch.arange(N**2, device=nodes.device)
    J_range = torch.arange(N**2, device=nodes.device)
    I, J = torch.meshgrid(I_range, J_range, indexing='ij')

    abs_diff = torch.abs(I - J)

    cond1 = (abs_diff == 1) | (abs_diff == N-1) | (abs_diff == N) | (abs_diff == N+1)
    cond2 = (J % N != N-1) | (I % N != 0)
    cond3 = (I % N != N-1) | (J % N != 0)

    mask = cond1 & cond2 & cond3

    batch_indices = torch.arange(batch_size, device=nodes.device)[:, None, None]
    J_masked = J[mask].view(1, -1)

    M_batch[batch_indices, mask] = nodes[batch_indices, 
                                         torch.div(J_masked, N, rounding_mode='trunc'), 
                                         J_masked % N]   
    return M_batch