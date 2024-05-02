import numpy as np
import heapq
import torch
from functools import partial
from comb_modules.utils import get_neighbourhood_func
from collections import namedtuple
import ray

DijkstraOutput = namedtuple("DijkstraOutput", ["shortest_path", "is_unique", "transitions"])


def dijkstra_original(matrix, neighbourhood_fn="8-grid", request_transitions=False):

    x_max, y_max = matrix.shape
    neighbors_func = partial(get_neighbourhood_func(neighbourhood_fn), x_max=x_max, y_max=y_max)

    costs = np.full_like(matrix, 1.0e10)
    costs[0][0] = matrix[0][0]
    num_path = np.zeros_like(matrix)
    num_path[0][0] = 1
    priority_queue = [(matrix[0][0], (0, 0))]
    certain = set()
    transitions = dict()

    while priority_queue:
        cur_cost, (cur_x, cur_y) = heapq.heappop(priority_queue)
        if (cur_x, cur_y) in certain:
            pass

        for x, y in neighbors_func(cur_x, cur_y):
            if (x, y) not in certain:
                if matrix[x][y] + costs[cur_x][cur_y] < costs[x][y]:
                    costs[x][y] = matrix[x][y] + costs[cur_x][cur_y]
                    heapq.heappush(priority_queue, (costs[x][y], (x, y)))
                    transitions[(x, y)] = (cur_x, cur_y)
                    num_path[x, y] = num_path[cur_x, cur_y]
                elif matrix[x][y] + costs[cur_x][cur_y] == costs[x][y]:
                    num_path[x, y] += 1

        certain.add((cur_x, cur_y))
    # retrieve the path
    cur_x, cur_y = x_max - 1, y_max - 1
    on_path = np.zeros_like(matrix)
    on_path[-1][-1] = 1
    while (cur_x, cur_y) != (0, 0):
        cur_x, cur_y = transitions[(cur_x, cur_y)]
        on_path[cur_x, cur_y] = 1.0

    is_unique = num_path[-1, -1] == 1

    if request_transitions:
        return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=transitions)
    else:
        return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=None)
    
    
    
def dijkstra(m_s_t, neighbourhood_fn="8-grid", request_transitions=False):
    
    matrix = m_s_t[0]
    source = m_s_t[1]
    target = m_s_t[2]
    
    x_max, y_max = matrix.shape
    neighbors_func = partial(get_neighbourhood_func(neighbourhood_fn), x_max=x_max, y_max=y_max)

    # Initialize the source
    source_x, source_y = source
    target_x, target_y = target
    
    source_x = int(source_x.item())
    source_y = int(source_y.item())
    target_x = int(target_x.item())
    target_y = int(target_y.item())

    
    costs = np.full_like(matrix, 1.0e10)
    costs[source_x][source_y] = matrix[source_x][source_y]

    num_path = np.zeros_like(matrix)
    num_path[source_x][source_y] = 1

    priority_queue = [(matrix[source_x][source_y], (source_x, source_y))]
    certain = set()
    transitions = dict()

    while priority_queue:
        cur_cost, (cur_x, cur_y) = heapq.heappop(priority_queue)
        if (cur_x, cur_y) in certain:
            pass

        for x, y in neighbors_func(cur_x, cur_y):
            if (x, y) not in certain:
                if matrix[x][y] + costs[cur_x][cur_y] < costs[x][y]:
                    costs[x][y] = matrix[x][y] + costs[cur_x][cur_y]
                    heapq.heappush(priority_queue, (costs[x][y], (x, y)))
                    transitions[(x, y)] = (cur_x, cur_y)
                    num_path[x, y] = num_path[cur_x, cur_y]
                elif matrix[x][y] + costs[cur_x][cur_y] == costs[x][y]:
                    num_path[x, y] += 1

        certain.add((cur_x, cur_y))

    # retrieve the path from target to source
    cur_x, cur_y = target_x, target_y
    on_path = np.zeros_like(matrix)
    on_path[target_x][target_y] = 1
    while (cur_x, cur_y) != (source_x, source_y):
        cur_x, cur_y = transitions[(cur_x, cur_y)]
        on_path[cur_x, cur_y] = 1.0

    is_unique = num_path[target_x, target_y] == 1

    if request_transitions:
        return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=transitions)
    else:
        return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=None)

    


def get_solver(neighbourhood_fn):
    def solver(m_s_t):
        return dijkstra(m_s_t, neighbourhood_fn).shortest_path

    return solver


def maybe_parallelize(function, arg_list):
    if ray.is_initialized():
        ray_fn = ray.remote(function)
        return ray.get([ray_fn.remote(arg) for arg in arg_list])
    else:
        return [function(arg) for arg in arg_list]

#class ShortestPath(torch.autograd.Function):
#    def __init__(self, lambda_val, neighbourhood_fn="8-grid"):
#        self.lambda_val = lambda_val
#        self.neighbourhood_fn = neighbourhood_fn
#        self.solver = get_solver(neighbourhood_fn)

#    def forward(self, weights):
#        self.weights = weights.detach().cpu().numpy()
#        self.suggested_tours = np.asarray(maybe_parallelize(self.solver, arg_list=list(self.weights)))
#        return torch.from_numpy(self.suggested_tours).float().to(weights.device)

#    def backward(self, grad_output):
#        assert grad_output.shape == self.suggested_tours.shape
#        grad_output_numpy = grad_output.detach().cpu().numpy()
#        weights_prime = np.maximum(self.weights + self.lambda_val * grad_output_numpy, 0.0)
#        better_paths = np.asarray(maybe_parallelize(self.solver, arg_list=list(weights_prime)))
#        gradient = -(self.suggested_tours - better_paths) / self.lambda_val
#        return torch.from_numpy(gradient).to(grad_output.device)
    
    
    
import torch
import numpy as np

class ShortestPath(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, source, target, lambda_val, neighbourhood_fn="8-grid"):      
        
        ctx.lambda_val = lambda_val
        ctx.neighbourhood_fn = neighbourhood_fn
        ctx.solver = get_solver(neighbourhood_fn)

        ctx.weights = weights.detach().cpu().numpy()
        ctx.source = source
        ctx.target = target
        
        args_for_dijkstra = [
            (ctx.weights[i], ctx.source[i], ctx.target[i]) for i in range(len(ctx.weights))
        ]
             
        ctx.suggested_tours = np.asarray(maybe_parallelize(ctx.solver, arg_list=args_for_dijkstra))
        return torch.from_numpy(ctx.suggested_tours).float().to(weights.device)

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.shape == ctx.suggested_tours.shape
        grad_output_numpy = grad_output.detach().cpu().numpy()
        weights_prime = np.maximum(ctx.weights + ctx.lambda_val * grad_output_numpy, 0.0)
        
        args_for_dijkstra = [
            (weights_prime[i], ctx.source[i], ctx.target[i]) for i in range(len(weights_prime))
        ]
        
        better_paths = np.asarray(maybe_parallelize(ctx.solver, arg_list=args_for_dijkstra))
        gradient = -(ctx.suggested_tours - better_paths) / ctx.lambda_val
        return torch.from_numpy(gradient).to(grad_output.device), None, None, None, None

