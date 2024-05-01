import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, input_size, output_size, n_hidden_layers=3, hl_sizes=[64, 64]):
        super().__init__()   
        
        self.act1 = nn.ReLU()
        self.linear1 = nn.Linear(input_size, hl_sizes[0])
        self.linear2 = nn.Linear(hl_sizes[0], hl_sizes[1])
        self.linear3 = nn.Linear(hl_sizes[1], hl_sizes[1])
        self.linear3A = nn.Linear(hl_sizes[1], hl_sizes[1])
        self.linear4 = nn.Linear(hl_sizes[1], output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)
        x = self.act1(x)
        x = self.linear3A(x)
        x = self.act1(x)
        y_avg = self.linear4(x)
        return y_avg