from turtle import forward
import torch.nn as nn
import torch

class HiddenState(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Conv2d(input_dim, hidden_dim, 3, padding='same', padding_mode='circular')

    def forward(self, x):
        return self.net(x)
