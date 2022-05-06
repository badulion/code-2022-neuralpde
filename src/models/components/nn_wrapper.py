from turtle import forward
import torch.nn as nn
import torch

class NeuralNetWrapper(nn.Module):
    def __init__(self,
                 net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, t, x):
        return self.net(x)