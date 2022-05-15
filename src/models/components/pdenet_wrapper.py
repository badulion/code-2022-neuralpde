import torch
import torch.nn as nn
from .pdenet.polypde import POLYPDE2D

class PDENet(nn.Module):
    def __init__(self,
                 input_channels,
                 kernel_size, 
                 max_order, 
                 constraint, 
                 hidden_layers=2, 
                 scheme='upwind',
                 dt=1, 
                 dx=1):
        super().__init__()
        abc = "abcdefghijklmnopqrstuvwxyz"
        channel_names_list = [abc[i] for i in range(input_channels)]
        channel_names_str = ','.join(channel_names_list)
        self.pdenet = POLYPDE2D(
            dt=dt,
            dx=dx,
            kernel_size=kernel_size,
            max_order=max_order,
            constraint=constraint,
            channel_names=channel_names_str,
            hidden_layers=hidden_layers,
            scheme=scheme
        ).to(torch.float32)

    def forward(self, x, num_steps):
        return self.pdenet.multistep(x, num_steps)