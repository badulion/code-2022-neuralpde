import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_layers,
                 hidden_channels,
                 kernel_size=3):
        super().__init__()
        hidden_list = []
        for i in range(hidden_layers):
            conv_layer = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding='same', padding_mode='circular')
            hidden_list.append(conv_layer)
            hidden_list.append(nn.SELU())
            
        self.model = nn.Sequential(
            nn.Conv2d(input_dim, hidden_channels, 3, padding='same', padding_mode='circular'),
            nn.SELU(),
            *hidden_list,
            nn.Conv2d(hidden_channels, input_dim, 3, padding='same', padding_mode='circular')
        )

    def forward(self, x):
        return self.model(x)