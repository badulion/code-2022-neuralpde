from re import M
from unicodedata import bidirectional
import torch.nn as nn
import torch

from .attention import SeqSelfAttention


class PDERNN(nn.Module):

    def __init__(self, input_dim, hidden_size, num_layers, use_attention=True, bias=True):
        super(PDERNN, self).__init__()

        self.input_features=input_dim
        self.use_attention=use_attention

        self.LSTM = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            bias=bias,
            bidirectional=True,
            batch_first=True,
        )

        if self.use_attention:
            self.attention = SeqSelfAttention(input_dim)
        else:
            self.attention = nn.Identity()


    def forward(self, x):
        in_shape = x.shape
        x = torch.reshape(x, (in_shape[0], in_shape[1], -1))
        x, _ = self.LSTM(x)
        x_forward, x_backward = torch.split(x, self.input_features, dim=-1)
        x = x_forward+x_backward
        x = self.attention(x)
        x = torch.reshape(x, in_shape)
        return x
