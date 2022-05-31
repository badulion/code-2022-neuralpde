from turtle import forward
import torch.nn as nn
import torch

class NeuralNetWrapper(nn.Module):
    def __init__(self,
                 net: nn.Module,
                 order=1):
        super().__init__()
        self.net = net
        self.order = order

    def forward(self, t, x):
        return self.net(x)

    def forward(self, t, x):
        #u_1 = du/dt
        #u_2 = du_1/dt = d2u/dt2
        #...
        #u_n-1 = du_n-1/dtn-1
        #x = [u, u_1, ..., u_n-1]

        #u_n = f(u, u_1, ..., u_n-1) = f(x)
        #dx/dt = [du/dt, du1_/dt, ..., du_n-1/dt] = [u_1, u_2, ..., u_n] = [u_1, ..., u_n-1, f(x)]
        data_dim = x.size(1) // self.order
        u_i = list(torch.split(x, data_dim, dim=1))
        x = self.net(x)
        x = torch.cat(u_i[1:]+[x], dim=1)
        return x

if __name__ == "__main__":
    x = torch.randn((10, 8, 2))
    s = torch.split(x, 8, dim=1)
    print(len(s))