import torch
from torch import nn
import typing
import numpy as np
import sys

def create_linear_torch(
    input_dims, output_dims, bias=False, relu=False, batch_norm=False
):
    layers = []
    lin = torch.nn.Linear(input_dims, output_dims, bias=bias)
    layers.append(lin)
    if batch_norm:
        layers.append(torch.nn.BatchNorm1d(output_dims))
    if relu:
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)

class Linear(nn.Module):
    def __init__(self, noise_dims, x_dims, y_dims, z_dims):
        super(Linear, self).__init__()
        self.noise_dims = noise_dims
        self.x_dims = x_dims
        self.y_dims = y_dims
        self.z_dims = z_dims

        self.linear1 = create_linear_torch(noise_dims, 64, relu=True)
        self.linear2 = create_linear_torch(64, 256, relu=True)
        self.linear3 = create_linear_torch(256, 512, relu=True)
        self.output_linear = create_linear_torch(512, x_dims*y_dims*z_dims)
        self.sigmoid = nn.Sigmoid()
        self.layers = [
            self.linear1,
            self.linear2,
            self.linear3,
            self.sigmoid
        ]
        self._net = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self._net(x)
        return x.view(x_dims, y_dims, z_dims)

    def generate(self, mean=0.0, std=1.0, seed=None):
        if seed is None:
            seed = np.random.randint(sys.maxsize)
        rs = np.random.RandomState(seed)
        noise = rs.normal(loc=mean, scale=std, size=(1, self.noise_dims))
        return self.forward(noise)