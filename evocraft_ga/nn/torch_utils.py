from torch import nn
import torch

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
