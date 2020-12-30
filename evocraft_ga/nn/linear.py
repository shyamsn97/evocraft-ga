import torch
from torch import nn
import numpy as np
from evocraft_ga.nn.torch_utils import *  # noqa


class Linear(nn.Module):
    def __init__(self, noise_dims, x_dims, y_dims, z_dims, num_classes=1):
        super(Linear, self).__init__()
        self.noise_dims = noise_dims
        self.x_dims = x_dims
        self.y_dims = y_dims
        self.z_dims = z_dims
        self.num_classes = num_classes

        self.linear1 = create_linear_torch(noise_dims, 64, relu=True)
        self.linear2 = create_linear_torch(64, 256, relu=True)
        self.output_linear = create_linear_torch(
            256, x_dims * y_dims * z_dims * num_classes
        )
        self.layers = [
            self.linear1,
            self.linear2,
            self.output_linear,
        ]
        self._net = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self._net(x)
        return x

    def generate(self, mean=0.0, std=1.0, to_numpy=False, squeeze=False, seed=None):
        if seed is None:
            seed = np.random.randint(1, high=2 ** 31 - 1)
        np.random.seed(seed)
        noise = torch.from_numpy(
            np.random.normal(loc=mean, scale=std, size=(1, self.noise_dims))
        ).float()
        out = self.forward(noise)
        out = out.view(self.x_dims, self.y_dims, self.z_dims, self.num_classes)
        out = nn.Softmax(dim=-1)(out)
        if squeeze:
            out = torch.squeeze(out)
        if to_numpy:
            out = out.detach().numpy()
        return out
