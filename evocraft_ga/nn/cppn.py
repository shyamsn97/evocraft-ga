import numpy as np
import torch
from torch import nn
from evocraft_ga.nn.torch_utils import *  # noqa


class CPPN(nn.Module):
    def __init__(self, noise_dims, x_dims, y_dims, z_dims, num_classes=1):
        super(CPPN, self).__init__()
        self.noise_dims = noise_dims
        self.num_classes = num_classes
        self.x_dims = x_dims
        self.y_dims = y_dims
        self.z_dims = z_dims

        # noise_dims + dimensions of the space + radius term
        self.linear1 = create_linear_torch(noise_dims + 3 + 1, 64, relu=True)
        self.linear2 = create_linear_torch(64, 256, relu=True)
        self.output_linear = create_linear_torch(256, num_classes)
        self.layers = [
            self.linear1,
            self.linear2,
            self.output_linear,
        ]
        self._net = nn.Sequential(*self.layers)

    def forward(self, x, y, z, numpy_noise):
        normalized_x = x / self.x_dims
        normalized_y = y / self.y_dims
        normalized_z = z / self.z_dims
        r = np.sqrt(normalized_x ** 2 + normalized_y ** 2 + normalized_z ** 2)
        inp = np.expand_dims(
            np.concatenate(
                (np.array([normalized_x, normalized_y, normalized_z, r]), numpy_noise),
                axis=0,
            ),
            0,
        )
        inp = torch.from_numpy(inp).float()
        x = self._net(inp)
        return x

    def generate(self, mean=0.0, std=1.0, to_numpy=False, squeeze=False, seed=None):
        if seed is None:
            seed = np.random.randint(1, high=2 ** 31 - 1)
        noise = np.random.normal(loc=mean, scale=std, size=(self.noise_dims,))
        out = torch.zeros(self.x_dims, self.y_dims, self.z_dims, self.num_classes)
        for coord in np.ndindex((self.x_dims, self.y_dims, self.z_dims)):
            x, y, z = coord
            coord_out = nn.Softmax(dim=-1)(torch.squeeze(self.forward(x, y, z, noise)))
            out[(x, y)] = coord_out
        out = out.view(self.x_dims, self.y_dims, self.z_dims, self.num_classes)
        if squeeze:
            out = torch.squeeze(out)
        if to_numpy:
            out = out.detach().numpy()
        return out
