from __future__ import annotations
import copy
import attr
import torch
import numpy as np
import typing

from evocraft_ga.nn.linear import Linear
from evocraft_ga.nn.cppn import CPPN

model_dict = {"linear": Linear, "cppn": CPPN}


@attr.s
class Genome:
    noise_dims: int = attr.ib()
    cube_dims: int = attr.ib()
    seeds: typing.List[int] = attr.ib(default=[])
    noise_stdev: float = attr.ib(default=1.0)
    num_classes: int = attr.ib(default=1)
    model_class: str = attr.ib(default="linear")
    # set later
    _model_class = attr.ib(init=False)
    model = attr.ib(init=False)

    def __attrs_post_init__(self):
        self._verify_seeds(self.seeds)
        self._model_class = model_dict.get(self.model_class, model_dict["linear"])
        self.model = self._model_class(
            self.noise_dims,
            self.cube_dims,
            self.cube_dims,
            self.cube_dims,
            num_classes=self.num_classes,
        )
        self.initialize_weights()

    def _verify_seeds(self, seeds):
        if len(seeds) == 0:
            seeds = [np.random.randint(1, high=2 ** 31 - 1)]
        self.seeds = seeds.copy()

    def initialize_weights(self) -> None:
        num_seeds = len(self.seeds)

        rs = np.random.RandomState(self.seeds[0])
        for n, W in self.model.named_parameters():
            if any(x in n for x in ["weight", "bias"]):
                weights = rs.normal(loc=0.0, scale=self.noise_stdev, size=W.size())
                W.data = torch.from_numpy(weights).float()

        for seed_num in range(1, num_seeds):
            rs = np.random.RandomState(self.seeds[seed_num])
            for name, W in self.model.named_parameters():
                if any(x in name for x in ["weight", "bias"]):
                    weights = (
                        rs.normal(loc=0.0, scale=1.0, size=W.size()) * self.noise_stdev
                    )
                    W.data = W.data + torch.from_numpy(weights).float()

    def mutate(self):
        new_seed = np.random.randint(1, high=2 ** 31 - 1)
        self.seeds.append(new_seed)
        self.initialize_weights()

    def generate(self, to_numpy=False, squeeze=False, seed=None):
        return self.model.generate(to_numpy=to_numpy, squeeze=squeeze, seed=seed)

    def copy_parameters(self, genome: Genome):
        self.seeds = copy.deepcopy(genome.seeds)
