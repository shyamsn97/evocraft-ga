import random
import attr
import numpy as np
import typing
from tqdm import tqdm
from evocraft_ga.genome.genome import Genome
from evocraft_ga.spawner.spawner import Spawner
from evocraft_ga.ns.novelty_search import NoveltySearch
from evocraft_ga.external.minecraft_pb2 import *  # noqa

class_dict = {0: REDSTONE_BLOCK, 1: SLIME, 2: COAL_BLOCK}


@attr.s
class GeneticAlgorithm:
    population_size: int = attr.ib(default=50)
    num_elites: int = attr.ib(default=5)
    noise_dims: int = attr.ib(default=100)
    cube_dims: int = attr.ib(default=10)
    noise_stdev: float = attr.ib(default=0.02)
    num_neighbors: int = attr.ib(default=5)
    num_bcs: int = attr.ib(default=200)
    clear_blocks_on_fail: bool = attr.ib(default=False)
    class_dict = attr.ib(default=class_dict)
    one_seed: bool = attr.ib(default=True)
    criteria: float = attr.ib(default=0.8)
    model_class: str = attr.ib(default="linear")

    elite_coords = attr.ib(default=(100, 10, 100))
    spawner: Spawner = attr.ib(default=None)
    genomes: typing.List[Genome] = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.populate()
        self.novelty_search = NoveltySearch(
            num_neighbors=self.num_neighbors, maxlen=self.num_bcs
        )
        if self.spawner is None:
            self.spawner = Spawner(cube_len=self.cube_dims, class_dict=self.class_dict)

    def populate(self):
        self.genomes = []
        for i in range(self.population_size):
            genome = Genome(
                noise_dims=self.noise_dims,
                cube_dims=self.cube_dims,
                noise_stdev=self.noise_stdev,
                model_class=self.model_class,
                num_classes=len(self.class_dict.keys()),
            )
            self.genomes.append(genome)

    def evolve(self, num_generations=100):
        seed = None
        if self.one_seed:
            seed = np.random.randint(1, high=2 ** 31 - 1)
        try:
            bar = tqdm(np.arange(num_generations))
            for i in bar:
                outputs = [
                    genome.generate(to_numpy=True, seed=seed) for genome in self.genomes
                ]
                fitnesses = self.novelty_search.apply(outputs)
                ranking = self.rank_outputs(fitnesses)
                max_fitness = np.max(fitnesses)
                bar.set_description(
                    "Generation: {} with max fitness: {}".format(i, max_fitness)
                )
                self.spawn(outputs)
                self.spawner.populate_arr(
                                        self.sample_block_class(outputs[ranking[0]]),
                                        self.elite_coords[0],
                                        self.elite_coords[1],
                                        self.elite_coords[2]
                )
                self.reproduce(fitnesses, ranking)
        except Exception as e:
            print("Caught exception: {}".format(e.traceback))
            if self.clear_blocks_on_fail:
                self.spawner.clear_population(self.population_size)

    def spawn(self, outputs, clear_population = True):
        sampled_outputs = []
        for output in outputs:
            sampled_outputs.append(self.sample_block_class(output))
        self.spawner.populate(sampled_outputs, clear_population=clear_population)

    def sample_block_class(self, output):
        arr = np.zeros((self.cube_dims, self.cube_dims, self.cube_dims))
        for coords in np.ndindex((self.cube_dims, self.cube_dims, self.cube_dims)):
            probs = output[coords][:]
            probs = probs / np.sum(probs)
            if np.max(probs) < self.criteria:
                arr[coords] = -1.0
            else:
                arr[coords] = np.random.choice(probs.shape[0], p=probs)
        return arr

    def rank_outputs(self, fitnesses):
        return np.argsort(fitnesses)[::-1]

    def reproduce(self, fitnesses, ranking) -> None:
        """
            reproduces by taking the top N elites and copying + mutating their seeds
        """
        elite_ranking = ranking[: self.num_elites]
        elite_ranking_set = set(elite_ranking)
        best_genome_index = elite_ranking[0]

        for genome_index in range(len(self.genomes)):
            # first mutate non elite agents
            if genome_index not in elite_ranking_set:
                random_elite_index = random.choice(elite_ranking)
                random_elite = self.genomes[random_elite_index]
                self.genomes[genome_index].copy_parameters(random_elite)  # copy seeds
                self.genomes[genome_index].mutate()

        for genome_index in elite_ranking_set:
            # mutate elites except the best one
            if genome_index != best_genome_index:
                self.genomes[genome_index].mutate()
        return ranking
