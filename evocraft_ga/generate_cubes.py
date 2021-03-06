import click
from evocraft_ga.ga.ga import GeneticAlgorithm
from evocraft_ga.genome.genome import model_dict

model_classes = list(model_dict.keys())


@click.command()
@click.option("--population_size", default=50, help="Population size")
@click.option("--num_generations", default=100, help="Number of generations to evolve")
@click.option("--num_elites", default=5, help="Number of elites for GA")
@click.option(
    "--noise_dims", default=100, help="Noise dimension to generate structures"
)
@click.option("--cube_dims", default=10, help="Dimension of cube to be created")
@click.option(
    "--num_neighbors", default=5, help="Number of neighbors to use for Novelty Search"
)
@click.option(
    "--num_bcs",
    default=200,
    help="Number of Behavior Constants to save in cache for Novelty Search",
)
@click.option("--noise_stdev", default=0.05, help="Stdev for Noise perturbations")
@click.option(
    "--criteria", default=0.8, help="Confidence critera for cube type assignment"
)
@click.option(
    "--model_class",
    default="linear",
    type=click.Choice(model_classes, case_sensitive=False),
)
def generate_cubes(
    population_size: int,
    num_generations: int,
    num_elites: int,
    noise_dims: int,
    cube_dims: int,
    num_neighbors: int,
    num_bcs: int,
    noise_stdev: float,
    criteria: float,
    model_class: str,
):
    g = GeneticAlgorithm(
        population_size=population_size,
        num_elites=num_elites,
        noise_dims=noise_dims,
        cube_dims=cube_dims,
        num_neighbors=num_neighbors,
        num_bcs=num_bcs,
        noise_stdev=noise_stdev,
        criteria=criteria,
        model_class=model_class,
        clear_blocks_on_fail=True,
    )
    g.evolve(num_generations)


if __name__ == "__main__":
    generate_cubes()
