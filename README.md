evocraft-ga
================
## (WIP) more algorithms / models coming soon
## Exploring deep genetic algorithms in Minecraft using https://github.com/real-itu/Evocraft-py

## Examples
### 1. Generating random cube structures with DNN using DeepGA [code](evocraft_ga/generate_cubes.py)
### 2. (WIP) Generating random structures with CPPN using DeepGA

Installation
------------

### 1. Bring in source from https://github.com/real-itu/Evocraft-py and apply some processing with setup.sh
```
    sh setup.sh
```
### 2. setup server (Only need to run this once)
```
    sh setup_server.sh
```

Run the generate cubes example
-----
### 1. Start server
```
    sh start_server.sh
```
### 2. Then log into Minecraft, follow steps in https://github.com/real-itu/Evocraft-py and join the server in `localhost`

### 3. To run the simple example
```
    python evocraft_ga/generate_cubes.py 

    Options:
    --population_size INTEGER  Population size
    --num_generations INTEGER  Number of generations to evolve
    --num_elites INTEGER       Number of elites for GA
    --noise_dims INTEGER       Noise dimension to generate structures
    --cube_dims INTEGER        Dimension of cube to be created
    --num_neighbors INTEGER    Number of neighbors to use for Novelty Search
    --num_bcs INTEGER          Number of BCS to save in cache for Novelty Search
    --noise_stdev FLOAT        Stdev for Noise perturbations
    --help                     Show this message and exit.

```
### 4. Watch the cubes form!
## Some examples below
### Start of generation:
![Alt Text](gifs/out_first.gif)
### Cubes starting to form:
![Alt Text](gifs/out_second.gif)
### Cubes formed:
![Alt Text](gifs/out_final.gif)

Authors
-------

### `evocraft-ga` was written by `Shyam Sudhakaran <shyamsnair97@gmail.com>`.