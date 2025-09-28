from itertools import chain
import numpy as np
from numpy.typing import NDArray
from random import random, randrange, sample, shuffle

# This whole file was adapted from the previous genetic algorithm implementation (genetic.py)
# with some parts from the VNS implementation (vns.py), some new code and updated lingo

# General framework for memetic algorithm:
#
# Generate initial population of memes
# Evaluate all memes
# Optimize each meme
# For each iteration:
#   Recombine pairs of memes into new memes
#   Create new memes by mutating the old ones
#   Optimize each new meme by local search
#   Select memes for the next iteration

# The problem:
# max   sum(v_i * x_i)
# s.t   sum(w_i * x_i) <= 10

# Auxiliary arrays
v: NDArray = np.array([8, 7, 6, 5, 4, 3, 2])
w: NDArray = np.array([5, 4, 3, 3, 2, 2, 1])


# Objective function
def objective(x: NDArray) -> int:
    return np.sum(np.multiply(x, v))


# Check if the only restriction for the problem is met
def viable(x: NDArray) -> bool:
    return np.sum(np.multiply(x, w)) <= 10


# Number of individuals in the population
population_size: int = 8

# Population
population: list[NDArray] = [
    np.zeros((7,), dtype=np.int8) for _ in range(population_size)
]

# Initial population: initialize every variable in every individual randomly
# while restriction is still satisfied
print("Initial solution:", end="\n\t")
for meme in population:
    for i in range(7):
        if random() < 0.5:
            meme[i] = 1
        if not viable(meme):
            meme[i] = 0
    print(meme, objective(meme), end="\n\t")
print()


# Size of the elite, i.e. number of the best individuals in each generation
# preserved to the next unchanged
# This bit is preserved from the genetic algorithm even though it was not explicitly
# required in the presented version of the memetic algorithm, but could be removed
# if the memetic algorithm converges without it by simply setting it to 0.
elite_size: int = 3

# Chance for individual to mutate
mutation_chance: float = 0.2


# Genetic Operators:


# Recombination
def recombination(mom: NDArray, dad: NDArray) -> list[NDArray]:
    cut_point: int = randrange(1, 6)
    offspring: list[NDArray] = []

    # First offspring: first half from mom, second from dad
    offspring_1: NDArray = np.array(list(mom[:cut_point]) + list(dad[cut_point:]))
    if viable(offspring_1):
        offspring.append(offspring_1)

    # Second offspring: first half from dad, second from mom
    offspring_2: NDArray = np.array(list(dad[:cut_point]) + list(mom[cut_point:]))
    if viable(offspring_2):
        offspring.append(offspring_2)

    return offspring


# Mutation
def mutation(x: NDArray) -> NDArray | None:
    mutated: NDArray = x.copy()

    # Every variable (bit) has a certain chance to flip
    for i in range(7):
        if random() < 0.3:
            mutated[i] = 1 - mutated[i]

    # Only return a viable mutation
    if viable(mutated):
        return mutated


# Selection
def selection(population: list[NDArray]) -> list[NDArray]:
    selected = []

    # Selecion by tournament
    # Two random individuals pulled from population, the best gets to the next generation
    for _ in range(population_size - elite_size):
        ind1, ind2 = sample(population, 2)
        if objective(ind1) > objective(ind2):
            selected.append(ind1)
        else:
            selected.append(ind2)

    return selected


# The k=1 neighborhood algorithm from vns.py
def neighborhood_gen(x: NDArray):
    neighbors: list[NDArray] = []
    for i in range(7):
        candidate: NDArray = x.copy()
        candidate[i] = 1 - candidate[i]
        if viable(candidate):
            neighbors.append(candidate)

    return neighbors


# Optimize an individual through one round of local search
# Taken from vns.py
def optimize(x: NDArray) -> NDArray:
    best_value: int = objective(x)
    best_n: NDArray = x
    neighborhood: list[NDArray] = neighborhood_gen(x)

    for n in neighborhood:
        n_value: int = objective(n)
        if n_value > best_value:
            best_value = n_value
            best_n = n

    return best_n


# The number of local search optimization rounds
optimization_rounds: int = 3

# Optimize each meme in the population
for _ in range(optimization_rounds):
    population = [optimize(x) for x in population]


# Define the flow of a single iteration, and return a new generation for the next
def iteration(population: list[NDArray]) -> list[NDArray]:
    # Evaluate and sort population by the objective function
    population.sort(key=lambda x: objective(x), reverse=True)

    # Create next population as a copy of the current population
    new_population: list[NDArray] = population.copy()

    # Perform crossover in the entire population and save offsprings
    moms = population.copy()
    dads = population.copy()
    shuffle(moms)
    shuffle(dads)
    offspring: list[list[NDArray]] = [
        recombination(mom, dad) for mom, dad in zip(moms, dads)
    ]
    augmented_population: list[NDArray] = population + list(
        chain.from_iterable(offspring)
    )

    # Mutate members from the augmented_population and add them back
    for individual in augmented_population:
        if random() < mutation_chance:
            mutated: NDArray | None = mutation(individual)
            if mutated is not None:
                augmented_population.append(mutated)

    # Optimize the memes
    for _ in range(optimization_rounds):
        augmented_population = [optimize(x) for x in population]

    # Select individuals from the augmented_population for a new population
    # The best individuals in the elite are preserved unchanged
    new_population[elite_size:] = selection(augmented_population)

    return new_population


# Maximum number of iterations
max_iterations: int = 20

# Perform all iterations
for g in range(max_iterations):
    population = iteration(population)
    print(f"Iteration {g + 1}:", end="\n\t")
    for meme in population:
        print(meme, objective(meme), end="\n\t")
    print()

# Sort the population one last time
population.sort(key=lambda x: objective(x), reverse=True)

print("Final solution:", population[0], objective(population[0]))
