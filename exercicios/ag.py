from itertools import chain
import numpy as np
from numpy.typing import NDArray
from random import random, randrange, sample, shuffle

# General framework for genetic algorithm:
#
# Generate initial population
# For each generation:
#   Evaluate each individual
#   Save the elite
#   Genetic operators:
#       Crossover
#       Mutation
#       Selection
#   Generate new population

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
for individual in population:
    for i in range(7):
        if random() < 0.5:
            individual[i] = 1
        if not viable(individual):
            individual[i] = 0
    print(individual, objective(individual), end="\n\t")
print()


# Size of the elite, i.e. number of the best individuals in each generation
# preserved to the next unchanged
elite_size: int = 3

# Chance for individual to mutate
mutation_chance: float = 0.2


# Genetic Operators:


# Crossover
def crossover(mom: NDArray, dad: NDArray) -> list[NDArray]:
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


# Define the flow of a single generation, and return a new generation for the next iteration
def generation(population: list[NDArray]) -> list[NDArray]:
    # Evaluate and sort population by fitness (objective)
    population.sort(key=lambda x: objective(x), reverse=True)

    # Create next population as a copy of the current population
    new_population: list[NDArray] = population.copy()

    # Perform crossover in the entire population and save offsprings
    moms = population.copy()
    dads = population.copy()
    shuffle(moms)
    shuffle(dads)
    offspring: list[list[NDArray]] = [
        crossover(mom, dad) for mom, dad in zip(moms, dads)
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

    # Select individuals from the augmented_population for a new population
    # The best individuals in the elite are preserved unchanged
    new_population[elite_size:] = selection(augmented_population)

    return new_population


# Maximum number of generations
max_generations: int = 30

# Perform all generations iteratively
for g in range(max_generations):
    population = generation(population)
    print(f"Population {g + 1}:", end="\n\t")
    for individual in population:
        print(individual, objective(individual), end="\n\t")
    print()

print("Final solution:", population[0], objective(population[0]))
