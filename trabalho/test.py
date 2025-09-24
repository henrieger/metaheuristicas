from copy import deepcopy
from random import random, randint, randrange
from typing import Callable
from variable import Variable
from restriction import Restriction
from solvers.test import TestSolver
from solvers.genetic import Chromosome, GeneticAlgorithm


x: list[Variable] = [Variable(f"x{i}", min=0, max=1) for i in range(7)]

v: list[int] = [8, 7, 6, 5, 4, 3, 2]
w: list[int] = [5, 4, 3, 3, 2, 2, 1]


def objective(x: list[Variable]) -> float:
    sum = 0
    for x_i, v_i in zip(x, v):
        sum += x_i * v_i
    return sum


def restriction_func(*x: Variable) -> bool:
    sum = 0
    for x_i, w_i in zip(x, w):
        sum += x_i * w_i
    return 0 < sum <= 10


r: Restriction = Restriction(x, restriction_func)

# Test Solver
test_solver: TestSolver = TestSolver(x, [r], objective, maximize=True)
test_solver.solve()
print("TestSolver:", [x_i.value() for x_i in x], objective(x))


# Genetic Algorithm


def crossover(mom: Chromosome, dad: Chromosome) -> tuple[Chromosome, Chromosome]:
    cutting_point = randrange(1, len(mom))
    child1 = mom[:cutting_point] + dad[cutting_point:]
    child2 = dad[:cutting_point] + mom[cutting_point:]

    return (child1, child2)


def mutation(chromosome: Chromosome) -> Chromosome:
    new: Chromosome = deepcopy(chromosome)
    for position in range(len(chromosome)):
        if random() < 0.5:
            new[position] = 1 - chromosome[position]

    return new


def fitness(
    chromosome: Chromosome,
    objective: Callable[[Chromosome], float],
    viability: Callable[[Chromosome], bool],
):
    return objective(chromosome) - 50 * (1 - viability(chromosome))


def random_chromosome(variables: list[Variable]) -> Chromosome:
    return [randint(0, 1) for _ in variables]


genetic_algorithm: GeneticAlgorithm = GeneticAlgorithm(
    x,
    [r],
    objective,
    maximize=True,
    crossover_op=crossover,
    mutation_op=mutation,
    fitness_func=fitness,
    mutation_prob=0.1,
    stop_criteria=lambda g, _: g > 100,
    population_size=20,
    elite_size=1,
    initial_chromosome_func=random_chromosome,
)
genetic_algorithm.solve()
print(
    "GeneticAlgorithm:",
    [x_i.value() for x_i in x],
    objective(x),
)
