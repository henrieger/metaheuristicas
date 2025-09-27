from copy import deepcopy
from random import random, randint, randrange
from typing import Callable, Generator
from number import Number
from variable import Variable
from restriction import Restriction
from solvers.test import TestSolver
from solvers.genetic import Chromosome, GeneticAlgorithm
from solvers.vns import VariableNeighborhoodSearch


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


# Variable Neighborhood Search


def random_solution(variables: list[Variable]) -> list[Number]:
    return [randint(0, 1) for _ in variables]


# def neighborhood1(solution: list[Number]) -> Generator[list[Number], None, None]:
#     tail = solution[3:]
#     yield [0, 0, 0] + tail
#     yield [0, 0, 1] + tail
#     yield [0, 1, 0] + tail
#     yield [0, 1, 1] + tail
#     yield [1, 0, 0] + tail
#     yield [1, 1, 0] + tail
#     yield [1, 0, 1] + tail
#     yield [1, 0, 1] + tail
#     yield [1, 1, 1] + tail
#
#
# def neighborhood2(solution: list[Number]) -> Generator[list[Number], None, None]:
#     head = solution[0:3]
#     tail = solution[5:]
#
#     yield head + [0, 0] + tail
#     yield head + [0, 1] + tail
#     yield head + [1, 0] + tail
#     yield head + [1, 1] + tail
#
#
# def neighborhood3(solution: list[Number]) -> Generator[list[Number], None, None]:
#     head = solution[0:5]
#
#     yield head + [0, 0]
#     yield head + [0, 1]
#     yield head + [1, 0]
#     yield head + [1, 1]


def neighborhood1(solution: list[Number]) -> Generator[list[Number], None, None]:
    for i, bit in enumerate(solution):
        head = solution[:i]
        tail = solution[i + 1:]
        yield head + [1 - bit] + tail


def neighborhood2(solution: list[Number]) -> Generator[list[Number], None, None]:
    for i, bit1 in enumerate(solution):
        head = solution[:i]
        body = solution[i + 1:]
        for j, bit2 in enumerate(body, i + 1):
            middle = solution[i + 1: j]
            tail = solution[j + 1:]
            yield head + [1 - bit1] + middle + [1 - bit2] + tail


vns_solver = VariableNeighborhoodSearch(
    x,
    [r],
    objective,
    maximize=True,
    initial_solution_func=random_solution,
    neighborhoods=[neighborhood1, neighborhood2],
    local_searches=3,
)
vns_solver.solve()
print(
    "VariableNeighborhoodSearch:",
    [x_i.value() for x_i in x],
    objective(x),
    vns_solver.bests,
)
