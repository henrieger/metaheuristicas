import numpy as np
from typing import Callable
from numpy.typing import NDArray
from random import random

# General framework for variable neighborhood search:
#
# Generate initial solution
# While solution is not the best in neighborhood:
#   Generate the neighborhoods of solution
#   For each neighborhood:
#       res := local search in neighborhood
#       if res is better than solution:
#           solution := res
#           break
#   If loop ended naturally:
#       solution is the best


# The problem:
# max   sum(v_i * x_i)
# s.t   sum(w_i * x_i) <= 10

# Solution
x: NDArray = np.zeros((7,), dtype=np.int8)

# Auxiliary arrays
v: NDArray = np.array([8, 7, 6, 5, 4, 3, 2])
w: NDArray = np.array([5, 4, 3, 3, 2, 2, 1])


# Objective function
def objective(x: NDArray) -> int:
    return np.sum(np.multiply(x, v))


# Check if the only restriction for the problem is met
def viable(x: NDArray) -> bool:
    return np.sum(np.multiply(x, w)) <= 10


# Initial solution: initialize every variable randomly
# while restriction is still satisfied
for i in range(7):
    if random() < 0.5:
        x[i] = 1
    if not viable(x):
        x[i] = 0
print("Initial solution:", x, objective(x))


# Define neighborhoods
# k = 1: Set of all viable solutions with exactly 1 different variable value (i.e 1 bit-flip)
def neighborhood_1(x: NDArray) -> list[NDArray]:
    neighbors: list[NDArray] = []
    for i in range(7):
        candidate: NDArray = x.copy()
        candidate[i] = 1 - candidate[i]
        if viable(candidate):
            neighbors.append(candidate)

    return neighbors


# k = 2: Set of all viable solutions with exactly 2 different variable values (i.e 2 bit-flips)
def neighborhood_2(x: NDArray) -> list[NDArray]:
    neighbors: list[NDArray] = []
    for i in range(7):
        for j in range(i + 1, 7):
            candidate: NDArray = x.copy()
            candidate[i] = 1 - candidate[i]
            candidate[j] = 1 - candidate[j]
            if viable(candidate):
                neighbors.append(candidate)

    return neighbors


# Define a local search through neighborhood
def local_search(
    x: NDArray, neighborhood_function: Callable[[NDArray], list[NDArray]]
) -> tuple[NDArray, int]:
    best_value: int = objective(x)
    best_n: NDArray = x
    neighborhood: list[NDArray] = neighborhood_function(x)

    for n in neighborhood:
        n_value: int = objective(n)
        if n_value > best_value:
            best_value = n_value
            best_n = n

    return (best_n, best_value)


# Number of local searches
local_search_rounds: int = 5

# Begin iterative search
iterations: int = 1
while ...:
    # k = 1
    search_result: NDArray = x
    search_value: int = 0
    curr_value: int = objective(x)
    for _ in range(local_search_rounds):
        search_result, search_value = local_search(search_result, neighborhood_1)
    if search_value > curr_value:
        x = search_result
        print(f"Iteration {iterations}, neighborhood 1, Solution: {x} {search_value}")
        iterations += 1
        continue

    # k = 2
    search_result: NDArray = x
    search_value: int = 0
    curr_value: int = objective(x)
    for _ in range(local_search_rounds):
        search_result, search_value = local_search(search_result, neighborhood_2)
    if search_value > curr_value:
        x = search_result
        print(f"Iteration {iterations}, neighborhood 2, Solution: {x} {search_value}")
        iterations += 1
        continue

    print(f"Iteration {iterations} found nothing better. Stopping")
    break

print("Final solution:", x, objective(x))
