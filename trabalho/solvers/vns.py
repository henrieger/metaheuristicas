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


from solver import Solver
from number import Number
from variable import Variable
from restriction import Restriction
from typing import Callable, Generator


class VariableNeighborhoodSearch(Solver):
    # The amount of times to perform a local search
    local_searches: int

    # How to generate an initial solution
    initial_solution_func: Callable[[list[Variable]], list[Number]]

    # Functions that generate neighborhoods
    neighborhoods: list[Callable[[list[Number]],
                                 Generator[list[Number], None, None]]]

    def __init__(
        self,
        variables: list[Variable],
        restrictions: list[Restriction],
        objective: Callable[..., float],
        *,
        maximize: bool = False,
        local_searches: int = 10,
        initial_solution_func: Callable[[list[Variable]], list[Number]] = lambda x: [
            0 for _ in x
        ],
        neighborhoods: list[
            Callable[[list[Number]], Generator[list[Number], None, None]]
        ],
    ):
        super().__init__(variables, restrictions, objective, maximize=maximize)

        self.local_searches = local_searches
        self.initial_solution_func = initial_solution_func
        self.neighborhoods = neighborhoods
        self.objective = lambda x: objective(
            x) - 50 * (1 - self.viable_solution(x))

    def solve(self):
        solution = self.initial_solution_func(self.variables)
        score = self.objective(solution)
        self.bests = [score]

        found_best_in_neighborhood: bool = False
        while not found_best_in_neighborhood:
            for i, neighborhood in enumerate(self.neighborhoods):
                print(f"Testing neighborhood {i}")
                nbest, nscore = self.local_search(solution, neighborhood)
                if (self.maximize and nscore > score) or (
                    not self.maximize and nscore < score
                ):
                    solution = nbest
                    score = nscore
                    self.bests.append(nscore)
                    break
                found_best_in_neighborhood = True

        for i, variable in enumerate(self.variables):
            variable.set_value(solution[i])

    def local_search(
        self,
        solution: list[Number],
        neighborhood: Callable[[list[Number]], Generator[list[Number], None, None]],
    ) -> tuple[list[Number], Number]:
        """
        Perform a local search and return the best solution found
        in the neighborhood, and its score
        """
        best = solution
        best_score = self.objective(solution)

        for i in range(self.local_searches):
            print(f"Local search {i}")
            best = solution
            best_score = self.objective(solution)
            print(f"Starting search with solution {
                  solution} (score {best_score})")
            for neighbor in neighborhood(solution):
                print(f"Testing neighbor {neighbor}")
                nscore = self.objective(neighbor)
                if self.better(nscore, best_score):
                    print(f"Found as new best!: Score {nscore}")
                    best_score = nscore
                    best = neighbor
            solution = best

        return (best, best_score)

    def better(self, a: Number, b: Number) -> bool:
        if self.maximize:
            return a > b
        return a < b

    def viable_solution(self, solution: list[Number]) -> bool:
        for variable, value in zip(self.variables, solution):
            variable.set_value(value)
        return self.satisfied()
