from abc import abstractmethod
from functools import reduce
from typing import Any, Callable
from variable import Variable
from restriction import Restriction


class Solver:
    variables: list[Variable]
    restrictions: list[Restriction]
    objective: Callable[..., float]
    maximize: bool

    def __init__(
        self,
        variables: list[Variable],
        restrictions: list[Restriction],
        objective: Callable[..., float],
        *,
        maximize: bool = False,
        **kwargs,
    ):
        self.variables = variables
        self.restrictions = restrictions
        self.objective = objective
        self.maximize = maximize

    def satisfied(self) -> bool:
        return reduce(
            lambda x, y: (x and y), (r.satisfied() for r in self.restrictions)
        )

    def answer(self) -> dict[Variable, Any]:
        return {v.name: v.value() for v in self.variables}

    @abstractmethod
    def solve(self) -> None:
        raise NotImplementedError
