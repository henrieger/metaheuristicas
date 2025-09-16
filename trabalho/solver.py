from abc import abstractmethod
from functools import reduce
from typing import Any, Callable
from variable import Variable
from restriction import Restriction


class Solver:
    configured: bool = False
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
    ):
        self.variables = variables
        self.restrictions = restrictions
        self.objective = objective
        self.maximize = maximize

    def check_configure(self) -> None:
        if not self.configured:
            raise Exception("Solver has not been configured")

    def satisfied(self) -> bool:
        self.check_configure()
        return reduce(
            lambda x, y: (x and y), (r.satisfied() for r in self.restrictions)
        )

    def answer(self) -> dict[Variable, Any]:
        self.check_configure()
        return {v.name: v.value() for v in self.variables}

    @abstractmethod
    def configure(self, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def solve(self) -> None:
        raise NotImplementedError
