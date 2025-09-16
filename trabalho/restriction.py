from typing import Callable
from variable import Variable


class Restriction:
    func: Callable[..., bool]

    def __init__(self, vars: list[Variable], func: Callable[..., bool]) -> None:
        self.vars = vars
        self.func = func

    def satisfied(self) -> bool:
        return self.func(*self.vars)
