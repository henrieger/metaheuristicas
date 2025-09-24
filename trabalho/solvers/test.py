from solver import Solver
from random import random, randrange, choice

from variable import Variable
from restriction import Restriction


class TestSolver(Solver):
    def solve(self) -> None:
        while not self.satisfied():
            for variable in self.variables:
                if variable.value_set is not None:
                    variable.set_value(choice(list(variable.value_set)))
                elif variable.min is not None and variable.max is not None:
                    variable.set_value(
                        randrange(int(variable.min), int(variable.max + 1))
                    )
                else:
                    variable.set_value(random())


if __name__ == "__main__":
    x1: Variable = Variable("x1", min=1, max=5)
    x2: Variable = Variable("x2", min=1, max=5)
    x3: Variable = Variable("x3", min=1, max=5)

    r1: Restriction = Restriction([x1, x2], lambda x1, x2: x1 + x2 <= 4)
    r2: Restriction = Restriction([x1, x3], lambda x1, x3: x1 + x3 <= 7)
    r3: Restriction = Restriction([x2, x3], lambda x2, x3: x2 - x3 >= 2)

    solver: TestSolver = TestSolver(
        [x1, x2, x3], [r1, r2, r3], lambda x1, x2, x3: x1 + x2 + x3, maximize=True
    )

    solver.solve()

    print(solver.answer())
