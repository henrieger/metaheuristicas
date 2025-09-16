from solver import Solver
from random import randrange

from variable import Variable
from restriction import Restriction


class TestSolver(Solver):
    def configure(self, **kwargs) -> None:
        self.configured = True
        pass

    def solve(self) -> None:
        self.check_configure()
        while not self.satisfied():
            for variable in self.variables:
                variable.set_value(randrange(variable.min, variable.max))


if __name__ == "__main__":
    x1: Variable[int] = Variable[int]("x1", min=1, max=5)
    x2: Variable[int] = Variable[int]("x2", min=1, max=5)
    x3: Variable[int] = Variable[int]("x3", min=1, max=5)

    r1: Restriction = Restriction([x1, x2], lambda x1, x2: x1 + x2 <= 4)
    r2: Restriction = Restriction([x1, x3], lambda x1, x3: x1 + x3 <= 7)
    r3: Restriction = Restriction([x2, x3], lambda x2, x3: x2 - x3 >= 2)

    solver: TestSolver = TestSolver(
        [x1, x2, x3], [r1, r2, r3], lambda x1, x2, x3: x1 + x2 + x3, maximize=True
    )

    solver.configure()
    solver.solve()

    print(solver.answer())
