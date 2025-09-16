from numbers import Number


class Variable[T]:
    name: str
    min: T | None
    max: T | None
    value_set: set[T] | None
    _value: T

    def __init__(
        self,
        name: str = "undefined",
        *,
        min: T | None = None,
        max: T | None = None,
        value_set: set[T] | None = None,
    ) -> None:
        self.name = name
        self.min = min
        self.max = max
        self.value_set = value_set
        self._value = 0

    def set_value(self, value: T) -> None:
        if self.value_set is not None and value not in self.value_set:
            raise Exception(f"Value cannot be outside of {self.value_set}")

        if self.min is not None and value < self.min:
            raise Exception(f"Value cannot be less than {min}")

        if self.max is not None and value > self.max:
            raise Exception(f"Value cannot be more than {max}")

        self._value = value

    def value(self) -> T:
        return self._value

    def __eq__(self, other: object, /) -> bool:
        if isinstance(other, type(self)):
            return self._value == other._value

        raise NotImplementedError

    def __add__(self, other: object, /) -> float:
        if isinstance(other, Variable):
            return self.value() + other.value()

        if isinstance(other, Number):
            return self.value() + other

        raise NotImplementedError

    def __sub__(self, other: object, /) -> float:
        if isinstance(other, Variable):
            return self.value() - other.value()

        if isinstance(other, Number):
            return self.value() - other

        raise NotImplementedError

    def __mul__(self, other: object, /) -> float:
        if isinstance(other, Variable):
            return self.value() * other.value()

        if isinstance(other, Number):
            return self.value() * other

        raise NotImplementedError

    def __div__(self, other: object, /) -> float:
        if isinstance(other, Variable):
            return self.value() / other.value()

        if isinstance(other, Number):
            return self.value() / other

        raise NotImplementedError

    def __str__(self) -> str:
        return self.name
