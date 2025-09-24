from number import Number


class Variable:
    name: str
    min: Number | None
    max: Number | None
    value_set: set[Number] | None
    _value: Number

    def __init__(
        self,
        name: str = "undefined",
        *,
        min: Number | None = None,
        max: Number | None = None,
        value_set: set[Number] | None = None,
    ) -> None:
        self.name = name
        self.min = min
        self.max = max
        self.value_set = value_set
        self._value = 0

    def set_value(self, value: Number) -> None:
        if self.value_set is not None and value not in self.value_set:
            raise Exception(f"Value cannot be outside of {self.value_set}")

        if self.min is not None and value < self.min:
            raise Exception(f"Value cannot be less than {min}")

        if self.max is not None and value > self.max:
            raise Exception(f"Value cannot be more than {max}")

        self._value = value

    def value(self) -> Number:
        return self._value

    def __eq__(self, other: object, /) -> bool:
        if isinstance(other, Variable):
            return self._value == other._value

        raise TypeError(
            f"Cannot use operator == between Variable and {type(other)}")

    def __add__(self, other: object, /) -> Number:
        if isinstance(other, Variable):
            return self.value() + other.value()

        if isinstance(other, Number):
            return self.value() + other

        raise TypeError(
            f"Cannot use operator + between Variable and {type(other)}")

    def __sub__(self, other: object, /) -> Number:
        if isinstance(other, Variable):
            return self.value() - other.value()

        if isinstance(other, Number):
            return self.value() - other

        raise TypeError(
            f"Cannot use operator - between Variable and {type(other)}")

    def __mul__(self, other: object, /) -> Number:
        if isinstance(other, Variable):
            return self.value() * other.value()

        if isinstance(other, Number):
            return self.value() * other

        raise TypeError(
            f"Cannot use operator * between Variable and {type(other)}")

    def __div__(self, other: object, /) -> Number:
        if isinstance(other, Variable):
            return self.value() / other.value()

        if isinstance(other, Number):
            return self.value() / other

        raise TypeError(
            f"Cannot use operator / between Variable and {type(other)}")

    def __str__(self) -> str:
        return self.name
