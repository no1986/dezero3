from __future__ import annotations

from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import List

import numpy as np


# =============================================================================
# Variable
# =============================================================================
class Variable:
    def __init__(self, data: np.ndarray | List[Number] | Number | None) -> None:
        if not isinstance(data, np.ndarray):
            if (isinstance(data, List) and all(isinstance(d, Number) for d in data)) or (
                isinstance(data, Number)
            ):
                data = np.array(data)
            elif data is not None:
                raise TypeError(f"{type(data)} is not supported")
                pass
            pass

        self.data: np.ndarray | None = data
        self.grad: np.ndarray | None = None
        self.creator: Function | None = None
        return

    def set_creator(self, func: Function) -> None:
        self.creator = func
        return

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)
            pass

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)
                pass
            pass
        return

    pass


# =============================================================================
# Function
# =============================================================================
class Function(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.input: Variable | None = None
        self.output: Variable | None = None
        return

    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    pass
