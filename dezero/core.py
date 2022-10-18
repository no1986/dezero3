from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpy as np


# =============================================================================
# Variable
# =============================================================================
class Variable:
    def __init__(self, data: np.ndarray | None) -> None:
        self.data: np.ndarray | None = data
        self.grad: np.ndarray | None = None
        self.creator: Function | None = None
        return

    def set_creator(self, func: Function) -> None:
        self.creator = func
        return

    def backward(self) -> None:
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()
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
