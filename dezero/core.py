from abc import ABCMeta, abstractmethod

import numpy as np


# =============================================================================
# Variable
# =============================================================================
class Variable:
    def __init__(self, data: np.ndarray | None) -> None:
        self.data: np.ndarray | None = data
        return

    pass


# =============================================================================
# Function
# =============================================================================
class Function(metaclass=ABCMeta):
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    @abstractmethod
    def forward(self):
        pass

    pass
