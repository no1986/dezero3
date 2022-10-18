import numpy as np

from dezero.core import Function


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**2

    pass
