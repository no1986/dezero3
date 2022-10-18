import numpy as np


class Variable:
    def __init__(self, data: np.ndarray | None) -> None:
        self.data: np.ndarray | None = data
        return

    pass
