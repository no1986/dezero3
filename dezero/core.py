from __future__ import annotations

from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import List, Tuple

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
        self.generation: int = 0
        return

    def set_creator(self, func: Function) -> None:
        self.creator = func
        self.generation = func.generation + 1
        return

    def backward(self) -> None:
        def add_func(f: Function) -> None:
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
                pass
            return

        if self.grad is None:
            self.grad = np.ones_like(self.data)
            pass

        funcs = []
        seen_set = set()
        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, Tuple):
                gxs = (gxs,)
                pass

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                    pass

                if x.creator is not None:
                    add_func(x.creator)
                    pass
                pass
            pass
        return

    def cleargrad(self) -> None:
        self.grad = None
        return

    pass


# =============================================================================
# Function
# =============================================================================
class Function(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.input: Variable | None = None
        self.output: Variable | None = None
        self.generation: int = 0
        return

    def __call__(self, *inputs: Tuple[Variable]) -> Variable:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, Tuple):
            ys = (ys,)
            pass
        outputs = [Variable(y) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
            pass

        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    pass


# =============================================================================
# 四則演算 / 演算子のオーバーロード
# =============================================================================
class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 + x1

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
        return (gy, gy)

    pass


def add(x0: Variable, x1: Variable) -> Variable:
    return Add()(x0, x1)
