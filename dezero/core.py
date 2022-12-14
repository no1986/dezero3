from __future__ import annotations

import contextlib
import weakref
from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import List, Tuple

import numpy as np


# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True
    pass


@contextlib.contextmanager
def using_config(name: str, value: any) -> None:
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)
        pass
    return


def no_grad() -> None:
    return using_config("enable_backprop", False)


# =============================================================================
# Variable
# =============================================================================
class Variable:
    __array_priority__ = 200

    def __init__(self, data: np.ndarray | List[Number] | Number | None, name: str = None) -> None:
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except Exception:
                raise TypeError(f"{type(data)} is not supported")
                pass
            pass

        self.data: np.ndarray | None = data
        self.name: str | None = name
        self.grad: np.ndarray | None = None
        self.creator: Function | None = None
        self.generation: int = 0
        return

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"variable({p})"

    def set_creator(self, func: Function) -> None:
        self.creator = func
        self.generation = func.generation + 1
        return

    def backward(self, retain_grad: bool = False) -> None:
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
            gys = [output().grad for output in f.outputs]
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

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None
                    pass
                pass
            pass
        return

    def cleargrad(self) -> None:
        self.grad = None
        return

    pass


def as_variable(obj):
    if not isinstance(obj, Variable):
        obj = Variable(obj)
        pass
    return obj


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
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, Tuple):
            ys = (ys,)
            pass
        outputs = [Variable(y) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
                pass

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
            pass

        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    pass


# =============================================================================
# ???????????? / ?????????????????????????????????
# =============================================================================
class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 + x1

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
        return (gy, gy)

    pass


def add(x0: Variable, x1: Variable | np.ndarray | list | Number) -> Variable:
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 * x1

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return (gy * x1, gy * x0)

    pass


def mul(x0: Variable, x1: Variable | np.ndarray | list | Number) -> Variable:
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x

    def backward(self, gy: np.ndarray) -> np.ndarray:
        return -gy

    pass


def neg(x: Variable) -> Variable:
    return Neg()(x)


class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 - x1

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
        return (gy, -gy)


def sub(x0: Variable, x1: Variable | np.ndarray | list | Number) -> Variable:
    return Sub()(x0, x1)


def rsub(x0: Variable, x1: np.ndarray | list | Number) -> Variable:
    x1 = as_variable(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        return x0 / x1

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return (gx0, gx1)


def div(x0: Variable, x1: Variable | np.ndarray | list | Number) -> Variable:
    x1 = as_variable(x1)
    return Div()(x0, x1)


def rdiv(x0: Variable, x1: np.ndarray | list | Number) -> Variable:
    x1 = as_variable(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c: Number) -> None:
        self.c: Number = c
        return

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**self.c

    def backward(self, gy: np.ndarray) -> np.ndarray:
        x = self.inputs[0].data
        c = self.c
        return c * x ** (c - 1) * gy


def pow(x: Variable, c: Variable | list | Number) -> Variable:
    return Pow(c)(x)


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
