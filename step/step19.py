import numpy as np  # noqa

from dezero import Variable, no_grad  # noqa
from dezero import functions as F  # noqa


def main():
    x = Variable([[1, 2, 3], [4, 5, 6]])
    print(type(x.shape), x.shape)
    print(type(x.ndim), x.ndim)
    print(type(x.size), x.size)
    print(type(x.dtype), x.dtype)
    print(type(len(x)), len(x))
    print(x)
    return


if __name__ == "__main__":
    main()
