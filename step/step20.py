import numpy as np  # noqa

import dezero as dz  # noqa
from dezero import functions as F  # noqa


def main():
    a = dz.Variable(3.0)
    b = dz.Variable(2.0)
    c = dz.Variable(1.0)

    y = a * b + c
    y.backward()

    print(y)
    print(a.grad, b.grad)
    return


if __name__ == "__main__":
    main()
