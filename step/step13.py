import numpy as np  # noqa

import dezero as dz  # noqa
from dezero import functions as F  # noqa


def main():
    x0 = dz.Variable(2.0)
    x1 = dz.Variable(3.0)

    y = dz.add(F.square(x0), F.square(x1))
    y.backward()
    print(y.data)
    print(x0.grad, x1.grad)
    return


if __name__ == "__main__":
    main()
