import numpy as np  # noqa

import dezero as dz  # noqa
from dezero import functions as F  # noqa


def main():
    x = dz.Variable(3.0)
    y = dz.add(x, x)
    y.backward()
    print(x.grad)

    x.cleargrad()
    y = dz.add(dz.add(x, x), x)
    y.backward()
    print(x.grad)
    return


if __name__ == "__main__":
    main()
