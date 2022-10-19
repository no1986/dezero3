import numpy as np  # noqa

from dezero import Variable, no_grad  # noqa
from dezero import functions as F  # noqa


def main():
    x = Variable(2.0)
    y = -x
    print(y)

    y0 = 4 / x
    y1 = x / 2
    print(y0, y1)

    y0 = 2.0 - x
    y1 = x - 1.0
    print(y0, y1)

    y = x**3
    print(y)
    return


if __name__ == "__main__":
    main()
