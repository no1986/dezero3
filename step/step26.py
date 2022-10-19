import numpy as np  # noqa

from dezero import Variable, no_grad  # noqa
from dezero import functions as F  # noqa
from dezero.utils import plot_dot_graph # noqa


def goldstein(x, y):
    z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)) * (
        30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )
    return z


def main():
    x = Variable(1.0, "x")
    y = Variable(1.0, "y")
    z = goldstein(x, y)
    z.backward()

    z.name = "z"
    plot_dot_graph(z, verbose=False, to_file="goldstein", format="png")
    return


if __name__ == "__main__":
    main()
