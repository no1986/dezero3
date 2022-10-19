import numpy as np  # noqa

from dezero import Variable, no_grad  # noqa
from dezero import functions as F  # noqa


def main():
    x = Variable(2.0)
    y = 3.0 * x + 1.0
    y.backward()

    print(y, x.grad)

    x.cleargrad()
    y = np.array([2.0]) * x
    y.backward()
    print(y, x.grad)
    return


if __name__ == "__main__":
    main()
