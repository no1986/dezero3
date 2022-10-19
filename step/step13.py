import numpy as np  # noqa

from dezero import Variable  # noqa
from dezero import functions as F  # noqa
from dezero.core import add  # noqa


def main():
    x0 = Variable(2.0)
    x1 = Variable(3.0)

    y = add(F.square(x0), F.square(x1))
    y.backward()
    print(y.data)
    print(x0.grad, x1.grad)
    return


if __name__ == "__main__":
    main()
