import numpy as np  # noqa

from dezero import Variable  # noqa
from dezero import functions as F  # noqa
from dezero.core import add  # noqa


def main():
    x = Variable(3.0)
    y = add(x, x)
    y.backward()
    print(x.grad)

    x.cleargrad()
    y = add(add(x, x), x)
    y.backward()
    print(x.grad)
    return


if __name__ == "__main__":
    main()
