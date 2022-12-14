import numpy as np  # noqa

from dezero import Variable, no_grad  # noqa
from dezero import functions as F  # noqa


def main():
    a = Variable(3.0)
    b = Variable(2.0)
    c = Variable(1.0)

    y = a * b + c
    y.backward()

    print(y)
    print(a.grad, b.grad)
    return


if __name__ == "__main__":
    main()
