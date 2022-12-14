import numpy as np  # noqa

from dezero import Variable  # noqa
from dezero import functions as F  # noqa
from dezero.core import add  # noqa


def main():
    x = Variable(2.0)
    a = F.square(x)
    y = add(F.square(a), F.square(a))
    y.backward()
    print(y.data)
    print(x.grad)
    return


if __name__ == "__main__":
    main()
