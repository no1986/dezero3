import numpy as np  # noqa

from dezero import Variable  # noqa
from dezero import functions as F # noqa


def main():
    x = Variable(np.array(10.0))
    f = F.Square()
    y = f(x)
    print(type(y))
    print(y.data)
    return


if __name__ == "__main__":
    main()
