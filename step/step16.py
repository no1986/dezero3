import numpy as np  # noqa

import dezero as dz  # noqa
from dezero import functions as F  # noqa


def main():
    x = dz.Variable(2.0)
    a = F.square(x)
    y = dz.add(F.square(a), F.square(a))
    y.backward()
    print(y.data)
    print(x.grad)
    return


if __name__ == "__main__":
    main()
