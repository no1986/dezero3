import numpy as np  # noqa

import dezero as dz # noqa
from dezero import functions as F # noqa


def main():
    x = dz.Variable(np.array(10.0))
    f = F.Square()
    y = f(x)
    print(type(y))
    print(y.data)
    return


if __name__ == "__main__":
    main()
