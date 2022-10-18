import numpy as np  # noqa

import dezero as dz # noqa
from dezero import functions as F # noqa


def main():
    A = F.Square()
    B = F.Exp()
    C = F.Square()

    x = dz.Variable(np.array(0.5))

    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
    return


if __name__ == "__main__":
    main()
