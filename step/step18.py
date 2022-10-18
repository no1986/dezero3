import numpy as np  # noqa

import dezero as dz  # noqa
from dezero import functions as F  # noqa


def main():
    x0 = dz.Variable(1.0)
    x1 = dz.Variable(1.0)
    t = dz.add(x0, x1)
    y = dz.add(x0, t)
    y.backward()

    print(y.grad, t.grad)
    print(x0.grad, x1.grad)

    with dz.no_grad():
        x = dz.Variable(2.0)
        y = F.square(x)
        try:
            y.backward()
        except Exception as e:
            print(e)
            pass
        pass
    return


if __name__ == "__main__":
    main()
