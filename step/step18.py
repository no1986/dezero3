import numpy as np  # noqa

from dezero import Variable, no_grad  # noqa
from dezero import functions as F  # noqa
from dezero.core import add  # noqa


def main():
    x0 = Variable(1.0)
    x1 = Variable(1.0)
    t = add(x0, x1)
    y = add(x0, t)
    y.backward()

    print(y.grad, t.grad)
    print(x0.grad, x1.grad)

    with no_grad():
        x = Variable(2.0)
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
