import numpy as np  # noqa

import dezero as dz # noqa
from dezero import functions as F # noqa


def main():
    x = dz.Variable(0.5)
    y = F.square(F.exp(F.square(x)))
    y.backward()
    print(x.grad)
    print(type(y.data))

    x = dz.Variable(np.array(1.0))
    x = dz.Variable([1, 2, 3])
    x = dz.Variable(0.5)
    x = dz.Variable(None)
    try:
        x = dz.Variable("a")
    except Exception as e:
        print(e)
        pass
    return


if __name__ == "__main__":
    main()
