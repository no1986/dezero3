import numpy as np  # noqa

from dezero import Variable  # noqa
from dezero import functions as F # noqa


def main():
    x = Variable(0.5)
    y = F.square(F.exp(F.square(x)))
    y.backward()
    print(x.grad)
    print(type(y.data))

    x = Variable(np.array(1.0))
    x = Variable([1, 2, 3])
    x = Variable(0.5)
    x = Variable(None)
    try:
        x = Variable("a")
    except Exception as e:
        print(e)
        pass
    return


if __name__ == "__main__":
    main()
