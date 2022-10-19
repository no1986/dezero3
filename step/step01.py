import numpy as np  # noqa

from dezero import Variable  # noqa


def main():
    data = np.array(1.0)
    x = Variable(data)
    print(x.data)
    return


if __name__ == "__main__":
    main()
