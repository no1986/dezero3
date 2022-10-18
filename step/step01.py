import numpy as np  # noqa

import dezero as dz  # noqa


def main():
    data = np.array(1.0)
    x = dz.Variable(data)
    print(x.data)
    return


if __name__ == "__main__":
    main()
