from step01 import main as step01  # noqa
from step02 import main as step02  # noqa
from step03 import main as step03  # noqa
from step06 import main as step06  # noqa
from step07 import main as step07  # noqa
from step08 import main as step08  # noqa

if __name__ == "__main__":
    for f in list(globals()):
        if "step" in f:
            print(f"{'*' * 20}  {f} {'*' * 20}")
            globals()[f]()
            pass
        pass
    pass
