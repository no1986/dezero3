from step01 import main as step01  # noqa
from step02 import main as step02  # noqa
from step03 import main as step03  # noqa
from step06 import main as step06  # noqa
from step07 import main as step07  # noqa
from step08 import main as step08  # noqa
from step09 import main as step09  # noqa
from step13 import main as step13  # noqa
from step14 import main as step14  # noqa
from step16 import main as step16  # noqa
from step18 import main as step18  # noqa
from step19 import main as step19  # noqa
from step20 import main as step20  # noqa
from step21 import main as step21  # noqa
from step22 import main as step22  # noqa
from step26 import main as step26  # noqa

if __name__ == "__main__":
    for f in list(globals()):
        if "step" in f:
            print(f"{'*' * 20}  {f} {'*' * 20}")
            globals()[f]()
            pass
        pass
    pass
