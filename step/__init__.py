from step01 import main as step01  # noqa
from step02 import main as step02  # noqa

if __name__ == "__main__":
    for f in list(globals()):
        if "step" not in f:
            continue
        print(f"{'*' * 20}  {f} {'*' * 20}")
        if "step0" in f:
            globals()[f]()
            pass
        pass
    pass
