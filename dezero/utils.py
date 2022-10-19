from __future__ import annotations

import os

import imgcat
from graphviz import Digraph
from PIL import Image

from dezero import Function, Variable


def show(path: str) -> None:
    if os.getenv("DISPLAY") is None:
        imgcat.imgcat(open(path))
    else:
        img = Image.open(path)
        img.show()
        pass
    return


def plot_dot_graph(
    output: Variable, verbose: bool = True, to_file: str = "graph", format: str = "png"
) -> None:
    def add_func(f: Function) -> None:
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
            pass
        return

    def add_dot_v(v: Variable) -> None:
        name = "" if v.name is None else v.name
        if verbose:
            if v.name is not None:
                name += ": "
                pass
            if v.ndim == 0:
                name += "(0)"
            else:
                name += f"{str(v.shape)}"
                pass
            pass
        dg.node(str(id(v)), label=name, style="filled", color="orange")
        return

    def add_dot_f(f: Function) -> None:
        name = f.__class__.__name__
        dg.node(
            str(id(f)),
            label=name,
            shape="box",
            style="filled",
            color="lightblue",
        )
        for x in f.inputs:
            dg.edge(str(id(x)), str(id(f)))
            pass
        for y in f.outputs:
            dg.edge(str(id(f)), str(id(y())))
            pass
        return

    dg = Digraph(format=format)
    dg.attr("node", fontsize="24")
    funcs = []
    seen_set = set()

    add_dot_v(output)
    add_func(output.creator)
    while funcs:
        func = funcs.pop()
        add_dot_f(func)
        for x in func.inputs:
            add_dot_v(x)

            if x.creator is not None:
                add_func(x.creator)
                pass
            pass
        pass

    pwd = os.path.dirname(os.path.abspath(__file__))
    path = f"{pwd}/../graph"
    os.makedirs(path, exist_ok=True)
    dg.render(f"{path}/{to_file}", view=False)

    show(f"{path}/{to_file}.{format}")
    return
