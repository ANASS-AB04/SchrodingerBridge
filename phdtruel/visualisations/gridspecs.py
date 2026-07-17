import numpy as np
from matplotlib import pyplot as plt


def get_fig_gridspec(nrows: int, ncols: int) -> tuple[plt.Figure, plt.GridSpec]:
    fig = plt.figure(figsize=(ncols * 5, ncols * 5))
    gridspec = plt.GridSpec(nrows, ncols)
    return fig, gridspec


def bottom_corner_triangle_spec():
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    gridspec = plt.GridSpec(3, 3)
    axes = np.array(
        [
            fig.add_subplot(gridspec[0, 0]),
            fig.add_subplot(gridspec[2, 0]),
            fig.add_subplot(gridspec[2, 2]),
            fig.add_subplot(gridspec[1, 1]),
        ]
    )
    return fig, axes, gridspec


def isosceles_triangle_gridspec():
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    gridspec = plt.GridSpec(3, 3)
    axes = np.array(
        [
            fig.add_subplot(gridspec[0, 1]),
            fig.add_subplot(gridspec[2, 0]),
            fig.add_subplot(gridspec[2, 2]),
            fig.add_subplot(gridspec[1, 1]),
        ]
    )
    return fig, axes, gridspec


def five_fields_gridspec():
    fig = plt.figure()
    fig.set_size_inches(10, 6)
    gridspec = plt.GridSpec(2, 3)
    axes = np.array(
        [
            fig.add_subplot(gridspec[0, 0]),
            fig.add_subplot(gridspec[0, 2]),
            fig.add_subplot(gridspec[1, 0]),
            fig.add_subplot(gridspec[1, 2]),
            fig.add_subplot(gridspec[:, 1]),
        ]
    )
    return fig, axes, gridspec


def six_fields_gridspec():
    fig = plt.figure()
    fig.set_size_inches(9, 7)
    gridspec = plt.GridSpec(2, 3)
    axes = np.array(
        [
            fig.add_subplot(gridspec[0, 0]),
            fig.add_subplot(gridspec[0, 2]),
            fig.add_subplot(gridspec[1, 0]),
            fig.add_subplot(gridspec[1, 2]),
            fig.add_subplot(gridspec[0, 1]),
            fig.add_subplot(gridspec[1, 1]),
        ]
    )
    return fig, axes, gridspec


def ref_center_3_3_spec():
    fig = plt.figure()
    fig.set_size_inches(9, 9)
    gridspec = plt.GridSpec(3, 3)
    axes = np.array(
        [
            fig.add_subplot(gridspec[1, 1]),
        ]
    )
    return fig, axes, gridspec
