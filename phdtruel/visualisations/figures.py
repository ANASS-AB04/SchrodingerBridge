import logging

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from phdtruel import printable_path

logger = logging.getLogger(__name__)


def subplots(
    n_rows: int = 1,
    n_cols: int = 1,
    figsize: tuple[float, float] | None = None,
    axsize: tuple[float, float] | None = None,
):
    """Create a figure and a set of subplots.

    If axsize is provided, figsize is computed as (n_cols * axsize[0], n_rows * axsize[1]) (width, height) per axis.

    """
    if figsize is None:
        if axsize is None:
            axsize = (7.0, 7.0)  # default axis size
        figsize = (n_cols * axsize[0], n_rows * axsize[1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    return fig, axes


def get_linestyles(N=2):
    linestyles = []
    for k in range(N):
        space = 2 ** (k + 1)
        dotted = (0, (1, space))
        dashed = (0, (5, 5))
        dashdotted = (0, (3, 5, 1, 5))
        dashdotdotted = (0, (3, space, 1, space, 1, space))
        linestyles.append(dotted)
        linestyles.append(dashed)
        linestyles.append(dashdotted)
        linestyles.append(dashdotdotted)
    return linestyles


def savefig(fig, name, subfolder: str | None = None, dpi: int = 500, **kwargs):
    from phdtruel.visualisations import get_plot_subfolder

    plot_folder = get_plot_subfolder(subfolder)
    plot_folder.mkdir(exist_ok=True, parents=True)
    filepath = plot_folder / name
    fig.savefig(filepath, dpi=dpi, **kwargs)
    logger.info(f"Saving {printable_path(filepath)}")


def cmap_as_cbar_to_ax(fig, ax, cmap, label="", pad=0.05, **kwargs):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=pad)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cax, label=label, **kwargs)


def auto_plot_shape(n_plots: int):
    if n_plots <= 3:
        shape = (1, 3)
    elif n_plots == 4:
        shape = (2, 2)
    elif n_plots <= 6:
        shape = (2, 3)
    elif n_plots <= 9:
        shape = (3, 3)
    elif n_plots <= 12:
        shape = (3, 4)
    else:
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        shape = (n_rows, n_cols)
    return shape
