from typing import Any

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from phdtruel import visualisations
from phdtruel.fields import FieldOfInterest
from phdtruel.visualisations.figures import auto_plot_shape

OptimFFDMapping = Any


def plot_for_ffd_based_mapping(
    fields: list[FieldOfInterest],
    ffd_map: OptimFFDMapping,
    interpolation: FieldOfInterest | None = None,
    name="FFD",
):
    f0, f_ref, f1 = fields
    mesh = f_ref.mesh
    determinant = np.ones(mesh.shape)

    fig, axes = plot_determinant(mesh, determinant)
    visualisations.savefig(fig, f"{name}_det.svg")

    fig, ax = plot_control_points(ffd_map._ffd, mesh, determinant)
    visualisations.savefig(fig, f"{name}_ffd_control_points.svg")

    fig, axes = plot_ffd_traces(ffd_map._plot_data.traces, fig)
    visualisations.savefig(fig, f"{name}_norms.svg")


def plot_ffd_traces(traces, fig):
    traces = pd.DataFrame(traces.numeric_traces).set_index("step")
    shape = auto_plot_shape(len(traces.columns) + 2)
    fig, axes = visualisations.subplots(shape[0], shape[1])

    fig.suptitle("FFD traces")
    ax1, ax2 = axes.ravel()[0], axes.ravel()[1]
    for name, col in traces.items():
        ax1.plot(col, label=name)
        ax2.plot(col, label=name)
    ax1.grid()
    ax2.grid()
    ax1.legend()
    ax2.legend()
    ax1.set_title("Norm")
    ax1.set_ylim(0.0, 600.0)
    ax2.set_ylim(traces.min().min(), 1e4)
    ax2.set_title("Norm log")
    ax2.set_yscale("log")

    for ax, col in zip(axes.ravel()[2:], traces.items()):
        ax.plot(col[1])
        ax.set_title(col[0])
        ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel("Norm")
        ax.grid()
    fig.tight_layout()
    return fig, axes


def plot_control_points(ffd, mesh, determinant):
    fig, ax = plt.subplots()
    ax.set_title("$1/J(T^{-1})$ and displacement of control points")
    c2 = ax.contourf(
        mesh.x,
        mesh.y,
        1.0 / determinant.T,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(1.0, 0.0),
        levels=100,
    )
    fig.colorbar(c2, ax=ax)
    ax.plot(
        [
            ffd.control_points(deformed=False)[::2, 0],
            ffd.control_points(deformed=True)[::2, 0],
        ],
        [
            ffd.control_points(deformed=False)[::2, 1],
            ffd.control_points(deformed=True)[::2, 1],
        ],
        color="blue",
        linestyle="-",
        linewidth=0.5,
    )
    ax.plot(
        ffd.control_points(deformed=True)[::2, 0],
        ffd.control_points(deformed=True)[::2, 1],
        "rx",
        markersize=1.0,
    )
    ax.plot(
        ffd.control_points(deformed=False)[::2, 0],
        ffd.control_points(deformed=False)[::2, 1],
        "gx",
        markersize=1.0,
    )
    return fig, ax


def plot_determinant(mesh, determinant):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Det and inverse det")
    ax1.set_title("$J(T^{-1})$")
    ax2.set_title("$J(T)) = 1/det(J(T^{-1})$")
    c1 = ax1.contourf(
        mesh.x,
        mesh.y,
        determinant.T,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(1.0, 0.0),
        levels=100,
    )
    c2 = ax2.contourf(
        mesh.x,
        mesh.y,
        1.0 / determinant.T,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(1.0, 0.0),
        levels=100,
    )
    fig.colorbar(c1, ax=ax1)
    fig.colorbar(c2, ax=ax2)
    return fig, (ax1, ax2)
