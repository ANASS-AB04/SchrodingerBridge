import matplotlib.pyplot as plt
import numpy as np

from phdtruel.feature_identification import grid_on_field
from phdtruel.fields import FieldOfInterest
from phdtruel.mappings import GaussianOTMapping
from phdtruel.mappings.two_fields_mappings import (
    TwoFieldMappingModel,
    IdentityTwoFieldMappingModel,
)
from phdtruel.mappings.masayuki import MasayukiMapping
from phdtruel.visualisations.figures import cmap_as_cbar_to_ax
from phdtruel.visualisations.mappings.ffd_based import plot_for_ffd_based_mapping
from phdtruel.visualisations.mappings.ot_gaussian import plot_for_got_method
from phdtruel.visualisations.mappings.masayuki import plot_interpolation_and_pp


def do_plot_associated_to_mapping(
    fields: list[FieldOfInterest], mapping: TwoFieldMappingModel, name: str
):
    assert len(fields) == 3
    if isinstance(mapping, GaussianOTMapping):
        plot_for_got_method(fields, mapping)
    elif hasattr(mapping, "_ffd"):
        plot_for_ffd_based_mapping(fields, mapping, name=name)
    elif isinstance(mapping, IdentityTwoFieldMappingModel):
        pass
    elif isinstance(mapping, MasayukiMapping):
        plot_interpolation_and_pp(fields, mapping, method_name=name)


def mapping_quiver_visualisation(fig, axes, T, W, T_1, W_1, x, y):
    visu_points = grid_on_field(x, y, 20, 20)

    visu_T = T(visu_points)
    visu_W = W(visu_points)
    visu_T_1 = T_1(visu_points)
    visu_W_1 = W_1(visu_points)

    scale = (
        (visu_T - visu_points)[:, 0] ** 2 + (visu_T - visu_points)[:, 1] ** 2
    ).max() / 1.0
    xlim = (x.min(), x.max())
    ylim = (y.min(), y.max())

    for ax, label, points in zip(
        axes.ravel(),
        [r"$T$", r"$W$", r"$T^{-1}$", r"$W^{-1}$"],
        [visu_T, visu_W, visu_T_1, visu_W_1],
    ):
        ax.quiver(
            visu_points[:, 0],
            visu_points[:, 1],
            (points - visu_points)[:, 0],
            (points - visu_points)[:, 1],
            color="k",
            label=label,
            scale=scale,
            alpha=0.5,
        )
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.grid()
        ax.legend()
    return fig, axes


def plot_mapping_points_visualisation(
    x, y, T, W, T_1, W_1, feature_points_0, feature_points_1, n_steps=5
):
    fig, axes = plt.subplots(2, 2)
    cmap = plt.get_cmap("cividis_r")
    fig.set_size_inches(10, 8)

    visu_points = grid_on_field(x, y, 50, 50)
    visu_points_0 = np.vstack([visu_points, feature_points_0])
    visu_points_1 = np.vstack([visu_points, feature_points_1])

    for s in np.linspace(0, 1, n_steps):
        T_i = T(visu_points_0, s)
        W_i = W(visu_points_1, s)
        T_1_i = T_1(visu_points_1, s)
        W_1_i = W_1(visu_points_0, s)

        axes[0, 0].plot(
            T_i[:, 0],
            T_i[:, 1],
            label="Fitted contour",
            marker=".",
            linestyle="",
            markersize=2.0,
            color=cmap(s),
        )
        axes[0, 1].plot(
            W_i[:, 0],
            W_i[:, 1],
            label="Fitted contour",
            marker=".",
            linestyle="",
            markersize=2.0,
            color=cmap(s),
        )
        axes[1, 0].plot(
            T_1_i[:, 0],
            T_1_i[:, 1],
            label="Fitted contour",
            marker=".",
            linestyle="",
            markersize=2.0,
            color=cmap(s),
        )
        axes[1, 1].plot(
            W_1_i[:, 0],
            W_1_i[:, 1],
            label="Fitted contour",
            marker=".",
            linestyle="",
            markersize=2.0,
            color=cmap(s),
        )

    axes[0, 0].set_title(r"$\mathcal{T}$")
    axes[0, 1].set_title(r"$\mathcal{W}$")
    axes[1, 0].set_title(r"$\mathcal{T}^{-1}$")
    axes[1, 1].set_title(r"$\mathcal{W}^{-1}$")

    for ax in axes.ravel():
        ax.set_aspect("equal")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.grid()
    fig.suptitle("Interpolated mappings")

    cmap_as_cbar_to_ax(fig, axes[0, 0], cmap, label="$s$ value")
    cmap_as_cbar_to_ax(fig, axes[0, 1], cmap, label="$s$ value")
    cmap_as_cbar_to_ax(fig, axes[1, 0], cmap, label="$s$ value")
    cmap_as_cbar_to_ax(fig, axes[1, 1], cmap, label="$s$ value")

    fig.tight_layout()
    return fig, axes


def get_grid(step=2.0, xlim=(-10.0, 10.0), ylim=(-10.0, 10.0)):
    x, y = np.mgrid[xlim[0] : xlim[1] : step, ylim[0] : ylim[1] : step]
    points = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])
    return points


def get_visualisation_points(
    cross_positions: list[np.ndarray],
    xlim=(-10.0, 10.0),
    ylim=(-10.0, 10.0),
    step=2.0,
    cross_scale=1.0,
    cross_n_points=10,
):
    def make_cross(scale=1.0, n_points=10):
        bar_h = np.vstack([np.linspace(0, 1, n_points) - 0.5, np.zeros(n_points)]).T
        bar_v = np.vstack([np.zeros(n_points), np.linspace(0, 1, n_points) - 0.5]).T

        cross = np.vstack((bar_h, bar_v, np.array([[0, 0]])))
        cross = cross * scale
        return cross

    visu_points = get_grid(step=step, xlim=xlim, ylim=ylim)
    for mu in cross_positions:
        cross = make_cross(scale=cross_scale, n_points=cross_n_points)
        cross = cross + mu.reshape(1, 2)
        visu_points = np.vstack((visu_points, cross))

    return visu_points
