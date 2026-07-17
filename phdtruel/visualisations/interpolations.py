import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from typing import cast

from phdtruel import visualisations
from phdtruel.experiments.experiment_compagnon import ExperimentCompagnon
from phdtruel.fields import FieldOfInterest
from phdtruel.fields.parameters import ParameterNormalizer
from phdtruel.visualisations.errors import insert_line_breaks
from phdtruel.visualisations.fields import centered_norm
from phdtruel.visualisations.figures import auto_plot_shape
from phdtruel.visualisations.gridspecs import get_fig_gridspec


def minimum_distance(points: np.ndarray) -> float:
    """
    Find the minimum distance between a set of points.

    Parameters:
    points (ndarray): An array of shape (n_points, n_dim) representing the points.

    Returns:
    float: The minimum distance between any two points.
    """
    n_points = points.shape[0]
    min_dist = np.inf  # Start with infinity as the minimum distance

    for i in range(n_points - 1):
        for j in range(i + 1, n_points):
            # Compute Euclidean distance between points[i] and points[j]
            dist = np.linalg.norm(points[i] - points[j])
            if dist < min_dist:
                min_dist = dist

    return float(min_dist)


def plot_interpolation_map(
    field_map: list[FieldOfInterest],
    reference_field_to_subtract: FieldOfInterest | None = None,
    **kwargs,
):
    if reference_field_to_subtract is not None:
        pre_title = "Error"
        norm = "centered_zero"
    else:
        pre_title = "Interpolation"
        norm = "centered_mean"

    parameters = [field.parameter for field in field_map]

    normalized_parameters = ParameterNormalizer.fit_predict(parameters)
    normalized_parameters[:, 1] = 1.0 - normalized_parameters[:, 1]
    normalized_parameters[:, [1, 0]] = normalized_parameters[:, [0, 1]]

    min_dist = minimum_distance(normalized_parameters)
    plot_size = min_dist / (min_dist + 1.0)

    margin_ratio = 0.85
    w, h = plot_size * margin_ratio, plot_size * margin_ratio  # Shape of each subplot
    normalized_parameters = normalized_parameters * (1.0 - plot_size)

    fig = plt.figure()
    inches_per_plot = 20.0
    fig.set_size_inches((inches_per_plot, inches_per_plot))
    for k, point in enumerate(normalized_parameters):
        position_and_size = cast(
            tuple[float, float, float, float],
            (float(point[1]), float(point[0]), float(w), float(h)),
        )
        ax = fig.add_axes(position_and_size)
        ax.set_aspect("auto", anchor="NW")
        field = field_map[k]
        ax.set_title(f"{pre_title} {field.parameter}")

        if reference_field_to_subtract is not None:
            field = field - reference_field_to_subtract

        visualisations.contourf(fig, ax, field, norm=norm, **kwargs)
        ax.set_position(position_and_size)

    return fig


def plot_linear_interpolation_and_snapshots(
    exp: ExperimentCompagnon, interpolated_foi: FieldOfInterest
):
    f0 = exp.fields[0]
    f1 = exp.fields[1]
    target = exp.target_field
    points = exp.test_points

    fig, gs = get_fig_gridspec(2, 3)
    fig.set_size_inches(16, 8)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
    ]
    norm = mcolors.CenteredNorm(vcenter=np.mean(target(points)))
    axes[0].set_title(f"Solution for {f0.parameter}")
    axes[1].set_title(f"Reference solution for {target.parameter}")
    axes[2].set_title(f"Solution for {f1.parameter}")
    axes[3].set_title(f"Projeted u for {target.parameter} on POD basis")
    visualisations.contourf(fig, axes[0], f0, norm=norm, truncation=exp.truncation)
    visualisations.contourf(fig, axes[1], target, norm=norm, truncation=exp.truncation)
    visualisations.contourf(fig, axes[2], f1, norm=norm, truncation=exp.truncation)
    visualisations.contourf(
        fig,
        axes[3],
        interpolated_foi,
        truncation=exp.truncation,
    )
    for ax in axes:
        ax.grid(False)
    fig.tight_layout()
    if exp.savefig is not None:
        exp.savefig(fig, "linear_interp_vs_snapshots.png", dpi=300)


def plot_many_fields(
    values_to_plot: dict[str, FieldOfInterest],
    truncation=None,
    norm=None,
    **kwargs,
):
    shape = auto_plot_shape(len(values_to_plot))
    if norm is None:
        if truncation is None:
            norm = centered_norm(next(iter(values_to_plot.values())))
        else:
            norm = centered_norm(
                next(iter(values_to_plot.values())), truncation=truncation
            )

    fig, axes = plt.subplots(*shape)
    fig.set_size_inches(4 * shape[1], 4 * shape[0])
    for k, (name, field) in enumerate(values_to_plot.items()):
        ax = axes.ravel()[k]
        ax.set_title(insert_line_breaks(name, 40))
        visualisations.contourf(
            fig, ax, field, norm=norm, truncation=truncation, **kwargs
        )
    fig.tight_layout()
    return fig, axes
