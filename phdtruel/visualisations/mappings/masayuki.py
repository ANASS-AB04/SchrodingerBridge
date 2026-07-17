import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm

from phdtruel import visualisations
from phdtruel.fields import FieldOfInterest
from phdtruel.fields.parameters import ParameterSet
from phdtruel.interpolations.interpolations import DirectInterpolator
from phdtruel.mappings.masayuki import MasayukiMapping
from scripts.mapping_two_fileds_studies.plots import PLOT_FOLDER


logger = logging.getLogger(__name__)


def plot_interpolation_and_pp(
    fields: list[FieldOfInterest],
    masayuki_mapping: MasayukiMapping,
    method_name: str,
) -> tuple:
    f0, f_ref, f1 = fields
    mesh = f_ref.mesh

    all_principal_points_0 = masayuki_mapping._principal_points_source
    all_principal_points_1 = masayuki_mapping._principal_points_target

    inverse_mapping = masayuki_mapping.get_mapping_function(inverse=True)
    mapping = masayuki_mapping.get_mapping_function()

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    ax = axes[0]
    ax.set_title("transformed 0 and principal points \n + contour of 0 and 1")

    # Use item() to convert numpy scalars to Python float
    f1_mean = np.mean(f1.values).item()
    f1_vwidth = (f1.values.max() - np.mean(f1.values)).item()
    norm = CenteredNorm(vcenter=f1_mean, halfrange=f1_vwidth)
    visualisations.contourf(
        fig,
        ax,
        FieldOfInterest(ParameterSet(), mesh, f0(inverse_mapping(mesh.points, 1.0))),
        norm=norm,
    )
    visualisations.contour(fig, ax, f0, cmap="Blues", linewidths=0.5, norm=norm)
    visualisations.contour(fig, ax, f1, cmap="Reds", linewidths=0.5, norm=norm)
    ax.scatter(
        all_principal_points_0[:, 0], all_principal_points_0[:, 1], c="b", marker="o"
    )
    ax.scatter(
        all_principal_points_1[:, 0], all_principal_points_1[:, 1], c="r", marker="x"
    )
    for s in np.linspace(0, 1, 20):
        displaced_0 = mapping(all_principal_points_0, float(s))
        ax.scatter(displaced_0[:, 0], displaced_0[:, 1], c="green", marker="+")
    ax = axes[1]
    visualisations.contourf(fig, ax, f1, norm=norm)
    ax.set_title("field 1")
    fig.suptitle(f"{f0.name} fields")

    try:
        bd_p = masayuki_mapping._border_points
        axes[0].plot(bd_p[:, 0], bd_p[:, 1], "xk")
        axes[1].plot(bd_p[:, 0], bd_p[:, 1], "xk")
    except AttributeError:
        logger.debug(f"no border points available for {f0.name}")

    visualisations.savefig(fig, f"{method_name}_moved_0_1_and_pp.png")
    return fig, axes


def plot_principal_points_and_interpolations_both_orders(
    fields: list[FieldOfInterest],
    direct_masayuki_method: DirectInterpolator,
    plot_folder: str = PLOT_FOLDER,
) -> None:
    f0, f_ref, f1 = fields
    mappings = direct_masayuki_method._mappings

    if mappings is None:
        logger.warning("No mappings available in DirectInterpolator")
        return

    m0 = mappings[0]
    m1 = mappings[1]

    # Type narrow to MasayukiMapping
    if not isinstance(m0, MasayukiMapping):
        logger.warning("Expected MasayukiMapping but got %s", type(m0).__name__)
        return
    if not isinstance(m1, MasayukiMapping):
        logger.warning("Expected MasayukiMapping but got %s", type(m1).__name__)
        return

    plot_interpolation_and_pp(
        [f0, f_ref, f1],
        m0,
        direct_masayuki_method.name,
    )
    plot_interpolation_and_pp(
        [f1, f_ref, f0],
        m1,
        direct_masayuki_method.name,
    )
