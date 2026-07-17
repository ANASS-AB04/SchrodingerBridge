import logging
from functools import wraps
from typing import cast
import warnings

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from matplotloom import Loom

from phdtruel import visualisations
from phdtruel.experiments.experiment_compagnon import ExperimentCompagnon
from phdtruel.fields import DynamicOfInterest, FieldOfInterest, Gaussian
from phdtruel.fields.parameters import ParameterSet
from phdtruel.interpolations.avg_descriptors_cdi import AveragedDescriptorsCDI
from phdtruel.interpolations.cdi import CDIInterpolator
from phdtruel.visualisations.fields import plot_fields
from phdtruel.visualisations.figures import auto_plot_shape
from phdtruel.visualisations.gridspecs import five_fields_gridspec

logger = logging.getLogger(__name__)


def _deprecated(message: str):
    """Decorator to mark functions as deprecated."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                message,
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


@_deprecated("only used for multi parametric experiments")
def plot_setup(exp: ExperimentCompagnon, timestep: int | None = None):
    if not exp.is_steady:
        raise ValueError("Experiment is not steady.")
    num_fields = len(exp.fields)

    fields_dict = {f.parameter: f for f in exp.fields}
    if num_fields == 4:
        fig, axes, _ = five_fields_gridspec()
        if exp.norm is not None:
            fig, axes = plot_fields(
                fig,
                axes,
                fields_dict | {exp.test_parameter: exp.target_field},
                norm=exp.norm,
                cmap=exp.cmap,
                truncation=exp.truncation,
            )
        else:
            fig, axes = plot_fields(
                fig,
                axes,
                fields_dict | {exp.test_parameter: exp.target_field},
                cmap=exp.cmap,
                truncation=exp.truncation,
            )
        ax = axes[-1]
        ax.set_title(f"Target for {exp.test_parameter}")
    elif num_fields == 2:
        fig = plt.figure(figsize=(15, 6))
        gridspec = plt.GridSpec(1, 3)
        axes = np.array(
            [
                fig.add_subplot(gridspec[0]),
                fig.add_subplot(gridspec[2]),
                fig.add_subplot(gridspec[1]),
            ]
        )
        for (parameter, field), ax in zip(fields_dict.items(), axes[[0, 1]]):
            visualisations.contourf(
                fig,
                ax,
                field,
                norm=exp.norm,
                cmap=exp.cmap,
                truncation=exp.truncation,
            )
            ax.set_title(f"Parameter {parameter}")
        ax = axes[-1]
        visualisations.contourf(
            fig,
            ax,
            exp.target_field,
            norm=exp.norm,
            cmap=exp.cmap,
            truncation=exp.truncation,
        )
        ax.set_title(f"Target for {exp.test_parameter}")
    else:
        nrows, ncols = auto_plot_shape(num_fields + 1)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, ncols * 5))
        kwargs = exp.get_contourf_kwargs()
        if kwargs.get("norm") is None:
            kwargs.pop("norm", None)
        fig, axes = plot_fields(
            fig,
            axes.ravel(),
            fields_dict | {exp.test_parameter: exp.target_field},
            exp.test_points,
            **kwargs,
            truncation=exp.truncation,
        )
    fig.tight_layout()
    if exp.savefig is not None:
        exp.savefig(fig, "setup_fields.png")
    return fig, axes


def plot_gaussians(
    got: AveragedDescriptorsCDI | CDIInterpolator,
    exp: ExperimentCompagnon,
    other_gaussians_on_target: dict[str, Gaussian] | None = None,
    other_gaussians_on_fields: dict[str, list[Gaussian]] | None = None,
):
    fig, axes = plot_setup(exp)
    gaussians = got._descriptors

    for i, g in enumerate(gaussians):
        axes[i].contour(
            exp.x,
            exp.y,
            g.sample(exp.test_points).reshape(exp.test_mesh.shape).transpose(),
            colors="k",
            levels=5,
        )
    ax = axes[-1]
    ax.set_title(f"Target for ${exp.test_parameter}$")

    field = exp.target_field
    # if got._truncation is not None:
    #     field = exp.target_field.truncate(exp.truncation)

    target_gaussian = Gaussian.from_field(
        field.mesh.points, got._sensor_method(field).values
    )
    ax.contour(
        exp.x,
        exp.y,
        target_gaussian.sample(exp.test_points)
        .reshape(exp.test_mesh.shape)
        .transpose(),
        colors="k",
        levels=5,
        linestyles="solid",
    )

    patches = [
        mpatches.Patch(color="k", label="Gaussian Fit", linestyle="solid", fill=False),
    ]
    file_modif = ""
    linestyles = ["dotted", "dashed", "dashdot"]
    if other_gaussians_on_target is not None:
        for (text, gaussian), dashes in zip(
            other_gaussians_on_target.items(), linestyles
        ):
            ax.contour(
                exp.x,
                exp.y,
                gaussian.sample(exp.test_points)
                .reshape(exp.test_mesh.shape)
                .transpose(),
                colors="green",
                levels=5,
                linestyles=dashes,
            )
            patches.append(
                mpatches.Patch(color="green", label=text, linestyle=dashes, fill=False)
            )
            file_modif += f"+ {text}"

    if other_gaussians_on_fields is not None:
        for (text, other_gaussians), dashes in zip(
            other_gaussians_on_fields.items(), linestyles
        ):
            for k, g in enumerate(other_gaussians):
                axes[k].contour(
                    exp.x,
                    exp.y,
                    g.sample(exp.test_points).reshape(exp.test_mesh.shape).transpose(),
                    colors="green",
                    levels=5,
                    linestyles=dashes,
                )

    fig.legend(handles=patches, loc="outside upper right", ncols=2)
    fig.suptitle("Gaussian fitted over fields of interest")
    if exp.savefig is not None:
        exp.savefig(fig, f"fitted_gaussians{file_modif}.png")


def plot_result(exp: ExperimentCompagnon, name: str, field: FieldOfInterest):
    fig, ax = plt.subplots()
    visualisations.contourf(
        fig, ax, field, norm=exp.norm, cmap=exp.cmap, truncation=exp.truncation
    )
    ax.set_title(name)
    fig.tight_layout()
    if exp.plot_folder_name is not None:
        visualisations.savefig(
            fig, f"interpolation_{name}.png", subfolder=exp.plot_folder_name
        )


def comparison_dynamic_animation(
    reference_dynamic: DynamicOfInterest,
    dynamic: DynamicOfInterest,
    exp: ExperimentCompagnon,
    filename="out.mp4",
    add_gaussians: list[Gaussian] | None = None,
):
    if len(reference_dynamic) != len(dynamic):
        logger.warning("The two dynamics are not the same length.")
    n_fields = min(len(reference_dynamic), len(dynamic))

    if exp.plot_folder_name is not None:
        plot_folder = visualisations.get_plot_subfolder(exp.plot_folder_name)
        anim_path = plot_folder / filename
        with Loom(anim_path, fps=3, overwrite=True) as loom:
            for k in range(n_fields):
                field_ref = reference_dynamic[k]
                field = dynamic[k]

                t_ref = reference_dynamic.timesteps[k]
                t = dynamic.timesteps[k]

                fig, axes = visualisations.subplots(1, 2, figsize=(10, 5))
                ax_interp, ax_ref = axes
                fig.suptitle(f"Frame {k} — interpolation vs reference")

                visualisations.contourf(
                    fig,
                    ax_interp,
                    field,
                    norm=exp.norm,
                    truncation=exp.truncation,
                )
                ax_interp.set_title(f"Interpolation t={t:.2f}")
                if add_gaussians is not None:
                    g = add_gaussians[k]
                    ax_interp.contour(
                        exp.x,
                        exp.y,
                        g.sample(exp.test_points).reshape(exp.x.shape),
                        linewidths=0.5,
                        colors="k",
                    )

                visualisations.contourf(
                    fig,
                    ax_ref,
                    field_ref,
                    norm=exp.norm,
                    truncation=exp.truncation,
                )
                ax_ref.set_title(f"Reference t={t_ref:.2f}")

                fig.tight_layout()
                loom.save_frame(fig)


def plot_parameter_space(
    train_parameters: list[ParameterSet],
    test_parameters: list[ParameterSet] | ParameterSet,
    field_name: str,
):
    if isinstance(test_parameters, list):
        test_parameters_list = cast(list[ParameterSet], test_parameters)
    else:
        test_parameters_list = [test_parameters]

    import pandas as pd

    train_df = pd.DataFrame([param.as_dict() for param in train_parameters])
    test_df = pd.DataFrame([param.as_dict() for param in test_parameters_list])

    parameter_dim = train_df.shape[1]

    if parameter_dim == 1:
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 2)
        p = train_df.columns[0]
        ax.scatter(train_df[p], [0] * train_df.shape[0], label="train parameters")
        ax.scatter(
            test_df[p],
            [0] * test_df.shape[0],
            color="red",
            label="test parameters",
            marker="x",
        )
        ax.set_xlabel(p)
        ax.legend(loc="upper center", ncol=5)
        # To remove the y-axis
        # ax.spines['left'].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.tick_params(left=False)
        fig.tight_layout()
    elif parameter_dim == 2:
        fig, ax = plt.subplots()
        p1, p2 = train_df.columns
        ax.scatter(train_df[p1], train_df[p2], label="train parameters")
        ax.scatter(
            test_df[p1], test_df[p2], color="red", label="test parameters", marker="x"
        )
        ax.set(xlabel=p1, ylabel=p2)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5)
        # ax.set_aspect("equal")
        fig.tight_layout()
    elif parameter_dim == 3:
        p1, p2, p3 = train_df.columns

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(train_df[p1], train_df[p2], train_df[p3], label="train parameters")
        ax.scatter(
            test_df[p1],
            test_df[p2],
            test_df[p3],
            color="red",
            label="test parameters",
            marker="x",
        )
        ax.set(xlabel=p1, ylabel=p2, zlabel=p3)

    fig.suptitle(f"Parameter space on {field_name} fields")
    fig.tight_layout()
    return fig, ax
