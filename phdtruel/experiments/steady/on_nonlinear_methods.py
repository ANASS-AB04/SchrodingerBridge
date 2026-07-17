import logging
from functools import partial
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotloom import Loom

from phdtruel import visualisations
from phdtruel.errors import (
    evaluate_interpolation_on_parameters,
    compute_error_map,
    get_error,
    mesure_errors,
)
from phdtruel.experiments.experiment_compagnon import (
    ExperimentCompagnon,
    compute_error_table,
    minimize,
)
from phdtruel.experiments.steady.on_linear_methods import pod_projection_experiment
from phdtruel.fields import FieldOfInterest
from phdtruel.fields.parameters import (
    generate_evaluation_params_in_rectangle,
    ParameterSet,
)
from phdtruel.interpolations.interpolations import (
    FieldInterpolator,
    TwoFieldsInterpolator,
)
from phdtruel.visualisations.errors import plot_error_map, plot_errors
from phdtruel.visualisations.experiment import plot_result
from phdtruel.visualisations.interpolations import (
    plot_interpolation_map,
)

logger = logging.getLogger(__name__)


def two_fields_eval_experiment(
    exp: ExperimentCompagnon,
    method: TwoFieldsInterpolator,
):
    exp.add_result(
        method.name,
        method(exp.test_parameter),
        hook=plot_result,
    )
    do_s_influence_study(exp, method)


def evaluate_method_experiment(
    exp: ExperimentCompagnon,
    method: FieldInterpolator,
    do_optimal_l2: bool = False,
    do_optimal_w2: bool = False,
):
    logger.info(f"{method.name}")

    field = method(exp.test_parameter, exp.test_mesh)
    exp.add_result(method.name, field, hook=plot_result)

    method_with_mesh = partial(
        method,
        mesh=exp.test_mesh,
        abs_tol=None,
    )

    if do_optimal_l2:
        field_opt_l2 = minimize(method_with_mesh, exp, "l_2")
        exp.add_result(f"{method.name} opt $l_2$", field_opt_l2, hook=plot_result)

    if do_optimal_w2:
        field = minimize(method_with_mesh, exp, "w_2")
        exp.add_result(f"{method.name} opt $w_2$", field, hook=plot_result)


def enrichment_experiments(
    exp: ExperimentCompagnon,
    method: FieldInterpolator,
    map_shape: tuple[int, int],
    field_opt_l2: FieldOfInterest,
):
    generated_fields = map_experiments(exp, method, map_shape)
    pod_projection_experiment(
        exp.fields + generated_fields,
        exp,
        name=f"{len(generated_fields)} {method.name} generated + {len(exp.fields)} originals",
    )
    pod_projection_experiment(
        exp.fields + generated_fields + [field_opt_l2],
        exp,
        name=f"{len(generated_fields)} {method.name} generated + {method.name} optimal $l_2$ + {len(exp.fields)} originals",
    )
    pod_projection_experiment(
        exp.fields + [field_opt_l2],
        exp,
        name=f"{method.name} optimal $l_2$ + {len(exp.fields)} originals",
    )


def map_experiments(
    exp: ExperimentCompagnon, method: FieldInterpolator, map_shape: tuple[int, int]
):
    eval_params = generate_evaluation_params_in_rectangle(map_shape)
    got_generated_fields = evaluate_interpolation_on_parameters(
        method=partial(method, mesh=exp.test_mesh), parameters=eval_params
    )

    fig = plot_interpolation_map(got_generated_fields)
    fig.suptitle(f"{method.name}")
    if exp.savefig is not None:
        exp.savefig(fig, f"{method.name}_map.png")
    fig = plot_interpolation_map(
        got_generated_fields,
        cmap="RdBu_r",
        reference_field_to_subtract=exp.target_field,
        truncation=exp.truncation,
    )
    fig.suptitle(f"{method.name}")
    if exp.savefig is not None:
        exp.savefig(fig, f"{method.name}_err_map.png")

    for err in ("l_2",):  # "w_2"):
        logger.info(f"Error map {err}")
        err_map = compute_error_map(
            got_generated_fields,
            reference_field=exp.target_field,
            points=exp.test_points,
            error_metric=get_error(
                name=err, mesh=exp.test_mesh, truncation=exp.truncation
            ),
        )
        fig, ax = plot_error_map(
            err_map.reshape(map_shape),
            err_name=err,
        )
        if exp.savefig is not None:
            exp.savefig(fig, f"{method.name}_{err}_map.png")
    return got_generated_fields


def plot_error_bars(exp, name=""):
    test_case = "nascar"
    df_of_errors = compute_error_table(exp, test_case)
    apply_colors = {
        "orange": [
            "Aligned wake stripes",
            "GOT Bary with",
            "GOT Bary rbf with",
            "GOT Grid opt",
            "GOT Bary rbf opt",
            "GOT Bary opt",
        ],
        "red": ["Projection in POD"],
    }
    legend = {
        "blue": "Direct prediction (without apriori)",
        "orange": "Direct prediction (with apriori)",
        "red": "Projection (with apriori)",
    }
    fig, ax = plot_errors(
        df_of_errors,
        test_case=test_case,
        error_norm="l_2",
        apply_colors=apply_colors,
        legend=legend,
    )
    exp.savefig(fig, f"error_bars_l2{name}.png")
    fig, ax = plot_errors(
        df_of_errors,
        test_case=test_case,
        error_norm="w_2",
        apply_colors=apply_colors,
        legend=legend,
    )
    exp.savefig(fig, f"error_bars_w2{name}.png")


def do_s_influence_study(exp: ExperimentCompagnon, method: TwoFieldsInterpolator):
    points = exp.test_points
    mesh = exp.test_mesh
    target_field = exp.target_field
    truncation = exp.truncation

    s_values = np.linspace(0, 1, 60)
    all_errs_cdi = []
    all_errs_linear = []
    all_interpolations = []
    for s in s_values:
        values_cdi = method.interpolate(points=points, s=float(s))
        values_linear = (1 - s) * exp.fields[0](points) + s * exp.fields[1](points)
        all_interpolations.append(values_cdi)
        errors_cdi = mesure_errors(
            target_field(points), values_cdi, mesh, truncation=truncation
        )
        errors_linear = mesure_errors(
            target_field(points), values_linear, mesh, truncation=truncation
        )
        all_errs_cdi.append(errors_cdi)
        all_errs_linear.append(errors_linear)
    import pandas as pd

    all_errs_cdi = pd.DataFrame(all_errs_cdi).drop(columns="truncated")
    all_errs_linear = pd.DataFrame(all_errs_linear).drop(columns="truncated")

    err = "l_2"
    min_row = all_errs_cdi.loc[all_errs_cdi[err].idxmin()]
    min_idx = int(all_errs_cdi[err].argmin())
    min_s = s_values[min_idx]
    min_err = min_row[err]

    fig, ax = plt.subplots()
    # for col in all_errs_cdi.columns:
    ax.set_title(f"${err}$ error between interpolation and reference")
    ax.plot(s_values, all_errs_cdi[err], "b-", label="CDI")
    ax.plot(s_values, all_errs_linear[err], "r--", label="Linear")
    ax.scatter([min_s], [min_err], color="red")
    ax.set_xlabel("$s$")
    ax.set_ylabel("Error value")
    ax.legend()
    ax.grid()
    if exp.savefig is not None:
        exp.savefig(fig, f"error_evolution {method.name}.png")

    plot_folder = visualisations.get_plot_subfolder(exp.plot_folder_name)
    mp4_path = plot_folder / f"interpolations {method.name}.mp4"
    gif_path = plot_folder / f"interpolations {method.name}.gif"

    def _record_animation(path: str | Path) -> None:
        with Loom(path, fps=9, overwrite=True) as loom:
            for s, values_cdi in zip(s_values, all_interpolations):
                fig, axes = visualisations.subplots(1, 2)
                ax_interp, ax_ref = axes
                visualisations.contourf(
                    fig,
                    ax_interp,
                    FieldOfInterest(ParameterSet(s=s), mesh, values_cdi),
                    norm=exp.norm,
                )
                ax_interp.set_title(f"Interpolation $s={s:.2f}$")

                visualisations.contourf(fig, ax_ref, target_field, norm=exp.norm)
                ax_ref.set_title(f"Reference {exp.test_parameter}")
                fig.tight_layout()
                loom.save_frame(fig)

    _record_animation(mp4_path)
    _record_animation(gif_path)
