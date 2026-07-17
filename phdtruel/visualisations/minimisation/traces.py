from collections.abc import Mapping

import numpy as np

from phdtruel import visualisations


def _to_trace_dict(traces: Mapping | object) -> dict[str, np.ndarray]:
    if isinstance(traces, Mapping):
        return {str(name): np.asarray(values) for name, values in traces.items()}

    if hasattr(traces, "to_dict"):
        trace_dict = traces.to_dict(orient="list")
        return {str(name): np.asarray(values) for name, values in trace_dict.items()}

    raise TypeError("traces must be a mapping or a dataframe-like object.")


def plot_solver_trace_panels(
    traces: Mapping | object,
    *,
    fig=None,
    axes=None,
    solver_step: int | None = None,
    gradient_key: str = "gradient_norms",
    yscale: str = "log",
) -> tuple:
    traces_dict = _to_trace_dict(traces)
    if solver_step is None:
        solver_step = (
            max((len(values) for values in traces_dict.values()), default=1) - 1
        )

    if fig is None or axes is None:
        fig, axes = visualisations.subplots(1, 3, figsize=(18, 5))

    flat_axes = np.asarray(axes).ravel()
    if flat_axes.size < 3:
        raise ValueError("plot_solver_trace_panels requires at least three axes.")

    cost_ax = flat_axes[0]
    for key, cost in traces_dict.items():
        if key == gradient_key or key in ("ls_steps", "iter_time"):
            continue
        cost_ax.plot(cost, label=key)
    cost_ax.set_title("Cost function evolution")
    cost_ax.set_xlabel("Iteration")
    cost_ax.set_ylabel("Cost")
    cost_ax.grid(True)
    cost_ax.set_yscale(yscale)
    cost_ax.legend(fontsize=8)

    gradients = traces_dict.get(gradient_key, None)

    grad_ax = flat_axes[1]
    if gradients is None:
        grad_ax.set_title("Gradient/Value norms not available")
        grad_ax.text(
            0.5,
            0.5,
            f"The gradient was not found as {gradient_key!r} in the traces.",
            ha="center",
            va="center",
            transform=grad_ax.transAxes,
        )
        grad_ax.grid(True)
    else:
        grad_ax.plot(gradients, label="Gradient norm", color="k")
        grad_ax.set_title("Gradient norm evolution")
        grad_ax.set_xlabel("Iteration")
        grad_ax.set_ylabel("Gradient norm")
        grad_ax.grid(True)
        grad_ax.set_yscale(yscale)

    # Third axis: time per ls_step and ls_iteration
    time_ax = flat_axes[2]
    itertime = traces_dict.get("iter_time", None)
    ls_steps = traces_dict.get("ls_steps", None)

    if itertime is not None and ls_steps is not None:
        # Compute time per ls_step, avoid division by zero
        time_per_step = np.divide(
            itertime,
            ls_steps,
            where=(ls_steps != 0),
            out=np.zeros_like(itertime, dtype=float),
        )
        time_ax.plot(
            time_per_step, label="time/ls_step", color="blue", marker="o", markersize=3
        )
        time_ax.set_ylabel("time/ls_step", color="blue")
        time_ax.set_yscale("log")
        time_ax.tick_params(axis="y", labelcolor="blue")

    ls_steps = traces_dict.get("ls_steps", None)
    if ls_steps is not None:
        twin_ax = time_ax.twinx()
        twin_ax.plot(ls_steps, label="ls_steps", color="red", marker="s", markersize=3)
        twin_ax.set_ylabel("ls_steps", color="red")
        twin_ax.set_yscale("log")
        twin_ax.tick_params(axis="y", labelcolor="red")

    time_ax.set_title("Time per line search step and number of ls per iteration")
    time_ax.set_xlabel("Iteration")
    time_ax.grid(True, alpha=0.3)

    return fig, (cost_ax, grad_ax, time_ax)
