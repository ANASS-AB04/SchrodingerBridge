import numpy as np
from matplotlib import pyplot as plt

from phdtruel.experiments.experiment_compagnon import ExperimentCompagnon
from phdtruel.visualisations.figures import auto_plot_shape


def plot_pod_modes(
    pod_basis: np.ndarray,
    exp: ExperimentCompagnon,
):
    n_modes = pod_basis.shape[1]
    figure_shape = auto_plot_shape(n_modes)

    fig, axes = plt.subplots(figure_shape[0], figure_shape[1])
    fig.set_size_inches((5 * figure_shape[1], 5 * figure_shape[0]))
    fig.suptitle("POD basis vectors")
    for k in range(n_modes):
        shape = exp.test_mesh.shape
        ax = axes.flatten()[k]
        ax.set_title(f"POD vector n°{k}")
        ax.contourf(
            exp.x,
            exp.y,
            pod_basis[:, k].reshape(shape[1], shape[0]),
            norm=exp.norm,
        )
    fig.tight_layout()
    return fig, axes


def plot_singular_values(s):
    fig, ax = plt.subplots()
    ax.set_title("Singular values")
    ax.set_yscale("log")
    # ax.set_xticks(list(range(0, pod.singular_values.shape[0], pod.singular_values.shape[0] // 20)))
    ax.plot(s, ls="--", marker="o")
    # ax.plot([0, s.shape[0]], [s[s_pod.size - 1]] * 2)
    # ax.set_title("Décroissance des valeurs singulières")
    ax.set_xlabel("N de valeur singulière")
    ax.set_ylabel("Valeur singulière")
    ax.grid()
    return fig, ax
