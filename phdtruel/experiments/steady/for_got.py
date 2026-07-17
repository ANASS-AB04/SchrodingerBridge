import numpy as np

from phdtruel.errors import get_error, optimize_parameters
from phdtruel.experiments.experiment_compagnon import ExperimentCompagnon
from phdtruel.fields import Gaussian
from phdtruel.interpolations.avg_descriptors_cdi import AveragedDescriptorsCDI
from phdtruel.sensors.sensors import localize_field_fit_gaussian
from phdtruel.visualisations.experiment import plot_gaussians, plot_result


def avg_gaussian_experiments(exp: ExperimentCompagnon, got: AveragedDescriptorsCDI):
    try:
        mean_gaussian_without_apriori: Gaussian = got._last_mean_gaussian  # type: ignore[attr-defined, assignment]
    except AttributeError:
        raise AttributeError(f"Please run {got} method at least once.")
    plot_gaussians(
        got, exp, other_gaussians_on_target={"mean": mean_gaussian_without_apriori}
    )
    override_gaussian = localize_field_fit_gaussian(
        exp.target_field, truncation=exp.truncation
    )
    field = got(
        exp.test_parameter,
        exp.test_mesh,
        _override_mean_descriptor=override_gaussian,
    )
    exp.add_result(
        f"{got.name} with gaussian fitted on target", field, hook=plot_result
    )
    _, best_gauss_l2 = optimize_target_gauss(
        exp, got, error="l_2", start_gauss=mean_gaussian_without_apriori
    )
    _, best_gauss_w2 = optimize_target_gauss(
        exp, got, error="w_2", start_gauss=mean_gaussian_without_apriori
    )
    plot_gaussians(
        got,
        exp,
        other_gaussians_on_target={
            "mean": mean_gaussian_without_apriori,
            "opt $l_2$": best_gauss_l2,
            # "opt $w_2$": best_gauss_w2,
        },
    )
    _, all_best_gaussians = optimize_all_gaussians(
        exp,
        got,
        error="l_2",
        start_gaussians=got.descriptors + [mean_gaussian_without_apriori],
    )
    plot_gaussians(
        got,
        exp,
        other_gaussians_on_target={
            "mean": mean_gaussian_without_apriori,
            "opt $l_2$": best_gauss_l2,
            "all opt $l_2$": all_best_gaussians[-1],
        },
        other_gaussians_on_fields={
            "all opt $l_2$": all_best_gaussians[:-1],
        },
    )


def optimize_all_gaussians(
    exp: ExperimentCompagnon,
    got: AveragedDescriptorsCDI,
    error: str,
    start_gaussians: list[Gaussian],
):
    def extract_gaussians(m_array, c_array) -> list[Gaussian]:
        n_gauss = 5
        means = m_array.reshape(n_gauss, 2)
        covs = c_array.reshape(n_gauss, 3)
        gaussians = []
        for k in range(n_gauss):
            L = np.array(
                [
                    [np.squeeze(covs[k, 0]), np.squeeze(covs[k, 1])],
                    [0.0, np.squeeze(covs[k, 2])],
                ]
            )
            gaussians.append(Gaussian(means[k], L @ L.T))
        return gaussians

    def gaussians_to_array(gaussians: list[Gaussian]) -> tuple[np.ndarray, np.ndarray]:
        n_gauss = len(gaussians)
        m_array = np.zeros((n_gauss, 2))
        c_array = np.zeros((n_gauss, 3))
        for i in range(n_gauss):
            m_array[i, :] = gaussians[i].mu.ravel()

            L = np.linalg.cholesky(gaussians[i].sigma)
            c_array[i, :] = L.ravel()[:3]
        return m_array.ravel(), c_array.ravel()

    def got_imposed_gaussian(means: np.ndarray, covariances: np.ndarray):
        gaussians = extract_gaussians(means, covariances)
        return got(
            target_parameter=exp.test_parameter,
            mesh=exp.test_mesh,
            abs_tol=None,
            _override_mean_descriptor=gaussians[-1],
            _override_source_descriptors=gaussians[:-1],
        )

    for k in range(5):
        start_m_array, start_c_array = gaussians_to_array(start_gaussians)
        initial_param = {
            "means": start_m_array,
            "covariances": start_c_array,
        }

    error_func = get_error(name=error, mesh=exp.test_mesh, truncation=exp.truncation)  # type: ignore[arg-type]
    optimal_param, result_got_optimized = optimize_parameters(
        method=got_imposed_gaussian,
        reference=exp.target_field,
        parameters_to_optimize=initial_param,
        bounds={
            "covariances": (np.ones(5 * 3) * -5.0, np.ones(5 * 3) * 5.0),
        },
        error_mesure=error_func,
        return_best=True,
        maxiter=100,
    )
    best_gaussians = extract_gaussians(
        optimal_param["means"], optimal_param["covariances"]
    )
    exp.add_result(
        f"{got.name} with all opt gaussian ${error}$",
        result_got_optimized,
        hook=plot_result,
    )

    return result_got_optimized, best_gaussians


def optimize_target_gauss(
    exp, got, error: str = "l_2", start_gauss: Gaussian | None = None
) -> tuple:
    """Optimize Gaussian parameters for target field."""

    def got_imposed_gaussian(m0, m1, s0, s1, s2):
        L = np.array([[np.squeeze(s0), np.squeeze(s1)], [0.0, np.squeeze(s2)]])
        sig = L @ L.T
        gauss = Gaussian(np.array([m0, m1]), sigma=sig)
        return got(
            target_parameter=exp.test_parameter,
            mesh=exp.test_mesh,
            abs_tol=None,
            _override_mean_descriptor=gauss,
        )

    if start_gauss is None:
        start_gauss = Gaussian(np.array([0.0, 0.0]), np.array([[0.1, 0.0], [0.0, 0.1]]))

    L = np.linalg.cholesky(start_gauss.sigma)
    initial_param = {
        "m0": np.array(start_gauss.mu[0]),
        "m1": np.array(start_gauss.mu[1]),
        "s0": np.array([L[0, 0]]),
        "s1": np.array([L[1, 0]]),
        "s2": np.array([L[1, 1]]),
    }

    error_func = get_error(name=error, mesh=exp.test_mesh, truncation=exp.truncation)  # type: ignore[arg-type]
    optimal_param, result_got_optimized = optimize_parameters(
        method=got_imposed_gaussian,
        reference=exp.target_field,
        parameters_to_optimize=initial_param,
        bounds={
            "s0": (np.array([-5.0]), np.array([5.0])),
            "s1": (np.array([-5.0]), np.array([5.0])),
            "s2": (np.array([-5.0]), np.array([5.0])),
        },
        error_mesure=error_func,
        return_best=True,
        maxiter=100,
    )
    L = np.array(
        [
            [np.squeeze(optimal_param["s0"]), 0.0],
            [np.squeeze(optimal_param["s1"]), np.squeeze(optimal_param["s2"])],
        ]
    )
    best_sigma = L @ L.T
    best_gauss = Gaussian(
        np.array([optimal_param["m0"], optimal_param["m1"]]), sigma=best_sigma
    )
    exp.add_result(
        f"{got.name} with opt gaussian ${error}$", result_got_optimized, plot_result
    )

    return result_got_optimized, best_gauss
