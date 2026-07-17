import logging
from collections import defaultdict
from datetime import datetime
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, Callable, Literal, overload

import numpy as np
import pandas as pd

import phdtruel
from phdtruel import config
from phdtruel.fields import DynamicOfInterest, FieldOfInterest, Gaussian
from phdtruel.fields.meshes import DomainBounds, Mesh
from phdtruel.fields.parameters import ParameterSet
from phdtruel.fields.rejection_sampling import rejection_sampling
from phdtruel.mappings import wasserstein_distance_gaussians

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


SupportedErrorsNames = Literal["l_1", "l_2", "l_{inf}", "l_{inf} abs", "w_2"]

SESSION_DATE = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
SESSION_ERROR_DICT = defaultdict(dict)
ErrorFunction = Callable[[np.ndarray, np.ndarray], float]


def absolute_error(u_ref: np.ndarray, u: np.ndarray, order: int | float = 2):
    return np.linalg.norm(u - u_ref, ord=order)


@overload
def relative_error(
    u_ref: FieldOfInterest, u: FieldOfInterest, order: int | float = 2
) -> float: ...


@overload
def relative_error(
    u_ref: np.ndarray, u: np.ndarray, order: int | float = 2
) -> np.floating: ...


def relative_error(
    u_ref: np.ndarray | FieldOfInterest,
    u: np.ndarray | FieldOfInterest,
    order: int | float = 2,
) -> float | np.floating:
    # Extract values from FieldOfInterest if needed
    if isinstance(u_ref, FieldOfInterest):
        u_ref_values = u_ref.values
        assert u_ref_values is not None, "u_ref.values is None"
    else:
        u_ref_values = u_ref

    if isinstance(u, FieldOfInterest):
        u_values = u.values
        assert u_values is not None, "u.values is None"
    else:
        u_values = u

    norm = np.linalg.norm(u_ref_values, ord=order)
    return absolute_error(u_ref_values, u_values, order) / norm


def wasserstein_distance_point_clouds(
    points_u_ref: np.ndarray, points_u: np.ndarray, blur: float = 0.05
):
    # logger.warning(
    #     "W2 function tests cases are not valid on CFD  fields yet."
    #     " (Only valid for gaussians)"
    # )

    import time

    import torch
    from geomloss import SamplesLoss

    t0 = time.perf_counter()
    loss = SamplesLoss(loss="sinkhorn", p=1, blur=blur)
    points_ref: Any = torch.tensor(points_u_ref)
    points_u_tens: Any = torch.tensor(points_u)
    if torch.cuda.is_available():
        points_ref = points_ref.cuda()
        points_u_tens = points_u_tens.cuda()
    w2 = float(loss(points_ref, points_u_tens))
    duration = time.perf_counter() - t0
    if duration > 1.0:
        logger.warning(f"W2 took {duration}")
    return w2


def averaged_wasserstein_distance_values(
    u_ref: np.ndarray,
    u: np.ndarray,
    points: np.ndarray,
    blur: float = 0.001,
    n_points: int = 1000,
    n_repetitions: int = 5,
    seed: int | None = None,
):
    w2_avg = 0.0
    for _ in range(n_repetitions):
        w2_avg += wasserstein_distance_values(
            u_ref=u_ref,
            u=u,
            points=points,
            blur=blur,
            n_points=n_points,
            seed=seed,
        )
    return w2_avg / n_repetitions


def wasserstein_distance_values(
    u_ref: np.ndarray,
    u: np.ndarray,
    points: np.ndarray,
    blur: float = 0.001,
    n_points: int = 1000,
    seed: int | None = None,
):
    import time

    u_ref_normalized = normalise_values(np.abs(u_ref - np.median(u_ref)))
    u_normalized = normalise_values(np.abs(u - np.median(u_ref)))

    t0 = time.perf_counter()
    sampling_kwargs = {} if seed is None else {"seed": seed}
    points_u_ref = rejection_sampling(
        n_points=n_points,
        normalized_field_values=u_ref_normalized,
        field_values_coordinates=points,
        p=3,
        **sampling_kwargs,
    )

    points_u = rejection_sampling(
        n_points=n_points,
        normalized_field_values=u_normalized,
        field_values_coordinates=points,
        p=3,
        **sampling_kwargs,
    )
    duration = time.perf_counter() - t0
    if duration > 2.5:
        logger.warning(f"Sampling took {duration}")
    return wasserstein_distance_point_clouds(points_u_ref, points_u, blur=blur)


def wasserstein_distance_fields_of_interest(
    field_ref: FieldOfInterest,
    field: FieldOfInterest,
    blur: float = 0.005,
    n_points: int = 1_000,
    seed: int | None = None,
):
    points = field_ref.mesh.points
    return wasserstein_distance_values(
        u_ref=field_ref.values,
        u=field.values,
        points=points,
        blur=blur,
        n_points=n_points,
        seed=seed,
    )


def gaussian_based_w2(u0: FieldOfInterest, u1: FieldOfInterest):
    g0 = Gaussian.from_field(u0.mesh.points, u0.values)
    g1 = Gaussian.from_field(u1.mesh.points, u1.values)
    return wasserstein_distance_gaussians(g0, g1)


def gmm_based_w2(u0: FieldOfInterest, u1: FieldOfInterest):
    point_cloud_0 = rejection_sampling(500, u0.values, u0.mesh.points, seed=1)
    point_cloud_1 = rejection_sampling(500, u1.values, u1.mesh.points)
    return gmm_based_w2_cloud_points(point_cloud_0, point_cloud_1)


def gmm_based_w2_cloud_points(x0: np.ndarray, x1: np.ndarray):
    from sklearn.mixture import BayesianGaussianMixture

    gmm0 = BayesianGaussianMixture(n_components=30).fit(x0)
    gmm1 = BayesianGaussianMixture(n_components=30).fit(x1)

    m_s = gmm0.means_
    C_s = gmm0.covariances_
    w_s = gmm0.weights_

    m_t = gmm1.means_
    C_t = gmm1.covariances_
    w_t = gmm1.weights_

    import ot

    loss = np.sqrt(ot.gmm.gmm_ot_loss(m_s, m_t, C_s, C_t, w_s, w_t, log=False))

    return loss


def get_error(
    name: SupportedErrorsNames,
    mesh: Mesh | None = None,
    truncation: DomainBounds | None = None,
):
    match name:
        case "l_1":
            error = partial(relative_error, order=1)
        case "l_2":
            error = partial(relative_error, order=2)
        case "l_{inf}":
            error = partial(relative_error, order=np.inf)
        case "l_{inf} abs":
            error = partial(absolute_error, order=np.inf)
        case "w_2":
            if mesh is None:
                raise ValueError("Mesh is needed for w_2 error")

            points = mesh.points
            if truncation is not None:
                points = mesh.truncate_points(truncation)
            error = partial(wasserstein_distance_values, points=points)
        case _:
            raise ValueError(f"Error type {name} not defined")

    if truncation is not None:
        if mesh is None:
            raise ValueError("Mesh is needed when truncation is provided")
        truncation_mesh: Mesh = mesh

        def truncated_error(u_ref, u):
            return error(
                truncation_mesh.truncate_values(truncation, u_ref),
                truncation_mesh.truncate_values(truncation, u),
            )

        return truncated_error
    else:
        return error


def mesure_errors(
    u_ref: np.ndarray,
    u: np.ndarray,
    mesh: Mesh,
    *,
    truncation: DomainBounds | None = None,
    return_only: str | None = None,
):
    if truncation is not None:
        u_ref = mesh.truncate_values(truncation, u_ref)
        u = mesh.truncate_values(truncation, u)
        mesh = mesh.truncate(truncation)

    out = {
        "l_1": relative_error(u_ref, u, order=1),
        "l_2": relative_error(u_ref, u, order=2),
        "l_{inf}": relative_error(u_ref, u, order=np.inf),
        "l_{inf} abs": absolute_error(u_ref, u, order=np.inf),
        "w_2": wasserstein_distance_values(u_ref, u, mesh.points),
    }

    if truncation is not None:
        out["truncated"] = True

    if return_only is not None:
        return out[return_only]
    return out


def evaluate_interpolation_on_parameters(
    method: Callable[[ParameterSet | np.ndarray], FieldOfInterest],
    parameters: list[ParameterSet] | list[np.ndarray],
) -> list[FieldOfInterest]:
    out_fields = list[FieldOfInterest]()

    for param in parameters:
        field = method(param)
        out_fields.append(field)
    return out_fields


def compute_error_map(
    fields_map: list[FieldOfInterest],
    reference_field: FieldOfInterest,
    points: np.ndarray,
    error_metric: ErrorFunction | None = None,
):
    error_function: ErrorFunction
    if error_metric is None:
        error_function = partial(relative_error, order=2)  # type: ignore
    else:
        error_function = error_metric

    error_map = np.zeros(len(fields_map))
    for i, field in enumerate(fields_map):
        error_map[i] = error_function(reference_field(points), field(points))
    return error_map


def commit_errors_to_db(
    errors: dict[str, float | str],
    *,
    method: str,
    test_case: str,
    filepath: Path | str | None = None,
    **kwargs,
):
    if filepath is None:
        db_path = phdtruel.config.get("errors_db_path")
        if db_path is None:
            raise ValueError("errors_db_path not configured")
        filepath = Path(db_path)

    if isinstance(filepath, str):
        filepath = Path(filepath)

    errors["method"] = method
    errors["test_case"] = test_case
    errors["date"] = SESSION_DATE
    errors.update(kwargs)

    new_df = pd.DataFrame(errors, index=[0])
    old_df = read_errors_db(filepath)

    df = pd.concat([old_df, new_df], ignore_index=True)

    df.to_csv(filepath, index=True)
    SESSION_ERROR_DICT[method] = errors
    return new_df


def read_errors_db(filepath: Path | str | None = None):
    if filepath is None:
        db_path = config.get("errors_db_path")
        if db_path is None:
            raise ValueError("errors_db_path not configured")
        filepath = Path(db_path)
    elif isinstance(filepath, str):
        filepath = Path(filepath)

    if filepath.exists():
        return pd.read_csv(filepath, index_col=0)
    else:
        return pd.DataFrame()


def optimize_parameters(
    *,
    method: Callable,
    parameters_to_optimize: dict[str, float | np.ndarray],
    reference: FieldOfInterest,
    error_mesure: ErrorFunction,
    bounds: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    return_best: bool = False,
    maxiter: int = 20,
):
    """
    Optimize the parameters of a given method to minimize the error between the method's output and a reference.

    This function takes a callable method and a set of parameters to optimize. It uses an optimization
    algorithm to find the parameter values that minimize the difference between the output of the method
    (with the optimized parameters) and a reference array.

    Args:
        method (Callable):
            A callable which accepts parameters defined in `parameters_to_optimize` and
            returns a result that will be compared to the reference.

        parameters_to_optimize (dict[str, float | np.ndarray]):
            A dictionary where keys are parameter names to optimize and values are initial values
            (float or numpy arrays).

        reference (np.ndarray):
            The reference output to which the method's result will be compared for error calculation.

        error_mesure (ErrorFunction):
            A callable that takes the reference and the result of the method, and returns a numerical error measure.

        bounds (dict[str, tuple[np.ndarray, np.ndarray]], optional):
            A dictionary where keys are parameter names and values are tuples containing lower and upper
            bounds for the parameters. If None, parameters are allowed to take any value.

        return_best (bool, optional):
            If True, the function also returns the best result obtained from the method.
            Default is False.

    Returns:
        dict: A dictionary containing the optimal parameter values.

        If `return_best` is True, a tuple (dict, np.ndarray) is returned where the first element
        is the optimal parameters and the second element is the result of the method with
        these optimal parameters.

    Examples:
        optimal_param_l2, result_got_grid_optimized_l2 = optimize_parameters(
            method=partial(
                got_grid_2_param,
                points=points,
                points_shape=shape,
                fields=fields,
            ),
            reference=target_field(points),
            parameters_to_optimize={
                "s_line": 0.5,
                "s_col": 0.5,
            },
            error_mesure=partial(
                mesure_errors,
                points=points,
                truncation=truncation,
                return_only="l_2",
            ),
            return_best=True,
        )
    """

    def unravel_param_values(x: np.ndarray, sizes: list[int]) -> list[np.ndarray]:
        unraveled_values = []
        count = 0
        for size in sizes:
            unraveled_values.append(x[count : size + count])
            count += size
        return unraveled_values

    def ravel_param_values(x: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
        """Ravel list of array and return the way to do the inverse operation"""
        sizes = []
        for arr in x:
            sizes.append(arr.size)

        arr = np.concatenate(x).ravel()
        return arr, sizes

    from scipy.optimize import minimize

    parameter_arrays: dict[str, np.ndarray] = {
        k: np.atleast_1d(np.asarray(v, dtype=float))
        for k, v in parameters_to_optimize.items()
    }

    initial_param_values, unravel_size = ravel_param_values(
        list(parameter_arrays.values())
    )

    def f(x: np.ndarray):
        logger.debug(f"called f with {x}")
        unraveled_values = unravel_param_values(x, sizes=unravel_size)
        parameters = {
            key: unraveled_values[i] for i, key in enumerate(parameter_arrays.keys())
        }
        # print(parameters)
        result = method(**parameters)
        error = error_mesure(reference.values, result.values)
        # print(error)

        return error

    bound_tuples = None
    if bounds is not None:
        bound_tuples = []
        for k, param in enumerate(parameter_arrays.keys()):
            if param in bounds.keys():
                for low_bound, high_bound in zip(bounds[param][0], bounds[param][1]):
                    bound_tuples.append((low_bound, high_bound))
            else:
                size = unravel_size[k]
                for _ in range(size):
                    bound_tuples.append((-np.inf, np.inf))

    res = minimize(
        f,
        initial_param_values,
        bounds=bound_tuples,
        options={"disp": False, "maxiter": maxiter},
    )

    # unravel_param_values
    unraveled_values = unravel_param_values(res.x, sizes=unravel_size)
    optimal_parameters = {
        key: unraveled_values[i] for i, key in enumerate(parameter_arrays.keys())
    }
    best_result = method(**optimal_parameters)
    if return_best:
        return optimal_parameters, best_result
    return optimal_parameters


def find_optimal_s(
    function: Callable[[float], FieldOfInterest | np.ndarray],
    reference_field: FieldOfInterest,
    initial_s: float = 0.5,
    bounds: tuple[float, float] = (0.0, 1.0),
    error_fn: ErrorFunction = relative_error,
    maxiter: int = 20,
) -> tuple[float, float]:
    """Find the scalar interpolation parameter ``s`` minimizing error to a reference.

    Typical use case: pick the blending parameter of a convex displacement
    interpolation (or any other single-parameter interpolation family) that
    best matches a reference field, via a bounded 1D scipy minimization.

    Args:
        function: Maps a scalar ``s`` to an interpolated field/array.
        reference_field: Ground truth to compare against.
        initial_s: Initial guess for ``s``.
        bounds: ``(low, high)`` bounds for ``s``.
        error_fn: Error metric called as ``error_fn(interpolated, reference_field)``.
        maxiter: Maximum number of optimizer iterations.

    Returns:
        A ``(optimal_s, optimal_error)`` tuple.
    """
    from scipy.optimize import minimize

    def objective(s: np.ndarray | float) -> float:
        s_value = float(s[0]) if isinstance(s, np.ndarray) else float(s)
        interpolated = function(s_value)
        error = error_fn(interpolated, reference_field)
        return float(error)

    res = minimize(
        fun=objective,
        x0=initial_s,
        bounds=[bounds],
        options={"disp": False, "maxiter": maxiter},
    )
    optimal_s = float(res.x[0])
    optimal_error = float(res.fun)
    return optimal_s, optimal_error


def normalise_values(u):
    u = np.abs(u)
    u = u - np.min(u)
    if np.max(u) == 0:
        return np.ones_like(u)
    u_norm = u / np.max(u)
    return u_norm


def remove_corners_from_map(
    map_list: list[FieldOfInterest],
    field_map: tuple[int, ...],
) -> list[FieldOfInterest]:
    """Remove corner elements from map array.

    Args:
        map_list: List of FieldOfInterest objects to reshape
        field_map: Shape tuple for reshaping

    Returns:
        Filtered list with corner elements removed
    """
    dim = len(field_map)
    corners_idx = [[0, -1]] * dim

    # Convert to numpy array with object dtype to allow None assignment
    map_array = np.array(map_list, dtype=object).reshape(field_map)
    for corner in product(*corners_idx):
        map_array[corner] = None

    # Filter out None values and return as list
    return [field for field in map_array.ravel() if isinstance(field, FieldOfInterest)]


def error_along_time(
    reference_dynamic: DynamicOfInterest,
    dynamic: DynamicOfInterest,
    test_points: np.ndarray,
    error_function: ErrorFunction,
) -> np.ndarray:
    if len(reference_dynamic) != len(dynamic):
        logger.warning("The two dynamics are not the same length.")

    n_fields = min(len(reference_dynamic), len(dynamic))
    errors = []
    for k in range(n_fields):
        field_ref = reference_dynamic[k]
        field = dynamic[k]
        err = error_function(field_ref(test_points), field(test_points))
        errors.append(err)
    return np.array(errors)


def error_dynamics(d_ref: DynamicOfInterest, d: DynamicOfInterest) -> np.ndarray:
    errors = []
    n_fields = min(len(d_ref), len(d))
    for idx in range(n_fields):
        f_ref = d_ref[idx]
        f = d[idx]
        err = float(relative_error(f_ref.values, f.values))
        errors.append(err)
    return np.array(errors)
