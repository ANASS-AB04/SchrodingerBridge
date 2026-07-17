import numpy as np
from scipy.interpolate import NearestNDInterpolator

from phdtruel.feature_identification import DEFAULT_SEED


def rejection_sampling(
    n_points: int,
    normalized_field_values,
    field_values_coordinates,
    p=1,
    seed=DEFAULT_SEED,
):
    """Sample points according to any normalized field.

    The sampling is continuous on the interpolated normalized field.
    Uses the rejection sampling method.
    """
    np.random.seed(seed)
    if len(field_values_coordinates.shape) < 2:
        field_values_coordinates = field_values_coordinates.reshape(-1, 1)

    dim = field_values_coordinates.shape[1]

    coords_minima = np.min(field_values_coordinates, axis=0)
    coords_maxima = np.max(field_values_coordinates, axis=0)

    field_value_interp = NearestNDInterpolator(
        field_values_coordinates, normalized_field_values
    )

    samples = np.zeros((0, dim))
    oversample_ratio = 100  # To sample faster
    np.random.seed(seed)
    while len(samples) < n_points:
        # Sample candidates on uniform distrib
        sample_candidates = np.random.rand(n_points * oversample_ratio, dim)
        sample_candidates = (
            sample_candidates * (coords_maxima - coords_minima) + coords_minima
        )

        candidates_values = field_value_interp(sample_candidates)
        # Retain candidates that are at big field values using a uniform distribution.
        to_retain = candidates_values.ravel() >= np.random.rand(
            n_points * oversample_ratio
        ) ** (1.0 / p)

        samples = np.vstack((samples, sample_candidates[to_retain]))
    samples = samples[:n_points]
    return samples
