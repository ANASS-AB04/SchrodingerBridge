import logging
import warnings
from collections.abc import Sequence
from typing import Any, Callable, Literal, cast, override

import numpy as np
from scipy.optimize import lsq_linear

from phdtruel.descriptors.descriptors import (
    DescriptorCreationMethod,
    FieldDescriptor,
    FieldDescriptorFactory,
)
from phdtruel.fields import FieldOfInterest
from phdtruel.fields.meshes import Mesh
from phdtruel.interpolations.interpolations import (
    FieldInterpolator,
)
from phdtruel.mappings.two_fields_mappings import MappingCreationMethod, MappingFactory
from phdtruel.sensors.sensors import SensorMethod

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


PossibleWeightMethods = Literal["rbf", "idw"]
RbfKernel = Literal[
    "thin_plate_spline",
    "linear",
    "cubic",
    "quintic",
    "multiquadric",
    "inverse_multiquadric",
    "inverse_quadratic",
]


def rbf_interpolated_weights(
    points: np.ndarray,
    centers: np.ndarray,
    kernel: RbfKernel = "linear",
    degree: int | None = None,
    epsilon: float = 1.0,
):
    points = np.asarray(points, dtype=float)
    centers = np.asarray(centers, dtype=float)
    n_points = points.shape[0]
    dimension = points.shape[1]

    if dimension != centers.shape[1]:
        raise ValueError("Dimension missmatch")

    # TODO Check if duplicate in points
    from scipy.interpolate import RBFInterpolator

    rbf_constructor = cast(Any, RBFInterpolator)

    if degree is None:
        degree = 0

    imposed_w_values = np.eye(n_points)
    interpolators = [
        rbf_constructor(
            points,
            imposed_w_values[i, :].astype(float),
            degree=degree,
            kernel=kernel,
            epsilon=epsilon,
        )
        for i in range(n_points)
    ]
    weights = np.array([interp(centers) for interp in interpolators])
    return weights


def idw_interpolated_weights(
    points: np.ndarray,
    centers: np.ndarray,
    power: float = 2.0,
    zero_tolerance: float = 1e-12,
):
    """Simple inverse-distance weighting on normalized parameter locations."""

    if points.shape[1] != centers.shape[1]:
        raise ValueError("Dimension missmatch")

    points = np.asarray(points, dtype=float)
    centers = np.asarray(centers, dtype=float)

    if centers.ndim == 1:
        centers = centers.reshape(1, -1)

    weights = np.zeros((points.shape[0], centers.shape[0]), dtype=float)
    for idx, center in enumerate(centers):
        distances = np.linalg.norm(points - center, axis=1)
        zero_mask = distances <= zero_tolerance

        if np.any(zero_mask):
            w = np.zeros_like(distances)
            w[zero_mask] = 1.0 / zero_mask.sum()
        else:
            inv_dist = 1.0 / np.power(distances, power)
            w = inv_dist / np.sum(inv_dist)

        weights[:, idx] = w

    return weights


def solve_lsq_barycentric_weights(points: np.ndarray, center: np.ndarray):
    n_points = points.shape[0]
    n_dim = points.shape[1]
    center = center.reshape(1, 2)
    if points.shape[1] != center.shape[1]:
        raise ValueError("Dimension missmatch")
    if n_dim + 1 != n_points:
        warnings.warn("Barycentric weight solution will not be optimal")

    A = points.transpose() - center.reshape(-1, 1)
    A = np.vstack((A, np.ones((1, n_points))))

    b = np.zeros(n_dim + 1)
    b[-1] = 1

    lsq_res = lsq_linear(A, b)
    weights = lsq_res.x
    return weights


class AveragedDescriptorsCDI(FieldInterpolator):
    def __init__(
        self,
        fields: Sequence[FieldOfInterest],
        sensor_method: SensorMethod,
        descriptor_factory: FieldDescriptorFactory | DescriptorCreationMethod,
        mapping_factory: MappingFactory | MappingCreationMethod,
        normalization_ranges: dict[str, tuple[float, float]] | None = None,
        weights_method: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        name: str = "AveragedDescriptorsCDI",
    ):
        super().__init__(
            fields,
            sensor_method,
            descriptor_factory,
            mapping_factory,
            normalization_ranges,
            name=name,
        )

        self._transformed_fields = [self._sensor_method(f) for f in self._fields]
        self._descriptors = [
            self._descriptor_factory(f) for f in self._transformed_fields
        ]

        if weights_method is None:
            self._weights_method = idw_interpolated_weights
        else:
            self._weights_method = weights_method

    @override
    def _predict(
        self,
        normalized_parameter: np.ndarray,
        mesh: Mesh,
        **kwargs,
    ) -> np.ndarray:
        abs_tol = cast(float, kwargs.get("abs_tol", 1e-3))
        _override_mean_descriptor = cast(
            FieldDescriptor | None,
            kwargs.get("_override_mean_descriptor")
            or kwargs.get("_override_mean_gaussian"),
        )
        _override_source_descriptors = cast(
            list[FieldDescriptor | None] | None,
            kwargs.get("_override_source_descriptors"),
        )

        interpolation_weights = self._weights_method(
            self._parameter_normalizer.normalize(self.parameters),
            normalized_parameter,
        )
        interpolation_weights = np.asarray(interpolation_weights, dtype=float)
        if interpolation_weights.ndim == 2 and interpolation_weights.shape[1] == 1:
            interpolation_weights = interpolation_weights[:, 0]

        if interpolation_weights.ndim != 1:
            raise ValueError(
                "Weight computation returned unexpected shape; expected 1D weights"
            )

        if abs_tol is None:
            where_non_zero = np.arange(0, interpolation_weights.shape[0], dtype=int)
        else:
            where_non_zero = np.where(interpolation_weights > abs_tol)[0]

        if where_non_zero.size == 0:
            where_non_zero = np.array([int(np.argmax(interpolation_weights))])

        interpolation_weights_subset = interpolation_weights[where_non_zero]
        weights_fields_subset = interpolation_weights_subset
        weights_descriptors_subset = interpolation_weights_subset
        mapping_travel_parameters_subset = np.ones(len(where_non_zero))
        indices = [int(k) for k in where_non_zero]
        fields_subset = [self._fields[k] for k in indices]
        logger.debug(
            f"GOT Bary on {where_non_zero} with {interpolation_weights_subset}"
        )

        if _override_source_descriptors is not None:
            source_descriptors = [
                od if od is not None else d
                for od, d in zip(_override_source_descriptors, self._descriptors)
            ]
        else:
            source_descriptors = self._descriptors
        descriptors_subset = [source_descriptors[k] for k in indices]

        if _override_mean_descriptor:
            mean_descriptor = _override_mean_descriptor
        else:
            weighted_descriptors = [
                float(w) * d
                for w, d in zip(weights_descriptors_subset, descriptors_subset)
            ]
            mean_descriptor = weighted_descriptors[0]
            for descriptor in weighted_descriptors[1:]:
                mean_descriptor = mean_descriptor + descriptor

        if self._mapping_factory is None:
            raise ValueError("mapping_factory cannot be None")
        mappings = [
            self._mapping_factory(d, mean_descriptor) for d in descriptors_subset
        ]
        inverse_mappings = [m.get_ts_mapping_function() for m in mappings]

        interpolated_values = np.zeros((mesh.points.shape[0], 1))
        for field, weight, inv_map, s in zip(
            fields_subset,
            weights_fields_subset,
            inverse_mappings,
            mapping_travel_parameters_subset,
        ):
            interpolated_values += weight * field(inv_map(mesh.points, s))

        self._last_interpolation_weights = interpolation_weights
        self._last_mean_gaussian = mean_descriptor
        return interpolated_values

    @property
    def descriptors(self):
        return self._descriptors
