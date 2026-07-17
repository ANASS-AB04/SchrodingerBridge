from typing import Literal, Self

import numpy as np
import ot
from scipy.interpolate import RBFInterpolator
from sklearn.cluster import HDBSCAN

from phdtruel.fields import FieldOfInterest
from phdtruel.fields.gaussians import (
    principal_point_of_gaussian,
    Gaussian,
    compute_principal_points,
)
from phdtruel.fields.meshes import DomainBounds
from phdtruel.fields.rejection_sampling import rejection_sampling
from phdtruel.mappings import DisplacementDefinedMapping
from phdtruel.mappings.mappings import (
    Mapping,
    construct_inverse_mapping_on_mesh,
)
from phdtruel.mappings.two_fields_mappings import TwoFieldMappingModel
from phdtruel.ot1d import (
    ot1d_mappings,
    to_hess_reg,
    to_local_support_density,
    to_density,
)
from phdtruel.sensors import sensors

LocalSupportDensityStrategies = Literal[
    "hessian", "absolute_fluctuations", "normalisation"
]


def compute_border_displacement(
    points: np.ndarray,
    source_field: FieldOfInterest,
    target_field: FieldOfInterest,
    ax: int,
    density_strategy: LocalSupportDensityStrategies = "hessian",
) -> np.ndarray:
    x = points[:, ax]
    if density_strategy == "hessian":
        to_local_support_transformation = to_hess_reg
    elif density_strategy == "absolute_fluctuations":
        to_local_support_transformation = to_local_support_density
    elif density_strategy == "normalisation":
        to_local_support_transformation = to_density
    else:
        raise ValueError(f"Unknown density strategy {density_strategy}")

    left_values_0 = to_local_support_transformation(
        x, source_field(points).ravel(), const=1e-6
    )
    left_values_1 = to_local_support_transformation(
        x, target_field(points).ravel(), const=1e-6
    )
    T, _ = ot1d_mappings(x, x, left_values_0, left_values_1)
    disp = T - x
    displacement = np.zeros_like(points)
    displacement[:, ax] = disp
    return displacement


class MasayukiMapping(TwoFieldMappingModel):
    _name: str = "Masayuki"

    def get_mapping_function(self, inverse: bool = False) -> Mapping:
        if self._disp_mapping is None:
            raise ValueError("Model must be fitted before calling get_mapping_function")
        disp_mapping = self._disp_mapping
        if inverse:
            return self.compute_inverse()

        # Return a wrapper that matches the Mapping protocol signature
        def mapping_wrapper(points: np.ndarray, s: float | None = None) -> np.ndarray:
            s_val = 1.0 if s is None else s
            return disp_mapping(points, s=s_val)

        return mapping_wrapper

    def fit(self, u0: FieldOfInterest, u1: FieldOfInterest) -> Self:
        self._source_field = u0
        self._target_field = u1

        self._fit_masayuki(
            self._number_of_gaussians,
            self._n_boundary_points,
            self._do_right_border,
            self._do_transport_border_points,
            self._boundary_margin,
            self._hdbscan_parameters,
            self._rbf_parameters,
            self._border_density_strategy,
        )
        return self

    def __init__(
        self,
        support_points: np.ndarray,
        number_of_gaussians: int = 1,
        n_boundary_points: int = 0,
        boundary_margin: float = 0.0,
        do_right_border: bool = True,
        do_transport_border_points: bool = False,
        border_density_strategy: LocalSupportDensityStrategies = "hessian",
        hdbscan_parameters: dict | None = None,
        rbf_parameters: dict | None = None,
        **kwargs,
    ):
        self._source_field: FieldOfInterest | None = None
        self._target_field: FieldOfInterest | None = None
        self._support_points = support_points
        self._disp_mapping: DisplacementDefinedMapping | None = None

        self._number_of_gaussians = number_of_gaussians
        self._n_boundary_points = n_boundary_points
        self._do_right_border = do_right_border
        self._do_transport_border_points = do_transport_border_points
        self._boundary_margin = boundary_margin
        self._hdbscan_parameters = hdbscan_parameters
        self._rbf_parameters = rbf_parameters
        self._border_density_strategy = border_density_strategy

    def copy(self) -> Self:
        copy = MasayukiMapping(
            self._support_points,
            number_of_gaussians=self._number_of_gaussians,
            n_boundary_points=self._n_boundary_points,
            do_right_border=self._do_right_border,
            do_transport_border_points=self._do_transport_border_points,
            boundary_margin=self._boundary_margin,
            hdbscan_parameters=self._hdbscan_parameters,
            rbf_parameters=self._rbf_parameters,
        )
        copy._principal_points_source = self._principal_points_source
        copy._principal_points_target = self._principal_points_target
        copy._border_points = self._border_points
        raise NotImplementedError
        return copy

    def meta_parameters(self) -> dict:
        return {
            "number_of_gaussians": self._number_of_gaussians,
            "n_boundary_points": self._n_boundary_points,
            "do_right_border": self._do_right_border,
            "do_transport_border_points": self._do_transport_border_points,
            "boundary_margin": self._boundary_margin,
            "hdbscan_parameters": self._hdbscan_parameters,
            "rbf_parameters": self._rbf_parameters,
        }

    def _fit_masayuki(
        self,
        number_of_gaussians: int,
        n_boundary_points: int,
        do_right_border: bool,
        do_transport_border_points: bool,
        boundary_margin: float,
        hdbscan_parameters: dict | None,
        rbf_parameters: dict | None,
        density_strategy: LocalSupportDensityStrategies,
    ):
        source_field = self._source_field
        target_field = self._target_field
        support_points = self._support_points

        # TYPE GUARD: Narrow types from union to non-None
        if source_field is None or target_field is None:
            raise ValueError(
                "Source and target fields must be set before calling _fit_masayuki"
            )

        if number_of_gaussians == 1:
            g0 = Gaussian.from_field(
                support_points,
                sensors.to_localized_density_sensor(source_field)(support_points),
            )
            g1 = Gaussian.from_field(
                support_points,
                sensors.to_localized_density_sensor(target_field)(support_points),
            )
            self._principal_points_source = principal_point_of_gaussian(g0)
            self._principal_points_target = principal_point_of_gaussian(g1)
        elif number_of_gaussians > 1:
            # Handle None case for hdbscan_parameters
            if hdbscan_parameters is None:
                hdbscan_parameters = {}
            self._generate_principal_points_gmm(hdbscan_parameters=hdbscan_parameters)

        principal_points_displacements = (
            self._principal_points_target - self._principal_points_source
        )

        self._border_points = np.zeros((0, 2))
        bounds = source_field.mesh.get_bounds()
        if do_borders := n_boundary_points > 0:
            expanded_bounds = bounds.shrink(-boundary_margin)
            left_expanded, right_exp, top_exp, bottom_exp = get_four_borders(
                expanded_bounds,
                n=n_boundary_points,
            )

            borders = [left_expanded, top_exp, bottom_exp]
            if do_right_border:
                borders.append(right_exp)
            self._border_points = np.vstack(borders)
        border_displacements = np.zeros_like(self._border_points)
        if do_transport_border_points and do_borders:
            left, right, top, bottom = get_four_borders(bounds, n=n_boundary_points)
            left_displacement = compute_border_displacement(
                left,
                source_field,
                target_field,
                ax=1,
                density_strategy=density_strategy,
            )
            right_displacement = compute_border_displacement(
                right,
                source_field,
                target_field,
                ax=1,
                density_strategy=density_strategy,
            )
            top_displacement = compute_border_displacement(
                top, source_field, target_field, ax=0, density_strategy=density_strategy
            )
            bottom_displacement = compute_border_displacement(
                bottom,
                source_field,
                target_field,
                ax=0,
                density_strategy=density_strategy,
            )
            displacements = [left_displacement, top_displacement, bottom_displacement]
            if do_right_border:
                displacements.append(right_displacement)
            border_displacements = np.vstack(displacements)
        train_x = np.vstack((self._principal_points_source, self._border_points))
        train_x, duplicated_idx = np.unique(train_x, axis=0, return_index=True)
        train_y = np.vstack((principal_points_displacements, border_displacements))
        train_y = train_y[duplicated_idx, :]

        if rbf_parameters is None:
            rbf_parameters = {
                "kernel": "thin_plate_spline",
                "degree": 1,
            }

        self._rbf = RBFInterpolator(
            train_x,
            train_y,
            **rbf_parameters,
        )
        self._disp_mapping = DisplacementDefinedMapping(
            displacement=self._rbf(self._support_points),
            support_points=self._support_points,
        )

    def _generate_principal_points_gmm(
        self,
        n_points: int = 3000,
        p: int = 2,
        hdbscan_parameters: dict | None = None,
    ):
        # TYPE GUARD: Narrow types
        if self._source_field is None or self._target_field is None:
            raise ValueError(
                "Source and target fields must be set before generating principal points"
            )

        # Now source_field and target_field are guaranteed to be FieldOfInterest
        f0 = sensors.to_localized_density_sensor(self._source_field)
        f1 = sensors.to_localized_density_sensor(self._target_field)

        cloud_points_0 = rejection_sampling(n_points, f0.values, f0.mesh.points, p)
        cloud_points_1 = rejection_sampling(n_points, f1.values, f1.mesh.points, p)

        if hdbscan_parameters is None:
            hdbscan_parameters = {}
        hdbscan_defaults = {
            "min_cluster_size": 100,
            "min_samples": 3,
            "allow_single_cluster": True,
        }

        hdbscan_0 = HDBSCAN(**(hdbscan_defaults | hdbscan_parameters))
        hdbscan_1 = HDBSCAN(**(hdbscan_defaults | hdbscan_parameters))

        hdbscan_0.fit(cloud_points_0)
        hdbscan_1.fit(cloud_points_1)

        labels_0 = hdbscan_0.labels_
        labels_1 = hdbscan_1.labels_

        clusters_0 = [cloud_points_0[labels_0 == i] for i in range(max(labels_0) + 1)]
        clusters_1 = [cloud_points_1[labels_1 == i] for i in range(max(labels_1) + 1)]

        gmm0 = [
            Gaussian.from_field(cluster_points, np.ones(cluster_points.shape[0]))
            for cluster_points in clusters_0
        ]
        gmm1 = [
            Gaussian.from_field(cluster_points, np.ones(cluster_points.shape[0]))
            for cluster_points in clusters_1
        ]

        means_0 = np.hstack([g.mu for g in gmm0]).T
        means_1 = np.hstack([g.mu for g in gmm1]).T

        cov_0 = np.array([g.sigma for g in gmm0])
        cov_1 = np.array([g.sigma for g in gmm1])

        M = ot.dist(means_0, means_1)
        a, b = np.ones(len(gmm0)) / len(gmm0), np.ones(len(gmm1)) / len(gmm1)
        G0 = ot.emd(a, b, M)
        to_indices = np.argmax(G0, axis=1)
        means_1_reordered = means_1[to_indices]
        cov_1_reordered = cov_1[to_indices]

        all_principal_points_0 = [
            compute_principal_points(mu, sigma) for mu, sigma in zip(means_0, cov_0)
        ]
        all_principal_points_0 = np.concatenate(all_principal_points_0, axis=0)

        all_principal_points_1 = [
            compute_principal_points(mu, sigma)
            for mu, sigma in zip(means_1_reordered, cov_1_reordered)
        ]
        all_principal_points_1 = np.concatenate(all_principal_points_1, axis=0)

        self._gmm_cloud_points_0 = cloud_points_0
        self._gmm_cloud_points_1 = cloud_points_1
        self._gmm_labels_0 = labels_0
        self._gmm_labels_1 = labels_1
        self._gmm_G0 = G0
        self._gmm_means_0 = means_0
        self._gmm_means_1 = means_1
        self._gmm_gmm0 = gmm0
        self._gmm_gmm1 = gmm1
        self._gmm_means_1_reordered = means_1_reordered

        self._principal_points_source = all_principal_points_0
        self._principal_points_target = all_principal_points_1

    def __call__(self, points: np.ndarray, s: float = 1.0):
        if self._disp_mapping is None:
            raise ValueError("Model must be fitted before calling __call__")
        result = self._disp_mapping(points, s=s)
        return result

    def check(self):
        inverse_mapping = self.compute_inverse()
        return np.allclose(
            inverse_mapping(self(self._principal_points_source)),
            self._principal_points_source,
            atol=1e-3,
        )

    def compute_inverse(self) -> Mapping:
        if self._disp_mapping is None:
            raise ValueError("Model must be fitted before calling compute_inverse")
        disp_mapping = self._disp_mapping
        support_points = self._support_points

        # Create the inverse mapping using construct_inverse_mapping_on_mesh
        # which expects a Mapping protocol compatible object
        def mapping_wrapper(points: np.ndarray, s: float | None = None) -> np.ndarray:
            s_val = 1.0 if s is None else s
            return disp_mapping(points, s=s_val)

        return construct_inverse_mapping_on_mesh(mapping_wrapper, support_points)


def make_line(
    start: tuple[float, ...], end: tuple[float, ...], n: int = 50
) -> np.ndarray:
    dim = len(start)
    assert dim == len(end)

    points = np.vstack([np.linspace(start[i], end[i], num=n) for i in range(dim)])
    return points.transpose()


def get_four_borders(
    bounds: DomainBounds, n: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = (
        bounds.minima[0],
        bounds.maxima[0],
        bounds.minima[1],
        bounds.maxima[1],
    )
    left = make_line((xmin, ymin), (xmin, ymax), n)
    right = make_line((xmax, ymin), (xmax, ymax), n)
    top = make_line((xmin, ymax), (xmax, ymax), n)
    bottom = make_line((xmin, ymin), (xmax, ymin), n)
    return left, right, top, bottom


def get_border_points(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    n: int = 100,
    do_right_border: bool = True,
) -> np.ndarray:
    x = [
        np.linspace(xmin, xmax, n),
        np.linspace(xmax, xmin, n),
        np.ones(n) * xmin,
    ]
    y = [
        np.ones(n) * ymin,
        np.ones(n) * ymax,
        np.linspace(ymax, ymin, n),
    ]
    if do_right_border:
        x.append(np.ones(n) * xmax)
        y.append(np.linspace(ymin, ymax, n))

    X = np.concatenate(x)
    Y = np.concatenate(y)
    return np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))


def get_border_points_with_margin(
    bounds: DomainBounds, margin: float, n: int = 10, do_right_border: bool = True
) -> np.ndarray:
    xmin = bounds.minima[0] - margin
    xmax = bounds.maxima[0] + margin
    ymin = bounds.minima[1] - margin
    ymax = bounds.maxima[1] + margin
    return get_border_points(
        xmin, xmax, ymin, ymax, n=n, do_right_border=do_right_border
    )
