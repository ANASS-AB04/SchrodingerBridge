from typing import Callable

import numpy as np
from contourpy import contour_generator
from scipy.optimize import minimize_scalar

from phdtruel.fields import FieldOfInterest, Gaussian
from phdtruel.fields.gaussians import to_localized_density
from phdtruel.fields.meshes import DomainBounds

SensorMethod = Callable[[FieldOfInterest], FieldOfInterest]


def identity_sensor(field: FieldOfInterest) -> FieldOfInterest:
    return field


def to_localized_density_sensor(field: FieldOfInterest, p: int = 2) -> FieldOfInterest:
    values = np.abs(field.values - np.median(field.values)) ** p
    return FieldOfInterest(
        field.parameter, field.mesh, values, field.name, field.is_interpolated
    )


def localize_field_fit_gaussian(
    field: FieldOfInterest,
    p: int = 2,
    truncation: DomainBounds | None = None,
) -> Gaussian:
    if truncation is not None:
        field = field.truncate(truncation)

    return Gaussian.from_field(
        field.mesh.points,
        to_localized_density(field.values, p=p),
    )


def contours_with_n_points(
    field: FieldOfInterest, n_points_target: int, return_level: bool = False
) -> np.ndarray | tuple[np.ndarray, float]:
    cont_gen = contour_generator(
        field.mesh.x, field.mesh.y, field.values.reshape(field.mesh.shape).T
    )

    def func(level: float):
        contours = cont_gen.create_contour(max(field.values) * level)
        if len(contours) == 0:
            n_points = 0
        else:
            points = np.vstack([np.asarray(contour) for contour in contours])
            n_points = points.shape[0]
        return abs(n_points - n_points_target)

    res = minimize_scalar(
        func,
        bracket=None,
        bounds=(0.01, 1.0),
        args=(),
        method="bounded",
    )
    level = res.x
    contours = cont_gen.create_contour(max(field.values) * level)
    points = np.vstack([np.asarray(contour) for contour in contours])
    if return_level:
        return points, level
    return points


def contours_at_level(field: FieldOfInterest, level: float) -> np.ndarray:
    cont_gen = contour_generator(
        field.mesh.x, field.mesh.y, field.values.reshape(field.mesh.shape).T
    )
    contours = cont_gen.create_contour(max(field.values) * level)
    points = np.vstack([np.asarray(contour) for contour in contours])
    return points


def boundary_box_of_foi(
    field: FieldOfInterest,
    n_points_per_side: int = 10,
    margin: float = 0.0,
) -> np.ndarray:
    """Only in 2d"""
    x_min, x_max = field.mesh.x.min(), field.mesh.x.max()
    y_min, y_max = field.mesh.y.min(), field.mesh.y.max()

    grid = np.meshgrid(
        np.linspace(x_min - margin, x_max + margin, n_points_per_side),
        np.linspace(y_min - margin, y_max + margin, n_points_per_side),
    )
    points = np.vstack((grid[0].ravel(), grid[1].ravel()))
    return points.T
