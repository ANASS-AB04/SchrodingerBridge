import numpy as np
from typing import Literal

from scipy.interpolate import interpn


def interpolate_in_regular_grid(
    _points: np.ndarray,
    values: np.ndarray,
    xi: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    shape: tuple,
    method: Literal[
        "linear", "nearest", "slinear", "cubic", "quintic", "pchip"
    ] = "linear",
    fill_value=np.nan,
) -> np.ndarray:
    points = (x_axis, y_axis)
    values = values.reshape(shape).transpose()
    return interpn(
        points, values, xi, method=method, bounds_error=False, fill_value=fill_value
    )


def constant_extrapolation(_points, _values, xi, fill_value=0.0):
    return np.zeros(xi.shape[0]) * fill_value
