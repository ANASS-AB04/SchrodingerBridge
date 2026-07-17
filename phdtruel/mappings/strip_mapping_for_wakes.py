from typing import Callable, Self

import numpy as np
from scipy.interpolate import (
    PchipInterpolator,
    RegularGridInterpolator,
)

from phdtruel.fields import FieldOfInterest
from phdtruel.fields.meshes import RegularGrid
from phdtruel.mappings.mappings import (
    Mapping,
)
from phdtruel.mappings.two_fields_mappings import TwoFieldMappingModel
from phdtruel.ot1d import ot1d_mappings, to_hess_reg


class StripMappingForWakes(TwoFieldMappingModel):
    _name: str = "Stripes"

    def __init__(self, sensor_method_1d: Callable | None = None):
        if sensor_method_1d is None:
            self._sensor_method_1d = to_hess_reg
        else:
            self._sensor_method_1d = sensor_method_1d
        self._T_displacements: np.ndarray | None = None
        self._W_displacements: np.ndarray | None = None
        self._mesh: RegularGrid | None = None

    def copy(self) -> "StripMappingForWakes":
        new_mapping = StripMappingForWakes(self._sensor_method_1d)
        new_mapping._T_displacements = (
            None if self._T_displacements is None else np.copy(self._T_displacements)
        )
        new_mapping._W_displacements = (
            None if self._W_displacements is None else np.copy(self._W_displacements)
        )
        new_mapping._mesh = self._mesh
        return new_mapping

    def meta_parameters(self) -> dict:
        return {"sensor_method_1d": self._sensor_method_1d.__name__}

    def fit(self, u0: FieldOfInterest, u1: FieldOfInterest) -> Self:
        mesh_0 = u0.mesh
        mesh_1 = u1.mesh

        if not isinstance(mesh_0, RegularGrid) or not isinstance(mesh_1, RegularGrid):
            raise ValueError("The meshes must be RegularGrid instances.")

        if mesh_0.shape[1] != mesh_1.shape[1]:
            raise ValueError("The two fields must have the same mesh shape.")

        self._mesh = mesh_0
        self._T_displacements, self._W_displacements = self._construct_stripes_mappings(
            u0, u1
        )
        return self

    def _construct_stripes_mappings(
        self, u0: FieldOfInterest, u1: FieldOfInterest
    ) -> tuple[np.ndarray, np.ndarray]:
        mesh = self._mesh
        if mesh is None:
            raise ValueError("Mapping must be fitted before constructing stripes")
        shape = mesh.shape

        v0 = u0.values.reshape(shape)
        v1 = u1.values.reshape(shape)
        x0 = mesh.y
        x1 = mesh.y
        T_displacements = np.zeros((*shape, 2))
        W_displacements = np.zeros((*shape, 2))

        for i in range(len(mesh.x)):
            a_raw = v0[i, :]
            b_raw = v1[i, :]

            a = self._sensor_method_1d(x0, a_raw, const=1e-6)
            b = self._sensor_method_1d(x1, b_raw, const=1e-6)

            T_1d, W_1d = ot1d_mappings(x0, x1, a, b)
            T_displacements[i, :, 0] = 0
            T_displacements[i, :, 1] = T_1d - x0
            W_displacements[i, :, 0] = 0
            W_displacements[i, :, 1] = W_1d - x1
        return T_displacements, W_displacements

    def get_ts_mapping_function(self) -> Mapping:
        T = self._T_displacements
        mesh = self._mesh

        # Type guard: narrow union types
        if T is None or mesh is None:
            raise ValueError("Model must be fitted before getting mapping function")

        # Now T and mesh are guaranteed non-None
        def mapping(points: np.ndarray, s: float | None = None) -> np.ndarray:
            # Handle None default
            if s is None:
                s = 1.0
            s_T = s * T  # s is guaranteed float, T is guaranteed ndarray
            interp = RegularGridInterpolator(
                mesh.axes, s_T, bounds_error=False, fill_value=0.0
            )
            displacement = interp(points)
            return points + displacement

        return mapping

    def get_ts_inverse_mapping_function(self) -> Mapping:
        T = self._T_displacements
        mesh = self._mesh

        # Type guard
        if T is None or mesh is None:
            raise ValueError(
                "Model must be fitted before getting inverse mapping function"
            )

        def inverse_mapping(points: np.ndarray, s: float | None = None) -> np.ndarray:
            if s is None:
                s = 1.0
            s_T = s * T
            T_s_inv_displacements = np.zeros_like(s_T)

            # Only need to invert y-direction (index 1) since x-displacement is 0
            for i in range(s_T.shape[0]):  # Iterate over x-axis strips
                x0 = mesh.y
                # s_T[i, :, 1] is the y-displacement for strip i
                T_s_inv_displacements[i, :, 1] = (
                    PchipInterpolator(x0 + s_T[i, :, 1], x0)(x0) - x0
                )
                # x-displacement remains 0
                T_s_inv_displacements[i, :, 0] = 0

            interp = RegularGridInterpolator(
                mesh.axes,
                T_s_inv_displacements,
                bounds_error=False,
                fill_value=0.0,
            )
            displacement = interp(points)
            return points + displacement

        return inverse_mapping

    def get_ws_mapping_function(self) -> Mapping:
        W = self._W_displacements
        mesh = self._mesh

        # Type guard
        if W is None or mesh is None:
            raise ValueError("Model must be fitted before getting mapping function")

        def mapping(points: np.ndarray, s: float | None = None) -> np.ndarray:
            if s is None:
                s = 1.0
            s_W = s * W
            interp = RegularGridInterpolator(
                mesh.axes, s_W, bounds_error=False, fill_value=0.0
            )
            displacement = interp(points)
            return points + displacement

        return mapping

    def get_ws_inverse_mapping_function(self) -> Mapping:
        W = self._W_displacements
        mesh = self._mesh

        # Type guard
        if W is None or mesh is None:
            raise ValueError(
                "Model must be fitted before getting inverse mapping function"
            )

        def inverse_mapping(points: np.ndarray, s: float | None = None) -> np.ndarray:
            if s is None:
                s = 1.0
            s_W = s * W
            W_s_inv_displacements = np.zeros_like(s_W)

            # Only need to invert y-direction (index 1) since x-displacement is 0
            for i in range(s_W.shape[0]):  # Iterate over x-axis strips
                x1 = mesh.y
                # s_W[i, :, 1] is the y-displacement for strip i
                W_s_inv_displacements[i, :, 1] = (
                    PchipInterpolator(x1 + s_W[i, :, 1], x1)(x1) - x1
                )
                # x-displacement remains 0
                W_s_inv_displacements[i, :, 0] = 0

            interp = RegularGridInterpolator(
                mesh.axes,
                W_s_inv_displacements,
                bounds_error=False,
                fill_value=0.0,
            )
            displacement = interp(points)
            return points + displacement

        return inverse_mapping


def distance_between_maxima(x0, p0, x1, p1):
    # norm_0 = np.abs(p0 - np.mean(p0))
    # norm_0 = normalise_values(norm_0)
    # norm_1 = np.abs(p1 - np.mean(p1))
    # norm_1 = normalise_values(norm_1)

    i0_max = np.argmax(p0)
    i1_max = np.argmax(p1)

    displacement = x1[i1_max] - x0[i0_max]
    return displacement


def stripes_disp_mapping_maxima(field_0: FieldOfInterest, field_1: FieldOfInterest):
    mesh_0 = field_0.mesh
    mesh_1 = field_1.mesh

    if not isinstance(mesh_0, RegularGrid) or not isinstance(mesh_1, RegularGrid):
        raise TypeError("stripes_disp_mapping_maxima requires RegularGrid meshes.")

    if mesh_0.shape != mesh_1.shape:
        raise ValueError("The two fields must have the same mesh shape.")

    v0 = field_0.values.reshape((mesh_0.shape[0], mesh_0.shape[1]), order="C")
    v1 = field_1.values.reshape((mesh_1.shape[0], mesh_1.shape[1]), order="C")

    x0 = mesh_0.y
    x1 = mesh_1.y

    displacements = np.zeros((*v0.shape, 2))
    scalings = np.zeros(v0.shape)
    for i in range(len(mesh_0.x)):
        p0 = v0[i, :]
        p1 = v1[i, :]
        assert len(p0) == v0.shape[1]

        disp = distance_between_maxima(x0, p0, x1, p1)

        displacements[i, :, 0] = 0
        displacements[i, :, 1] = disp
        scalings[i, :] = p1.max() / (p0.max() + 1e-16)

    disp_vert = np.hstack(
        (
            displacements[:, :, 0].reshape(-1, 1),
            displacements[:, :, 1].reshape(-1, 1),
        )
    )
    return disp_vert, scalings.reshape(-1, 1)
