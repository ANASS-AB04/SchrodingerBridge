from collections.abc import Callable, Sequence
from typing import Literal, override, overload

import numpy as np

from phdtruel.descriptors.descriptors import (
    DescriptorCreationMethod,
    FieldDescriptorFactory,
)
from phdtruel.fields import FieldOfInterest
from phdtruel.fields.parameters import ParameterSet
from phdtruel.interpolations.interpolations import TwoFieldsInterpolator
from phdtruel.mappings.mappings import (
    Mapping,
)
from phdtruel.mappings.two_fields_mappings import (
    TwoFieldMappingModel,
    MappingCreationMethod,
    MappingFactory,
)
from phdtruel.sensors.sensors import SensorMethod


class CDIInterpolator(TwoFieldsInterpolator):
    def __init__(
        self,
        fields: Sequence[FieldOfInterest],
        sensor_method: SensorMethod,
        descriptor_factory: FieldDescriptorFactory | DescriptorCreationMethod,
        mapping_factory: MappingFactory | MappingCreationMethod,
        name: str | None = None,
    ):
        super().__init__(
            fields, sensor_method, descriptor_factory, mapping_factory, name
        )

        d0, d1 = self._descriptors
        self._mappings: list[TwoFieldMappingModel] = [
            mapping_factory(d0, d1),
            mapping_factory(d1, d0),
        ]

    @override
    def interpolate(self, points: np.ndarray, s: float, **kwargs) -> np.ndarray:
        map_t = self._mappings[0]
        map_w = self._mappings[1]
        t_inverse = map_t.get_ts_mapping_function()
        w_inverse = map_w.get_ws_mapping_function()

        values = convex_displacement_interpolation(
            points,
            self._fields[0],
            self._fields[1],
            w_inverse,
            t_inverse,
            s,
        )
        return values


class CustomCDIInterpolator(TwoFieldsInterpolator):
    def __init__(
        self,
        fields: Sequence[FieldOfInterest],
        sensor_method: SensorMethod,
        descriptor_factory: FieldDescriptorFactory | DescriptorCreationMethod,
        mapping_factory_0: MappingFactory | MappingCreationMethod,
        mapping_factory_1: MappingFactory | MappingCreationMethod,
        name: str | None = None,
    ):
        super().__init__(fields, sensor_method, descriptor_factory, None, name)

        d0, d1 = self._descriptors
        self._mappings: list[TwoFieldMappingModel] = [
            mapping_factory_0(d0, d1),
            mapping_factory_1(d1, d0),
        ]

    @override
    def interpolate(self, points: np.ndarray, s: float, **kwargs) -> np.ndarray:
        map_t = self._mappings[0]
        map_w = self._mappings[1]
        t_inverse = map_t.get_ts_mapping_function()
        w_inverse = map_w.get_ws_mapping_function()

        values = convex_displacement_interpolation(
            points,
            self._fields[0],
            self._fields[1],
            w_inverse,
            t_inverse,
            s,
        )
        return values


class ApproximatedCDIInterpolator(CDIInterpolator):
    @override
    def interpolate(self, points: np.ndarray, s: float, **kwargs) -> np.ndarray:
        map_t = self._mappings[0]
        map_w = self._mappings[1]
        t_map = map_t.get_ts_mapping_function()
        w_map = map_w.get_ws_mapping_function()

        values = convex_displacement_interpolation(
            points,
            self._fields[0],
            self._fields[1],
            w_map,
            t_map,
            s,
        )
        return values


@overload
def convex_displacement_interpolation(
    points: np.ndarray,
    u0: FieldOfInterest,
    u1: FieldOfInterest,
    ws_inverse: Mapping,
    ts_inverse: Mapping,
    s: float,
    s_t_inv: float | None = ...,
    s_w_inv: float | None = ...,
    *,
    as_foi: Literal[True],
) -> FieldOfInterest: ...


@overload
def convex_displacement_interpolation(
    points: np.ndarray,
    u0: FieldOfInterest,
    u1: FieldOfInterest,
    ws_inverse: Mapping,
    ts_inverse: Mapping,
    s: float,
    s_t_inv: float | None = ...,
    s_w_inv: float | None = ...,
    as_foi: Literal[False] = ...,
) -> np.ndarray: ...


def convex_displacement_interpolation(
    points: np.ndarray,
    u0: FieldOfInterest,
    u1: FieldOfInterest,
    ws_inverse: Mapping,
    ts_inverse: Mapping,
    s: float,
    s_t_inv: float | None = None,
    s_w_inv: float | None = None,
    as_foi: bool = False,
) -> np.ndarray | FieldOfInterest:
    """Convex displacement interpolation.

    Defined as:
    .. math::
        u(s) = (1 - s) u_0(t^{-1}(x, s)) + s u_1(w^{-1}(x, 1 - s))
    """
    if s_t_inv is None:
        s_t_inv = s

    if s_w_inv is None:
        s_w_inv = s

    values = (1 - s) * u0(ts_inverse(points, s_t_inv)) + s * u1(
        ws_inverse(points, 1 - s_w_inv)
    )
    if as_foi:
        return FieldOfInterest(
            ParameterSet(s=s),
            u0.mesh,
            values,
            name=f"CDI s={s:.2f}",
        )
    return values


def mccann_displacement_interpolation(
    points: np.ndarray,
    u0: FieldOfInterest,
    domain_shape: tuple,
    m_inverse: Mapping,
    determinant: Callable[[np.ndarray, tuple, float], np.ndarray],
    s: float,
) -> np.ndarray:
    return u0(m_inverse(points, s)) * determinant(points, domain_shape, s).reshape(
        -1, 1
    )
