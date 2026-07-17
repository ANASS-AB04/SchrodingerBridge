import logging
from typing import Callable, Literal, Sequence, cast, override

import numpy as np

from phdtruel.descriptors.descriptors import (
    DescriptorCreationMethod,
    FieldDescriptorFactory,
)
from phdtruel.fields import FieldOfInterest
from phdtruel.fields.meshes import Mesh
from phdtruel.fields.parameters import ParameterSet
from phdtruel.interpolations.interpolations import DirectInterpolator
from phdtruel.mappings.two_fields_mappings import MappingCreationMethod, MappingFactory
from phdtruel.ot1d import cdi_1d, direct_interpolation_1d, ot1d_mappings
from phdtruel.sensors.sensors import SensorMethod

logger = logging.getLogger(__name__)


class AlignStripesInterpolator(DirectInterpolator):
    def __init__(
        self,
        fields: Sequence[FieldOfInterest],
        sensor_method: SensorMethod,
        descriptor_factory: FieldDescriptorFactory | DescriptorCreationMethod,
        mapping_factory: MappingFactory | MappingCreationMethod | None = None,
        name: str | None = None,
        density_transform_1d: Callable[..., np.ndarray] | None = None,
        do_direct: Literal["cdi", "on_01", "on_10"] = "cdi",
    ):
        if mapping_factory is not None:
            logger.warning("No mapping is used in AlignStripesInterpolator")

        super().__init__(
            fields,
            sensor_method,
            descriptor_factory,
            mapping_factory,
            name=name,
        )
        self._density_transform = density_transform_1d
        self._do_direct = do_direct

    @override
    def _predict(
        self, normalized_parameter: np.ndarray, mesh: Mesh, **kwargs
    ) -> np.ndarray:
        s = float(normalized_parameter[0])

        if self.param_dimension != 1:
            raise ValueError(
                f"The CDI pred with parameters only works for 1 parameter. (here {self.param_dimension})"
            )

        return self.interpolate(points=mesh.points, s=s)

    @override
    def interpolate(
        self,
        points: np.ndarray,
        s: float,
        **kwargs,
    ) -> np.ndarray:
        return self._align_stripes(points, s).values

    def _align_stripes(
        self,
        points: np.ndarray,
        s: float,
        do_direct: Literal["cdi", "on_01", "on_10"] | None = None,
    ) -> FieldOfInterest:
        del points
        if do_direct is None:
            do_direct = self._do_direct
        source = cast(FieldOfInterest, self._descriptors[0])
        target = cast(FieldOfInterest, self._descriptors[1])
        if self._density_transform is None:

            def density_transform(
                x: np.ndarray,
                v: np.ndarray,
                const: float = 1e-6,
            ) -> np.ndarray:
                del x
                return v + const
        else:
            density_transform = self._density_transform
        shape = source.mesh.shape
        interpolated_values = np.zeros(shape)
        v0 = source.values.reshape(shape)
        v1 = target.values.reshape(shape)
        x0 = source.mesh.y
        x1 = target.mesh.y
        for i in range(len(source.mesh.x)):
            a_raw = v0[i, :]
            b_raw = v1[i, :]

            a = density_transform(x0, a_raw, const=1e-6)
            b = density_transform(x1, b_raw, const=1e-6)

            T, T_inv = ot1d_mappings(x0, x1, a, b)
            if do_direct == "cdi":
                interp_1d = cdi_1d(x0, x1, a_raw, b_raw, T, T_inv, s)
            elif do_direct == "on_01":
                interp_1d = direct_interpolation_1d(x0, a_raw, T_inv, s)
            elif do_direct == "on_10":
                interp_1d = direct_interpolation_1d(x1, b_raw, T, 1 - s)
            interpolated_values[i, :] = interp_1d
        return FieldOfInterest(
            ParameterSet(s=s),
            source.mesh,
            interpolated_values.reshape(-1, 1),
            name=f"Stripes s={s:.2f} {do_direct}",
        )

    @override
    def interpolation_0_1(self, points: np.ndarray, s: float):
        return self._align_stripes(points, s, do_direct="on_01").values

    @override
    def interpolation_1_0(self, points: np.ndarray, s: float):
        return self._align_stripes(points, s, do_direct="on_10").values
