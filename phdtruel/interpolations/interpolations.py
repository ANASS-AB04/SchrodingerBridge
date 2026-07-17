from abc import abstractmethod, ABC
from typing import Sequence, cast, override

import numpy as np

from phdtruel.descriptors.descriptors import (
    FieldDescriptorFactory,
    DescriptorCreationMethod,
    FieldDescriptor,
)
from phdtruel.fields import FieldOfInterest
from phdtruel.fields.meshes import Mesh
from phdtruel.fields.parameters import (
    ParameterSet,
    ParameterNormalizer,
    DistanceNormalizer,
)
from phdtruel.mappings.two_fields_mappings import (
    TwoFieldMappingModel,
    MappingCreationMethod,
    MappingFactory,
)
from phdtruel.sensors.sensors import SensorMethod


class FieldInterpolator(ABC):
    def __init__(
        self,
        fields: Sequence[FieldOfInterest],
        sensor_method: SensorMethod,
        descriptor_factory: FieldDescriptorFactory | DescriptorCreationMethod,
        mapping_factory: MappingFactory | MappingCreationMethod | None,
        normalization_ranges: dict[str, tuple[float, float]] | None = None,
        name: str = "FieldInterpolator",
    ):
        self._fields = fields
        self._sensor_method = sensor_method
        self._descriptor_factory = descriptor_factory
        self._mapping_factory = mapping_factory
        self._name = name

        self._do_input_checks()
        self._spacial_dimension = fields[0].mesh.dimension
        self._param_dimension = len(fields[0].parameter.parameters)

        self._parameter_normalizer = ParameterNormalizer(
            self.parameters, normalization_ranges
        )

    def _do_input_checks(self):
        dimensions = [field.mesh.dimension for field in self._fields]
        if dimensions.count(dimensions[0]) != len(dimensions):
            raise ValueError("All fields must have the same dimension")

        n_params = [len(p.parameters) for p in self.parameters]
        if len(n_params) != n_params.count(n_params[0]):
            raise ValueError("Inconsistent number of parameters in fields ")

    @property
    def param_dimension(self):
        return self._param_dimension

    @property
    def spacial_dimension(self):
        return self._spacial_dimension

    @property
    def parameters(self):
        return [f.parameter for f in self._fields]

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def __call__(
        self,
        target_parameter: ParameterSet | np.ndarray,
        mesh: Mesh | None = None,
        **kwargs,
    ) -> FieldOfInterest:
        if mesh is None:
            mesh = self._fields[0].mesh

        if isinstance(target_parameter, ParameterSet):
            normalized_param = self._parameter_normalizer.normalize(target_parameter)
        else:
            normalized_param = target_parameter
        normalized_param = normalized_param.reshape(1, -1)

        values = self._predict(normalized_param, mesh, **kwargs)

        parameter_unormalized = self._parameter_normalizer.unormalize(normalized_param)
        if isinstance(parameter_unormalized, list):
            if len(parameter_unormalized) != 1:
                raise ValueError("Expected exactly one parameter set for prediction")
            parameter = cast(ParameterSet, parameter_unormalized[0])
        else:
            parameter = cast(ParameterSet, parameter_unormalized)
        field = FieldOfInterest(
            parameter,
            mesh,
            values,
            name=self._fields[0].name,
            interpolated=True,
        )
        return field

    @abstractmethod
    def _predict(
        self, normalized_parameter: np.ndarray, mesh: Mesh, **kwargs
    ) -> np.ndarray:
        raise NotImplementedError


class TwoFieldsInterpolator(FieldInterpolator, ABC):
    def __init__(
        self,
        fields: Sequence[FieldOfInterest],
        sensor_method: SensorMethod,
        descriptor_factory: FieldDescriptorFactory | DescriptorCreationMethod,
        mapping_factory: MappingFactory | MappingCreationMethod | None,
        name: str | None = None,
    ):
        interpolator_name = "TwoFieldsInterpolator" if name is None else name
        super().__init__(
            fields,
            sensor_method,
            descriptor_factory,
            mapping_factory,
            name=interpolator_name,
        )

        if len(fields) != 2:
            raise ValueError("Only works between 2 fields")

        self._parameter_normalizer = DistanceNormalizer(self.parameters)

        f0, f1 = fields
        self._transformed_fields: list[FieldOfInterest] = [
            sensor_method(f0),
            sensor_method(f1),
        ]

        tf0, tf1 = self._transformed_fields
        self._descriptors: list[FieldDescriptor] = [
            descriptor_factory(tf0),
            descriptor_factory(tf1),
        ]

        self._mappings = None

    @abstractmethod
    def interpolate(self, *, points: np.ndarray, s: float, **kwargs) -> np.ndarray:
        raise NotImplementedError

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


class DirectInterpolator(TwoFieldsInterpolator):
    def __init__(
        self,
        fields: Sequence[FieldOfInterest],
        sensor_method: SensorMethod,
        descriptor_factory: FieldDescriptorFactory | DescriptorCreationMethod,
        mapping_factory: MappingFactory | MappingCreationMethod | None = None,
        name: str | None = None,
    ):
        super().__init__(
            fields, sensor_method, descriptor_factory, mapping_factory, name
        )

        d0, d1 = self._descriptors
        if mapping_factory is not None:
            self._mappings: list[TwoFieldMappingModel] = [
                mapping_factory(d0, d1),
                mapping_factory(d1, d0),
            ]

    @override
    def interpolate(self, points: np.ndarray, s: float, **kwargs) -> np.ndarray:
        return self.interpolation_0_1(points, s)

    def interpolation_0_1(self, points: np.ndarray, s: float):
        map_t = self._mappings[0]
        t_inverse = map_t.get_ts_mapping_function()
        u0 = self._fields[0]
        return u0(t_inverse(points, s))

    def interpolation_1_0(self, points: np.ndarray, s: float):
        map_w = self._mappings[1]
        w_inverse = map_w.get_ws_mapping_function()
        u1 = self._fields[1]
        return u1(w_inverse(points, 1 - s))


class ApproximatedDirectInterpolator(DirectInterpolator):
    @override
    def interpolation_0_1(self, points: np.ndarray, s: float):
        map_w = self._mappings[1].get_ws_mapping_function()
        u0 = self._fields[0]
        return u0(map_w(points, s))

    @override
    def interpolation_1_0(self, points: np.ndarray, s: float):
        map_t = self._mappings[0].get_ts_mapping_function()
        u1 = self._fields[1]
        return u1(map_t(points, 1 - s))
