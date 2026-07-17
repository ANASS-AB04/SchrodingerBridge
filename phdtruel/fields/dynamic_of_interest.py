from __future__ import annotations

import numpy as np

from phdtruel.fields.field_of_interest import (
    FieldOfInterest,
    derive_foi,
)
from phdtruel.fields.meshes import DomainBounds, RegularGrid
from phdtruel.fields.parameters import ParameterSet


class DynamicOfInterest:
    @classmethod
    def from_nascar_dynamic(
        cls,
        nascar_dynamic,
        field_name,
        parameter: ParameterSet | None = None,
    ):
        if parameter is None:
            parameter = ParameterSet()

        nx = nascar_dynamic.dimensions.nx
        ny = nascar_dynamic.dimensions.ny

        mesh = RegularGrid([nascar_dynamic.x_axis, nascar_dynamic.y_axis])
        timesteps = nascar_dynamic.t
        space_time_values = getattr(nascar_dynamic, field_name)

        fois = []
        for k, t in enumerate(timesteps):
            # in nascar dynamic the shape is not indexed like meshgrid "ij"
            values = space_time_values[k, :]
            values = values.reshape(ny, nx, order="C").T
            values = values.reshape(-1, 1, order="C")
            param_with_time = ParameterSet(**parameter.as_dict() | {"t": k})
            field = FieldOfInterest(
                param_with_time,
                mesh,
                values,
                name=field_name,
            )
            fois.append(field)
        return cls(parameter, fois, timesteps)

    def __init__(
        self,
        parameter: ParameterSet | None,
        fields_of_interest: list[FieldOfInterest],
        timesteps: np.ndarray | None,
        name: str = "",
    ):
        if parameter is None:
            parameter = ParameterSet()
        self._parameter = parameter
        self._fields_of_interest = fields_of_interest
        self._name = name
        if timesteps is None:
            self._timesteps = np.array([f.parameter for f in fields_of_interest])
        else:
            self._timesteps = timesteps

    def truncate(self, bounds: DomainBounds) -> "DynamicOfInterest":
        new_fields = [f.truncate(bounds) for f in self._fields_of_interest]
        return DynamicOfInterest(
            self._parameter,
            new_fields,
            self._timesteps,
            self._name,
        )

    def as_array(self) -> np.ndarray:
        n_fields = len(self._fields_of_interest)
        n_points = self._fields_of_interest[0].values.shape[0]
        array = np.zeros((n_points, n_fields))
        for i, f in enumerate(self._fields_of_interest):
            array[:, i] = f.values.ravel()
        return array

    @property
    def parameter(self) -> ParameterSet:
        return self._parameter

    @property
    def fields(self) -> list[FieldOfInterest]:
        return self._fields_of_interest

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    def with_name(self, name: str) -> "DynamicOfInterest":
        return DynamicOfInterest(
            self._parameter,
            self._fields_of_interest,
            self._timesteps,
            name,
        )

    def with_parameter(self, parameter: ParameterSet) -> "DynamicOfInterest":
        return DynamicOfInterest(
            parameter,
            self._fields_of_interest,
            self._timesteps,
            self._name,
        )

    @property
    def all_parameters(self) -> list[ParameterSet]:
        return [f.parameter for f in self._fields_of_interest]

    @property
    def timesteps(self) -> np.ndarray:
        return self._timesteps

    def __getitem__(self, item: int) -> FieldOfInterest:
        return self._fields_of_interest[item]

    def __len__(self) -> int:
        return len(self._fields_of_interest)

    def __add__(self, other: DynamicOfInterest | float) -> DynamicOfInterest:
        raise NotImplementedError

    def __sub__(self, other: DynamicOfInterest | float) -> DynamicOfInterest:
        raise NotImplementedError

    def __mul__(self, other: float) -> DynamicOfInterest:
        raise NotImplementedError

    def __call__(self, points: np.ndarray) -> DynamicOfInterest:
        new_fields = []
        for f in self._fields_of_interest:
            v = f.eval(points)
            n_f = derive_foi(f, v)
            new_fields.append(n_f)
        return DynamicOfInterest(
            self._parameter,
            new_fields,
            self._timesteps,
            self._name,
        )

    def as_list_of_foi(self) -> list[FieldOfInterest]:
        return self._fields_of_interest

    def to_space_time_foi(self) -> FieldOfInterest:
        raise NotImplementedError

    def get_subset(self, sl: slice) -> DynamicOfInterest:
        raise NotImplementedError


def compute_temporal_mean(dynamic: DynamicOfInterest) -> FieldOfInterest:
    values = np.zeros_like(dynamic[0].values)
    for f in dynamic:
        values += f.values
    values /= len(dynamic)
    mean = derive_foi(dynamic[0], values)
    return mean


#
# @dataclass
# class DynamicOfInterest(CachableItem):
#     @property
#     def nbytes(self):
#         return sum((f.nbytes for f in self._fields))
#
#     @property
#     def cache_dir(self):
#         return self._fields.cache_dir
#
#     @classmethod
#     def from_arrays(
#         cls,
#         timesteps: np.ndarray,
#         points: np.ndarray,
#         space_time_values: np.ndarray,
#         field_shape: tuple,
#         parameter: ParameterSet = None,
#         allow_extrapolation: bool = True,
#         extrapolation_method: PossibleExtrapolationMethods = "rbf",
#         name: str = "",
#     ):
#         if parameter is None:
#             parameter = ParameterSet()
#
#         fois = CachableOrderedDict()
#         for k, t in enumerate(timesteps):
#             values = space_time_values[k, :]
#             param_with_time = parameter.copy()
#             param_with_time.add_parameter(name="t", value=k, state=False)
#             field = FieldOfInterest(
#                 points,
#                 values,
#                 field_shape,
#                 allow_extrapolation=allow_extrapolation,
#                 extrapolation_method=extrapolation_method,
#                 parameter=param_with_time,
#                 name=name,
#             )
#             fois.append(field)
#         return cls(timesteps, fois, parameter)
#
#     @classmethod
#     def from_time_fields(cls, fields: CachableOrderedDict):
#         timesteps = [param.t for param in fields.parameters]
#
#         parameter = fields.parameters[0].copy()
#         del parameter["t"]
#
#         return cls(timesteps, fields, parameter)
#
#     def __init__(
#         self,
#         timesteps: np.ndarray[float],
#         fields_of_interests: list[FieldOfInterest] | CachableOrderedDict,
#         parameter: ParameterSet = None,
#     ):
#         if parameter is None:
#             parameter = ParameterSet()
#         if not isinstance(fields_of_interests, CachableOrderedDict):
#             fields_of_interests = CachableOrderedDict(*fields_of_interests)
#
#         self.timesteps = timesteps
#         self._fields: CachableOrderedDict = fields_of_interests
#         self._parameter = parameter
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}({self.parameter} - t={self.timesteps}) "
#
#     @property
#     def parameter(self):
#         return self._parameter
#
#     @property
#     def fields(self):
#         return self._fields
#
#     def to_space_time_foi(self):
#         f0: FieldOfInterest = self.fields[0]
#
#         shape = f0.shape
#         n_ts = len(self.timesteps)
#         axes = [self.timesteps] + f0.cartesian_axes
#         space_time_shape = (n_ts, *shape)
#         dim = len(space_time_shape)
#         space_time_values = np.zeros(space_time_shape)
#
#         for ts in range(n_ts):
#             space_time_values[ts] = self.fields[ts].values.reshape(shape)
#
#         space_time_points = np.zeros((*space_time_shape, dim))
#         for k in range(dim):
#             _axis = list(range(0, k)) + list(range(k + 1, len(space_time_shape)))
#             space_time_points[..., k] = np.expand_dims(axes[k], axis=_axis)
#
#         space_time_foi = FieldOfInterest(
#             points=space_time_points,
#             values=space_time_values,
#             shape=space_time_shape,
#             parameter=self.parameter,
#         )
#         return space_time_foi
#
#     def get_subset(self, sl: slice):
#         new_fields = [self.fields[i] for i in range(len(self))[sl]]
#         return DynamicOfInterest(self.timesteps[sl], new_fields, self.parameter)
#
#     def __call__(self, points: np.ndarray) -> np.ndarray[float]:
#         return self.sample(points)
#
#     def sample(self, points: np.ndarray):
#         # TODO return array shape (Np,Nt)
#         out = np.zeros((points.shape[0], self.timesteps.shape[0]))
#         for i in range(len(self)):
#             out[:, i] = self[i](points).ravel()
#         return out
#
#     @property
#     def name(self) -> str:
#         return self.fields[0].name
#
#     def __getitem__(self, item):
#         return self._fields[item]
#
#     def __len__(self):
#         return len(self._fields)
#
#     def __add__(self, other):
#         if len(self) != len(other):
#             raise ValueError("Cannot add fields of different lengths")
#         new_fields = [f1 + f2 for f1, f2 in zip(self.fields, other.fields)]
#         return DynamicOfInterest(self.timesteps, new_fields, self.parameter)
#
#     def __sub__(self, other):
#         if len(self) != len(other):
#             raise ValueError("Cannot add fields of different lengths")
#         new_fields = [f1 - f2 for f1, f2 in zip(self.fields, other.fields)]
#         return DynamicOfInterest(self.timesteps, new_fields, self.parameter)
