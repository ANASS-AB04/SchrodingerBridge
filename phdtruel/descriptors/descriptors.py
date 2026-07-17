from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, Self, Type

import numpy as np

from phdtruel.fields import FieldOfInterest
from phdtruel.fields.rejection_sampling import rejection_sampling


class FieldDescriptor(ABC):
    """An object that represents the main features of a field.

    It must be :
    - Usable in a weighted average to construct an average descriptor

    It will be used :
    - with a second instance to build a mapping
    """

    @classmethod
    def factory_constructor(cls, field: FieldOfInterest, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __add__(self, other: Self | Literal[0]):
        raise NotImplementedError

    def __radd__(self, other: Self):
        return self.__add__(other)

    @abstractmethod
    def __mul__(self, other: float):
        raise NotImplementedError

    def __rmul__(self, other: float):
        return self.__mul__(other)


DescriptorCreationMethod = Callable[[FieldOfInterest], FieldDescriptor]


class FieldDescriptorFactory:
    def __init__(
        self,
        descriptor_class: Type[FieldDescriptor],
        **kwargs,
    ):
        self._descriptor_class = descriptor_class
        self._kwargs = kwargs

    def __call__(self, field: FieldOfInterest) -> FieldDescriptor:
        return self._descriptor_class.factory_constructor(field, **self._kwargs)


class FOIasDescriptor(FieldOfInterest, FieldDescriptor):
    @classmethod
    def factory_constructor(cls, field: FieldOfInterest, **kwargs):
        return FOIasDescriptor(field)

    def __init__(self, field: FieldOfInterest):
        super().__init__(
            field.parameter,
            field.mesh,
            field.values,
            field.name,
            field.is_interpolated,
        )


class FOIandMappingasDescriptor(FieldOfInterest, FieldDescriptor):
    @classmethod
    def factory_constructor(cls, field: FieldOfInterest, **kwargs):
        return FOIandMappingasDescriptor(field, kwargs.get("mapping"))

    def __init__(self, field: FieldOfInterest, mapping: Any):
        super().__init__(
            field.parameter,
            field.mesh,
            field.values,
            field.name,
            field.is_interpolated,
        )
        self._mapping = mapping


class PointCloudasDescriptor(FieldDescriptor):
    @classmethod
    def factory_constructor(cls, field: FieldOfInterest, **kwargs):
        return PointCloudasDescriptor(field, **kwargs)

    def __init__(
        self,
        normalized_foi: FieldOfInterest,
        n_points: int = 1000,
        support_points: np.ndarray | None = None,
        p: int = 2,
    ):
        super().__init__()
        self._field = normalized_foi
        if support_points is None:
            support_points = normalized_foi.mesh.points

        values = normalized_foi(support_points)

        self._point_cloud = rejection_sampling(n_points, values, support_points, p)

    def __add__(self, other: FieldDescriptor | Literal[0]):
        raise NotImplementedError

    def __mul__(self, other: float):
        raise NotImplementedError


class ContourPointsDescriptor(FieldDescriptor):
    @classmethod
    def factory_constructor(cls, field: FieldOfInterest, **kwargs):
        pass

    def __init__(self):
        pass
