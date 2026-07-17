from abc import ABC, abstractmethod
from typing import Self, Callable, Type

import h5py
from hickle import hickle

from phdtruel.descriptors.descriptors import FieldDescriptor
from phdtruel.fields import FieldOfInterest
from phdtruel.mappings.mappings import (
    Mapping,
    logger,
    ComposedMapping,
    identity_mapping,
)


class TwoFieldMappingModel(ABC):
    """Abstract base class for mappings between two fields."""

    def __init__(self):
        self._name: str = self.__class__.__name__

    @abstractmethod
    def fit(self, u0: FieldOfInterest, u1: FieldOfInterest) -> Self:
        raise NotImplementedError

    @abstractmethod
    def get_ts_mapping_function(self) -> Mapping:
        raise NotImplementedError

    @abstractmethod
    def get_ws_mapping_function(self) -> Mapping:
        raise NotImplementedError

    @abstractmethod
    def copy(self) -> Self:
        raise NotImplementedError

    @abstractmethod
    def meta_parameters(self) -> dict:
        """Useful for saving the mapping and hash."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    def with_name(self, name: str) -> Self:
        """Set the name of the mapping."""
        self.name = name
        return self

    @classmethod
    def load_h5(cls, g: h5py.Group) -> Self:
        obj = hickle.load(g)
        return obj

    def save_h5(self, g: h5py.Group):
        hickle.dump(self, g, mode="w", compression="gzip")


class ComposedTwoFieldMappingModel(TwoFieldMappingModel):
    """A mapping that composes two two-field mapping models."""

    def __init__(self, map: TwoFieldMappingModel, compose_with: TwoFieldMappingModel):
        super().__init__()
        self.map = map
        self.compose_with = compose_with

    def fit(self, u0: FieldOfInterest, u1: FieldOfInterest) -> Self:
        logger.warning("ComposedTwoFieldMappingModel.fit does nothing")
        return self

    def get_ts_mapping_function(self) -> Mapping:
        """Return a composed mapping function for target space."""
        first_map = self.map.get_ts_mapping_function()
        second_map = self.compose_with.get_ts_mapping_function()
        return ComposedMapping(first_map, second_map)

    def get_ws_mapping_function(self) -> Mapping:
        """Return a composed mapping function for world space."""
        first_map = self.map.get_ws_mapping_function()
        second_map = self.compose_with.get_ws_mapping_function()
        return ComposedMapping(first_map, second_map)

    def copy(self) -> "ComposedTwoFieldMappingModel":
        """Return a copy of this composed mapping."""
        return ComposedTwoFieldMappingModel(self.map.copy(), self.compose_with.copy())

    def meta_parameters(self) -> dict:
        """Return metadata parameters from both composed mappings."""
        return {
            "first_map": self.map.meta_parameters(),
            "second_map": self.compose_with.meta_parameters(),
        }


class IdentityTwoFieldMappingModel(TwoFieldMappingModel):
    """A mapping that returns the identity (no transformation)."""

    def __init__(self):
        super().__init__()

    def fit(self, u0: FieldOfInterest, u1: FieldOfInterest) -> Self:
        """Fit does nothing for identity mapping."""
        return self

    def get_ts_mapping_function(self) -> Mapping:
        """Return the identity mapping function."""
        return identity_mapping

    def get_ws_mapping_function(self) -> Mapping:
        """Return the identity mapping function."""
        return identity_mapping

    def copy(self) -> "IdentityTwoFieldMappingModel":
        """Return a copy of this mapping."""
        return IdentityTwoFieldMappingModel()

    def meta_parameters(self) -> dict:
        """Return metadata parameters (empty for identity mapping)."""
        return {}


MappingCreationMethod = Callable[
    [FieldDescriptor, FieldDescriptor], TwoFieldMappingModel
]


class MappingFactory:
    def __init__(self, mapping_class: Type[TwoFieldMappingModel], **kwargs):
        self._mapping_class = mapping_class
        self._kwargs = kwargs

    def __call__(
        self,
        source_descriptor: FieldDescriptor,
        target_descriptor: FieldDescriptor,
        **kwargs,
    ) -> TwoFieldMappingModel:
        return self._mapping_class.factory_constructor(
            source_descriptor, target_descriptor, **self._kwargs
        )
