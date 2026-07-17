from typing import override, Literal

import numpy as np

from phdtruel.descriptors.descriptors import FieldDescriptor
from phdtruel.fields import Gaussian, FieldOfInterest


class GaussianDescriptor(FieldDescriptor, Gaussian):
    @classmethod
    def factory_constructor(cls, field: FieldOfInterest, **kwargs):
        return cls.from_field(field.mesh.points, field.values)

    @override
    def __add__(self, other: FieldDescriptor | Literal[0]):
        try:
            is_zero = other == 0.0
        except TypeError:
            is_zero = False

        if is_zero:
            other_mu = np.zeros_like(self.mu)
            other_sigma = np.ones_like(self.sigma)
        elif isinstance(other, GaussianDescriptor):
            other_mu = other.mu
            other_sigma = other.sigma
        else:
            msg = f"Cannot add GaussianDescriptor with {type(other)!r}"
            raise TypeError(msg)
        return GaussianDescriptor(mu=self.mu + other_mu, sigma=self.sigma + other_sigma)

    @override
    def __mul__(self, other: float):
        return GaussianDescriptor(mu=self.mu * other, sigma=self.sigma * other)
