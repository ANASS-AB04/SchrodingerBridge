from typing import Iterable

import numpy as np

from phdtruel.fields import FieldOfInterest
from phdtruel.fields.meshes import Mesh
from phdtruel.fields.parameters import ParameterSet


def linear_interpolation(fields: Iterable[FieldOfInterest], mesh: Mesh):
    fields = list(fields)

    interpolation = np.zeros_like(fields[0](mesh.points))
    for field in fields:
        interpolation += field(mesh.points)
    interpolation = interpolation / len(fields)
    return FieldOfInterest(ParameterSet(), mesh, interpolation)
