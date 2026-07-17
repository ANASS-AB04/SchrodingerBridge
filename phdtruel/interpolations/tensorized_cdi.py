import itertools
import math
from collections.abc import Sequence
from typing import Type, override

import numpy as np

from phdtruel.descriptors.descriptors import (
    FieldDescriptorFactory,
    DescriptorCreationMethod,
)
from phdtruel.fields import FieldOfInterest
from phdtruel.fields.meshes import Mesh
from phdtruel.fields.parameters import ParameterSet
from phdtruel.interpolations.interpolations import (
    FieldInterpolator,
    TwoFieldsInterpolator,
)
from phdtruel.mappings.two_fields_mappings import MappingCreationMethod, MappingFactory
from phdtruel.sensors.sensors import SensorMethod


class TensorisedInterpolator(FieldInterpolator):
    def __init__(
        self,
        fields: Sequence[FieldOfInterest],
        sensor_method: SensorMethod,
        descriptor_factory: FieldDescriptorFactory | DescriptorCreationMethod,
        mapping_factory: MappingFactory | MappingCreationMethod | None,
        two_field_interpolation_method: Type[TwoFieldsInterpolator],
        normalization_ranges: dict[str, tuple[float, float]] | None = None,
        name: str = "Tensorised CDI",
    ):
        super().__init__(
            fields,
            sensor_method,
            descriptor_factory,
            mapping_factory,
            normalization_ranges,
            name=name,
        )

        self._two_field_interpolation_method = two_field_interpolation_method

    @override
    def _predict(
        self, normalized_parameter: np.ndarray, mesh: Mesh, **kwargs
    ) -> np.ndarray:
        axes_permutations = permute_axes(self.param_dimension)
        final_interpolations = []
        for axes_order in axes_permutations:
            fields: dict[int, list[FieldOfInterest]] = {
                self.param_dimension: list(self._fields)
            }
            for dim in range(self.param_dimension, 0, -1):
                indices = list(range(len(fields[dim])))
                current_ax = axes_order[0]
                transport_plan = compute_transport_plan(indices, current_ax)
                s = get_s_value(
                    list(axes_order),
                    dim,
                    list(normalized_parameter.ravel()),
                )

                interpolated_fields = []
                for i, j in transport_plan:
                    f0 = fields[dim][i]
                    f1 = fields[dim][j]

                    pair_interpolation_method = self._two_field_interpolation_method(
                        fields=[f0, f1],
                        sensor_method=self._sensor_method,
                        descriptor_factory=self._descriptor_factory,
                        mapping_factory=self._mapping_factory,
                    )

                    values = pair_interpolation_method.interpolate(
                        points=mesh.points, s=s
                    )
                    fi = FieldOfInterest(
                        ParameterSet(s=s),
                        mesh,
                        values,
                        name="interpolated",
                        interpolated=True,
                    )
                    interpolated_fields.append(fi)

                fields[dim - 1] = interpolated_fields
                axes_order = reduce_dim_for_axes_order(axes_order)
            final_interpolations.append(fields[0][0])

        interpolated_values = np.zeros((mesh.points.shape[0], 1))
        for f in final_interpolations:
            interpolated_values += f(mesh.points)
        interpolated_values /= len(final_interpolations)
        return interpolated_values


def _get_bit(value: int, bit_index: int) -> int:
    return value & (1 << bit_index)


def _set_bit(value: int, bit_index: int) -> int:
    return value | (1 << bit_index)


def _print_bin(iterable) -> None:
    print([bin(v) for v in iterable])


def permute_axes(dimension: int) -> list[tuple[int, ...]]:
    if dimension <= 0:
        raise ValueError("Dimension must be greater than zero")

    return list(itertools.permutations(range(dimension), r=dimension))


def indices_for_cube(dimension: int) -> list[int]:
    """
    Generate indices for the vertices of a hypercube in the given dimension.

    The indices are constructed in binary form. See examples below.

    Parameters:
    dimension (int): The dimension of the hypercube.

    Returns:
    List[int]: A list of indices representing the vertices of the hypercube.

    Examples of binary labeling in hypercubes :

    1D (Line with 2 vertices):
    0 --- 1

    2D (Square with 4 vertices):
    00 --- 01
     |      |
    10 --- 11

    3D (Cube with 8 vertices):
        100 ------- 101
       / |          / |
     000 ------- 001  |
      |  |        |   |
      |  110 ------|- 111
      | /         |  /
     010 ------- 011

    Note: The function currently contains a placeholder implementation.

    Returns:
    list: A list containing the indices for the vertices of the hypercube.
    """
    return list(range(2**dimension))


def get_indices_to_move_along_ax(vertices_indices: list[int], ax: int) -> list[int]:
    check_dimensions(ax, vertices_indices)
    return list(filter(lambda x: _get_bit(x, ax) == 0, vertices_indices))


def move_indices_along_ax(movable_indices: list[int], ax: int) -> list[int]:
    return [(i + _set_bit(0b0, ax)) for i in movable_indices]


def compute_transport_plan(indices: list[int], ax: int):
    starts = get_indices_to_move_along_ax(indices, ax)
    ends = move_indices_along_ax(starts, ax)
    return [(s, e) for s, e in zip(starts, ends)]


def check_dimensions(ax, indices):
    max_dim = math.floor(math.log2(max(indices + [1])) + 1.0)
    if math.log2(len(indices)) != max_dim:
        raise ValueError("Not enough indices to define the hypercube")
    if ax >= max_dim:
        raise ValueError(
            f"Axis {ax} exceeds maximum dimension {max_dim} for given indices"
        )


def reduce_dim_for_axes_order(axes_order):
    return [ax - min(axes_order[1:]) for ax in axes_order[1:]]


def get_s_value(
    axes_orders: list[int],
    current_dim: int,
    s_vector: list[float],
) -> float:
    start_dim = max(axes_orders) + 1
    index = start_dim - current_dim
    ax = axes_orders[index]
    return s_vector[ax]
