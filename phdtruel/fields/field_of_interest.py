from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Literal, Union, cast, overload

import h5py
import numpy as np
from scipy.interpolate import (
    LinearNDInterpolator,
    RBFInterpolator,
    RegularGridInterpolator,
)

from phdtruel.fields.meshes import DomainBounds, Mesh, RegularGrid, load_mesh_from_hdf5
from phdtruel.fields.parameters import ParameterSet

if TYPE_CHECKING:
    from phdtruel.mappings.mappings import Mapping as Mapping


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

InterpolationMethod = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
ExtrapolationMethod = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
PossibleInterpolationMethods = Literal["grid", "linear", "rbf"]
PossibleExtrapolationMethods = Literal["constant", "rbf", "aslam"]


PossibleInterpolators = Union[RegularGridInterpolator, RBFInterpolator]


class InterpolatorConfigError(Exception):
    pass


def extract_axes(points, shape):
    """
    Generalize the extraction of axes from an n-dimensional points array.

    Parameters:
    points -- A points array of shape (num_points, num_dims)
    shape -- The target multidimensional shape for reshaping

    Returns:
    A list containing axes values for each dimension.
    """
    num_dims = points.shape[1]  # Number of axes (dimensions in the data)
    axes = []

    for dim in range(num_dims):
        # Reshape the points corresponding to the dim-th coordinate
        reshaped_dim = points[:, dim].reshape(shape)

        # Extract the axis by slicing: keep full elements only along the current dimension (dim)
        slices = tuple(0 if d != dim else slice(None) for d in range(len(shape)))
        axes.append(reshaped_dim[slices])

    return axes


class FieldOfInterest:
    @classmethod
    def from_hdf5(cls, f: h5py.Group | h5py.File) -> "FieldOfInterest":
        """
        Load the FieldOfInterest from an HDF5 file.
        """
        mesh_group = cast(h5py.Group, f["mesh"])
        parameter_group = cast(h5py.Group, f["parameter"])
        mesh = load_mesh_from_hdf5(mesh_group)
        parameter = ParameterSet.load_from_hdf5(parameter_group)
        values_dataset = cast(h5py.Dataset, f["values"])
        values = np.array(values_dataset)
        name = cast(str, f.attrs["name"])
        foi = cls(
            parameter=parameter,
            mesh=mesh,
            values=values,
            name=name,
            interpolated=False,
        )
        return foi

    def save_in_hdf5(self, f: h5py.Group | h5py.File, name: str | None = None):
        """
        Save the FieldOfInterest in an HDF5 file.
        """
        if name is None:
            name = self.name

        self.mesh.save_in_hdf5(f.create_group("mesh"))
        self.parameter.save_to_hdf5(f.create_group("parameter"))
        f.attrs["name"] = name
        f.create_dataset("values", data=self.values)

    def __init__(
        self,
        parameter: ParameterSet | None,
        mesh: Mesh,
        values: np.ndarray,
        name: str = "",
        interpolated: bool = True,
        fill_value: float | None = None,
    ):
        if parameter is None:
            parameter = ParameterSet()
        self._parameter = parameter
        self._mesh = mesh
        self._name = name

        self._is_interpolated = False
        self._interpolator: PossibleInterpolators | LinearNDInterpolator | None = None
        self._extrapolator: PossibleInterpolators | LinearNDInterpolator | None = None
        self._fill_value = fill_value
        self._values: np.ndarray | None = values

        if interpolated:
            self.setup_interpolation()

    def setup_interpolation(
        self,
        interpolator: PossibleInterpolators | None = None,
        extrapolator: PossibleInterpolators | Literal["constant", "rbf"] = "constant",
    ):
        if interpolator is None:
            self._setup_interpolator_based_on_mesh()
        else:
            self._interpolator = interpolator

        if isinstance(extrapolator, str):
            self._setup_extrapolator(
                method=cast(Literal["constant", "rbf"], extrapolator)
            )
        else:
            self._extrapolator = extrapolator

        self._is_interpolated = True
        self._values = None

    def _setup_interpolator_based_on_mesh(self):
        mesh = self._mesh
        if isinstance(mesh, RegularGrid):
            interpolator = RegularGridInterpolator(
                points=mesh.axes,
                values=self.values.reshape(mesh.shape),
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
        else:
            interpolator = LinearNDInterpolator(
                points=mesh.points,
                values=self.values,
                fill_value=np.nan,
            )

        self._interpolator = interpolator

    def _setup_extrapolator(self, method: Literal["constant", "rbf"]):
        mesh = self._mesh
        if method == "constant":
            constant_fill_value = (
                self._fill_value if self._fill_value is not None else 0.0
            )

            if isinstance(mesh, RegularGrid):
                extrapolator = RegularGridInterpolator(
                    points=mesh.axes,
                    values=self.values.reshape(mesh.shape),
                    method="linear",
                    bounds_error=False,
                    fill_value=constant_fill_value,
                )
            else:
                extrapolator = LinearNDInterpolator(
                    points=mesh.points,
                    values=self.values,
                    fill_value=constant_fill_value,
                )
        elif method == "rbf":
            extrapolator = RBFInterpolator(
                y=mesh.points,
                d=self.values,
                neighbors=100,
            )
        else:
            raise InterpolatorConfigError(f"Unsupported mesh type: {type(mesh)}")

        self._extrapolator = extrapolator

    @property
    def is_interpolated(self):
        return self._is_interpolated

    @property
    def parameter(self):
        return self._parameter

    @property
    def mesh(self):
        return self._mesh

    @overload
    def eval(
        self,
        points: np.ndarray,
        as_foi: Literal[True],
        with_name: str | None = None,
    ) -> FieldOfInterest: ...

    @overload
    def eval(
        self,
        points: np.ndarray,
        as_foi: Literal[False] = False,
        with_name: str | None = None,
    ) -> np.ndarray: ...

    def eval(
        self,
        points: np.ndarray,
        as_foi: bool = False,
        with_name: str | None = None,
    ) -> np.ndarray | FieldOfInterest:
        if not self._is_interpolated:
            raise InterpolatorConfigError("Interpolation has not been setup")
        if self._interpolator is None or self._extrapolator is None:
            raise InterpolatorConfigError("Interpolation has not been setup")

        values = self._interpolator(points).flatten()
        nan_indices = np.isnan(values)  # Nan's means interpolation failed
        if not np.not_equal(True, nan_indices).all():
            points_where_nan = points[nan_indices]

            extrapolated_values = self._extrapolator(points_where_nan).flatten()
            values[nan_indices] = extrapolated_values

        if as_foi:
            if with_name is None:
                with_name = self.name + "_eval"
            return FieldOfInterest(self.parameter, self.mesh, values, with_name)
        return values.reshape(-1, 1)

    def eval_mapped(
        self,
        mapping: Mapping,
        on: np.ndarray,
        with_s: float | None = None,
        as_foi: bool = False,
        with_name: str | None = None,
    ) -> np.ndarray | FieldOfInterest:
        if with_s is None:
            points = mapping(on)
        else:
            points = mapping(on, with_s)
        if as_foi and with_name is None:
            mapping_name = getattr(mapping, "name", None)
            if mapping_name:
                with_name = self._compose_name(mapping_name)
        return self.eval(points, as_foi=as_foi, with_name=with_name)

    def _compose_name(self, mapping_name: str) -> str:
        name = self.name
        if mapping_name.startswith("$") and mapping_name.endswith("$"):
            mapping_name = mapping_name[1:-1]
        if name.startswith("$") and name.endswith("$") and len(name) >= 2:
            inner = name[1:-1]
            return f"${inner} \\circ {mapping_name}$"
        return f"{name} comp {mapping_name}"

    @property
    def values(self) -> np.ndarray:
        if not self._is_interpolated:
            if self._values is None:
                raise ValueError("Values are not available")
            return self._values
        else:
            result = self.eval(self.mesh.points)
            if isinstance(result, FieldOfInterest):
                raise ValueError("Unexpected FieldOfInterest result")
            return result

    def truncate(self, bounds: DomainBounds) -> "FieldOfInterest":
        new_mesh = self.mesh.truncate(bounds)
        new_points_ref = self.mesh.truncate_points(bounds)
        new_values = self.mesh.truncate_values(bounds, self.values)

        if not np.allclose(new_points_ref, new_mesh.points):
            raise ValueError("Truncation of values is different of mesh truncation")

        new_mesh = RegularGrid.from_points(
            new_points_ref, shape=cast(RegularGrid, new_mesh).shape
        )  # TODO solve bug ?
        return FieldOfInterest(
            self.parameter,
            new_mesh,
            new_values,
            self._name,
            interpolated=self.is_interpolated,
        )

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    def with_name(self, new_name: str) -> FieldOfInterest:
        return FieldOfInterest(
            self.parameter,
            self.mesh,
            self.values,
            new_name,
            interpolated=self.is_interpolated,
        )

    def __call__(self, points: np.ndarray) -> np.ndarray:
        result = self.eval(points)
        if isinstance(result, FieldOfInterest):
            raise ValueError("Unexpected FieldOfInterest result")
        return result

    def __add__(self, other: FieldOfInterest | float):
        to_add = (
            other(self.mesh.points) if isinstance(other, FieldOfInterest) else other
        )

        return derive_foi(self, self.values + to_add)

    def __sub__(self, other: FieldOfInterest | float):
        to_subtract = (
            other(self.mesh.points) if isinstance(other, FieldOfInterest) else other
        )

        return derive_foi(self, self.values - to_subtract)

    def __mul__(self, other: float):
        if not isinstance(other, (int, float)):
            raise ValueError("Multiplication only supported with scalar values")
        return derive_foi(self, self.values * other)

    def __len__(self):
        return len(self.values)

    def __str__(self):
        return f"FieldOfInterest(name={self.name}, parameter={self.parameter}, mesh={self.mesh})"

    def __repr__(self):
        return (
            "FOI("
            f"name={self.name}, parameter={self.parameter}, mesh={self.mesh}, "
            f"values_shape={self.values.shape}, "
            f"number_nan={np.count_nonzero(np.isnan(self.values))} "
            f"interp={self.is_interpolated}"
            ")"
        )

    def __format__(self, format_spec):
        return self.__str__()


class VectorFieldOfInterest:
    @classmethod
    def from_hdf5(cls, f: h5py.Group | h5py.File) -> VectorFieldOfInterest:
        fields = [
            FieldOfInterest.from_hdf5(cast(h5py.Group, f[key])) for key in f.keys()
        ]
        return cls(fields, name=cast(str, f.attrs["name"]))

    @classmethod
    def from_array(
        cls,
        param: ParameterSet | None,
        mesh: Mesh,
        values: np.ndarray,
        name: str = "",
    ) -> VectorFieldOfInterest:
        if values.shape[0] != len(mesh):
            raise ValueError(
                f"Values shape {values.shape} does not match mesh length {len(mesh)}"
            )

        fields = [
            FieldOfInterest(
                parameter=param, mesh=mesh, values=values[:, i], name=f"{name}_{i}"
            )
            for i in range(values.shape[1])
        ]
        return cls(fields, name=name)

    def __init__(
        self,
        fields: list[FieldOfInterest],
        name: str = "",
    ):
        self._name = name
        self._fields_of_interest = fields

    @property
    def name(self):
        return self._name

    @property
    def fields(self):
        return self._fields_of_interest

    def save_in_hdf5(self, f: h5py.Group | h5py.File, name: str | None = None):
        if name is None:
            name = self.name
        for field in self._fields_of_interest:
            field.save_in_hdf5(f.create_group(field.name))
        f.attrs["name"] = name

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        return np.array([field.eval(points) for field in self._fields_of_interest])

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return self.evaluate(points)

    def __add__(self, other: VectorFieldOfInterest | float):
        if isinstance(other, VectorFieldOfInterest):
            if len(self.fields) != len(other.fields):
                raise ValueError("Cannot add vector fields of different sizes")
            return VectorFieldOfInterest(
                [f1 + f2 for f1, f2 in zip(self._fields_of_interest, other.fields)]
            )
        return VectorFieldOfInterest(
            [field + other for field in self._fields_of_interest]
        )

    def __sub__(self, other: VectorFieldOfInterest | float):
        if isinstance(other, VectorFieldOfInterest):
            if len(self.fields) != len(other.fields):
                raise ValueError("Cannot subtract vector fields of different sizes")
            return VectorFieldOfInterest(
                [f1 - f2 for f1, f2 in zip(self._fields_of_interest, other.fields)]
            )
        return VectorFieldOfInterest(
            [field - other for field in self._fields_of_interest]
        )

    def __mul__(self, other: float):
        return VectorFieldOfInterest(
            [field * other for field in self._fields_of_interest]
        )

    def __len__(self):
        return len(self._fields_of_interest[0])

    def norm(self) -> FieldOfInterest:
        values = self.fields[0].values ** 2
        for f in self.fields[1:]:
            values += f.values**2

        values = np.sqrt(values)
        return FieldOfInterest(
            None, self.fields[0].mesh, values, name=f"Magnitude of {self.name}"
        )


def derive_foi(foi: FieldOfInterest, new_values: np.ndarray) -> FieldOfInterest:
    return FieldOfInterest(
        parameter=foi.parameter,
        mesh=foi.mesh,
        values=new_values,
        name=foi.name,
        interpolated=foi.is_interpolated,
    )


def get_domain_bounds(domain_points: np.ndarray) -> DomainBounds:
    x_min, x_max = np.min(domain_points[:, 0]), np.max(domain_points[:, 0])
    y_min, y_max = np.min(domain_points[:, 1]), np.max(domain_points[:, 1])
    return DomainBounds.by_axis((x_min, x_max), (y_min, y_max))


def get_smallest_domain_bounds(domain_bounds: list[DomainBounds]) -> DomainBounds:
    x_min = max([b.minima[0] for b in domain_bounds])
    x_max = min([b.maxima[0] for b in domain_bounds])
    y_min = max([b.minima[1] for b in domain_bounds])
    y_max = min([b.maxima[1] for b in domain_bounds])
    return DomainBounds.by_axis((x_min, x_max), (y_min, y_max))


# def get_indices_to_crop_subdomain_xy(
#     x: np.ndarray, y: np.ndarray, domain_bound: DomainBounds
# ) -> TruncationIndices:
#     i_min = np.where(x > domain_bound.x_min)[1].min()
#     j_min = np.where(y > domain_bound.y_min)[0].min()
#     i_max = np.where(x < domain_bound.x_max)[1].max()
#     j_max = np.where(y < domain_bound.y_max)[0].max()
#     return TruncationIndices(i_min, i_max, j_min, j_max, original_shape=x.shape)
#
#
# def get_indices_to_crop_subdomain_points(
#     domain_points: np.ndarray, domain_bound: DomainBounds, domain_shape: tuple
# ) -> TruncationIndices:
#     x = domain_points[:, 0].reshape(domain_shape)
#     y = domain_points[:, 1].reshape(domain_shape)
#     return get_indices_to_crop_subdomain_xy(x, y, domain_bound)


def _sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"
