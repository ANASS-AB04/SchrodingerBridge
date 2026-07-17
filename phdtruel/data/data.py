"""Data loading utilities and protocols for field and dynamic loading.

This module provides:
- Protocol definitions for field and dynamic loaders
- Utility functions for parameter validation and time extraction
- Helper functions for generating synthetic test fields
"""

import logging
from typing import Literal, Protocol, cast, runtime_checkable

import h5py
import numpy as np
from scipy.stats import multivariate_normal

from phdtruel.fields import DynamicOfInterest, FieldOfInterest, Gaussian
from phdtruel.fields.meshes import Mesh
from phdtruel.fields.parameters import ParameterSet

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols for loaders
# ============================================================================


@runtime_checkable
class FieldLoader(Protocol):
    """Protocol for callable objects that load fields.

    Field loaders are callables that take a ParameterSet and return a FieldOfInterest.
    They can be functions or classes with __call__ method.
    """

    def __call__(self, parameter: ParameterSet, **kwargs) -> FieldOfInterest:
        """Load a field for the given parameters.

        Args:
            parameter: Parameter set defining the field to load
            **kwargs: Additional loader-specific arguments

        Returns:
            The loaded field
        """
        ...


@runtime_checkable
class DynamicLoader(Protocol):
    """Protocol for callable objects that load dynamics (time series of fields).

    Dynamic loaders are callables that take a ParameterSet and return a DynamicOfInterest.
    """

    def __call__(self, parameter: ParameterSet, **kwargs) -> DynamicOfInterest:
        """Load a dynamic (time series) for the given parameters.

        Args:
            parameter: Parameter set defining the dynamic to load
            **kwargs: Additional loader-specific arguments

        Returns:
            The loaded dynamic
        """
        ...


# ============================================================================
# Utility functions
# ============================================================================


def extract_time_from_parameter(
    parameter: ParameterSet,
) -> tuple[float | Literal["average", "last"] | None, ParameterSet]:
    """Extract time value from a parameter set.

    Args:
        parameter: Parameter set potentially containing a 't' parameter

    Returns:
        Tuple of (time_value, parameter_without_time)
        - time_value can be a float, "average", "last", or None if not present
        - parameter_without_time is the input with 't' removed
    """
    if "t" not in parameter.parameters:
        return None, parameter

    value = parameter.t
    param_without_t = parameter.without("t")

    if isinstance(value, str):
        if value in ("average", "last"):
            return cast(Literal["average", "last"], value), param_without_t
        # Try to parse as float
        try:
            return float(value), param_without_t
        except ValueError:
            raise ValueError(f"Invalid time parameter value: {value}")

    return float(value), param_without_t


# ============================================================================
# Synthetic field generation utilities
# ============================================================================


def generate_random_gaussian_fields(
    grid_points: np.ndarray, num_gaussians: int
) -> np.ndarray:
    """
    Generates a mixture of Gaussians for a given set of points.

    Parameters:
    num_gaussians (int): The number of Gaussians to include in the mixture.
    grid_points (numpy array): The points at which to evaluate the mixture of Gaussians.

    Returns:
    numpy array: The values of the mixture of Gaussians at the given points.
    """
    # Randomly generate the parameters of the Gaussians
    means = np.random.uniform(
        np.min(grid_points, axis=0),
        np.max(grid_points, axis=0),
        size=(num_gaussians, grid_points.shape[1]),
    )
    covariances = (
        np.array(
            [
                np.eye(grid_points.shape[1]) * np.random.uniform(0.1, 1.0)
                for _ in range(num_gaussians)
            ]
        )
        * 0.1
        * (np.max(grid_points[:, 0]) - np.min(grid_points[:, 0]))
    )
    weights = np.random.uniform(0.8, 1.0, size=num_gaussians)
    weights /= np.sum(weights)

    values = np.zeros((grid_points.shape[0], 1))
    for i in range(num_gaussians):
        g = Gaussian(means[i], covariances[i])
        values += g.sample(grid_points) * weights[i]
    return values


def construct_random_gmm_fields_of_interest(
    mesh: Mesh, num_fields: int, num_gaussians: int
):
    field_values = [
        generate_random_gaussian_fields(mesh.points, num_gaussians)
        for _ in range(num_fields)
    ]
    return [FieldOfInterest(ParameterSet(), mesh, values) for values in field_values]


def sillage(
    points,
    y_position=0.0,
    wavelength=5.0,
    amplitude=1.0,
    curvature=0.0,
):
    """Construct a field with gaussians onlong an oscillating mean line."""
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    x_line = np.linspace(x_min + np.abs(x_min * 0.1), x_max + np.abs(x_max * 0.1), 200)
    mean_line = np.sin(x_line * 2 * np.pi / wavelength)
    mean_line = mean_line + curvature * x_line**2 / (10**2)
    mean_line = mean_line + y_position

    out = np.zeros((len(points), 1))
    for mu_x, mu_y in zip(x_line, mean_line):
        mv = multivariate_normal(mean=[mu_x, mu_y], cov=np.eye(2))
        out += amplitude * mv.pdf(points).reshape(-1, 1)

    out = out / np.max(out)
    return out


# ============================================================================
# HDF5 utilities
# ============================================================================


def hdf5_tree(filename: str) -> str:
    """Return a tree-like representation of an HDF5 file structure.

    Args:
        filename: Path to the HDF5 file

    Returns:
        String with tree representation showing:
        - Groups as name/
        - Datasets as name [dtype=..., shape=...]
        - Attributes as @attr_name [dtype=..., shape=...]

    Note: Does not print actual data content, only structure and metadata.
    """

    def format_attr(name, val):
        try:
            arr = np.asarray(val)
            dtype = str(arr.dtype)
            shape = arr.shape
        except Exception:
            dtype = type(val).__name__
            try:
                shape = np.shape(val)
            except Exception:
                shape = ()
        return f"@{name} [dtype={dtype}, shape={shape}]"

    def format_dataset(ds):
        try:
            dtype = str(ds.dtype)
        except Exception:
            dtype = "unknown"
        try:
            shape = tuple(ds.shape)
        except Exception:
            shape = ()
        return f"[dtype={dtype}, shape={shape}]"

    def is_group(obj):
        return isinstance(obj, h5py.Group)

    def is_dataset(obj):
        return isinstance(obj, h5py.Dataset)

    def walk(obj, name, prefix, is_last):
        """
        Render a single node (group or dataset) and its attributes/children.
        `prefix` is the accumulated indent string.
        `is_last` controls whether to draw └── or ├── for this node.
        """
        lines = []
        connector = "└── " if is_last else "├── "

        # Header line
        if is_group(obj):
            lines.append(prefix + connector + f"{name}/")
        else:
            # dataset
            lines.append(prefix + connector + f"{name} {format_dataset(obj)}")

        # Child prefix for attributes/children
        child_prefix = prefix + ("    " if is_last else "│   ")

        # Attributes (always available for Group and Dataset)
        attr_keys = sorted(list(obj.attrs.keys()))
        # Children only exist for groups
        child_names = sorted(list(obj.keys())) if is_group(obj) else []

        total = len(attr_keys) + len(child_names)

        # Print attributes first (they are considered children in ordering)
        for i, akey in enumerate(attr_keys):
            a_is_last = i == total - 1
            lines.append(
                child_prefix
                + ("└── " if a_is_last else "├── ")
                + format_attr(akey, obj.attrs[akey])
            )

        # Then print group/dataset children (for groups only)
        for i, child_name in enumerate(child_names):
            overall_index = len(attr_keys) + i
            child_is_last = overall_index == total - 1
            try:
                child_obj = obj[child_name]
            except Exception:
                # broken link or inaccessible
                lines.append(
                    child_prefix
                    + ("└── " if child_is_last else "├── ")
                    + f"{child_name} -> <unable to open>"
                )
                continue

            # Recurse for child (works whether child_obj is group or dataset)
            lines.extend(walk(child_obj, child_name, child_prefix, child_is_last))

        return lines

    # Open file and start from root
    try:
        with h5py.File(filename, "r") as f:
            root = f["/"]
            tree_lines = [filename]

            # Root attributes and children (same ordering logic as walk)
            root_attr_keys = sorted(list(root.attrs.keys()))
            root_child_names = sorted(list(root.keys()))
            total_root = len(root_attr_keys) + len(root_child_names)

            for i, akey in enumerate(root_attr_keys):
                is_last_attr = i == total_root - 1
                tree_lines.append(
                    ("└── " if is_last_attr else "├── ")
                    + format_attr(akey, root.attrs[akey])
                )

            for i, child_name in enumerate(root_child_names):
                overall_index = len(root_attr_keys) + i
                child_is_last = overall_index == total_root - 1
                try:
                    child_obj = root[child_name]
                except Exception:
                    tree_lines.append(
                        ("└── " if child_is_last else "├── ")
                        + f"{child_name} -> <unable to open>"
                    )
                    continue
                tree_lines.extend(walk(child_obj, child_name, "", child_is_last))

            return "\n".join(tree_lines)
    except Exception as e:
        raise RuntimeError(f"Could not open HDF5 file '{filename}': {e}")


def generate_artificial_sillages(num):
    pass


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hdf5_tree.py <hdf5_file>")
        sys.exit(1)

    filename = sys.argv[1]
    tree_str = hdf5_tree(filename)
    print(tree_str)


if __name__ == "__main__":
    main()
