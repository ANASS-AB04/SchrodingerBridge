"""Shared types for cost_functional module.

This module defines common types used across the functional cost function
implementation, providing JAX-compatible data structures.

Types:
    MeshJaxed: JAX-compatible mesh representation with points and axes.

Example:
    >>> from phdtruel.mappings.cost_functional.types import MeshJaxed
    >>> import jax.numpy as jnp
    >>>
    >>> # Create a simple 2D grid mesh
    >>> x = jnp.linspace(0, 1, 50)
    >>> y = jnp.linspace(0, 1, 50)
    >>> X, Y = jnp.meshgrid(x, y, indexing='ij')
    >>> points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    >>>
    >>> mesh = MeshJaxed(points=points, axes=(x, y))
"""

from typing import NamedTuple

from jax import Array
from jaxtyping import Float


class MeshJaxed(NamedTuple):
    """JAX-compatible mesh representation.

    This NamedTuple stores mesh data in a format suitable for JAX operations,
    particularly for interpolation and field evaluation.

    Attributes:
        points: Array of mesh point coordinates with shape (n_points, dim).
            For a 2D grid of size (nx, ny), this has shape (nx*ny, 2).
        axes: Tuple of 1D arrays defining grid axes. For a regular grid,
            this contains the x and y axis arrays used for interpolation.

    Example:
        >>> # Create mesh for a 10x10 regular grid
        >>> x = jnp.linspace(0, 1, 10)
        >>> y = jnp.linspace(0, 1, 10)
        >>> X, Y = jnp.meshgrid(x, y, indexing='ij')
        >>> points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
        >>> mesh = MeshJaxed(points=points, axes=(x, y))
        >>> mesh.points.shape
        (100, 2)
    """

    points: Float[Array, "n_points dim"]
    axes: tuple[Float[Array, " n_along_ax"], ...]
