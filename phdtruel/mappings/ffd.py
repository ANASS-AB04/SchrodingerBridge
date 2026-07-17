from math import comb as math_comb
from typing import NamedTuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array as JaxArray
from jax.scipy.special import gammaln
from jaxtyping import Float, Int

from phdtruel.backends import get_backend

# Custom type for arrays that can be either NumPy or JAX arrays
npLikeArray = Union[np.ndarray, JaxArray]


class FFDParameters(NamedTuple):
    box_origin: Float[npLikeArray, "dim"]
    box_length: Float[npLikeArray, "dim"]
    n_control_points: Int[npLikeArray, "dim"]

    @property
    def dim(self) -> int:
        return len(self.box_origin)

    def is_inside(
        self,
        points: Float[npLikeArray, "n_points dim"],
    ) -> Float[npLikeArray, " n_points"]:
        """For each point, return True if it is inside the FFD box defined by box_origin and box_length."""

        nx = get_backend(points, self.box_origin, self.box_length)
        lower_bounds = self.box_origin
        upper_bounds = self.box_origin + self.box_length

        inside_mask = nx.all(
            (points >= lower_bounds) & (points <= upper_bounds), axis=-1
        )
        return inside_mask


class BenchmarkResults(NamedTuple):
    time_of_first_call: float
    time_of_second_call_with_same_args: float
    time_of_second_call_with_different_args: float
    times_of_subsequent_calls_same_args: list
    times_of_subsequent_calls_different_args: list
    times_of_jacobian_det_calls: list


def binom(x, y):
    nx = get_backend(x, y)
    if nx is jnp:
        return nx.exp(gammaln(x + 1) - gammaln(y + 1) - gammaln(x - y + 1))
    else:
        # For numpy, use scipy.special.gammaln
        from scipy.special import gammaln as scipy_gammaln

        return nx.exp(
            scipy_gammaln(x + 1) - scipy_gammaln(y + 1) - scipy_gammaln(x - y + 1)
        )


def get_control_points(
    box_origin: Float[npLikeArray, " dim"],
    box_length: Float[npLikeArray, " dim"],
    n_control_points: Int[npLikeArray, " dim"],
) -> Float[npLikeArray, "*n_control_points dim"]:
    nx = get_backend(box_origin, box_length, n_control_points)
    axes = []
    for length, size in zip(box_length, n_control_points):
        axes.append(nx.linspace(0, length, size))

    axes_coords = nx.meshgrid(*axes, indexing="ij")

    control_points = box_origin + nx.stack(axes_coords, axis=-1)
    return control_points


def get_control_points_from_ffd_params(
    params: FFDParameters,
) -> Float[npLikeArray, "*n_control_points dim"]:
    return get_control_points(
        box_origin=params.box_origin,
        box_length=params.box_length,
        n_control_points=params.n_control_points,
    )


def map_to_unit_cube(
    points: Float[npLikeArray, "n_points dim"],
    params: FFDParameters,
) -> Float[npLikeArray, "n_points dim"]:
    """
    Map points from the FFD bounding box to the unit cube [0,1]^dim.

    Args:
        points: Points in the original coordinate system
        params: FFD parameters containing box_origin and box_length

    Returns:
        Points mapped to the unit cube [0,1]^dim
    """
    _ = get_backend(points, params.box_origin, params.box_length)
    # Translate points to have origin at (0,0,...)
    translated_points = points - params.box_origin

    # Scale to unit cube by dividing by box_length
    unit_cube_points = translated_points / params.box_length

    return unit_cube_points


def map_from_unit_cube(
    points: Float[npLikeArray, "n_points dim"],
    params: FFDParameters,
) -> Float[npLikeArray, "n_points dim"]:
    """
    Map points from the unit cube [0,1]^dim back to the FFD bounding box.

    Args:
        points: Points in the unit cube coordinate system
        params: FFD parameters containing box_origin and box_length

    Returns:
        Points mapped back to the original coordinate system
    """
    _ = get_backend(points, params.box_origin, params.box_length)
    # Scale from unit cube by multiplying by box_length
    scaled_points = points * params.box_length

    # Translate to original position by adding box_origin
    original_points = scaled_points + params.box_origin

    return original_points


def _bernstein_basis_1d(coord, degree, nx):
    """
    coord: shape (N,) array of points in [0,1]
    degree: number of control points along this axis (== l, m or n)
    nx: either numpy or jax.numpy
    returns: B, shape (degree, N), where
      B[i,p] = C(degree-1, i) * (1 - coord[p])**(degree-1-i) * coord[p]**i
    """

    # 1) k = [0,1,…,degree-1]
    k = nx.arange(degree)  # (degree,)
    # 2) build the binomial coefficients C(degree-1,i)
    #    we do this once in Python – degree is typically small
    binoms = nx.array(
        [math_comb(degree - 1, int(i)) for i in range(degree)], dtype=coord.dtype
    )  # (degree,)
    # 3) broadcast the powers out to shape (degree, N)
    #    (degree,1) ** (1,N)  →  (degree, N)
    p1 = nx.power((1.0 - coord)[None, :], (degree - 1 - k)[:, None])
    p2 = nx.power(coord[None, :], k[:, None])
    return binoms[:, None] * p1 * p2  # (degree, N)


def ffd_mapping(
    points: Float[npLikeArray, "n_points 3"],
    normalized_control_points_displacement: Float[npLikeArray, "l m n 3"],
    params: FFDParameters,
) -> Float[npLikeArray, "n_points 3"]:
    """
    3D FFD with no Python loops (apart from tiny binomial‐array builds).
    Supports both numpy and jax.numpy backends interchangeably.
    """
    # pick your numpy‐style backend
    nx = get_backend(
        points,
        normalized_control_points_displacement,
        params.box_origin,
        params.box_length,
    )

    # 1) map input points into the unit‐cube [0,1]^3
    u = map_to_unit_cube(points, params)  # shape (N,3)
    x, y, z = u[:, 0], u[:, 1], u[:, 2]  # each is (N,)

    # 2) how many control‐points along each axis?
    l, m, n = normalized_control_points_displacement.shape[:-1]  # noqa E741
    _ = x.shape[0]

    # 3) build the three Bernstein‐matrices, shapes (l,N), (m,N), (n,N)
    Bx = _bernstein_basis_1d(x, l, nx)
    By = _bernstein_basis_1d(y, m, nx)
    Bz = _bernstein_basis_1d(z, n, nx)

    # 4) form the full tensor‐product weights, shape (l,m,n,N):
    #      W[i,j,k,p] = Bx[i,p]*By[j,p]*Bz[k,p]
    W = (
        Bx[:, None, None, :]  # (l,1,1,N)
        * By[None, :, None, :]  # (1,m,1,N)
        * Bz[None, None, :, :]
    )  # (1,1,n,N)

    # 5) now contract W against your 4-D control-point‐displacement array of shape (l,m,n,3)
    #    to get per‐point, per‐axis total shift (still “normalized” units):
    #    result has shape (N,3)
    shift_normalized = nx.tensordot(
        W, normalized_control_points_displacement, axes=([0, 1, 2], [0, 1, 2])
    )
    # 6) apply that shift in the unit‐cube, then map back out:
    u_deformed = u + shift_normalized  # (N,3)
    return map_from_unit_cube(u_deformed, params)


def ffd_mapping_2d(
    points: Float[npLikeArray, "n_points 2"],
    control_points_displacement: Float[npLikeArray, "nx ny 2"],
    params: FFDParameters,
) -> Float[npLikeArray, "n_points 2"]:
    """
    Apply a 2D Free-Form Deformation (FFD) to a set of points.

    Args:
        points: A 2D array of shape (n_points, 2) containing the coordinates of the points to be deformed.
        control_points_displacement: A 3D array of shape (n_x, n_y, 2) representing the displacement of control points.
        params: 2D FFD parameters containing box_origin, box_length, and n_control_points

    Returns:
        deformed_points: A 2D array of shape (n_points, 2) containing the deformed coordinates of the input points.
    """
    nx = get_backend(points)
    # Convert 2D points to 3D by appending a zero z-coordinate
    points_3d = nx.concatenate([points, nx.zeros((points.shape[0], 1))], axis=-1)

    # Create 3D FFD parameters from 2D parameters
    box_origin_3d = nx.concatenate([params.box_origin, nx.array([0.0])])
    box_length_3d = nx.concatenate(
        [params.box_length, nx.array([1.0])]
    )  # Unit length in z direction
    n_control_points_3d = nx.concatenate(
        [params.n_control_points, nx.array([1])]
    )  # Single layer in z

    params_3d = FFDParameters(
        box_origin=box_origin_3d,
        box_length=box_length_3d,
        n_control_points=n_control_points_3d,
    )

    # The control_points_displacement is 3D (n_x, n_y, 2), so we need to pad it to match the shape (n_x, n_y, 1, 3) expected by ffd_mapping
    # First expand to (n_x, n_y, 1, 2) by adding a singleton dimension
    control_points_displacement_expanded = nx.expand_dims(
        control_points_displacement, axis=2
    )
    # Then pad the last dimension from 2 to 3: (n_x, n_y, 1, 2) -> (n_x, n_y, 1, 3)
    control_points_displacement_3d = nx.pad(
        control_points_displacement_expanded, ((0, 0), (0, 0), (0, 0), (0, 1))
    )

    # Apply the 3D FFD to the 3D points using the 3D parameters
    normalized_cp_displacements = control_points_displacement_3d / params_3d.box_length
    deformed_points_3d = ffd_mapping(points_3d, normalized_cp_displacements, params_3d)

    # Discard the z-coordinate to get back to 2D
    deformed_points = deformed_points_3d[:, :2]

    return deformed_points


def jacobian_matrix_2d(
    points: Float[npLikeArray, "n_points 2"],
    control_points_displacement: Float[npLikeArray, "nx ny 2"],
    params: FFDParameters,
) -> Float[npLikeArray, "n_points 2 2"]:
    """Compute the Jacobian matrix of the 2D FFD mapping at each point.

    Args:
        points: Points where to evaluate the Jacobian, shape (n_points, 2)
        control_points_displacement: Control point displacements, shape (nx, ny, 2)
        params: 2D FFD parameters

    Returns:
        Jacobian matrices at each point, shape (n_points, 2, 2)
    """
    nx = get_backend(points)

    # Create a function that maps a single point
    def single_point_mapping(point):
        return ffd_mapping_2d(point[None, :], control_points_displacement, params)[0]

    # Compute the Jacobian matrices using JAX's automatic differentiation
    if nx is jnp:
        return jax.vmap(jax.jacobian(single_point_mapping))(points)
    else:
        # For numpy, we need to use numerical differentiation or raise an error
        raise NotImplementedError(
            "Jacobian computation with numpy arrays not implemented. Use JAX arrays for automatic differentiation."
        )


def jacobian_det_2d(
    points: Float[npLikeArray, "n_points 2"],
    control_points_displacement: Float[npLikeArray, "nx ny 2"],
    params: FFDParameters,
) -> Float[npLikeArray, " n_points"]:
    """Compute the determinant of the Jacobian matrix at each point.

    Args:
        points: Points where to evaluate the Jacobian determinant, shape (n_points, 2)
        control_points_displacement: Control point displacements, shape (nx, ny, 2)
        params: 2D FFD parameters

    Returns:
        Determinant of the Jacobian matrix at each point, shape (n_points,)
    """
    _ = get_backend(
        points, control_points_displacement, params.box_origin, params.box_length
    )
    J = jacobian_matrix_2d(points, control_points_displacement, params)
    return J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]


def piecewise_jacobian_matrix_2d(
    points: Float[npLikeArray, "n_points 2"],
    all_control_points_displacement: tuple[Float[npLikeArray, "nx ny 2"], ...],
    all_params: tuple[FFDParameters, ...],
) -> Float[npLikeArray, "n_points 2 2"]:
    """Compute the Jacobian matrix of the piecewise 2D FFD mapping at each point.

    For points inside an FFD region, computes the Jacobian of that region's FFD.
    For points outside all regions, returns the identity matrix (Jacobian of identity mapping).
    If a point belongs to multiple boxes, the first matching box is used.

    Args:
        points: Points where to evaluate the Jacobian, shape (n_points, 2)
        all_control_points_displacement: Tuple of control point displacements for each FFD region.
        all_params: Tuple of FFD parameters defining each region's box.

    Returns:
        Jacobian matrices at each point, shape (n_points, 2, 2)
    """
    nx = get_backend(points)
    n_regions = len(all_params)

    if nx is np:
        raise NotImplementedError(
            "Jacobian computation with numpy arrays not implemented. Use JAX arrays for automatic differentiation."
        )

    # JAX version - fully traceable using jax.lax.switch

    # Build a tuple of functions, one per region + identity for "no match"
    def make_jacobian_branch(region_idx):
        """Create a branch function for Jacobian of a specific region."""

        def branch_fn(point):
            def single_point_mapping(p):
                return ffd_mapping_2d(
                    p[None, :],
                    all_control_points_displacement[region_idx],
                    all_params[region_idx],
                )[0]

            return jax.jacobian(single_point_mapping)(point)

        return branch_fn

    # Identity Jacobian for points outside all regions
    def identity_jacobian(point):
        return jnp.eye(2)

    branches = tuple(make_jacobian_branch(i) for i in range(n_regions)) + (
        identity_jacobian,
    )

    def process_single_point(point):
        """Process a single point: find first matching region and compute its Jacobian."""
        # Check which regions contain this point
        inside_flags = jnp.array(
            [params.is_inside(point[None, :])[0] for params in all_params]
        )

        # Find first matching region index, or n_regions if none match
        any_match = jnp.any(inside_flags)
        first_match_idx = jnp.argmax(inside_flags)
        # If no match, use index n_regions (identity branch)
        region_idx = jnp.where(any_match, first_match_idx, n_regions)

        return jax.lax.switch(region_idx, branches, point)

    # vmap over all points
    return jax.vmap(process_single_point)(points)


def piecewise_jacobian_det_2d(
    points: Float[npLikeArray, "n_points 2"],
    all_control_points_displacement: tuple[Float[npLikeArray, "nx ny 2"], ...],
    all_params: tuple[FFDParameters, ...],
) -> Float[npLikeArray, " n_points"]:
    """Compute the determinant of the Jacobian matrix of the piecewise FFD at each point.

    For points inside an FFD region, computes the Jacobian determinant of that region's FFD.
    For points outside all regions, returns 1.0 (determinant of identity matrix).
    If a point belongs to multiple boxes, the first matching box is used.

    Args:
        points: Points where to evaluate the Jacobian determinant, shape (n_points, 2)
        all_control_points_displacement: Tuple of control point displacements for each FFD region.
        all_params: Tuple of FFD parameters defining each region's box.

    Returns:
        Determinant of the Jacobian matrix at each point, shape (n_points,)
    """
    J = piecewise_jacobian_matrix_2d(
        points, all_control_points_displacement, all_params
    )
    return J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]


def piecewise_ffd_mapping_2d(
    points: Float[npLikeArray, "n_points 2"],
    all_control_points_displacement: tuple[Float[npLikeArray, "nx ny 2"], ...],
    all_params: tuple[FFDParameters, ...],
) -> Float[npLikeArray, "n_points 2"]:
    """
    Apply piecewise 2D FFD mapping where each region has its own FFD parameters.

    Fully JAX-compatible version. For each point, the function checks which FFD box
    it belongs to and applies the corresponding FFD mapping. Points outside all boxes
    remain unchanged. If a point belongs to multiple boxes, the first matching box is used.

    Args:
        points: Array of shape (n_points, 2) containing the coordinates of the points.
        all_control_points_displacement: Tuple of control point displacements for each FFD region.
        all_params: Tuple of FFD parameters defining each region's box.

    Returns:
        deformed_points: Array of shape (n_points, 2) with deformed coordinates.
    """
    nx = get_backend(points)
    n_regions = len(all_params)

    if nx is np:
        # NumPy version with Python loops
        deformed_points = points.copy()
        already_mapped = np.zeros(points.shape[0], dtype=bool)

        for control_displacement, params in zip(
            all_control_points_displacement, all_params
        ):
            inside_mask = params.is_inside(points) & ~already_mapped

            if np.any(inside_mask):
                points_to_map = points[inside_mask]
                mapped_points = ffd_mapping_2d(
                    points_to_map, control_displacement, params
                )
                deformed_points[inside_mask] = mapped_points
                already_mapped[inside_mask] = True

        return deformed_points

    else:
        # JAX version - fully traceable using jax.lax.switch

        # Build a tuple of functions, one per region + identity for "no match"
        def make_ffd_branch(region_idx):
            """Create a branch function for a specific region."""

            def branch_fn(point):
                return ffd_mapping_2d(
                    point[None, :],
                    all_control_points_displacement[region_idx],
                    all_params[region_idx],
                )[0]

            return branch_fn

        branches = tuple(make_ffd_branch(i) for i in range(n_regions)) + (
            lambda point: point,
        )

        def process_single_point(point):
            """Process a single point: find first matching region and apply its FFD."""
            # Check which regions contain this point
            inside_flags = jnp.array(
                [params.is_inside(point[None, :])[0] for params in all_params]
            )

            # Find first matching region index, or n_regions if none match
            # argmax returns first True, but returns 0 if all False
            # So we need to handle the "no match" case
            any_match = jnp.any(inside_flags)
            first_match_idx = jnp.argmax(inside_flags)
            # If no match, use index n_regions (identity branch)
            region_idx = jnp.where(any_match, first_match_idx, n_regions)

            return jax.lax.switch(region_idx, branches, point)

        # vmap over all points
        return jax.vmap(process_single_point)(points)
