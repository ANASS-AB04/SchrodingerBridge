from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Callable, NamedTuple, Self, override

import numpy as np
from scipy.linalg import fractional_matrix_power

from phdtruel.descriptors.descriptors import FieldDescriptor
from phdtruel.fields import FieldOfInterest, Gaussian
from phdtruel.mappings.mappings import Mapping
from phdtruel.mappings.two_fields_mappings import TwoFieldMappingModel
from phdtruel.sensors.sensors import to_localized_density_sensor

if TYPE_CHECKING:
    import jax


class GotPlotData(NamedTuple):
    g0: Gaussian
    g1: Gaussian
    u0_hat: FieldOfInterest
    u1_hat: FieldOfInterest
    sensor_name: str


class GaussianOTMapping(TwoFieldMappingModel):
    """Maps a Gaussian to another Gaussian using optimal transport.

    The mapping is known analytically.
    """

    _name: str = "GOT"

    @property
    def meta_parameters(self) -> dict:
        return {
            "sensor_name": self._sensor_method.__name__,
        }

    def __init__(
        self,
        sensor_method: Callable[[FieldOfInterest], FieldOfInterest] | None = None,
    ) -> None:
        super().__init__()

        self._g0: Gaussian | None = None
        self._g1: Gaussian | None = None
        self._M: np.ndarray | None = None
        if sensor_method is None:
            sensor_method = to_localized_density_sensor

        self._sensor_method = sensor_method
        self.plot_data: GotPlotData | None = None

    @classmethod
    def factory_constructor(
        cls,
        source_descriptor: FieldDescriptor,
        target_descriptor: FieldDescriptor,
        **kwargs,
    ) -> "GaussianOTMapping":
        mapping = cls(**kwargs)
        mapping._initialise_from_descriptors(source_descriptor, target_descriptor)
        return mapping

    @override
    def copy(self) -> "GaussianOTMapping":
        """Return a copy of the mapping."""
        copy = GaussianOTMapping(self._sensor_method).with_name(self.name)
        copy._g0 = self._g0
        copy._g1 = self._g1
        copy._M = self._M
        copy.plot_data = self.plot_data
        return copy

    @staticmethod
    def _compute_m_matrix(gaussian_0: Gaussian, gaussian_1: Gaussian):
        sigma_0 = np.asarray(gaussian_0.sigma, dtype=float)
        sigma_1 = np.asarray(gaussian_1.sigma, dtype=float)
        return (
            fractional_matrix_power(sigma_0, -0.5)
            @ fractional_matrix_power(
                fractional_matrix_power(sigma_0, 0.5)
                @ sigma_1
                @ fractional_matrix_power(sigma_0, 0.5),
                0.5,
            )
            @ fractional_matrix_power(sigma_0, -0.5)
        )

    @override
    def fit(self, u0: FieldOfInterest, u1: FieldOfInterest) -> Self:
        u0_hat = self._sensor_method(u0)
        u1_hat = self._sensor_method(u1)

        g0 = Gaussian.from_field(u0_hat.mesh.points, u0_hat.values)
        g1 = Gaussian.from_field(u1_hat.mesh.points, u1_hat.values)

        self._set_gaussians(g0, g1)

        self.plot_data = GotPlotData(
            g0=g0,
            g1=g1,
            u0_hat=u0_hat,
            u1_hat=u1_hat,
            sensor_name=self._sensor_method.__name__,
        )
        return self

    @property
    def dimension(self) -> int:
        if self._g0 is None or self._g1 is None:
            raise ValueError("Mapping must be fitted or constructed before evaluation")
        mu_0 = np.asarray(self._g0.mu)
        mu_1 = np.asarray(self._g1.mu)
        dim = mu_0.shape[0]
        if dim != mu_1.shape[0]:
            raise ValueError("Gaussian dimensions do not match")
        return dim

    def _reset_cached_mappings(self) -> None:
        try:
            self.get_mapping_function.cache_clear()
        except AttributeError:
            pass

    def _set_gaussians(self, gaussian_0: Gaussian, gaussian_1: Gaussian) -> None:
        M = self._compute_m_matrix(gaussian_0, gaussian_1)
        self._g0 = gaussian_0
        self._g1 = gaussian_1
        # Sometimes M is complex type due to fractional matrix power computations.
        self._M = np.asarray(M, dtype=float)
        self._reset_cached_mappings()

    def _initialise_from_descriptors(
        self, source_descriptor: FieldDescriptor, target_descriptor: FieldDescriptor
    ) -> None:
        if not isinstance(source_descriptor, Gaussian) or not isinstance(
            target_descriptor, Gaussian
        ):
            raise TypeError(
                "GaussianOTMapping factory expects Gaussian-compatible descriptors"
            )
        g0 = Gaussian(
            mu=np.array(source_descriptor.mu), sigma=np.array(source_descriptor.sigma)
        )
        g1 = Gaussian(
            mu=np.array(target_descriptor.mu), sigma=np.array(target_descriptor.sigma)
        )
        self._set_gaussians(g0, g1)

    def _ensure_ready(self) -> None:
        if self._g0 is None or self._g1 is None or self._M is None:
            raise ValueError("Mapping must be fitted or constructed before evaluation")

    def _build_affine_mapping(self, inverse: bool) -> Mapping:
        self._ensure_ready()
        assert self._g0 is not None and self._g1 is not None and self._M is not None
        mu_0 = np.asarray(self._g0.mu)
        mu_1 = np.asarray(self._g1.mu)
        M = self._M
        dimension = mu_0.shape[0]

        def mapping(points: np.ndarray, s: float | None = None) -> np.ndarray:
            s_val = 1.0 if s is None else s
            return np.transpose(
                (1 - s_val) * points.T + s_val * (mu_1 + M @ (points.T - mu_0))
            )

        def inverse_mapping(points: np.ndarray, s: float | None = None) -> np.ndarray:
            s_val = 1.0 if s is None else s
            A = (1 - s_val) * np.eye(dimension) + s_val * M
            return np.transpose(
                np.linalg.inv(A) @ (points.T - s_val * (mu_1 - M @ mu_0))
            )

        return inverse_mapping if inverse else mapping

    @lru_cache
    @override
    def get_mapping_function(self, *, inverse: bool = False) -> Mapping:
        return self._build_affine_mapping(inverse)

    def compute_inverse(self) -> Mapping:
        return self._build_affine_mapping(inverse=True)

    def get_ts_mapping_function(self) -> Mapping:
        return self._build_affine_mapping(inverse=False)

    def get_ws_mapping_function(self) -> Mapping:
        return self._build_affine_mapping(inverse=True)


def _get_det_of_jac_of_phi_relation(mu_0, mu_1, sigma_0, sigma_1):
    from sympy import Matrix, symbols

    s = symbols("s")

    x1, x2 = symbols("x1 x2")
    x = Matrix([x1, x2])
    m0 = Matrix(mu_0)
    m1 = Matrix(mu_1)

    A = fractional_matrix_power(sigma_0, -1.0) @ fractional_matrix_power(
        sigma_0 @ sigma_1, 0.5
    )
    A = Matrix(A)
    T = m1 + A * (x - m0)
    T_tilde = x + s * (T - x)
    jacobian = T_tilde.jacobian(x)

    return jacobian.det()


def determinant_mapping(s_value: float, mu_0, mu_1, sigma_0, sigma_1):
    from sympy import symbols

    det_jac_phi = _get_det_of_jac_of_phi_relation(mu_0, mu_1, sigma_0, sigma_1)

    s = symbols("s")
    return det_jac_phi.evalf(subs={s: s_value})


def wasserstein_distance_gaussians(g0: Gaussian, g1: Gaussian):
    return wasserstein_distance_analytical(
        np.asarray(g0.mu),
        np.asarray(g1.mu),
        np.asarray(g0.sigma),
        np.asarray(g1.sigma),
    )


def wasserstein_distance_analytical(
    mu_0: np.ndarray,
    mu_1: np.ndarray,
    sigma_0: np.ndarray,
    sigma_1: np.ndarray,
):
    mu_0 = np.array(mu_0)
    mu_1 = np.array(mu_1)
    dim = mu_0.shape[0]
    sigma_0 = np.array(sigma_0).reshape((dim, dim))
    sigma_1 = np.array(sigma_1).reshape((dim, dim))

    if mu_0.shape != mu_1.shape:
        raise ValueError("mu_0 and mu_1 must have the same shape")

    if sigma_0.shape != sigma_1.shape:
        raise ValueError("sigma_0 and sigma_1 must have the same shape")

    if sigma_0.shape[0] != dim:
        raise ValueError("Incompatible shapes")

    sigma_0 = np.asarray(sigma_0, dtype=float)
    sigma_1 = np.asarray(sigma_1, dtype=float)
    return np.sqrt(
        np.linalg.norm(mu_0 - mu_1) ** 2
        + np.trace(
            sigma_0
            + sigma_1
            - 2
            * fractional_matrix_power(
                fractional_matrix_power(sigma_0, 0.5)
                @ sigma_1
                @ fractional_matrix_power(sigma_0, 0.5),
                0.5,
            )
        )
    )


def wasserstein_distance_analytical_jax(
    mu_0: "jax.Array",
    mu_1: "jax.Array",
    sigma_0: "jax.Array",
    sigma_1: "jax.Array",
):
    """JAX version of Wasserstein distance between two Gaussians.

    Uses eigendecomposition-based fractional matrix power for symmetric positive
    semi-definite matrices, which is JAX-compatible and differentiable.

    Args:
        mu_0: Mean of first Gaussian (shape: (dim,))
        mu_1: Mean of second Gaussian (shape: (dim,))
        sigma_0: Covariance of first Gaussian (shape: (dim, dim))
        sigma_1: Covariance of second Gaussian (shape: (dim, dim))

    Returns:
        Wasserstein-2 distance between the two Gaussians
    """
    import chex
    import jax.numpy as jnp

    # Shape validation using chex
    chex.assert_rank(mu_0, 1)
    chex.assert_rank(mu_1, 1)
    chex.assert_rank(sigma_0, 2)
    chex.assert_rank(sigma_1, 2)

    dim = mu_0.shape[0]
    chex.assert_shape(mu_1, (dim,))
    chex.assert_shape(sigma_0, (dim, dim))
    chex.assert_shape(sigma_1, (dim, dim))

    def fractional_matrix_power_jax(matrix: "jax.Array", power: float) -> "jax.Array":
        """Compute fractional matrix power using eigendecomposition with safeguards.

        For symmetric positive semi-definite matrix A, computes
        A^p = Q @ diag(λ^p) @ Q.T where Q contains eigenvectors and λ are
        eigenvalues. Includes safeguards for numerical stability.
        """
        # Ensure the matrix is symmetric
        matrix = (matrix + matrix.T) / 2.0

        # Use eigh for symmetric matrices (more efficient and stable)
        eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)

        # Clip eigenvalues that are too small or negative for numerical stability
        # Use machine epsilon based on dtype and scale for robustness
        dtype = matrix.dtype
        eps = jnp.finfo(dtype).eps * 1000  # Much larger tolerance
        min_eigval = jnp.maximum(jnp.std(matrix) * 1e-6, eps)

        # Use a soft threshold to avoid abrupt clipping
        eigenvalues_clipped = jnp.maximum(eigenvalues, min_eigval)

        # Ensure we don't get zero eigenvalues
        eigenvalues_safe = jnp.where(
            eigenvalues_clipped <= 0, min_eigval, eigenvalues_clipped
        )

        # Handle NaN/inf in eigenvalues
        eigenvalues_safe = jnp.nan_to_num(
            eigenvalues_safe, nan=min_eigval, posinf=1e4, neginf=min_eigval
        )

        eigenvalues_powered = jnp.power(eigenvalues_safe, power)

        # Reconstruct: A^p = Q @ diag(λ^p) @ Q.T
        result = eigenvectors @ jnp.diag(eigenvalues_powered) @ eigenvectors.T

        # Ensure the result is finite and regularize if needed
        result = jnp.nan_to_num(
            result, nan=jnp.eye(matrix.shape[0], dtype=dtype), posinf=1e8, neginf=1e-8
        )

        # Ensure positive definiteness
        reg_val = jnp.finfo(dtype).eps * matrix.shape[0]
        result = result + jnp.eye(matrix.shape[0], dtype=dtype) * reg_val

        return result

    # Additional safeguards for sigma matrices
    sigma_0_reg = sigma_0 + 1e-6 * jnp.eye(sigma_0.shape[0], dtype=sigma_0.dtype)
    sigma_1_reg = sigma_1 + 1e-6 * jnp.eye(sigma_1.shape[0], dtype=sigma_1.dtype)

    # Ensure matrices are symmetric and regularized
    sigma_0_reg = (sigma_0_reg + sigma_0_reg.T) / 2.0
    sigma_1_reg = (sigma_1_reg + sigma_1_reg.T) / 2.0

    sigma_0_half = fractional_matrix_power_jax(sigma_0_reg, 0.5)
    middle_matrix = sigma_0_half @ sigma_1_reg @ sigma_0_half
    middle_sqrt = fractional_matrix_power_jax(middle_matrix, 0.5)

    # Compute squared distance with additional safeguards
    mu_distance = jnp.sum((mu_0 - mu_1) ** 2)
    trace_term = jnp.trace(sigma_0_reg + sigma_1_reg - 2 * middle_sqrt)

    # Ensure trace_term is not negative due to numerical precision
    trace_term = jnp.maximum(trace_term, 0.0)
    w2_squared = mu_distance + trace_term

    # Final safeguard: ensure sqrt argument is non-negative
    w2_squared = jnp.maximum(w2_squared, jnp.finfo(w2_squared.dtype).eps)
    return jnp.sqrt(w2_squared)
