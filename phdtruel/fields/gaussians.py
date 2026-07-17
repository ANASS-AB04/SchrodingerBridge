from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy import linalg
from scipy.linalg import fractional_matrix_power
from scipy.stats import multivariate_normal


@dataclass
class Gaussian:
    mu: np.ndarray
    sigma: np.ndarray

    @classmethod
    def from_field(cls, points: np.ndarray, values: np.ndarray):
        """Mu and sigma are computed from the field"""
        mu = np.average(points, axis=0, weights=values.ravel())
        sigma = np.cov(points.T, aweights=values.ravel())
        return cls(mu, sigma)

    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=float)
        self.mu = self.mu.reshape(self.mu.size, -1)
        if np.isscalar(self.sigma):
            if self.dim > 1:
                self.sigma = np.diag(np.ones(self.dim) * float(self.sigma))
            else:
                self.sigma = np.array([[float(self.sigma)]], dtype=float)
        else:
            self.sigma = np.asarray(self.sigma, dtype=float)
            if self.sigma.ndim == 0:
                if self.dim > 1:
                    self.sigma = np.diag(np.ones(self.dim) * float(self.sigma))
                else:
                    self.sigma = np.array([[float(self.sigma)]], dtype=float)
            elif self.sigma.ndim == 1:
                if self.dim == 1:
                    self.sigma = self.sigma.reshape(1, 1)
                elif self.sigma.size == 1:
                    self.sigma = np.diag(np.ones(self.dim) * float(self.sigma[0]))

        if not np.allclose(self.sigma, self.sigma.transpose()):
            raise ValueError(f"Matrix {self.sigma} is not symmetric")

        if np.linalg.det(self.sigma) < 0:
            raise ValueError("Sigma det is not positive")

    @property
    def dim(self):
        return self.mu.size

    def __call__(self, points: np.ndarray):
        return self.sample(points)

    def sample(self, points: np.ndarray):
        if len(points.shape) != self.dim:
            raise ValueError("Dimension mismatch")
        return multivariate_normal.pdf(points, self.mu.ravel(), self.sigma).reshape(
            -1, 1
        )

    def __eq__(self, other):
        if not isinstance(other, Gaussian):
            return False
        return np.allclose(self.mu, other.mu) and np.allclose(self.sigma, other.sigma)

    def __repr__(self) -> str:
        return f"Gaussian: mu={self.mu}, sigma={self.sigma}"


def to_localized_density(values: np.ndarray, p: int = 2):
    return np.abs(values - np.median(values)) ** p


@dataclass
class MultipleGaussians:
    gaussians: list[Gaussian]

    def __post_init__(self):
        dim = None
        for g in self.gaussians:
            if dim is not None and dim != g.dim:
                raise ValueError("Dimension mismatch")
            dim = g.dim

    def __call__(self, points: np.ndarray):
        return self.sample(points)

    def sample(self, points: np.ndarray):
        values = np.zeros((points.shape[0], 1))
        for g in self.gaussians:
            values += g(points)
        return values


def interpolate_gaussians_linear(s: float, g0: Gaussian, g1: Gaussian) -> Gaussian:
    """Return the gaussian interpolating between g0 and g1 with parameter s"""
    return Gaussian(
        (1 - s) * g0.mu + s * g1.mu,
        (1 - s) * g0.sigma + s * g1.sigma,
    )


def interpolate_gaussians_ot(s: float, g0: Gaussian, g1: Gaussian) -> Gaussian:
    cov_0 = g1.sigma
    cov_1 = g0.sigma
    cov = (
        fractional_matrix_power(cov_0, -0.5)
        @ fractional_matrix_power(
            s * cov_0
            + (1 - s)
            * fractional_matrix_power(
                fractional_matrix_power(cov_0, 0.5)
                @ cov_1
                @ fractional_matrix_power(cov_0, 0.5),
                0.5,
            ),
            2,
        )
        @ fractional_matrix_power(cov_0, -0.5)
    )
    mu = s * g1.mu + (1 - s) * g0.mu
    return Gaussian(mu, cov)


InterpolationStrategy = Callable[[float, Gaussian, Gaussian], Gaussian]


def interpolate_gaussians(
    s: float,
    g0: Gaussian | MultipleGaussians,
    g1: Gaussian | MultipleGaussians,
    interpolation_strategy: Literal["linear", "ot"] = "ot",
) -> Gaussian | MultipleGaussians:
    if interpolation_strategy == "linear":
        strategy: InterpolationStrategy = interpolate_gaussians_linear
    elif interpolation_strategy == "ot":
        strategy = interpolate_gaussians_ot
    else:
        raise ValueError("Unknown interpolation strategy")

    if not isinstance(g0, MultipleGaussians):
        g0 = MultipleGaussians([g0])

    if not isinstance(g1, MultipleGaussians):
        g1 = MultipleGaussians([g1])

    if len(g0.gaussians) != len(g1.gaussians):
        raise ValueError("Number of gaussians mismatch")

    new_gaussians = [strategy(s, g0, g1) for g0, g1 in zip(g0.gaussians, g1.gaussians)]
    if len(new_gaussians) == 1:
        return new_gaussians[0]
    return MultipleGaussians(new_gaussians)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    g = Gaussian(np.array([0.0]), np.array([[1.0]]))
    g_vals = g.sample(np.linspace(-10, 10, 100))
    plt.plot(g_vals)

    # 2D
    g = Gaussian(np.array([0.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0]]))

    x, y = np.mgrid[-10:10:0.1, -10:10:0.1]

    points = np.dstack((x.reshape(200 * 200), y.reshape(200 * 200))).reshape(-1, 2)
    g_vals = g.sample(points)
    plt.figure()
    plt.contourf(x, y, g_vals.reshape(200, 200))

    plt.show()


def principal_point_of_gaussian(gaussian: Gaussian) -> np.ndarray:
    return compute_principal_points(gaussian.mu, gaussian.sigma)


def compute_principal_points(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    mu = mu.reshape(1, -1)

    v, w = linalg.eigh(sigma)
    principal_points = np.zeros((5, 2))
    principal_points[0] = mu
    principal_points[1] = mu + np.sqrt(v[0]) * w[0]
    principal_points[2] = mu + np.sqrt(v[1]) * w[1]
    principal_points[3] = mu - np.sqrt(v[0]) * w[0]
    principal_points[4] = mu - np.sqrt(v[1]) * w[1]
    return principal_points
