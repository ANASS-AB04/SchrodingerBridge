from typing import Literal, overload

import numpy as np

from phdtruel.fields import (
    DynamicOfInterest,
    FieldOfInterest,
    Gaussian,
    Square,
)
from phdtruel.fields.meshes import DomainBounds, RegularGrid
from phdtruel.fields.parameters import ParameterSet

ParameterAsDict = dict[str, int] | dict[str, float]


@overload
def construct_square_test_fields(
    source_parameters: list[ParameterSet],
    target_parameter: None = None,
    x_space: np.ndarray | None = None,
    y_space: np.ndarray | None = None,
    return_squares: bool = False,
) -> list[FieldOfInterest]: ...


@overload
def construct_square_test_fields(
    source_parameters: list[ParameterSet],
    target_parameter: ParameterSet,
    x_space: np.ndarray | None = None,
    y_space: np.ndarray | None = None,
    return_squares: Literal[False] = False,
) -> tuple[list[FieldOfInterest], FieldOfInterest]: ...


@overload
def construct_square_test_fields(
    source_parameters: list[ParameterSet],
    target_parameter: ParameterSet,
    x_space: np.ndarray | None = None,
    y_space: np.ndarray | None = None,
    return_squares: Literal[True] = True,
) -> tuple[list[FieldOfInterest], FieldOfInterest, list[Square]]: ...


def construct_square_test_fields(
    source_parameters: list[ParameterSet],
    target_parameter: ParameterSet | None = None,
    x_space: np.ndarray | None = None,
    y_space: np.ndarray | None = None,
    return_squares: bool = False,
) -> (
    list[FieldOfInterest]
    | tuple[list[FieldOfInterest], FieldOfInterest]
    | tuple[list[FieldOfInterest], FieldOfInterest, list[Square]]
):
    """Constructs test fields with square distributions.

    Parameters should be x and y coordinates and size of the square.
    p.x, p.y, p.size, p.value
    """
    if x_space is None:
        x_space = np.linspace(-1.0, 1.0, 200)
    if y_space is None:
        y_space = np.linspace(-1.0, 1.0, 200)

    mesh = RegularGrid((x_space, y_space))

    squares = [
        Square.from_center(
            np.array([float(p.x), float(p.y)]),
            size=float(p.side_lenght),
            value=float(p.value),
        )
        for p in source_parameters
    ]

    fields = [
        FieldOfInterest(mu, mesh, sq(mesh.points), fill_value=0.0)
        for mu, sq in zip(source_parameters, squares)
    ]
    if target_parameter is not None:
        target_sq = Square.from_center(
            np.array([float(target_parameter.x), float(target_parameter.y)]),
            float(target_parameter.side_lenght),
            value=float(target_parameter.value),
        )
        target_field = FieldOfInterest(
            target_parameter, mesh, target_sq(mesh.points), fill_value=0.0
        )

        if return_squares:
            return fields, target_field, squares
        return fields, target_field
    else:
        return fields


@overload
def construct_gaussian_test_fields(
    source_parameters: list[ParameterSet],
    target_parameter: None = None,
    x_space: np.ndarray | None = None,
    y_space: np.ndarray | None = None,
    return_gaussians: bool = False,
) -> list[FieldOfInterest]: ...


@overload
def construct_gaussian_test_fields(
    source_parameters: list[ParameterSet],
    target_parameter: ParameterSet,
    x_space: np.ndarray | None = None,
    y_space: np.ndarray | None = None,
    return_gaussians: Literal[False] = False,
) -> tuple[list[FieldOfInterest], FieldOfInterest]: ...


@overload
def construct_gaussian_test_fields(
    source_parameters: list[ParameterSet],
    target_parameter: ParameterSet,
    x_space: np.ndarray | None = None,
    y_space: np.ndarray | None = None,
    return_gaussians: Literal[True] = True,
) -> tuple[list[FieldOfInterest], FieldOfInterest, list[Gaussian]]: ...


def construct_gaussian_test_fields(
    source_parameters: list[ParameterSet],
    target_parameter: ParameterSet | None = None,
    x_space: np.ndarray | None = None,
    y_space: np.ndarray | None = None,
    return_gaussians: bool = False,
) -> (
    list[FieldOfInterest]
    | tuple[list[FieldOfInterest], FieldOfInterest]
    | tuple[list[FieldOfInterest], FieldOfInterest, list[Gaussian]]
):
    """Constructs test fields with Gaussian distributions.

    Parameters should be x and y coordinates and if provided, the covariance.
    p.x, p.y, p.covariance
    """
    if x_space is None:
        x_space = np.linspace(-1, 1, 200)
    if y_space is None:
        y_space = np.linspace(-1, 1, 200)

    mesh = RegularGrid((x_space, y_space))

    gaussians = [
        Gaussian(
            mu=np.array([float(p.x), float(p.y)]),
            sigma=np.asarray(p.covariance, dtype=float),
        )
        for p in source_parameters
    ]

    fields = [
        FieldOfInterest(mu, mesh, g(mesh.points), fill_value=0.0)
        for mu, g in zip(source_parameters, gaussians)
    ]

    if target_parameter is not None:
        target_gaussian = Gaussian(
            np.array([float(target_parameter.x), float(target_parameter.y)]),
            np.asarray(target_parameter.covariance, dtype=float),
        )
        target_field = FieldOfInterest(
            target_parameter, mesh, target_gaussian(mesh.points), fill_value=0.0
        )
        if return_gaussians:
            return fields, target_field, gaussians
        return fields, target_field
    else:
        return fields


def create_regular_grid(x_min=-1.0, x_max=1.0, nx=100, y_min=-1.0, y_max=1.0, ny=100):
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    x, y = np.meshgrid(x, y)
    points = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    return points


PossibleGaussianParameters = Literal["m_0", "m_1", "sigma_0", "sigma_1", "sigma_2"]


class GaussianDataLoader:
    """Data loader for synthetic Gaussian fields.

    Generates fields from mixtures of Gaussian distributions.
    Useful for testing and synthetic experiments.
    """

    def __init__(self, **kwargs):
        self._points = create_regular_grid(**kwargs)
        self._weights = None
        self._gaussians = None

    def setup_as_mock(
        self,
        domain_bounds: DomainBounds,
        parameters: list[ParameterSet] | list[ParameterAsDict],
        weights: np.ndarray | None = None,
        nx: int = 100,
        ny: int = 100,
    ) -> None:
        self._points = create_regular_grid(
            x_min=float(domain_bounds.minima[0]),
            x_max=float(domain_bounds.maxima[0]),
            nx=nx,
            y_min=float(domain_bounds.minima[1]),
            y_max=float(domain_bounds.maxima[1]),
            ny=ny,
        )
        # TODO finish implementation of mock
        if weights is None:
            self._weights = np.ones(len(parameters))
        else:
            self._weights = weights
        parameter_dicts = [
            p.as_dict() if isinstance(p, ParameterSet) else p for p in parameters
        ]
        means = np.array(
            [[p["m_0"], p["m_1"]] for p in parameter_dicts],
            dtype=float,
        )
        covariances = np.array(
            [
                [[p["sigma_0"], p["sigma_1"]], [p["sigma_1"], p["sigma_2"]]]
                for p in parameter_dicts
            ],
            dtype=float,
        )
        self._gaussians = [Gaussian(mu, cov0) for mu, cov0 in zip(means, covariances)]

    def setup_random(
        self,
        dimension: int = 2,
        num_fields: int = 5,
        cov: np.ndarray | None = None,
    ) -> None:
        weights = np.random.uniform(0.8, 1.0, size=num_fields)
        weights /= np.sum(weights)

        means = 0.1 + np.random.random((num_fields, dimension)) * 0.8

        # covariances.shape == (num_fields,dimension,dimension)
        if cov is None:
            # TODO generate only 3 values ? -> ensure SDP matrix
            A = np.random.random((num_fields, dimension, dimension))
            covariances = np.transpose(A, axes=[0, 2, 1]) @ A
        elif cov.shape == (num_fields, dimension, dimension):
            covariances = cov
        elif cov.shape == (dimension, dimension):
            covariances = cov @ np.ones((num_fields, dimension, dimension))
        else:
            raise ValueError(f"Invalid cov argument: {cov}")

        self._weights = weights
        self._gaussians = [Gaussian(mu, cov0) for mu, cov0 in zip(means, covariances)]

    def get_field(
        self,
        parameter: ParameterSet | ParameterAsDict,
        timestep: int = 0,
        time_as_parameter: bool = False,
    ) -> FieldOfInterest:
        """Load a synthetic Gaussian field.

        Not yet fully implemented.
        """
        raise NotImplementedError("GaussianDataLoader.get_field not yet implemented")

    def get_dynamic(self, parameter: ParameterSet, **kwargs) -> DynamicOfInterest:
        """Load a synthetic Gaussian dynamic.

        Not yet implemented.
        """
        raise NotImplementedError("GaussianDataLoader.get_dynamic not yet implemented")
