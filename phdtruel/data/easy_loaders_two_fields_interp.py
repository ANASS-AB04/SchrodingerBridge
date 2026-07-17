import typing
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal

import numpy as np

import phdtruel
from phdtruel.data import letters_fields
from phdtruel.data.gaussian_dataloader import (
    construct_gaussian_test_fields,
)
from phdtruel.data.jaxfluids_dataloader import JaxFluidsFields
from phdtruel.data.letters_fields import load_pre_generated_image
from phdtruel.data.nascar_dataloader import PossibleNascarFields
from phdtruel.fields import FieldOfInterest, VectorFieldOfInterest
from phdtruel.fields.meshes import Mesh, RegularGrid
from phdtruel.fields.parameters import ParameterSet

# Type alias for all available test case names
TestCaseName = Literal[
    # Gaussian cases
    "gaussian_dilate_translate",
    "gaussian_translate",
    # Square cases
    "square_translate",
    "square_dilate_translate",
    "square_dilate_translate_nooverlap",
    "square_diff_values_dilate_translate",
    "square_very_diff_values_dilate_translate",
    # Letter cases
    "letters_AA_scaled_translated",
    "letters_AA_diff_values_scaled_translated",
    "letters_AA_rotated_diff_values_scaled_translated",
    # "letters_MW", # Reference field not easily defined
    # DAFoam cases
    "dafoam_T",
    # "dafoam_U", # Vector field not implemented
    "dafoam_p",
    "dafoam_rho",
    # JAX-Fluids cases
    "jaxfluids_bump_density",
    "jaxfluids_bump_pressure",
    "jaxfluids_bump_temperature",
    "jaxfluids_bump_velocity_u",
    "jaxfluids_bump_velocity_v",
    # NASCAR cases
    "nascar_vorticity",
    "nascar_velocity",
    "nascar_u",
    "nascar_v",
    "nascar_p",
]

AvailableTestCaseNames = typing.get_args(TestCaseName)


@dataclass
class TestFieldsBundle:
    """Container for a complete test case with two fields, reference, mesh, and velocities."""

    f0: FieldOfInterest
    f1: FieldOfInterest
    f_ref: FieldOfInterest
    mesh: RegularGrid
    velocity_0: VectorFieldOfInterest
    velocity_1: VectorFieldOfInterest
    description: str


def load_test_case_for_two_fields_interpolation(
    case_name: TestCaseName,
) -> TestFieldsBundle:
    """Get a predefined test case by name.

    This is the main entry point for loading test fields. All test cases are
    consistently defined and cached for reproducibility.

    Args:
        case_name: Name of the test case (e.g., 'gaussian', 'square_dilate_translate', 'nascar_vorticity')

    Returns:
        TestFieldsBundle containing f0, f1, f_ref, mesh, velocity_0, velocity_1

    Example:
        >>> bundle = load_test_case_for_two_fields_interpolation('gaussian')
        >>> print(bundle.f0.mesh.n_points)
    """
    # Look up loader in registry
    loader = _TEST_CASE_REGISTRY.get(case_name)
    if not loader:
        available = ", ".join(sorted(_TEST_CASE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown test case: '{case_name}'. Available cases: {available}"
        )

    return loader()


def _create_dummy_velocities(
    mesh: Mesh,
    velocity_vector: np.ndarray | None = None,
) -> tuple[VectorFieldOfInterest, VectorFieldOfInterest]:
    """Create dummy velocity fields (zeros in x, ones in y) for testing."""
    if velocity_vector is None:
        velocity_vector = np.array([0.0, 1.0])

    if velocity_vector.shape != (2,):
        raise ValueError("velocity_vector must be of shape (2,) representing (vx, vy)")

    n = len(mesh)
    velocity = VectorFieldOfInterest(
        [
            FieldOfInterest(None, mesh, np.full(n, velocity_vector[0]), fill_value=0.0),
            FieldOfInterest(None, mesh, np.full(n, velocity_vector[1]), fill_value=0.0),
        ]
    )
    return velocity, velocity


def _unwrap_dafoam_field(
    field: FieldOfInterest
    | tuple[FieldOfInterest, np.ndarray]
    | tuple[FieldOfInterest, np.ndarray, np.ndarray],
) -> FieldOfInterest:
    match field:
        case FieldOfInterest() as field_obj:
            return field_obj
        case (FieldOfInterest() as field_obj, _):
            return field_obj
        case (FieldOfInterest() as field_obj, _, _):
            return field_obj
        case _:
            raise TypeError("Unsupported DAFoam field result")


def setup_gaussians_wide_apart() -> TestFieldsBundle:
    fields = construct_gaussian_test_fields(
        [
            ParameterSet(x=-0.2, y=0.3, covariance=0.025),
            ParameterSet(x=0.2, y=-0.3, covariance=0.020),
            ParameterSet(x=0.0, y=0.0, covariance=0.0225),
        ],
    )
    g0, g1, g_ref = fields
    mesh = g0.mesh
    velocity_0, velocity_1 = _create_dummy_velocities(mesh)
    return TestFieldsBundle(
        f0=g0,
        f1=g1,
        f_ref=g_ref,
        mesh=mesh,
        velocity_0=velocity_0,
        velocity_1=velocity_1,
        description="Gaussian fields with different covariances",
    )


def setup_gaussian_translate_fields() -> TestFieldsBundle:
    fields = construct_gaussian_test_fields(
        [
            ParameterSet(x=0.0, y=0.2, covariance=0.03),
            ParameterSet(x=0.0, y=-0.2, covariance=0.03),
            ParameterSet(x=0.0, y=0.0, covariance=0.03),
        ],
    )

    g0, g1, g_ref = fields
    mesh = g0.mesh
    velocity_0, velocity_1 = _create_dummy_velocities(mesh)
    return TestFieldsBundle(
        f0=g0,
        f1=g1,
        f_ref=g_ref,
        mesh=mesh,
        velocity_0=velocity_0,
        velocity_1=velocity_1,
        description="Gaussian fields with same covariance (translation)",
    )


def setup_square_fields(
    values_are: Literal["all_ones", "all_tens", "one_five_nine"]
    | tuple[float, float, float],
    shapes_are: Literal["big_big_big", "small_small_small", "small_med_big"]
    | tuple[int, int, int],
    n_mesh: int = 200,
    overlap_squares: bool = True,
    centered_along_x: bool = True,
) -> TestFieldsBundle:
    values: tuple[float, float, float] | None = None
    radiuses: tuple[int, int, int] | None = None
    center_x: tuple[int, int, int] | None = None
    center_y: tuple[int, int, int] | None = None
    domain = ((-3.0, 3.0), (-3.0, 3.0))
    two_by_two_center = (
        n_mesh % 2 == 0
    )  # Odd number of points allows for a perfect center cell, even number leads to a 2x2 block center

    mesh = RegularGrid(
        [
            np.linspace(domain[0][0], domain[0][1], n_mesh),
            np.linspace(domain[1][0], domain[1][1], n_mesh),
        ],
    )

    match values_are:
        case "all_ones":
            values = (1.0, 1.0, 1.0)
        case "all_tens":
            values = (10.0, 10.0, 10.0)
        case "one_five_nine":
            values = (1.0, 5.0, 9.0)
        case _ if isinstance(values_are, tuple) and len(values_are) == 3:
            values = values_are
        case _:
            raise ValueError(f"Invalid values_are: {values_are}")

    mid_ax_y = n_mesh // 2
    match shapes_are:
        case "big_big_big":
            big_radius = n_mesh // 8
            radiuses = (big_radius, big_radius, big_radius)
            if overlap_squares:
                center_y = (
                    mid_ax_y + big_radius // 4,
                    mid_ax_y,
                    mid_ax_y - big_radius // 4,
                )
            else:
                center_y = (
                    mid_ax_y + mid_ax_y // 4,
                    mid_ax_y,
                    mid_ax_y - mid_ax_y // 4,
                )
        case "small_small_small":
            small_radius = n_mesh // 16
            radiuses = (small_radius, small_radius, small_radius)
            if overlap_squares:
                center_y = (
                    mid_ax_y + small_radius // 4,
                    mid_ax_y,
                    mid_ax_y - small_radius // 4,
                )
            else:
                center_y = (
                    mid_ax_y + mid_ax_y // 4,
                    mid_ax_y,
                    mid_ax_y - mid_ax_y // 4,
                )
        case "small_med_big":
            raise NotImplementedError("Not well defined yet")
            radiuses = (3, 4, 5)
            if overlap_squares:
                center_y = (mid_ax_y + 2, mid_ax_y, mid_ax_y - 3)
            else:
                center_y = (mid_ax_y + 6, mid_ax_y, mid_ax_y - 6)
        case _ if isinstance(shapes_are, tuple) and len(shapes_are) == 3:
            radiuses = shapes_are
            center_y = (
                mid_ax_y + radiuses[0] // 2,
                mid_ax_y,
                mid_ax_y - radiuses[2] // 2,
            )
        case _:
            raise ValueError(f"Invalid shapes_are: {shapes_are}")

    mid_ax_x = n_mesh // 2
    quarter_ax_x = n_mesh // 4
    if centered_along_x:
        center_x = (mid_ax_x, mid_ax_x, mid_ax_x)
    else:
        center_x = (quarter_ax_x, quarter_ax_x, quarter_ax_x)

    if two_by_two_center:
        center_x = tuple(i - 1 for i in center_x)
        center_y = tuple(i - 1 for i in center_y)

    # Create the fields
    f0 = construct_square_field_by_center_and_radius(
        mesh,
        center=(center_x[0], center_y[0]),
        radius=radiuses[0],
        value=values[0],
        two_by_two_center=two_by_two_center,
    ).with_name("$u_0$")
    f_ref = construct_square_field_by_center_and_radius(
        mesh,
        center=(center_x[1], center_y[1]),
        radius=radiuses[1],
        value=values[1],
        two_by_two_center=two_by_two_center,
    ).with_name("$u_1$")
    f1 = construct_square_field_by_center_and_radius(
        mesh,
        center=(center_x[2], center_y[2]),
        radius=radiuses[2],
        value=values[2],
        two_by_two_center=two_by_two_center,
    ).with_name("$u_{ref}$")

    mesh = f0.mesh
    velocity_0, velocity_1 = _create_dummy_velocities(mesh)

    desc = f"Square fields: values={values_are}, shapes={shapes_are}, overlap={overlap_squares}"
    return TestFieldsBundle(
        f0=f0,
        f1=f1,
        f_ref=f_ref,
        mesh=mesh,
        velocity_0=velocity_0,
        velocity_1=velocity_1,
        description=desc,
    )


def construct_square_field_by_indices(
    mesh: RegularGrid,
    width: int,
    lower_right_corner: tuple[int, int],
    value: float = 1.0,
) -> FieldOfInterest:
    """Create a square field directly defined by mesh indices.

    No ambiguity: the square is exactly the cells within the specified index bounds.

    Args:
        mesh: RegularGrid with at least 2 axes (2D mesh)
        width: Width of the square in number of mesh cells (width × width square)
        lower_right_corner: (i, j) indices of the lower-right corner in mesh space
                           i is x-axis index, j is y-axis index
        value: Scalar value to assign inside the square (default 1.0)

    Returns:
        FieldOfInterest with indicator function for the square region

    Example:
        >>> mesh = RegularGrid([np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)])
        >>> field = construct_square_field_by_indices(mesh, width=10, lower_right_corner=(45, 45), value=1.0)
    """
    if mesh.dimension < 2:
        raise ValueError(f"Mesh must be 2D or higher; got dimension {mesh.dimension}")

    if width <= 0:
        raise ValueError(f"width must be positive; got {width}")

    i_right, j_right = lower_right_corner
    n_x, n_y = mesh.shape[0], mesh.shape[1]

    # Compute index bounds: square from (i_left, j_bottom) to (i_right, j_top)
    i_left = i_right - width + 1
    j_bottom = j_right - width + 1
    j_top = j_right

    # Validate bounds
    if i_left < 0 or i_right >= n_x or j_bottom < 0 or j_top >= n_y:
        raise ValueError(
            f"Square indices out of bounds. Mesh shape: {mesh.shape}, "
            f"lower_right_corner: {lower_right_corner}, width: {width}. "
            f"Computed bounds: i=[{i_left}, {i_right}], j=[{j_bottom}, {j_top}]"
        )

    # Create indicator field: 1 inside square, 0 outside
    u = np.zeros(mesh.points.shape[0], dtype=float)

    # Map mesh indices to flat indices in the points array
    # For RegularGrid with indexing="ij", a point at (i, j) has flat index i * n_y + j
    for i in range(i_left, i_right + 1):
        for j in range(j_bottom, j_top + 1):
            flat_idx = i * n_y + j
            u[flat_idx] = value

    # Create a dummy ParameterSet for metadata (not used for index-defined squares)
    params = ParameterSet(
        x=mesh.x[i_left],
        y=mesh.y[j_bottom],
        side_lenght=float(width * (mesh.x[1] - mesh.x[0])),  # Approximate physical size
    )

    return FieldOfInterest(params, mesh, u.reshape(-1, 1), fill_value=0.0)


def construct_square_field_by_center_and_radius(
    mesh: RegularGrid,
    center: tuple[int, int],
    radius: int,
    value: float = 1.0,
    two_by_two_center: bool = False,
) -> FieldOfInterest:
    """Create a square field defined by center and radius in mesh indices.

    Two centering modes:
    - two_by_two_center=False (default): Center is a single cell at indices (i, j).
      Square spans from (i-radius, j-radius) to (i+radius, j+radius).
      Final square size: (2*radius + 1) × (2*radius + 1)

    - two_by_two_center=True: Center is defined by the 2×2 block of cells at
      (i, j), (i+1, j), (i, j+1), (i+1, j+1). The geometric center is at (i+0.5, j+0.5).
      Square spans from (i-radius, j-radius) to (i+1+radius, j+1+radius).
      Final square size: (2+2*radius) × (2+2*radius)

    Args:
        mesh: RegularGrid with at least 2 axes (2D mesh)
        center: (i, j) indices of the center in mesh space
               i is x-axis index, j is y-axis index
        radius: How many cells to span from center (≥ 0)
        value: Scalar value to assign inside the square (default 1.0)
        two_by_two_center: If True, the center is defined by a 2×2 block of cells.
                           If False, center is a single cell (default).

    Returns:
        FieldOfInterest with indicator function for the square region

    Example:
        >>> mesh = RegularGrid([np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)])
        >>> # Single-cell center: 21×21 square (radius=10) centered at cell (50, 50)
        >>> f1 = construct_square_field_by_center_and_radius(mesh, center=(50, 50), radius=10, value=1.0)
        >>> # 2×2-block center: 22×22 square (radius=10) centered at 2×2 block starting at (50, 50)
        >>> f2 = construct_square_field_by_center_and_radius(mesh, center=(50, 50), radius=10, value=2.0, two_by_two_center=True)
    """
    if mesh.dimension < 2:
        raise ValueError(f"Mesh must be 2D or higher; got dimension {mesh.dimension}")

    if radius < 0:
        raise ValueError(f"radius must be non-negative; got {radius}")

    i_center, j_center = center
    n_x, n_y = mesh.shape[0], mesh.shape[1]

    # Compute index bounds based on centering mode
    if two_by_two_center:
        # Center is the 2×2 block: cells (i, j), (i+1, j), (i, j+1), (i+1, j+1)
        # The geometric center is at (i+0.5, j+0.5)
        # Expand symmetrically around this center
        i_left = i_center - radius
        i_right = i_center + 1 + radius
        j_bottom = j_center - radius
        j_top = j_center + 1 + radius
    else:
        # Center is a single cell at (i_center, j_center)
        i_left = i_center - radius
        i_right = i_center + radius
        j_bottom = j_center - radius
        j_top = j_center + radius

    # Validate bounds
    if i_left < 0 or i_right >= n_x or j_bottom < 0 or j_top >= n_y:
        raise ValueError(
            f"Square indices out of bounds. Mesh shape: {mesh.shape}, "
            f"center: {center}, radius: {radius}. "
            f"Computed bounds: i=[{i_left}, {i_right}], j=[{j_bottom}, {j_top}]"
        )

    # Create indicator field: 1 inside square, 0 outside
    u = np.zeros(mesh.points.shape[0], dtype=float)

    # Map mesh indices to flat indices in the points array
    # For RegularGrid with indexing="ij", a point at (i, j) has flat index i * n_y + j
    for i in range(i_left, i_right + 1):
        for j in range(j_bottom, j_top + 1):
            flat_idx = i * n_y + j
            u[flat_idx] = value

    # Create a dummy ParameterSet for metadata (not used for index-defined squares)
    width = (2 * radius + 1) if not two_by_two_center else (2 + 2 * radius)

    if two_by_two_center:
        # Use the center of the 2×2 block for metadata
        x_coord = mesh.x[i_center] + 0.5 * (mesh.x[i_center + 1] - mesh.x[i_center])
        y_coord = mesh.y[j_center] + 0.5 * (mesh.y[j_center + 1] - mesh.y[j_center])
    else:
        # Use the single-cell center for metadata
        x_coord = mesh.x[i_center]
        y_coord = mesh.y[j_center]

    params = ParameterSet(
        x=x_coord,
        y=y_coord,
        side_lenght=float(width * (mesh.x[1] - mesh.x[0])),  # Approximate physical size
    )

    return FieldOfInterest(params, mesh, u.reshape(-1, 1), fill_value=0.0)


def setup_letters_fields(
    letters: Literal["Aa_diff", "Aa", "Aa_rotated", "MW"],
    letters_data_path: Path | None = None,
) -> TestFieldsBundle:
    """Setup letter-based test fields.

    Args:
        letters: Which letter combination to use
        letters_data_path: Path to letter images directory. Defaults to ~/data/
    """
    if letters_data_path is None:
        letters_data_path = Path.home() / "data"
    a_image_path = str(letters_data_path / "A.png")

    def load_transformation(
        translation: tuple[float, float],
        scale: float,
        weight: float,
        rotation_deg: float,
        interpolation_parameter: float,
    ) -> FieldOfInterest:
        return letters_fields.load_image_and_get_transformation(
            image_path=a_image_path,
            translation=translation,
            scale=scale,
            weight=weight,
            rotation_deg=rotation_deg,
            interpolation_parameter=interpolation_parameter,
        )

    match letters:
        case "Aa_diff":
            translation = (0.3, -0.3)
            scale = 0.7
            weight = 0.7
            rotation_deg = 0.0
            f0 = load_transformation(translation, scale, weight, rotation_deg, 0.0)
            f1 = load_transformation(translation, scale, weight, rotation_deg, 1.0)
            f_ref = load_transformation(translation, scale, weight, rotation_deg, 0.5)
        case "Aa_rotated":
            translation = (0.3, -0.3)
            scale = 0.7
            weight = 0.7
            rotation_deg = 45.0
            f0 = load_transformation(translation, scale, weight, rotation_deg, 0.0)
            f1 = load_transformation(translation, scale, weight, rotation_deg, 1.0)
            f_ref = load_transformation(translation, scale, weight, rotation_deg, 0.5)
        case "Aa":
            translation = (0.3, -0.3)
            scale = 0.7
            weight = 1.0
            rotation_deg = 0.0
            f0 = load_transformation(translation, scale, weight, rotation_deg, 0.0)
            f1 = load_transformation(translation, scale, weight, rotation_deg, 1.0)
            f_ref = load_transformation(translation, scale, weight, rotation_deg, 0.5)
        case "MW":
            f0 = load_pre_generated_image("M")
            f1 = load_pre_generated_image("W")
            f_ref = load_pre_generated_image("W")
        case _:
            raise ValueError(f"Unknown letters combination: {letters}")
    mesh = f0.mesh
    velocity_0, velocity_1 = _create_dummy_velocities(mesh)
    f0._name = "$u_0$"
    f1._name = "$u_1$"
    f_ref._name = "$u_{ref}$"

    return TestFieldsBundle(
        f0=f0,
        f1=f1,
        f_ref=f_ref,
        mesh=mesh,
        velocity_0=velocity_0,
        velocity_1=velocity_1,
        description=f"Letter fields: {letters}",
    )


def setup_nascar_fields(
    field_name: PossibleNascarFields,
) -> TestFieldsBundle:
    parameters = [{"F": 300, "A": 4500}, {"F": 500, "A": 4500}, {"F": 400, "A": 4500}]

    from phdtruel.data.nascar_dataloader import NascarDataLoader
    from scripts.mean_nascar.parameters_sets import HP_CONSTANT_PART

    nascar_data = NascarDataLoader(
        is_mean_dynamics=True,
        default_field=field_name,
        constant_parameters=HP_CONSTANT_PART,
    )
    fields = []
    velocities = []
    for p in parameters:
        field = nascar_data.get_field(p, field_parameters=["F"])
        u_field = nascar_data.get_field(p, field_name="u", field_parameters=["F", "A"])
        v_field = nascar_data.get_field(p, field_name="v", field_parameters=["F", "A"])
        smaller_bounds = field.mesh.get_bounds().shrink(0.2)
        fields.append(field.truncate(smaller_bounds))
        velocity = VectorFieldOfInterest(
            [u_field.truncate(smaller_bounds), v_field.truncate(smaller_bounds)],
            name=f"Velocity {p}",
        )
        velocities.append(velocity)

    f0: FieldOfInterest = fields[0]
    f1: FieldOfInterest = fields[1]
    f_ref: FieldOfInterest = fields[2]
    mesh = f0.mesh
    velocity_0 = velocities[0]
    velocity_1 = velocities[1]
    return TestFieldsBundle(
        f0=f0,
        f1=f1,
        f_ref=f_ref,
        mesh=mesh,
        velocity_0=velocity_0,
        velocity_1=velocity_1,
        description=f"NASCAR field: {field_name}",
    )


def setup_dafoam_fields(
    field_name: Literal["T", "U", "p", "rho"],
) -> TestFieldsBundle:
    from phdtruel.data.dafoam_dataloader import DAFoamDataloader

    dafoam_data = DAFoamDataloader()
    f0 = _unwrap_dafoam_field(dafoam_data.get_field({"speed": 240}, field=field_name))
    f1 = _unwrap_dafoam_field(dafoam_data.get_field({"speed": 260}, field=field_name))
    f_ref = _unwrap_dafoam_field(
        dafoam_data.get_field({"speed": 250}, field=field_name)
    )
    mesh = f0.mesh
    velocity_0, velocity_1 = _create_dummy_velocities(mesh)
    return TestFieldsBundle(
        f0=f0,
        f1=f1,
        f_ref=f_ref,
        mesh=mesh,
        velocity_0=velocity_0,
        velocity_1=velocity_1,
        description=f"DAFoam field: {field_name}",
    )


def setup_bump_fields(field_name: JaxFluidsFields) -> TestFieldsBundle:
    from phdtruel.data.jaxfluids_dataloader import JaxFluidsDataloader

    bump_path = phdtruel.config.get("bump_data_path")
    if bump_path is None:
        bump_path = phdtruel.require_config_path("data_dir") / "bump_results_subset"
    else:
        bump_path = Path(bump_path)
    dl = JaxFluidsDataloader(data_path=bump_path)

    f0 = dl.get_field(ParameterSet(mach=3.0, width=-80, height=0.1), field_name)
    f1 = dl.get_field(ParameterSet(mach=2.0, width=-80, height=0.1), field_name)
    f_ref = dl.get_field(ParameterSet(mach=2.5, width=-80, height=0.1), field_name)
    mesh = f0.mesh
    velocity_0, velocity_1 = _create_dummy_velocities(mesh)
    return TestFieldsBundle(
        f0=f0,
        f1=f1,
        f_ref=f_ref,
        mesh=mesh,
        velocity_0=velocity_0,
        velocity_1=velocity_1,
        description=f"JAX-Fluids bump field: {field_name}",
    )


# Registry mapping test case names to their loader functions
_TEST_CASE_REGISTRY: dict[str, Callable[[], TestFieldsBundle]] = {
    # Gaussian cases
    "gaussian_dilate_translate": setup_gaussians_wide_apart,
    "gaussian_translate": setup_gaussian_translate_fields,
    # Square cases
    "square_dilate_translate": partial(
        setup_square_fields, "all_same", "three_different", n_mesh=200
    ),
    "square_dilate_translate_nooverlap": partial(
        setup_square_fields,
        "all_same",
        "three_different",
        overlap_squares=False,
        n_mesh=200,
    ),
    "square_diff_values_dilate_translate": partial(
        setup_square_fields, "three_different", "three_different", n_mesh=200
    ),
    "square_very_diff_values_dilate_translate": partial(
        setup_square_fields, (100.0, 2.0, 51.0), "three_different", n_mesh=200
    ),
    "square_translate": partial(
        setup_square_fields, "all_same", "all_same", n_mesh=200
    ),
    # Letter cases
    "letters_AA_scaled_translated": partial(setup_letters_fields, "Aa"),
    "letters_AA_diff_values_scaled_translated": partial(
        setup_letters_fields, "Aa_diff"
    ),
    "letters_AA_rotated_diff_values_scaled_translated": partial(
        setup_letters_fields, "Aa_rotated"
    ),
    # "letters_MW": partial(setup_letters_fields, "MW"), # Referennce field not easily defined
    # DAFoam cases
    "dafoam_T": partial(setup_dafoam_fields, "T"),
    # "dafoam_U": partial(setup_dafoam_fields, "U"), # Vector field not implemented
    "dafoam_p": partial(setup_dafoam_fields, "p"),
    "dafoam_rho": partial(setup_dafoam_fields, "rho"),
    # JAX-Fluids cases
    "jaxfluids_bump_density": partial(setup_bump_fields, "density"),
    "jaxfluids_bump_pressure": partial(setup_bump_fields, "pressure"),
    "jaxfluids_bump_temperature": partial(setup_bump_fields, "temperature"),
    "jaxfluids_bump_velocity_u": partial(setup_bump_fields, "velocity_u"),
    "jaxfluids_bump_velocity_v": partial(setup_bump_fields, "velocity_v"),
    # NASCAR cases
    "nascar_vorticity": partial(setup_nascar_fields, "vorticity"),
    "nascar_velocity": partial(setup_nascar_fields, "velocity"),
    "nascar_u": partial(setup_nascar_fields, "u"),
    "nascar_v": partial(setup_nascar_fields, "v"),
    "nascar_p": partial(setup_nascar_fields, "p"),
}
# %%
if __name__ == "__main__":
    from phdtruel import visualisations

    case = setup_square_fields(
        values_are="one_five_nine",
        shapes_are="small_small_small",
        n_mesh=40,
        overlap_squares=False,
        centered_along_x=True,
    )
    fig, axes = visualisations.subplots(1, 3)
    visualisations.pcolormesh(fig, axes[0], case.f0, show_mesh=True)
    visualisations.pcolormesh(fig, axes[1], case.f_ref, show_mesh=True)
    visualisations.pcolormesh(fig, axes[2], case.f1, show_mesh=True)

    f = construct_square_field_by_center_and_radius(
        case.f0.mesh,
        center=(15, 15),
        radius=3,
        value=1.0,
        two_by_two_center=False,
    )

    # fig, ax = visualisations.subplots()
    # visualisations.pcolormesh(fig, ax, f, show_mesh=True)
