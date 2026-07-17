from pathlib import Path
from typing import Literal, cast

from matplotlib import pyplot as plt
from scipy.interpolate import (
    LinearNDInterpolator,
)

from phdtruel import visualisations
import h5py
import numpy as np

from phdtruel.fields import FieldOfInterest
from phdtruel.fields.meshes import DomainBounds, PointCloud, RegularGrid
from phdtruel.fields.parameters import ParameterSet
from scripts.mapping_study.plots import generate_checkerboard_field


class DAFoamDataloader:
    def __init__(
        self,
        data_folder: str | Path | None = None,
    ):
        if data_folder is None:
            import phdtruel

            data_folder = phdtruel.config["dafoam_fields_path"]
        if data_folder is None:
            raise ValueError("dafoam_fields_path is not set in the config")

        self.data_folder = Path(data_folder)
        self._last_loaded_parameter = None
        self._last_loaded_dynamic = None

        self.param_files = self._find_all_available_param_and_files()

    def _find_all_available_param_and_files(self) -> dict[ParameterSet, Path]:
        param_files: dict[ParameterSet, Path] = {}
        for path in sorted(self.data_folder.iterdir()):
            if path.suffix != ".h5":
                continue
            speed = int(path.stem[6:9])
            mu = ParameterSet(speed=speed)

            param_files[mu] = path
        return param_files

    def get_field(
        self,
        parameter: ParameterSet | dict[str, int] | dict[str, float],
        field: Literal["T", "U", "p", "rho"],
        return_naca_upper: bool = False,
        return_naca_lower: bool = False,
        x_axis: np.ndarray | None = None,
        y_axis: np.ndarray | None = None,
    ) -> (
        FieldOfInterest
        | tuple[FieldOfInterest, np.ndarray]
        | tuple[FieldOfInterest, np.ndarray, np.ndarray]
    ):
        if field == "U":
            raise NotImplementedError("Vector field not implemented yet")

        if isinstance(parameter, dict):
            parameter_dict = cast(dict[str, int | float], parameter)
            parameter = ParameterSet(**parameter_dict)

        if parameter not in self.param_files.keys():
            raise ValueError(f"Parameter {parameter} not found in dataset")

        file_path = self.param_files[parameter]

        with h5py.File(file_path, "r") as f:
            coords = np.array(f["mesh/points"])
            values = np.array(f[f"fields/{field}"])
            pressure_side = np.array(f["airfoil/pressure_side"])
            suction_side = np.array(f["airfoil/suction_side"])

        # Transform to cartesian mesh in 2d
        if x_axis is None:
            x_axis = np.linspace(-0.2, 1.5, 201)
        if y_axis is None:
            y_axis = np.linspace(-1.0, 1.0, 201)

        mesh = PointCloud(coords[:, :2])
        field_obj = FieldOfInterest(parameter, mesh, values)

        cartesian_mesh = RegularGrid([x_axis, y_axis])

        values_in_regular_grid = field_obj.eval(cartesian_mesh.points)
        field_in_cartesian_mesh = FieldOfInterest(
            parameter, cartesian_mesh, values_in_regular_grid
        )

        self._last_loaded_parameter = parameter
        self._last_loaded_dynamic = field_obj

        if return_naca_upper and return_naca_lower:
            return field_in_cartesian_mesh, suction_side, pressure_side
        if return_naca_upper:
            return field_in_cartesian_mesh, suction_side
        if return_naca_lower:
            return field_in_cartesian_mesh, pressure_side
        return field_in_cartesian_mesh

    def get_upper_part_squared_field(
        self, speed: int, field_name: Literal["T", "U", "p", "rho"]
    ):
        field_result = self.get_field(
            ParameterSet(speed=speed), field=field_name, return_naca_upper=True
        )
        field, naca_points = cast(tuple[FieldOfInterest, np.ndarray], field_result)


if __name__ == "__main__":
    dafo = DAFoamDataloader()

    field_result = dafo.get_field(
        ParameterSet(speed=240), field="rho", return_naca_upper=True
    )
    field, naca_points = cast(tuple[FieldOfInterest, np.ndarray], field_result)

    field: FieldOfInterest = field.truncate(
        DomainBounds.by_axis(
            (field.mesh.x.min(), field.mesh.x.max()),
            (0.0, 1.0),
        )
    )
    mesh: RegularGrid = field.mesh

    # left, right, top, bottom = get_four_borders(mesh.get_bounds(), n=20)

    naca_points_end = np.zeros_like(naca_points)
    naca_points_end[:, 0] = naca_points[:, 0]
    naca_points_end[:, 1] = 0.0
    naca_displacement = naca_points_end - naca_points

    borders = mesh.get_border_points()
    border_displacement = np.zeros_like(borders)

    control_points = np.vstack((naca_points, borders))
    cp_displacements = np.vstack((naca_displacement, border_displacement))

    # control_points = naca_points
    # cp_displacements = naca_displacement
    # LinearNDInterpolator
    mapping = LinearNDInterpolator(
        control_points,
        cp_displacements,
    )
    displaced_points = control_points + mapping(control_points)

    # Plot the naca boundary displacement
    fig, ax = visualisations.subplots(1, 1)
    ax.plot(
        control_points[:, 0], control_points[:, 1], "k.", markersize=0.4, label="cp"
    )
    ax.plot(
        naca_points_end[:, 0],
        naca_points_end[:, 1],
        "r.",
        markersize=0.4,
        label="naca displaced",
    )
    ax.plot(
        displaced_points[:, 0],
        displaced_points[:, 1],
        "b+",
        markersize=0.4,
        label="all displaced by rbf",
    )
    ax.legend(loc="upper right")
    ax.grid()
    plt.show()

    visu_mesh = RegularGrid(
        [
            np.linspace(mesh.x.min(), mesh.x.max(), 30),
            np.linspace(mesh.y.min(), mesh.y.max(), 30),
        ]
    )

    # Plot the mapping with quiver points on the visu mesh
    fig, ax = visualisations.subplots(1, 1)
    displaced_visu_mesh = visu_mesh.points + mapping(visu_mesh.points)
    ax.quiver(
        visu_mesh.points[:, 0],
        visu_mesh.points[:, 1],
        displaced_visu_mesh[:, 0] - visu_mesh.points[:, 0],
        displaced_visu_mesh[:, 1] - visu_mesh.points[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    ax.set_aspect("equal")
    plt.show()

    displaced_mesh = mesh.points + mapping(mesh.points)

    inverse_mapping = LinearNDInterpolator(displaced_mesh, mesh.points)
    damier = generate_checkerboard_field(mesh, taille_carre=20)

    # Plot the defomred damier with pcolormesh
    fig, ax = visualisations.subplots(1, 1)
    damier_deforme = damier.eval(inverse_mapping(mesh.points), as_foi=True)
    visualisations.pcolormesh(fig, ax, damier_deforme)
    ax.set_aspect("equal")
    plt.show()

    # Finnaly deform the field
    field_deformed = field.eval(inverse_mapping(mesh.points), as_foi=True)

    fig, (ax1, ax2, ax3, ax4) = visualisations.subplots(1, 4)
    visualisations.pcolormesh(
        fig,
        ax1,
        field_deformed,
        # norm="minmax",
        cmap="RdBu_r",
        # norm=Normalize(vmin=0, vmax=90000),
    )
    ax1.grid(False)

    visualisations.pcolormesh(fig, ax2, field, cmap="RdBu_r")
    ax2.plot(naca_points[:, 0], naca_points[:, 1], "k-", linewidth=0.5, label="naca")
    ax2.grid(False)

    visualisations.pcolormesh(
        fig,
        ax3,
        field_deformed.eval(displaced_mesh, as_foi=True),
        # norm="minmax",
        cmap="RdBu_r",
        # norm=Normalize(vmin=0, vmax=90000),
    )
    ax3.plot(naca_points[:, 0], naca_points[:, 1], "k-", linewidth=0.5, label="naca")

    visualisations.pcolormesh(
        fig,
        ax4,
        field - field_deformed.eval(displaced_mesh, as_foi=True),
        # norm="minmax",
        cmap="RdBu_r",
        # norm=Normalize(vmin=0, vmax=90000),
    )
    ax4.plot(naca_points[:, 0], naca_points[:, 1], "k-", linewidth=0.5, label="naca")

    visualisations.savefig(fig, "fields.png")

    plt.show()

    print(field)

    # field = dafo.get_upper_part_squared_field(speed=240, field_name="p")
