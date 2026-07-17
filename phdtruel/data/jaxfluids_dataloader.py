import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from phdtruel.data.data import extract_time_from_parameter
from phdtruel.fields import DynamicOfInterest, FieldOfInterest
from phdtruel.fields.meshes import RegularGrid
from phdtruel.fields.parameters import ParameterSet

logger = logging.getLogger(__name__)


type JaxFluidsFields = Literal[
    "density",
    "pressure",
    "temperature",
    "velocity_u",
    "velocity_v",
]
JaxFluidsData = dict[
    Literal[
        "x_axis",
        "y_axis",
    ]
    | JaxFluidsFields,
    np.ndarray,
]


def _load_timestep_data(timestep_file: Path) -> JaxFluidsData:
    """
    Load data from a specific timestep hdf5 file.

    Args:
        timestep_path (Path): The path to the hdf5 file for the specific timestep.
    Returns:
        MappingProxyType: A read-only view of the data contained in the hdf5 file.
    """
    with h5py.File(timestep_file, "r") as f:
        x_axis = f["domain/gridX"][:]
        y_axis = f["domain/gridY"][:]

        shape = (len(y_axis), len(x_axis))
        density = f["primitives/density"][:].reshape(shape, order="C")
        pressure = f["primitives/pressure"][:].reshape(shape, order="C")
        temperature = f["primitives/temperature"][:].reshape(shape, order="C")
        velocity = f["primitives/velocity"][:].reshape(shape + (2,), order="C")

        _ = f["levelset/levelset"][:].reshape(shape, order="C")
        _ = f["levelset/volume_fraction"][:].reshape(shape, order="C")

        velocity_u = velocity[..., 0]
        velocity_v = velocity[..., 1]

    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "density": density,
        "pressure": pressure,
        "temperature": temperature,
        "velocity_u": velocity_u,
        "velocity_v": velocity_v,
    }


def _parse_dynamic_folder_name(folder_name: str) -> ParameterSet:
    """
    Parse a folder name in the format p1_value1_p2_value2_... into a dictionary
    of parameter names and values.
    """
    parts = folder_name.split("_")

    # Expected format: p1_value1_p2_value2_...
    if len(parts) % 2 != 0:
        raise ValueError(
            f"Unexpected folder name format: {folder_name}\n"
            "Expected format: p1_value1_p2_value2_..."
        )

    param_dict = {}
    for i in range(0, len(parts), 2):
        param_name = parts[i]
        param_value = float(parts[i + 1])
        param_dict[param_name] = param_value

    return ParameterSet.from_dict(param_dict)


def _available_timesteps_for_run(run_folder: Path) -> dict[float, Path]:
    """
    List available timesteps for a given run folder.

    Args:
        run_folder (Path): The folder containing simulation results for a specific run.

    Returns:
        dict[float, Path]: A dictionary mapping available timesteps to their corresponding hdf5 result paths.
    """
    timesteps = {}

    for result_file in (run_folder / "bump/domain/").glob("*.h5"):
        # File are named like data_0.0100223344.h5
        name = result_file.stem
        if name.startswith("data_"):
            try:
                timestep = float(name[len("data_") :])
                timesteps[timestep] = result_file
            except ValueError:
                print(
                    f"Skipping file with unexpected name format: {result_file.name}\n"
                    "Expected format: data_<timestep>.h5"
                )
                continue
    timesteps = dict(sorted(timesteps.items()))
    return timesteps


class JaxFluidsDataloader:
    """Data loader for JAX-Fluids simulation results.

    Loads fields from HDF5 files organized in parameter-specific directories.
    Each directory contains multiple timesteps of simulation data.
    """

    def __init__(self, data_path: Path) -> None:
        self._data_path = Path(data_path)

        self._possible_parameter: dict[str, set] = defaultdict(set)
        self._dynamic_folders: dict[ParameterSet, Path] = {}
        for dynamic_folder in self._data_path.iterdir():
            if not dynamic_folder.is_dir():
                logger.warning(
                    f"Skipping non-directory item in data path: {dynamic_folder}"
                )
                continue

            try:
                parameter = _parse_dynamic_folder_name(dynamic_folder.name)
                for k, v in parameter.as_dict().items():
                    self._possible_parameter[k].add(v)
                self._dynamic_folders[parameter] = dynamic_folder
            except ValueError as e:
                logger.warning(str(e))
                continue

    def get_possible_parameters(self) -> list[str]:
        """Get list of parameter names available in this dataset."""
        return list(self._possible_parameter.keys())

    def get_possible_parameter_values(self, parameter_name: str) -> list[float]:
        """Get list of possible values for a specific parameter."""
        return list(sorted(self._possible_parameter[parameter_name]))

    def validate_parameter(self, parameter: ParameterSet) -> bool:
        """Check if a parameter set exists in the dataset."""
        return parameter in self._dynamic_folders

    def get_field(
        self,
        parameter: ParameterSet,
        field_name: JaxFluidsFields,
        time_as_parameter: bool = False,
    ) -> FieldOfInterest:
        """Load a field for the given parameters and time.

        Args:
            parameter: Parameter set (may include 't' for time)
            field_name: Name of the field to load
            time_as_parameter: If True, keep time in the returned field's parameter

        Returns:
            The loaded field
        """
        time, param_without_t = extract_time_from_parameter(parameter)

        if not self.validate_parameter(param_without_t):
            raise ValueError(f"Invalid parameter set: {parameter}")

        if time is None:
            logger.warning("No time parameter found, defaulting to last timestep.")
            time = "last"

        dynamic_folder = self._dynamic_folders.get(param_without_t, None)
        if dynamic_folder is None:
            raise ValueError(f"No dynamic found for parameter set: {param_without_t}")

        if time == "average":
            raise NotImplementedError("Average over time not implemented yet.")

        all_times_for_dynamic = _available_timesteps_for_run(dynamic_folder)
        if len(all_times_for_dynamic) == 0:
            raise ValueError(f"No timesteps found for dynamic: {dynamic_folder}")

        if time == "last":
            time = max(all_times_for_dynamic.keys())

        if time not in all_times_for_dynamic:
            raise ValueError(
                f"Timestep {time} not found for dynamic: {dynamic_folder}\n"
                f"Available timesteps: {list(all_times_for_dynamic.keys())}"
            )

        timestep_file = all_times_for_dynamic[time]
        data = _load_timestep_data(timestep_file)

        field_data = data[field_name].reshape(-1, 1, order="F")
        if np.count_nonzero(np.isnan(field_data)) > 0:
            raise ValueError(
                f"Field data contains NaN values for parameter set: {parameter}"
            )

        field_parameter = parameter if time_as_parameter else param_without_t
        mesh = RegularGrid([data["x_axis"], data["y_axis"]])
        return FieldOfInterest(field_parameter, mesh, field_data, name=field_name)

    def __call__(self, parameter: ParameterSet, **kwargs) -> FieldOfInterest:
        return self.get_field(parameter, **kwargs)

    def get_dynamic(self, parameter: ParameterSet, **kwargs) -> DynamicOfInterest:
        """Load a dynamic (time series) for the given parameters.

        Not yet implemented for JAX-Fluids data.
        """
        raise NotImplementedError()

    def get_dynamic_loader(self):
        return self.get_dynamic
