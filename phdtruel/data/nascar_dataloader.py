import logging
from pathlib import Path
from typing import Literal, Mapping, cast, overload

import nascar
import numpy as np
from matplotlib import pyplot as plt

import phdtruel
from phdtruel.fields import (
    DynamicOfInterest,
    FieldOfInterest,
)
from phdtruel.fields.field_of_interest import (
    PossibleExtrapolationMethods,
)
from phdtruel.fields.meshes import RegularGrid
from phdtruel.fields.parameters import ParameterSet, ParamValue

PossibleNascarParameters = Literal["F", "A", "H", "P"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PossibleNascarFields = Literal["u", "v", "p", "vorticity", "velocity"]

ParameterAsDict = Mapping[str, ParamValue]


class NascarDataLoader:
    """Data loader for NASCAR simulation results.

    Loads field and dynamic data from HDF5 files containing NASCAR simulation results.
    Supports mean dynamics and full time-resolved simulations.
    """

    """Data loader for NASCAR simulation results.
    
    Loads field and dynamic data from HDF5 files containing NASCAR simulation results.
    Supports mean dynamics and full time-resolved simulations.
    """

    def get_possible_parameters(self) -> list[str]:
        """Get list of parameter names available in this dataset."""
        return ["F", "A", "H", "P"]

    def get_possible_parameter_values(
        self, parameter_name: str
    ) -> dict[str, np.ndarray]:
        """Get dict mapping parameter names to arrays of possible values."""
        values = {k: set() for k in self.get_possible_parameters()}
        for param in self.param_files.keys():
            for k, v in param.as_dict().items():
                values[k].add(v)
        return {k: np.array(sorted(list(v))) for k, v in values.items()}

    def __init__(
        self,
        dynamics_folder: Path | str | None = None,
        is_mean_dynamics: bool = False,
        default_field: PossibleNascarFields = "velocity",
        constant_parameters: ParameterAsDict | ParameterSet | None = None,
    ) -> None:
        if dynamics_folder is None and is_mean_dynamics:
            config_path = phdtruel.config["nascar_mean_fields_path"]
            if config_path is None:
                raise ValueError("nascar_mean_fields_path is not set in the config")
            dynamics_folder = Path(str(config_path))
        elif dynamics_folder is None and not is_mean_dynamics:
            config_path = phdtruel.config["nascar_fields_path"]
            if config_path is None:
                raise ValueError("nascar_fields_path is not set in the config")
            dynamics_folder = Path(str(config_path))
        if dynamics_folder is None:
            raise ValueError("NASCAR dynamics folder is not set in the config")

        self.dynamics_folder = Path(dynamics_folder)
        self.default_field_name: PossibleNascarFields = default_field

        if constant_parameters is None:
            constant_parameters = {}
        elif isinstance(constant_parameters, ParameterSet):
            constant_parameters = constant_parameters.as_dict()
        self.constant_parameters = dict(constant_parameters)

        self.default_allow_extrapolation = True
        self.default_extrapolation_method: PossibleExtrapolationMethods = "constant"

        self._last_loaded_parameter = None
        self._last_loaded_dynamic = None

        self.param_files = self._find_all_available_param_and_files()

    @staticmethod
    def separate_time_and_parameter(
        parameter: ParameterSet | ParameterAsDict,
    ) -> tuple[ParameterSet, int | None]:
        if isinstance(parameter, ParameterSet):
            parameter_dict = parameter.as_dict()
        else:
            parameter_dict = dict(parameter)

        if "t" in parameter_dict:
            timestep_value = parameter_dict["t"]
            timestep = int(timestep_value) if timestep_value is not None else None
        else:
            timestep = None

        new_parameter = ParameterSet(
            **{k: parameter_dict[k] for k in parameter_dict.keys() if k != "t"}
        )
        return new_parameter, timestep

    def add_constant_parameter(
        self, parameter: ParameterSet | ParameterAsDict
    ) -> ParameterSet:
        if isinstance(parameter, ParameterSet):
            parameter_dict = parameter.as_dict()
        else:
            parameter_dict = dict(parameter)
        merged = {**self.constant_parameters, **parameter_dict}
        return ParameterSet(**merged)

    def is_parameter_in_dataset(self, value: ParameterAsDict | ParameterSet) -> bool:
        value = self.add_constant_parameter(value)
        for param in self.param_files.keys():
            if param == value:
                return True
        else:
            return False

    def _find_all_available_param_and_files(self) -> dict[ParameterSet, Path]:
        param_files = {}
        for path in sorted(list(self.dynamics_folder.iterdir())):
            if path.suffix != ".hdf5":
                continue
            values = nascar.parse_parameters(path.name)
            mu = ParameterSet(**{n: v for n, v in zip(["F", "A", "H", "P"], values)})

            param_files[mu] = path
        return param_files

    @overload
    def get_field(
        self,
        parameter: ParameterSet | ParameterAsDict,
        field_name: PossibleNascarFields | None = None,
        *,
        field_parameters: list[PossibleNascarParameters] | None = None,
        allow_extrapolation: bool = True,
        return_naca: Literal[True],
    ) -> tuple[FieldOfInterest, np.ndarray]: ...

    @overload
    def get_field(
        self,
        parameter: ParameterSet | ParameterAsDict,
        field_name: PossibleNascarFields | None = None,
        *,
        field_parameters: list[PossibleNascarParameters] | None = None,
        allow_extrapolation: bool = True,
        return_naca: Literal[False] = False,
    ) -> FieldOfInterest: ...

    def get_field(
        self,
        parameter: ParameterSet | ParameterAsDict,
        field_name: PossibleNascarFields | None = None,
        *,
        field_parameters: list[PossibleNascarParameters] | None = None,
        allow_extrapolation: bool = True,
        return_naca: bool = False,
    ) -> FieldOfInterest | tuple[FieldOfInterest, np.ndarray]:
        if field_name is None:
            field_name = self.default_field_name

        full_parameter = self.add_constant_parameter(parameter)
        if field_parameters is None:
            field_parameters = cast(
                list[PossibleNascarParameters], full_parameter.parameters
            )
        dataset_parameter, timestep = self.separate_time_and_parameter(full_parameter)
        if not self.is_parameter_in_dataset(dataset_parameter):
            raise KeyError("Parameter does not exist in dataset.")

        logger.info(f"Loading {self.param_files[dataset_parameter]} t={timestep}")
        dynamic_kwargs: dict[str, int] = {}
        if timestep is not None:
            dynamic_kwargs["specific_timestep"] = timestep
        nascar_dynamic = nascar.Dynamic.from_hdf5_filepath(
            filepath=self.param_files[dataset_parameter],
            **dynamic_kwargs,
        )

        active_parameter_values = {
            k: v for k, v in full_parameter.as_dict().items() if k in field_parameters
        }
        field_parameter_set = ParameterSet(**active_parameter_values)
        mesh = RegularGrid([nascar_dynamic.x_axis, nascar_dynamic.y_axis])

        nx = nascar_dynamic.dimensions.nx
        ny = nascar_dynamic.dimensions.ny

        # in nascar dynamic the shape is not indexed like meshgrid "ij"
        values = getattr(nascar_dynamic, field_name)
        values = values.reshape(ny, nx, order="C").T
        values = values.reshape(-1, 1, order="C")

        field = FieldOfInterest(
            field_parameter_set,
            mesh,
            values=values,
            name=f"{field_name} {field_parameter_set}",
            interpolated=allow_extrapolation,
        )
        if return_naca:
            naca = nascar_dynamic.naca_coordinates
            return field, naca
        return field

    @overload
    def get_dynamic(
        self,
        parameter: ParameterSet | ParameterAsDict,
        field_name: PossibleNascarFields | None = None,
        *,
        field_parameters: list[PossibleNascarParameters] | None = None,
        ratio_to_load: float = 1.0,
        nb_timesteps_to_load: int | None = None,
        specific_timestep: int | None = None,
        allow_extrapolation: bool | None = None,
        extrapolation_method: PossibleExtrapolationMethods | None = None,
        return_naca: Literal[True],
    ) -> tuple[DynamicOfInterest, np.ndarray]: ...

    @overload
    def get_dynamic(
        self,
        parameter: ParameterSet | ParameterAsDict,
        field_name: PossibleNascarFields | None = None,
        *,
        field_parameters: list[PossibleNascarParameters] | None = None,
        ratio_to_load: float = 1.0,
        nb_timesteps_to_load: int | None = None,
        specific_timestep: int | None = None,
        allow_extrapolation: bool | None = None,
        extrapolation_method: PossibleExtrapolationMethods | None = None,
        return_naca: Literal[False] = False,
    ) -> DynamicOfInterest: ...

    def get_dynamic(
        self,
        parameter: ParameterSet | ParameterAsDict,
        field_name: PossibleNascarFields | None = None,
        *,
        field_parameters: list[PossibleNascarParameters] | None = None,
        ratio_to_load: float = 1.0,
        nb_timesteps_to_load: int | None = None,
        specific_timestep: int | None = None,
        allow_extrapolation: bool | None = None,
        extrapolation_method: PossibleExtrapolationMethods | None = None,
        return_naca: bool = False,
    ) -> DynamicOfInterest | tuple[DynamicOfInterest, np.ndarray]:
        if field_name is None:
            field_name = self.default_field_name

        full_parameter = self.add_constant_parameter(parameter)
        if field_parameters is None:
            field_parameters = cast(
                list[PossibleNascarParameters], full_parameter.parameters
            )

        dataset_parameter = full_parameter
        if not self.is_parameter_in_dataset(dataset_parameter):
            raise KeyError("Parameter does not exist in dataset.")

        logger.info(f"Loading {self.param_files[dataset_parameter]}")
        if nb_timesteps_to_load is None and specific_timestep is None:
            nascar_dynamic = nascar.Dynamic.from_hdf5_filepath(
                filepath=self.param_files[dataset_parameter],
                ratio_to_load=ratio_to_load,
            )
        elif specific_timestep is None:
            assert nb_timesteps_to_load is not None
            nascar_dynamic = nascar.Dynamic.from_hdf5_filepath(
                filepath=self.param_files[dataset_parameter],
                ratio_to_load=ratio_to_load,
                nb_timesteps_to_load=nb_timesteps_to_load,
            )
        elif nb_timesteps_to_load is None:
            assert specific_timestep is not None
            nascar_dynamic = nascar.Dynamic.from_hdf5_filepath(
                filepath=self.param_files[dataset_parameter],
                ratio_to_load=ratio_to_load,
                specific_timestep=specific_timestep,
            )
        else:
            nascar_dynamic = nascar.Dynamic.from_hdf5_filepath(
                filepath=self.param_files[dataset_parameter],
                ratio_to_load=ratio_to_load,
                nb_timesteps_to_load=nb_timesteps_to_load,
                specific_timestep=specific_timestep,
            )

        active_parameter_values = {
            k: v for k, v in full_parameter.as_dict().items() if k in field_parameters
        }

        dynamic = DynamicOfInterest.from_nascar_dynamic(
            nascar_dynamic,
            field_name,
            ParameterSet.from_dict(active_parameter_values),
        )

        if return_naca:
            return dynamic, nascar_dynamic.naca_coordinates
        return dynamic

    def _set_defaults(
        self,
        allow_extrapolation: bool | None,
        extrapolation_method: PossibleExtrapolationMethods | None,
        field_name: PossibleNascarFields | None,
    ) -> tuple[bool, PossibleExtrapolationMethods, PossibleNascarFields]:
        if field_name is None:
            field_name = self.default_field_name
        if allow_extrapolation is None:
            allow_extrapolation = self.default_allow_extrapolation
        if extrapolation_method is None:
            extrapolation_method = self.default_extrapolation_method
        return allow_extrapolation, extrapolation_method, field_name


def load_nascar_fields(
    field_name: PossibleNascarFields,
    parameters: list[ParameterAsDict],
    target_parameter: ParameterAsDict,
    constant_parameters: ParameterAsDict,
    is_mean_dynamic: bool,
    active_parameters: list[PossibleNascarParameters] | None = None,
) -> tuple[list[FieldOfInterest], FieldOfInterest]:
    if active_parameters is None:
        active_parameters = cast(
            list[PossibleNascarParameters], list(set(target_parameter.keys()))
        )

    nascar_data = NascarDataLoader(
        is_mean_dynamics=is_mean_dynamic,
        default_field=field_name,
        constant_parameters=constant_parameters,
    )

    target_field = nascar_data.get_field(
        target_parameter, field_parameters=active_parameters
    )
    smaller_bounds = target_field.mesh.get_bounds().shrink(0.2)

    target_field = target_field.truncate(smaller_bounds)
    fields = []
    for p in parameters:
        field = nascar_data.get_field(p, field_parameters=active_parameters)
        fields.append(field.truncate(smaller_bounds))
    return fields, target_field


def write_two_snapshots_for_cpp():
    nascar_data = NascarDataLoader(is_mean_dynamics=True, default_field="vorticity")
    f0 = nascar_data.get_field({"A": 3500, "F": 400})
    f1 = nascar_data.get_field({"A": 4500, "F": 400})

    nx = 100
    x_axis = np.linspace(1.0, 7.0, nx)
    y_axis = np.linspace(-3.0, 3.0, nx)
    mesh = RegularGrid((x_axis, y_axis))
    points = mesh.points

    values_0 = np.abs(f0(points))
    values_0 = values_0 / np.sum(values_0)
    values_1 = np.abs(f1(points))
    values_1 = values_1 / np.sum(values_1)
    field_0 = FieldOfInterest(None, mesh, values_0)
    field_1 = FieldOfInterest(None, mesh, values_1)

    plt.contourf(
        mesh.x,
        mesh.y,
        field_0.values.reshape(mesh.shape),
        levels=100,
    )
    plt.colorbar()
    plt.show()
    plt.contourf(
        mesh.x,
        mesh.y,
        field_1.values.reshape(mesh.shape),
        levels=100,
    )
    plt.colorbar()
    plt.show()

    import h5py

    fields_dir = phdtruel.require_config_path("data_dir") / "fields"
    fields_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(fields_dir / "f0.hdf5", "w") as f0_file:
        field_0.save_in_hdf5(f0_file)
    with h5py.File(fields_dir / "f1.hdf5", "w") as f1_file:
        field_1.save_in_hdf5(f1_file)
