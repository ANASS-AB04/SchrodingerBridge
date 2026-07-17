import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from phdtruel import printable_path, visualisations
from phdtruel.errors import (
    commit_errors_to_db,
    mesure_errors,
    get_error,
    optimize_parameters,
)
from phdtruel.fields import (
    FieldOfInterest,
    DynamicOfInterest,
)
from phdtruel.fields.meshes import DomainBounds, RegularGrid
from phdtruel.fields.parameters import ParameterSet
from phdtruel.visualisations import name_generator, get_plot_subfolder

logger = logging.getLogger(__name__)


@dataclass
class ExperimentCompagnon:
    _train_data: list[FieldOfInterest] | list[DynamicOfInterest]
    _target_data: FieldOfInterest | DynamicOfInterest
    truncation: DomainBounds
    plot_folder_name: str | None = None
    norm: mcolors.Normalize | None = None
    cmap: mcolors.Colormap | None = None

    def __post_init__(self):
        self.is_steady = isinstance(self._target_data, FieldOfInterest)

        self._results: dict[str, np.ndarray | DynamicOfInterest | FieldOfInterest] = {}
        self._filehandler: logging.FileHandler | None = None
        self.savefig: Callable[..., Any] | None = None

        # Type narrowing: properly handle both types
        if self.is_steady:
            calibration_field = self._target_data
            assert isinstance(calibration_field, FieldOfInterest)
        else:
            dynamic_field = self._target_data
            assert isinstance(dynamic_field, DynamicOfInterest)
            calibration_field = dynamic_field[0]
        self._calibration_field: FieldOfInterest = calibration_field

        if self.norm is None:
            # TODO : Compute the norm on the truncation of the target values
            mean = np.mean(calibration_field.values)
            max_ = np.max(calibration_field.values)
            min_ = np.min(calibration_field.values)
            halfrange = max(max_ - mean, mean - min_)
            self.norm = mcolors.CenteredNorm(
                vcenter=float(mean), halfrange=float(halfrange)
            )
        if self.cmap is None:
            self.cmap = matplotlib.colormaps["YlOrRd"]
            self.cmap.set_under("yellow")
            self.cmap.set_over("red")
            self.cmap.set_bad("black")

    def setup_custom_plot_folder_name(self, plot_folder_name: str):
        self.plot_folder_name = plot_folder_name
        return self._init_plot_folder()

    def setup_plot_folder_from_data(
        self, method_name: str, data_source: str, parameters: list[ParameterSet]
    ) -> Path:
        self.plot_folder_name = name_generator(
            method=method_name,
            test_case=f"{data_source}_{self._target_data.name}",
            parameters=parameters,
            target_param=self._target_data.parameter,
        )
        return self._init_plot_folder()

    def _init_plot_folder(self) -> Path:
        assert self.plot_folder_name is not None
        plot_folder = get_plot_subfolder(self.plot_folder_name)
        plot_folder.mkdir(exist_ok=True)
        logger.info(f"Plot folder is {plot_folder}")
        self._add_file_handler_logs(plot_folder)
        self.savefig = partial(visualisations.savefig, subfolder=self.plot_folder_name)
        # TODO : Add a commit_to_db_for_this_case function here
        return plot_folder

    def _add_file_handler_logs(self, plot_folder: Path) -> None:
        logfile = plot_folder / "logs.txt"
        file_handler = logging.FileHandler(logfile)
        logger.info(
            f"Logs will also be saved under {printable_path(logfile.parent)} for this test case"
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
        self._filehandler = file_handler

    def __del__(self):
        plt.close("all")
        if self._filehandler is not None:
            logger.removeHandler(self._filehandler)

    @property
    def target_field(self):
        if not self.is_steady:
            raise ValueError("The experiment is not steady. Use self.target_dynamic.")
        return self._target_data

    @property
    def fields(self):
        if not self.is_steady:
            raise ValueError("The experiment is not steady. Use self.dynamics.")
        return self._train_data

    @property
    def target_dynamic(self):
        if self.is_steady:
            raise ValueError("The experiment is steady. Use self.target_field.")
        return self._target_data

    @property
    def dynamics(self):
        if self.is_steady:
            raise ValueError("The experiment is steady. Use self.fields.")
        return self._train_data

    def get_dynamic_from_fields(
        self,
        timesteps: list[int | float] | np.ndarray,
        parameter: ParameterSet | dict,
    ) -> DynamicOfInterest:
        """Convert list of fields to dynamic of interest.

        Args:
            timesteps: Time steps for each field
            parameter: Parameter set or dict

        Returns:
            DynamicOfInterest with the provided fields and timesteps
        """
        raise NotImplementedError("get_dynamic_from_fields not yet implemented")

    def add_result(self, name: str, values: FieldOfInterest | np.ndarray, hook=None):
        self._results[name] = values
        self._last_result_name = name
        if hook is not None:
            hook(self, name, values)

    @property
    def test_parameter(self):
        return self._calibration_field.parameter

    @property
    def test_points(self):
        return self._calibration_field.mesh.points

    @property
    def test_mesh(self):
        return self._calibration_field.mesh

    @property
    def test_points_shape(self):
        return self.x.shape

    @property
    def x(self) -> np.ndarray:
        mesh = self._calibration_field.mesh
        if not isinstance(mesh, RegularGrid):
            raise TypeError("ExperimentCompagnon.x requires a RegularGrid mesh")
        return mesh.x

    @property
    def y(self) -> np.ndarray:
        mesh = self._calibration_field.mesh
        if not isinstance(mesh, RegularGrid):
            raise TypeError("ExperimentCompagnon.y requires a RegularGrid mesh")
        return mesh.y

    @property
    def results(self):
        return self._results

    def get_contourf_kwargs(self) -> dict[str, Any]:
        return self.get_kwargs("x", "y", "norm", "cmap")

    def get_kwargs(self, *names) -> dict[str, Any]:
        return {n: getattr(self, n) for n in names}


def compute_error_table(exp: ExperimentCompagnon, test_case: str):
    commit_to_db_for_this_case = partial(
        commit_errors_to_db,
        test_case=test_case,
        field_name=exp.target_field.name,
        plot_folder=exp.plot_folder_name,
        truncation=f"{exp.truncation}",
    )
    errors = []
    for method, values in exp.results.items():
        err = commit_to_db_for_this_case(
            mesure_errors(
                u=values(exp.test_points),
                u_ref=exp.target_field(exp.test_points),
                mesh=exp.test_mesh,
                truncation=exp.truncation,
            ),
            method=method,
        )
        errors.append(err)
    df_of_errors = pd.concat(
        errors,
        ignore_index=True,
    )
    return df_of_errors


def minimize(
    method: Callable[[list[float]], FieldOfInterest],
    exp: ExperimentCompagnon,
    error: Literal["l_2", "w_2"] = "l_2",
):
    error_func = get_error(name=error, mesh=exp.test_mesh, truncation=exp.truncation)
    logger.info(f"Optimizing s with {error}")
    optimal_param, result_got_optimized = optimize_parameters(
        method=method,
        reference=exp.target_field,
        parameters_to_optimize={
            "target_parameter": np.ones(2) / 2.0,
        },
        bounds={
            "target_parameter": (np.zeros(2), np.ones(2)),
        },
        error_mesure=error_func,
        return_best=True,
    )
    return result_got_optimized
