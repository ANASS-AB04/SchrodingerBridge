"""Data loader for GenerativePDE raw simulation bundles."""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from phdtruel.data.data import extract_time_from_parameter
from phdtruel.fields import DynamicOfInterest, FieldOfInterest
from phdtruel.fields.meshes import PointCloud, RegularGrid
from phdtruel.fields.parameters import ParameterSet, ParamValue

logger = logging.getLogger(__name__)

GenerativePDEFields = Literal[
    "rho",
    "u",
    "v",
    "p",
    "mach",
    "mach_fluctuation",
    "grad_rho",
    "grad_u",
    "grad_v",
    "grad_p",
    "schlieren",
]

_SCALAR_FIELD_NAMES = frozenset(
    {"rho", "u", "v", "p", "mach", "mach_fluctuation", "schlieren"}
)
_GRAD_FIELD_NAMES = frozenset({"grad_rho", "grad_u", "grad_v", "grad_p"})

# Keys that are not physical parameters when discovered generically.
_NON_PARAMETER_KEYS = frozenset(
    {
        "case",
        "mesh_path",
        "n_cells",
        "flux",
        "reconstruction",
        "time_scheme",
        "time",
        "node_pos",
        "node_area",
        "conservatives",
        "primitives",
        "primitives_grad",
        "mach",
    }
)

# Map exported metadata keys to canonical parameter names.
_PARAMETER_ALIASES: dict[str, str] = {
    "mach_in": "mach",
    "aoa_in": "aoa",
    "width_in": "width",
    "height_in": "height",
}

_DIAMOND_FILENAME_RE = re.compile(
    r"AOA([+-]?[0-9.]+)_M([0-9.]+)_.*_t([0-9.]+)\.npz$"
)



def _read_scalar(arr: np.ndarray) -> float:
    return float(np.asarray(arr).reshape(-1)[0])


def _is_scalar_array(arr: np.ndarray) -> bool:
    return arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1)


def _discover_parameters_from_npz(
    data: np.lib.npyio.NpzFile,
) -> tuple[ParameterSet, float]:
    """Build a ParameterSet from scalar metadata stored in a bundle."""
    params: dict[str, ParamValue] = {}

    for src_key, param_name in _PARAMETER_ALIASES.items():
        if src_key in data:
            params[param_name] = _read_scalar(data[src_key])

    if "h" in data and "h" not in params:
        params["h"] = _read_scalar(data["h"])

    for key in data.files:
        if key in _NON_PARAMETER_KEYS or key in _PARAMETER_ALIASES or key == "h":
            continue
        arr = np.asarray(data[key])
        if not _is_scalar_array(arr):
            continue
        if not np.issubdtype(arr.dtype, np.number):
            continue
        params[key] = _read_scalar(arr)

    time = _read_scalar(data["time"]) if "time" in data else 0.0
    return ParameterSet(**params), time


def _parse_diamond_filename(path: Path) -> tuple[float, float, float] | None:
    match = _DIAMOND_FILENAME_RE.search(path.name)
    if match is None:
        return None
    aoa, mach, time = match.groups()
    return float(aoa), float(mach), float(time)


def _parse_h_from_folder(path: Path) -> float | None:
    for parent in path.parents:
        if parent.name.startswith("h"):
            try:
                return float(parent.name[1:])
            except ValueError:
                return None
    return None


def _parameter_tuple(parameter: ParameterSet) -> tuple[tuple[str, float], ...]:
    return tuple(
        sorted(
            (name, float(np.asarray(getattr(parameter, name)).item()))
            for name in parameter.parameters
        )
    )


@dataclass(frozen=True, slots=True)
class _IndexedRun:
    path: Path
    parameter: ParameterSet
    time: float
    case: str


def _load_field_array(
    data: np.lib.npyio.NpzFile,
    field_name: GenerativePDEFields,
    *,
    mach_reference: float | None = None,
) -> np.ndarray:
    prims = np.asarray(data["primitives"])
    grad = np.asarray(data["primitives_grad"])

    if field_name == "rho":
        return prims[:, 0]
    if field_name == "u":
        return prims[:, 1]
    if field_name == "v":
        return prims[:, 2]
    if field_name == "p":
        return prims[:, 3]
    if field_name == "mach":
        return np.asarray(data["mach"])
    if field_name == "mach_fluctuation":
        if mach_reference is None:
            raise ValueError(
                "mach_reference is required to compute 'mach_fluctuation'"
            )
        return np.asarray(data["mach"]) - mach_reference
    if field_name == "schlieren":
        grad_p = grad[:, 3, :]
        return np.log1p(np.linalg.norm(grad_p, axis=1))
    if field_name == "grad_rho":
        return grad[:, 0, :]
    if field_name == "grad_u":
        return grad[:, 1, :]
    if field_name == "grad_v":
        return grad[:, 2, :]
    if field_name == "grad_p":
        return grad[:, 3, :]

    raise ValueError(f"Unknown field '{field_name}'")


class GenerativePDEDataloader:
    """Data loader for GenerativePDE raw `.npz` simulation bundles.

    Parameters are discovered from bundle metadata when available. For the
    current ``diamond`` exports this yields ``h``, ``aoa``, and ``mach``; other
    cases can expose different scalar metadata without changing the loader API.
    """

    def __init__(
        self,
        data_path: Path | str | None = None,
        *,
        case: str | None = "diamond",
    ) -> None:
        if data_path is None:
            import phdtruel

            config_path = phdtruel.config.get("generative_pde_raw_path")
            if config_path is None:
                raise ValueError("generative_pde_raw_path is not set in the config")
            data_path = config_path

        self._data_path = Path(data_path)
        self._case_filter = case

        self._possible_parameter: dict[str, set[ParamValue]] = defaultdict(set)
        self._indexed_runs: list[_IndexedRun] = []
        self._discover_runs()

    def _match_parameter(self, parameter: ParameterSet) -> _IndexedRun | None:
        return self._match_parameter_in_runs(self._indexed_runs, parameter)

    @staticmethod
    def _match_parameter_in_runs(
        runs: list[_IndexedRun],
        parameter: ParameterSet,
    ) -> _IndexedRun | None:
        for run in runs:
            if run.parameter == parameter:
                return run
        return None

    @property
    def data_path(self) -> Path:
        return self._data_path

    @property
    def case_filter(self) -> str | None:
        return self._case_filter

    def _discover_runs(self) -> None:
        if not self._data_path.exists():
            raise FileNotFoundError(f"GenerativePDE raw path not found: {self._data_path}")

        case_dirs = []
        if self._case_filter is not None:
            case_root = self._data_path / self._case_filter
            if not case_root.exists():
                raise FileNotFoundError(
                    f"No case data under {case_root} for case={self._case_filter!r}"
                )
            case_dirs = [case_root]
        else:
            case_dirs = [
                path for path in sorted(self._data_path.iterdir()) if path.is_dir()
            ]
            if not case_dirs:
                raise FileNotFoundError(f"No case folders found under {self._data_path}")

        best: dict[tuple[str, tuple[tuple[str, float], ...]], _IndexedRun] = {}

        for case_root in case_dirs:
            case_name = case_root.name
            for npz_path in sorted(case_root.rglob("*.npz")):
                if npz_path.name == "graph.npz":
                    continue

                with np.load(npz_path) as data:
                    parameter, time = _discover_parameters_from_npz(data)

                if not parameter.parameters and case_name == "diamond":
                    parsed = _parse_diamond_filename(npz_path)
                    h_value = _parse_h_from_folder(npz_path)
                    if parsed is not None and h_value is not None:
                        aoa, mach, time = parsed
                        parameter = ParameterSet(h=h_value, aoa=aoa, mach=mach)

                if not parameter.parameters:
                    logger.warning("Skipping bundle with no discoverable parameters: %s", npz_path)
                    continue

                dedup_key = (case_name, _parameter_tuple(parameter))
                run = _IndexedRun(
                    path=npz_path,
                    parameter=parameter,
                    time=time,
                    case=case_name,
                )
                existing = best.get(dedup_key)
                if existing is None or run.time > existing.time:
                    best[dedup_key] = run

        if not best:
            raise FileNotFoundError(
                f"No GenerativePDE bundles found under {self._data_path}"
            )

        indexed_runs: list[_IndexedRun] = []
        for run in best.values():
            existing = self._match_parameter_in_runs(indexed_runs, run.parameter)
            if existing is None:
                indexed_runs.append(run)
                continue
            if run.time > existing.time:
                if self._case_filter is None and existing.case != run.case:
                    logger.warning(
                        "Parameter collision across cases %s and %s for %s; "
                        "keeping latest snapshot from %s.",
                        existing.case,
                        run.case,
                        run.parameter,
                        run.case,
                    )
                indexed_runs.remove(existing)
                indexed_runs.append(run)

        self._indexed_runs = indexed_runs
        self._possible_parameter = defaultdict(set)
        for run in self._indexed_runs:
            for name, value in run.parameter.as_dict().items():
                self._possible_parameter[name].add(value)

    def get_possible_parameters(self) -> list[str]:
        """Return parameter names available in the indexed dataset."""
        return sorted(self._possible_parameter.keys())

    def get_possible_parameter_values(self, parameter_name: str) -> list[ParamValue]:
        """Return sorted possible values for one parameter name."""
        if parameter_name not in self._possible_parameter:
            raise KeyError(f"Unknown parameter name: {parameter_name}")
        values = self._possible_parameter[parameter_name]
        return sorted(values, key=lambda value: (isinstance(value, str), value))

    def validate_parameter(self, parameter: ParameterSet) -> bool:
        """Check whether a parameter set exists in the indexed dataset."""
        _, param_without_t = extract_time_from_parameter(parameter)
        return self._match_parameter(param_without_t) is not None

    def is_parameter_in_dataset(
        self,
        parameter: ParameterSet | dict[str, ParamValue],
    ) -> bool:
        """Alias for ``validate_parameter`` with dict support."""
        if isinstance(parameter, dict):
            parameter = ParameterSet(**parameter)
        return self.validate_parameter(parameter)

    def get_field(
        self,
        parameter: ParameterSet,
        field_name: GenerativePDEFields,
        *,
        time_as_parameter: bool = False,
        x_axis: np.ndarray | None = None,
        y_axis: np.ndarray | None = None,
    ) -> FieldOfInterest:
        """Load one field for the given parameters."""
        if field_name not in _SCALAR_FIELD_NAMES | _GRAD_FIELD_NAMES:
            raise ValueError(f"Unknown field name: {field_name}")

        time, param_without_t = extract_time_from_parameter(parameter)
        indexed = self._match_parameter(param_without_t)
        if indexed is None:
            raise ValueError(f"Invalid parameter set: {parameter}")
        if time is not None and not np.isclose(time, indexed.time):
            raise ValueError(
                f"Requested time {time} not available for parameter set: {param_without_t}. "
                f"Latest indexed time is {indexed.time}."
            )

        mach_reference = (
            float(param_without_t.mach) if field_name == "mach_fluctuation" else None
        )
        with np.load(indexed.path) as data:
            node_pos = np.asarray(data["node_pos"])
            values = _load_field_array(data, field_name, mach_reference=mach_reference)

        if field_name in _SCALAR_FIELD_NAMES:
            field_values = values.reshape(-1, 1)
        else:
            field_values = values

        field_parameter = parameter if time_as_parameter else param_without_t
        mesh = PointCloud(node_pos)
        field = FieldOfInterest(
            field_parameter,
            mesh,
            field_values,
            name=field_name,
            interpolated=False,
        )

        if x_axis is not None and y_axis is not None:
            if field_name in _GRAD_FIELD_NAMES:
                raise NotImplementedError(
                    "Regridding is not implemented for vector gradient fields."
                )
            cartesian_mesh = RegularGrid([x_axis, y_axis])
            field_for_interp = FieldOfInterest(
                field_parameter,
                mesh,
                field_values,
                name=field_name,
            )
            values_on_grid = field_for_interp.eval(cartesian_mesh.points)
            return FieldOfInterest(
                field_parameter,
                cartesian_mesh,
                values_on_grid,
                name=field_name,
                interpolated=False,
            )

        return field

    def __call__(self, parameter: ParameterSet, **kwargs) -> FieldOfInterest:
        return self.get_field(parameter, **kwargs)

    def get_dynamic(self, parameter: ParameterSet, **kwargs) -> DynamicOfInterest:
        """Load a dynamic time series for the given parameters."""
        raise NotImplementedError(
            "GenerativePDE raw bundles currently expose one snapshot per parameter set."
        )

    def get_dynamic_loader(self):
        return self.get_dynamic
