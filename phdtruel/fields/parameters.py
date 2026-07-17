from typing import Mapping, TypeAlias, TypedDict, cast

import h5py
import numpy as np

ParamValue: TypeAlias = float | int | np.floating | np.integer


class _ParameterSetState(TypedDict):
    _parameters: list[str]
    _values: list[ParamValue]


class ParameterSet:
    @classmethod
    def from_dict(cls, param_dict: Mapping[str, ParamValue]) -> "ParameterSet":
        return cls(**dict(param_dict))

    @classmethod
    def from_names_values(
        cls, parameters: list[str], values: np.ndarray
    ) -> "ParameterSet":
        return cls(**{key: value for key, value in zip(parameters, values)})

    def save_to_hdf5(self, g: h5py.Group) -> None:
        """
        Save the ParameterSet in an HDF5 file.
        """
        for name, value in zip(self._parameters, self._values):
            g.attrs[name] = value

    @classmethod
    def load_from_hdf5(cls, g: h5py.Group) -> "ParameterSet":
        """
        Load the ParameterSet from an HDF5 file.
        """
        param_dict = {
            name: cast(ParamValue, np.asarray(g.attrs[name]).item()) for name in g.attrs
        }
        return cls.from_dict(param_dict)

    def __init__(self, **kwargs: ParamValue) -> None:
        self._parameters: list[str] = list(kwargs.keys())
        self._values: list[ParamValue] = list(kwargs.values())

    @property
    def values(self) -> list[ParamValue]:
        return self._values

    @property
    def parameters(self) -> list[str]:
        return self._parameters

    @property
    def size(self) -> int:
        return len(self._values)

    def as_dict(self) -> dict[str, ParamValue]:
        return dict(zip(self._parameters, self._values))

    def without(self, name: str, inplace: bool = False) -> "ParameterSet":
        if name not in self._parameters:
            return self

        d = self.as_dict()
        d.pop(name)

        if inplace:
            self._parameters.remove(name)
            self._values = [v for k, v in d.items()]
            return self
        return ParameterSet.from_dict(d)

    def __getstate__(self) -> _ParameterSetState:
        # Only serialize the necessary state.
        return {"_parameters": self._parameters, "_values": self._values}

    def __setstate__(self, state: _ParameterSetState) -> None:
        self._parameters = state["_parameters"]
        self._values = state["_values"]

    def __getattr__(self, name: str) -> ParamValue:
        if name in self._parameters:
            idx = self._parameters.index(name)
            return self._values[idx]
        raise AttributeError(f"{type(self).__name__} has no attribute {name}")

    def __getitem__(self, name: str) -> ParamValue:
        if name in self._parameters:
            idx = self._parameters.index(name)
            return self._values[idx]
        raise KeyError(f"{name} not found in parameters")

    def __repr__(self) -> str:
        out_str = "ParameterSet("
        for k, param in enumerate(self._parameters):
            out_str += f"{param} : {self._values[k]}, "
        out_str = out_str[:-2]
        out_str += ")"
        return out_str

    def __str__(self) -> str:
        out_str = ""
        for k, param in enumerate(self._parameters):
            out_str += f"{param}={self._values[k]}, "
        out_str = out_str[:-2]
        return out_str

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ParameterSet):
            is_equal = True
            is_equal &= len(self._values) == len(other._values)
            is_equal &= set(self._parameters) == set(other._parameters)
            for name in self._parameters:
                is_equal &= np.isclose(getattr(self, name), getattr(other, name))
            return is_equal
        return False

    def __hash__(self) -> int:
        items = tuple(
            sorted(
                (name, np.asarray(getattr(self, name)).item())
                for name in self._parameters
            )
        )
        return hash(items)


def generate_evaluation_params_in_rectangle(shape: tuple[int, int]) -> list[np.ndarray]:
    x = np.linspace(0, 1, shape[0])
    y = np.linspace(0, 1, shape[1])
    x, y = np.meshgrid(x, y)

    param_values = np.vstack([x.ravel(), y.ravel()]).transpose()
    return [p for p in param_values]


class ParameterNormalizer:
    def __init__(
        self,
        parameters: list[ParameterSet] | None = None,
        normalization_ranges: dict[str, tuple[float, float]] | None = None,
    ):
        if parameters is None:
            raise ValueError("parameters cannot be None")
        self._parameters_names = parameters[0].parameters
        p_values = np.asarray([p.values for p in parameters], dtype=float)
        self._min_param_values = np.min(p_values, axis=0)
        self._max_param_values = np.max(p_values, axis=0)

        normalization_ranges_default = {
            name: (0.0, 1.0) for name in self._parameters_names
        }
        if normalization_ranges is not None:
            normalization_ranges_default.update(normalization_ranges)
            normalization_ranges = normalization_ranges_default
        else:
            normalization_ranges = normalization_ranges_default

        self._min_normalization_ranges = np.array(
            [v[0] for n, v in normalization_ranges.items()]
        )
        self._max_normalization_ranges = np.array(
            [v[1] for n, v in normalization_ranges.items()]
        )

    @classmethod
    def fit_predict(cls, parameters: list[ParameterSet]) -> np.ndarray:
        return cls(parameters).normalize(parameters)

    def normalize(self, parameters: ParameterSet | list[ParameterSet]) -> np.ndarray:
        if isinstance(parameters, ParameterSet):
            parameters = [parameters]

        if parameters[0].size != len(self._parameters_names):
            raise ValueError(
                f"Expected {len(self._parameters_names)} parameters, got {parameters[0].size}"
            )

        p_values = np.asarray([p.values for p in parameters], dtype=float)
        normalized_values = (p_values - self._min_param_values) / (
            self._max_param_values - self._min_param_values
        )
        np.nan_to_num(normalized_values, copy=False, nan=1.0)
        normalized_values_in_range = (
            normalized_values
            * (self._max_normalization_ranges - self._min_normalization_ranges)
            + self._min_normalization_ranges
        )
        return normalized_values_in_range

    def unormalize(
        self, normalized_parameters: list[np.ndarray] | np.ndarray
    ) -> list[ParameterSet] | ParameterSet:
        if isinstance(normalized_parameters, list):
            normalized_parameters = np.array(normalized_parameters)

        if normalized_parameters.ndim == 1:
            normalized_parameters = normalized_parameters.reshape(
                1, -1
            )  # TODO Add dimension

        normalized_parameters = (
            normalized_parameters - self._min_normalization_ranges
        ) / (self._max_normalization_ranges - self._min_normalization_ranges)

        parameters_values = (
            normalized_parameters * (self._max_param_values - self._min_param_values)
            + self._min_param_values
        )
        parameters = []
        for values in parameters_values:
            param = ParameterSet.from_dict(
                {name: val for name, val in zip(self._parameters_names, values)}
            )
            parameters.append(param)

        if len(parameters) == 1:
            return parameters[0]
        return parameters


class DistanceNormalizer:
    def __init__(self, parameters: list[ParameterSet] | None = None):
        if parameters is None:
            raise ValueError("parameters cannot be None")
        self._parameter_name = parameters[0].parameters[0]
        self._origin = float(parameters[0].values[0])

        p_values = np.asarray([p.values for p in parameters], dtype=float)
        self._ratio = 1.0 / (float(p_values[1:].max()) - float(self._origin) + 1e-13)

    @classmethod
    def fit_predict(cls, parameters: list[ParameterSet]) -> np.ndarray:
        return cls(parameters).normalize(parameters)

    def normalize(self, parameters: ParameterSet | list[ParameterSet]) -> np.ndarray:
        if isinstance(parameters, ParameterSet):
            parameters = [parameters]

        p_values = np.asarray([p.values for p in parameters], dtype=float)
        distances = p_values - np.ones_like(p_values) * float(self._origin)
        normalized_param = distances * self._ratio
        return normalized_param

    def unormalize(
        self, normalized_parameters: list[np.ndarray] | np.ndarray
    ) -> list[ParameterSet] | ParameterSet:
        if isinstance(normalized_parameters, list):
            normalized_parameters = np.array(normalized_parameters)

        normalized_parameters = normalized_parameters.ravel()

        p_values = normalized_parameters / self._ratio + self._origin

        parameters = []
        for value in p_values:
            param = ParameterSet.from_dict({self._parameter_name: float(value)})
            parameters.append(param)

        if len(parameters) == 1:
            return parameters[0]
        return parameters
