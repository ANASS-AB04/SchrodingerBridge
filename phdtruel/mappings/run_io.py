"""Shared HDF5 persistence helpers for mapping-optimization experiment scripts.

These were previously copy-pasted (verbatim) across every script in
``scripts/minimized_mappings/`` that saves a piecewise FFD or RBF optimization
run to HDF5. They are generic with respect to the specific fields/mesh layout
being saved, so they live here instead of in each experiment script.
"""

import h5py
import jax
import numpy as np

from phdtruel.mappings.ot_gaussian import GaussianOTMapping


def to_numpy(value) -> np.ndarray:
    """Bring a (possibly device-resident) JAX/NumPy value to a host NumPy array."""
    return np.asarray(jax.device_get(value))


def to_hdf5_scalar(value) -> float | int | bool | str:
    """Extract a Python scalar from a 0-d array-like value, for use as an attr."""
    arr = to_numpy(value)
    if arr.shape == ():
        return arr.item()
    raise ValueError(f"Expected scalar value, got shape={arr.shape}")


def save_named_array_tuple(
    parent: h5py.Group,
    group_name: str,
    values: tuple,
    *,
    prefix: str = "region",
) -> None:
    """Save a tuple of arrays (e.g. per-region control point displacements)."""
    subgroup = parent.create_group(group_name)
    for idx, array in enumerate(values):
        subgroup.create_dataset(
            f"{prefix}_{idx}",
            data=to_numpy(array),
            compression="gzip",
        )


def save_solver_history(parent: h5py.Group, history: dict[str, np.ndarray]) -> None:
    """Save a solver's optimization trace/history dict as an HDF5 group."""
    history_group = parent.create_group("history")
    for key, values in history.items():
        arr = np.asarray(values)
        if arr.dtype == object:
            arr = np.asarray([str(v) for v in arr], dtype="S")
        history_group.create_dataset(key, data=arr, compression="gzip")


def save_got_parameters(parent: h5py.Group, got_mapping: GaussianOTMapping) -> None:
    """Save a fitted `GaussianOTMapping`'s parameters (used as mapping initializer)."""
    got_group = parent.create_group("initial_mapping_got")
    got_group.attrs["available"] = True
    got_group.attrs["got_class_name"] = got_mapping.__class__.__name__
    got_group.attrs["sensor_name"] = got_mapping.meta_parameters.get(
        "sensor_name", "unknown"
    )

    g0 = got_mapping._g0
    g1 = got_mapping._g1
    matrix = got_mapping._M

    if g0 is None or g1 is None or matrix is None:
        got_group.attrs["available"] = False
        return

    g0_group = got_group.create_group("g0")
    g0_group.create_dataset("mu", data=np.asarray(g0.mu), compression="gzip")
    g0_group.create_dataset("sigma", data=np.asarray(g0.sigma), compression="gzip")

    g1_group = got_group.create_group("g1")
    g1_group.create_dataset("mu", data=np.asarray(g1.mu), compression="gzip")
    g1_group.create_dataset("sigma", data=np.asarray(g1.sigma), compression="gzip")

    got_group.create_dataset("M", data=np.asarray(matrix), compression="gzip")

    if got_mapping.plot_data is not None:
        sensor_fields = got_group.create_group("sensor_fields")
        got_mapping.plot_data.u0_hat.save_in_hdf5(sensor_fields.create_group("u0_hat"))
        got_mapping.plot_data.u1_hat.save_in_hdf5(sensor_fields.create_group("u1_hat"))
