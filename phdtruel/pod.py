from typing import Iterable, Literal, overload

import numpy as np
from scipy.linalg import svd

from phdtruel.fields import DynamicOfInterest, FieldOfInterest
from phdtruel.fields.field_of_interest import derive_foi
from phdtruel.fields.meshes import DomainBounds


@overload
def pod(
    snapshots: np.ndarray,
    n_values: int | None = None,
    relative_criterion: float | None = None,
    return_all_values: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def pod(
    snapshots: np.ndarray,
    n_values: int | None = None,
    relative_criterion: float | None = None,
    return_all_values: Literal[True] = True,
) -> tuple[np.ndarray, np.ndarray, int | None]: ...


def pod(
    snapshots: np.ndarray,
    n_values: int | None = None,
    relative_criterion: float | None = None,
    return_all_values: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, int | None]:
    """Proper Orthogonal Decomposition

    Args:
        matrix (np.ndarray): Matrix to decompose
        n_values (int, optional): Number of singular values to keep. Defaults to None.
        relative_criterion (float, optional): Minimal sigular value to keep relative to the first singular value. Defaults to None.
        return_all_values (bool, optional): Returns the full singular value vector and the number of singular value to keep for the POD. Defaults to False.

    Raises:
        ValueError: If nvalue and relative_criterion are None

    Returns:
        np.ndarray, np.ndarray: Phi_matrix, singular_values, (k )
    """
    U, s, _ = svd(snapshots, full_matrices=False)

    n_values_by_criterion = snapshots.shape[1]
    if relative_criterion is not None:
        criterion = max(s) * 1e-3
        n_values_by_criterion = (s >= criterion).sum()

    if n_values_by_criterion is None and n_values is None:
        raise ValueError("Can't decide how many sigular values to keep")

    if n_values is None:
        n_values = n_values_by_criterion

    n_values = min(n_values, n_values_by_criterion)

    if n_values is not None and n_values_by_criterion is not None:
        n_values = min(n_values_by_criterion, n_values)

    Phi = U[:, :n_values]
    pod_values = s[:n_values]

    if return_all_values:
        return Phi, s, n_values
    else:
        return Phi, pod_values


def pod_proj_of_target_on_fields(
    fields: Iterable[FieldOfInterest],
    target_field: FieldOfInterest,
    points: np.ndarray,
    n_modes: int | None = None,
    truncation: DomainBounds | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    fields = list(fields)

    if truncation is not None:
        fields = [field.truncate(truncation) for field in fields]

    snapshot_matrix = np.concatenate([f(points) for f in fields], axis=1)
    if n_modes is None:
        n_modes = len(fields)

    pod_basis, s, k = pod(snapshot_matrix, n_values=n_modes, return_all_values=True)

    # s_pod = s[:k]
    # print(f"Kept {s_pod.size}/{snapshot_matrix.shape[1]} singular values ")
    # f"Les valeurs singulières sont : {s[0]:3.5} et {s[1]:3.5}" + ''.join(
    #     f's{index}={value:3.5} ' for index, value in enumerate(s))

    # Check pod basis
    assert np.allclose(pod_basis.transpose() @ pod_basis, np.identity(n_modes))

    reduced_coordinates = pod_basis.transpose() @ target_field(points)
    field_pod = pod_basis @ reduced_coordinates
    return field_pod, pod_basis


class PODInterpolator:
    def __init__(
        self,
        points: np.ndarray,
        fields: list[FieldOfInterest],
        n_modes: int | None = None,
    ):
        self.fields = fields

        if n_modes is None:
            n_modes = len(self.fields)

        self.points = points
        snapshot_matrix = np.concatenate([f(points) for f in fields], axis=1)

        pod_basis, s, k = pod(snapshot_matrix, n_values=n_modes, return_all_values=True)
        self.pod_basis = pod_basis
        self.singular_values = s
        self.num_modes = k

        assert np.allclose(pod_basis.transpose() @ pod_basis, np.identity(n_modes))

    def reduced_coordinates(self, field: FieldOfInterest) -> np.ndarray:
        return self.pod_basis.transpose() @ field(self.points)

    def projection(
        self, field: FieldOfInterest, as_foi: bool = False
    ) -> np.ndarray | FieldOfInterest:
        proj = self.pod_basis @ self.reduced_coordinates(field)
        if as_foi:
            return derive_foi(field, new_values=proj)
        else:
            return proj

    def solution(self, reduced_coordinates: np.ndarray) -> np.ndarray:
        return self.pod_basis @ reduced_coordinates


def project_fields(
    pod: PODInterpolator,
    target_fields: list[FieldOfInterest],
) -> list[FieldOfInterest]:
    projections = []
    for field in target_fields:
        field: FieldOfInterest
        proj = pod.projection(field, as_foi=True)
        projections.append(proj)
    return projections


def project_dynamic(
    pod: PODInterpolator, target_dynamic: DynamicOfInterest
) -> DynamicOfInterest:
    projections = project_fields(pod, target_dynamic.fields)
    return DynamicOfInterest(
        target_dynamic.parameter, projections, target_dynamic.timesteps
    )
