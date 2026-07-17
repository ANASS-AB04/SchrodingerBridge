from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Square:
    origin: np.ndarray  # bottom-left of the unrotated square
    size: float
    value: float = 1.0
    angle_deg: float | None = None
    angle_rad: float | None = None

    @classmethod
    def from_center(
        cls,
        center: np.ndarray,
        size: float,
        value: float = 1.0,
        angle_deg: float | None = None,
        angle_rad: float | None = None,
    ):
        """Create a square from its center."""
        center = np.array(center, dtype=float)
        origin = center - size / 2.0
        return cls(
            origin=origin,
            size=size,
            value=value,
            angle_deg=angle_deg,
            angle_rad=angle_rad,
        )

    def __post_init__(self):
        self.origin = np.array(self.origin, dtype=float)
        self.size = float(self.size)
        self.value = float(self.value)

        # Handle angle initialization and normalization
        if self.angle_deg is not None and self.angle_rad is not None:
            raise ValueError("Only one of angle_deg or angle_rad should be provided.")

        if self.angle_deg is None and self.angle_rad is None:
            # Default to 0
            self.angle_rad = 0.0
            self.angle_deg = 0.0
        elif self.angle_deg is not None:
            # Normalize angle_deg to [0, 360) range
            self.angle_deg = self.angle_deg % 360.0
            self.angle_rad = self.angle_deg * np.pi / 180.0
        elif self.angle_rad is not None:
            # Normalize angle_rad to [0, 2*pi) range
            self.angle_rad = self.angle_rad % (2.0 * np.pi)
            self.angle_deg = self.angle_rad * 180.0 / np.pi

    def __call__(self, points):
        return self.sample(points)

    @property
    def center(self) -> np.ndarray:
        """Center of the square."""
        return self.origin + np.array([self.size / 2.0, self.size / 2.0])

    def _rotation_matrix(self, theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=float)

    def _unit_vertices(self) -> np.ndarray:
        """Vertices of the square centered at 0, before rotation, size=self.size."""
        h = self.size / 2.0
        # Order: bottom-left, bottom-right, top-right, top-left
        return np.array(
            [
                [-h, -h],
                [h, -h],
                [h, h],
                [-h, h],
            ],
            dtype=float,
        )

    def get_vertices(self) -> np.ndarray:
        """Rotated vertices translated to the square's position."""
        angle_rad = self.angle_rad
        if angle_rad is None:
            raise ValueError("angle_rad must be set")
        R = self._rotation_matrix(angle_rad)
        verts_local = self._unit_vertices()  # centered at 0
        verts_rot = verts_local @ R.T
        return verts_rot + self.center

    def get_contour(self, num_points_per_side: int = 10) -> np.ndarray:
        """
        Return points sampled along the square's perimeter (rotated),
        with num_points_per_side points per side (no duplicate end-point).
        """
        if num_points_per_side <= 0:
            raise ValueError("num_points_per_side must be a positive integer.")

        verts = self.get_vertices()
        contour_segments = []
        for i in range(4):
            a = verts[i]
            b = verts[(i + 1) % 4]
            # Sample along the edge [a, b). Exclude the endpoint to avoid duplicates across edges.
            t = np.linspace(0.0, 1.0, num_points_per_side, endpoint=False)
            seg = a[None, :] + (b - a)[None, :] * t[:, None]
            contour_segments.append(seg)
        contour = np.vstack(contour_segments)
        return contour

    def sample(self, points: np.ndarray):
        """
        Sample the square's indicator function at given points (Nx2),
        taking rotation into account. Points strictly inside are set to self.value.
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be an array of shape (N, 2).")

        # Transform points to the square's local (unrotated) coordinates
        c = self.center
        angle_rad = self.angle_rad
        if angle_rad is None:
            raise ValueError("angle_rad must be set")
        R = self._rotation_matrix(angle_rad)
        # To unrotate points: rotate by -angle => use R.T
        local = (points - c) @ R

        h = self.size / 2.0
        inside_x = np.logical_and(local[:, 0] > -h, local[:, 0] < h)
        inside_y = np.logical_and(local[:, 1] > -h, local[:, 1] < h)
        inside = np.logical_and(inside_x, inside_y)

        u = np.zeros(points.shape[0], dtype=float)
        u[inside] = self.value
        return u.reshape(-1, 1)


def interpolate_square_linear_1_0(s, gaussian_0, gaussian_1):
    return Square(
        origin=s * gaussian_0.origin + (1 - s) * gaussian_1.origin,
        size=s * gaussian_0.size + (1 - s) * gaussian_1.size,
        value=s * gaussian_0.value + (1 - s) * gaussian_1.value,
    )


def interpolation_multiple_linear_1_0(
    s: float, sources: list[Square], targets: list[Square]
):
    if isinstance(sources, Iterable):
        sources = sources
        targets = targets
    else:
        sources = [sources]
        targets = [targets]

    interpolated_squares = []
    for source, target in zip(sources, targets):
        interpolated_squares.append(interpolate_square_linear_1_0(s, source, target))

    return interpolated_squares
