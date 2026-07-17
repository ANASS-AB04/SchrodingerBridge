import numpy as np
from scipy.spatial import cKDTree


class AslamExtrapolator:
    def __init__(self, points, values, method="fast_marching", order=2):
        self.known_points = np.asarray(points)
        self.known_values = np.asarray(values)
        self.method = method
        # self.order = order
        self._prepare_extrapolation()

    def _prepare_extrapolation(self):
        # Use a k-d tree for efficient nearest-neighbor lookup
        self.tree = cKDTree(self.known_points)

    def _extrapolate_fast_marching(self, points):
        dist, indices = self.tree.query(points)
        return self.known_values[indices]

    def __call__(self, xi):
        xi = np.asarray(xi)
        if self.method == "fast_marching":
            return self._extrapolate_fast_marching(xi)
        else:
            raise ValueError(f"Method {self.method} not implemented")

    def extrapolate(self, points):
        return self(points)
