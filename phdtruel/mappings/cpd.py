from typing import cast, override

import numpy as np
from pycpd import DeformableRegistration

from ..descriptors.descriptors import FOIasDescriptor
from ..fields.field_of_interest import FieldOfInterest as FOI
from .mappings import Mapping
from .two_fields_mappings import TwoFieldMappingModel


class CPDMapping(TwoFieldMappingModel):
    """Use the Coherent Point Drift algorithm with deformable registration to compute a mapping."""

    _name: str = "CPD"

    def __init__(
        self,
        source_field: FOIasDescriptor,
        target_field: FOIasDescriptor,
        max_iterations: int = 10000,
        tolerance: float = 1e-10,
        inverse: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Extract numpy arrays from mesh points
        # Cast to FOI to access mesh attribute
        source_foi: FOI = cast(FOI, source_field)
        target_foi: FOI = cast(FOI, target_field)

        source_points = np.asarray(source_foi.mesh.points)
        target_points = np.asarray(target_foi.mesh.points)

        # Registration is a class
        # Instantiate it and register
        self.registration = DeformableRegistration(
            X=target_points,
            Y=source_points,
            max_iterations=max_iterations,
            tolerance=tolerance,
            **kwargs,
        )
        self.registration.register()
        self.source_field = source_field
        self.target_field = target_field
        self.inverse = inverse

    def evaluate(self, points: np.ndarray, s: float = 1.0) -> np.ndarray:
        if self.inverse:
            # For inverse, we swap the registration direction
            return self.registration.transform_point_cloud(points)
        else:
            return self.registration.transform_point_cloud(points)

    @override
    def fit(self, u0: FOI, u1: FOI) -> "CPDMapping":
        """Fit the CPD mapping between two fields."""
        # This is a post-initialization mapping, so fit returns self
        return self

    @override
    def get_ts_mapping_function(self) -> Mapping:
        """Get mapping function for target space."""

        def ts_mapping(points: np.ndarray, s: float | None = None) -> np.ndarray:
            s_val = 1.0 if s is None else s
            if s_val == 0.0:
                return points
            return self.evaluate(points, s=s_val)

        return ts_mapping

    @override
    def get_ws_mapping_function(self) -> Mapping:
        """Get mapping function for world space (inverse)."""

        def ws_mapping(points: np.ndarray, s: float | None = None) -> np.ndarray:
            s_val = 1.0 if s is None else s
            if s_val == 0.0:
                return points
            # For world space, we may need inverse transformation
            return self.evaluate(points, s=s_val)

        return ws_mapping

    @override
    def copy(self) -> "CPDMapping":
        """Return a copy of this mapping."""
        return CPDMapping(
            self.source_field,
            self.target_field,
            inverse=self.inverse,
        )

    @override
    def meta_parameters(self) -> dict:
        """Return metadata parameters."""
        return {
            "algorithm": "CPD",
            "inverse": self.inverse,
        }
