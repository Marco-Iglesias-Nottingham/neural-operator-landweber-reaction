from __future__ import annotations

import numpy as np
from dolfinx import fem

from .grid import RegularGrid2D, StructuredLagrangeStateGridMap


class FullFieldObservationOperator:
    """Observe scalar pressure at every structured-mesh vertex.

    ``apply`` returns a channel-first array with shape ``(1,nx,ny)`` so that
    shards can be consumed directly by both the FNO and DeepONet trainers.
    ``apply_scalar`` provides the underlying ``(nx,ny)`` field for plotting
    and numerical checks.

    This object represents point-value extraction.  ``grid_to_state`` is a
    value interpolation available only for P1; it must not be mistaken for
    the adjoint of Euclidean point sampling.  Exact inverse code should state
    its observation-space quadrature/pairing explicitly.
    """

    channel_names = ("p",)

    def __init__(self, state_map: StructuredLagrangeStateGridMap) -> None:
        self.state_map = state_map
        self.grid: RegularGrid2D = state_map.grid

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return 1, self.grid.nx, self.grid.ny

    def apply_scalar(
        self,
        state: fem.Function,
        *,
        dtype=None,
    ) -> np.ndarray:
        values = self.state_map.state_to_grid(state)
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        return values

    def apply(
        self,
        state: fem.Function,
        *,
        dtype=np.float32,
    ) -> np.ndarray:
        return self.apply_scalar(state, dtype=dtype)[None, ...]

    observe = apply

    def grid_to_state(
        self,
        values: np.ndarray,
        *,
        name: str = "observed_state",
    ) -> fem.Function:
        array = np.asarray(values)
        if array.shape == self.output_shape:
            array = array[0]
        return self.state_map.grid_to_state(array, name=name)


__all__ = ["FullFieldObservationOperator"]
