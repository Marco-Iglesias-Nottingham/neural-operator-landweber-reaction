from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

# Do not initialise DOLFINx/PETSc merely because an emulator-only program
# imports RegularGrid2D.  The structured map classes load ``fem`` on demand.
# This matters on GPU inference nodes, where PyTorch is available but a PETSc
# runtime may intentionally be absent (or may not be allowed to initialise).
fem: Any | None = None

if TYPE_CHECKING:
    from .solver import ReactionSolver2D


@dataclass(frozen=True)
class RegularGrid2D:
    """Tensor-product grid with the project's ``(x-index, y-index)`` layout."""

    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be one-dimensional.")
        if x.size < 2 or y.size < 2:
            raise ValueError("Each grid axis must contain at least two points.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Grid coordinates must be finite.")
        if np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0):
            raise ValueError("Grid coordinates must be strictly increasing.")

        canonical_x = np.linspace(x[0], x[-1], x.size, dtype=np.float64)
        canonical_y = np.linspace(y[0], y[-1], y.size, dtype=np.float64)
        if not np.allclose(x, canonical_x, rtol=2.0e-7, atol=1.0e-12):
            raise ValueError("x must be uniformly spaced.")
        if not np.allclose(y, canonical_y, rtol=2.0e-7, atol=1.0e-12):
            raise ValueError("y must be uniformly spaced.")

        # Canonical float64 coordinates avoid tiny float32 spacing variations
        # after grids have made a round trip through HDF5.
        object.__setattr__(self, "x", canonical_x)
        object.__setattr__(self, "y", canonical_y)

    @classmethod
    def from_domain(
        cls,
        Lx: float,
        Ly: float,
        nx: int,
        ny: int,
    ) -> "RegularGrid2D":
        if Lx <= 0.0 or Ly <= 0.0:
            raise ValueError("Domain lengths must be positive.")
        if nx < 2 or ny < 2:
            raise ValueError("nx and ny must be at least two.")
        return cls(
            x=np.linspace(0.0, float(Lx), int(nx), dtype=np.float64),
            y=np.linspace(0.0, float(Ly), int(ny), dtype=np.float64),
        )

    @property
    def nx(self) -> int:
        return int(self.x.size)

    @property
    def ny(self) -> int:
        return int(self.y.size)

    @property
    def shape(self) -> tuple[int, int]:
        return self.nx, self.ny

    @property
    def size(self) -> int:
        return self.nx * self.ny

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def dy(self) -> float:
        return float(self.y[1] - self.y[0])

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y, indexing="ij")

    def points(self) -> np.ndarray:
        X, Y = self.mesh()
        return np.column_stack((X.reshape(-1), Y.reshape(-1)))


def _require_fem() -> Any:
    """Return DOLFINx FEM lazily, preserving emulator-only imports."""

    global fem
    if fem is None:
        try:
            from dolfinx import fem as imported_fem
        except ImportError as exc:
            raise ImportError(
                "DOLFINx is required for coefficient/state FEM grid maps. "
                "RegularGrid2D remains available without DOLFINx."
            ) from exc
        fem = imported_fem
    return fem


def _require_serial(solver: "ReactionSolver2D") -> None:
    _require_fem()
    if fem is None:  # pragma: no cover - satisfies static type narrowing.
        raise ImportError(
            "DOLFINx is required for coefficient/state FEM grid maps. "
            "RegularGrid2D remains available without DOLFINx."
        )
    # The grid tensor is replicated and shard generation is deliberately one
    # MPI rank per file.  A distributed gather/scatter would require an
    # explicit global ownership policy rather than silently returning only a
    # local patch.
    if solver.comm.size != 1:
        raise RuntimeError(
            "Structured neural-grid maps currently require one MPI rank. "
            "Parallelise data generation over independent shard processes."
        )


def _grid_indices_for_coordinates(
    coordinates: np.ndarray,
    grid: RegularGrid2D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("coordinates must have shape (n,2) or (n,3).")

    scaled_x = (coordinates[:, 0] - grid.x[0]) / grid.dx
    scaled_y = (coordinates[:, 1] - grid.y[0]) / grid.dy
    i = np.rint(scaled_x).astype(np.int64)
    j = np.rint(scaled_y).astype(np.int64)

    # Tolerance is expressed in grid-index coordinates and comfortably covers
    # floating-point mesh construction noise while rejecting P2 edge midpoints.
    on_x = np.abs(scaled_x - i) <= 2.0e-8
    on_y = np.abs(scaled_y - j) <= 2.0e-8
    in_bounds = (
        (i >= 0)
        & (i < grid.nx)
        & (j >= 0)
        & (j < grid.ny)
    )
    return i, j, on_x & on_y & in_bounds


class StructuredCoefficientGridMap:
    """Exact permutation between a structured grid and the nodal P1 ``u``.

    There is no interpolation: the neural grid is required to be precisely
    ``(fem_nx+1) x (fem_ny+1)`` and therefore coincides with every P1
    coefficient degree of freedom.
    """

    def __init__(
        self,
        solver: "ReactionSolver2D",
        grid: RegularGrid2D,
    ) -> None:
        _require_serial(solver)
        self.solver = solver
        self.grid = grid
        expected = (
            int(solver.config.fem_nx) + 1,
            int(solver.config.fem_ny) + 1,
        )
        if grid.shape != expected:
            raise ValueError(
                "The coefficient grid must coincide with FEM vertices: "
                f"expected {expected}, received {grid.shape}."
            )

        coordinates = np.asarray(
            solver.Q.tabulate_dof_coordinates(),
            dtype=np.float64,
        )
        i, j, is_vertex = _grid_indices_for_coordinates(coordinates, grid)
        if not np.all(is_vertex):
            bad = int(np.count_nonzero(~is_vertex))
            raise RuntimeError(
                f"The P1 coefficient space has {bad} DOFs off the neural grid."
            )
        self.dof_to_flat = i * grid.ny + j
        if self.dof_to_flat.size != grid.size:
            raise RuntimeError(
                f"Expected {grid.size} coefficient DOFs, found "
                f"{self.dof_to_flat.size}."
            )
        if np.unique(self.dof_to_flat).size != grid.size:
            raise RuntimeError("Coefficient-to-grid mapping is not one-to-one.")

    def grid_to_array(self, field_grid: np.ndarray) -> np.ndarray:
        field_grid = np.asarray(field_grid, dtype=np.float64)
        if field_grid.shape != self.grid.shape:
            raise ValueError(
                f"Expected coefficient grid {self.grid.shape}, received "
                f"{field_grid.shape}."
            )
        if not np.all(np.isfinite(field_grid)):
            raise ValueError("Coefficient grid contains non-finite values.")
        return np.asarray(
            field_grid.reshape(-1)[self.dof_to_flat],
            dtype=self.solver.u.x.array.dtype,
        )

    def array_to_grid(self, coefficient_values) -> np.ndarray:
        fem_module = _require_fem()
        if isinstance(coefficient_values, fem_module.Function):
            values = coefficient_values.x.array
        else:
            values = np.asarray(coefficient_values)
        values = np.asarray(values, dtype=np.float64)
        expected = self.solver.u.x.array.shape
        if values.shape != expected:
            raise ValueError(
                f"Expected coefficient storage shape {expected}, received "
                f"{values.shape}."
            )
        flat = np.empty(self.grid.size, dtype=np.float64)
        flat[self.dof_to_flat] = values
        return flat.reshape(self.grid.shape)

    function_to_grid = array_to_grid

    def grid_to_function(
        self,
        field_grid: np.ndarray,
        *,
        name: str = "reaction_coefficient_u",
    ) -> fem.Function:
        return self.solver.new_parameter_function(
            self.grid_to_array(field_grid),
            name=name,
        )


class StructuredLagrangeStateGridMap:
    """Extract exact vertex values from scalar P1 or P2 Lagrange states.

    For P1 this is a bijective permutation and ``grid_to_state`` is exact.
    For P2, edge-midpoint DOFs remain internal to the FEM solve; state output
    at the mesh vertices is still exact, but lifting a vertex grid alone back
    to a P2 Function is intentionally undefined.
    """

    channel_names = ("p",)

    def __init__(
        self,
        solver: "ReactionSolver2D",
        grid: RegularGrid2D,
    ) -> None:
        _require_serial(solver)
        self.solver = solver
        self.grid = grid
        self.state_degree = int(solver.config.state_degree)

        expected = (
            int(solver.config.fem_nx) + 1,
            int(solver.config.fem_ny) + 1,
        )
        if grid.shape != expected:
            raise ValueError(
                "The state observation grid must coincide with FEM vertices: "
                f"expected {expected}, received {grid.shape}."
            )

        coordinates = np.asarray(
            solver.V.tabulate_dof_coordinates(),
            dtype=np.float64,
        )
        i, j, is_vertex = _grid_indices_for_coordinates(coordinates, grid)
        vertex_dofs = np.flatnonzero(is_vertex).astype(np.int64)
        flat_indices = (i[is_vertex] * grid.ny + j[is_vertex]).astype(np.int64)

        if vertex_dofs.size != grid.size:
            raise RuntimeError(
                f"Expected {grid.size} state vertex DOFs, found "
                f"{vertex_dofs.size}."
            )
        if np.unique(flat_indices).size != grid.size:
            raise RuntimeError("State vertex-to-grid mapping is not one-to-one.")

        self.flat_to_state_dof = np.empty(grid.size, dtype=np.int64)
        self.flat_to_state_dof[flat_indices] = vertex_dofs
        self.state_vertex_dofs = vertex_dofs
        self.state_vertex_flat_indices = flat_indices

    def state_to_grid(self, state: fem.Function) -> np.ndarray:
        values = np.asarray(state.x.array, dtype=np.float64)
        expected = self.solver.V.dofmap.index_map.size_local + (
            self.solver.V.dofmap.index_map.num_ghosts
        )
        if values.size != expected:
            raise ValueError(
                f"Expected {expected} state values, received {values.size}."
            )
        return values[self.flat_to_state_dof].reshape(self.grid.shape).copy()

    def grid_to_state(
        self,
        field_grid: np.ndarray,
        *,
        name: str = "state_from_grid",
    ) -> fem.Function:
        if self.state_degree != 1:
            raise NotImplementedError(
                "grid_to_state is exact only for P1. A P2 state also requires "
                "edge-midpoint DOFs, so a vertex-only lift is ambiguous."
            )
        field_grid = np.asarray(field_grid, dtype=np.float64)
        if field_grid.shape != self.grid.shape:
            raise ValueError(
                f"Expected state grid {self.grid.shape}, received "
                f"{field_grid.shape}."
            )
        if not np.all(np.isfinite(field_grid)):
            raise ValueError("State grid contains non-finite values.")
        values = np.zeros_like(self.solver.new_state_function().x.array)
        values[self.flat_to_state_dof] = field_grid.reshape(-1)
        return self.solver.new_state_function(values, name=name)


# Names used by the MRE project and by early reaction scripts.
ParameterGridMap = StructuredCoefficientGridMap
CoefficientGridMap = StructuredCoefficientGridMap
StructuredP1StateGridMap = StructuredLagrangeStateGridMap


__all__ = [
    "CoefficientGridMap",
    "ParameterGridMap",
    "RegularGrid2D",
    "StructuredCoefficientGridMap",
    "StructuredLagrangeStateGridMap",
    "StructuredP1StateGridMap",
]
