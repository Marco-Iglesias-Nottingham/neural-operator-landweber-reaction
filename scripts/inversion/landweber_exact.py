#!/usr/bin/env python3
"""Exact multiresolution Landweber for the inverse problem.

This driver keeps the inverse parameter field on one fixed regular grid while
allowing the internal P1 FEM state mesh (and therefore the pressure observation
grid) to be refined independently.

For the benchmark used in the paper the intended hierarchy is

    parameter u : 120 x 120 points   (119 intervals per axis)
    state FEM 1x: 119 x 119 cells    -> 120 x 120 pressure vertices
    state FEM 2x: 238 x 238 cells    -> 239 x 239 pressure vertices
    state FEM 4x: 476 x 476 cells    -> 477 x 477 pressure vertices
    truth FEM   : 952 x 952 cells    -> 953 x 953 pressure vertices

The saved truth coefficient may be defined on a different/finer source grid
(e.g. 512 x 512 points).  The synthetic archive stores that truth evaluated on
the fine truth-FEM vertices.  For inversion, the parameter truth used only for
error diagnostics is the exact nested restriction to the fixed 120 x 120
parameter grid.

For a state mesh h, let B_h map the fixed parameter grid to the P1 coefficient
vertices by bilinear interpolation.  The discrete forward map is

    F_h(U) = S_h(B_h U),

and the parameter-space adjoint is therefore

    F_h'(U)^* = B_h^T S_h'(B_h U)^*.

The code implements B_h and its exact Euclidean transpose explicitly.  The
iteration-zero JVP/VJP test is performed in the *fixed parameter space*, so it
also checks the transpose transfer.

All interior pressure vertices of the chosen state mesh are used in the
weighted L2 objective.  The homogeneous Dirichlet boundary is excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from mpi4py import MPI
import numpy as np


PATCH_VERSION = (
    "2026-08-20-reaction-exact-landweber-fixed-parameter-multires-v2-"
    "nobt-default+projection-diagnostics-v1"
    "+discrepancy-crossing-snapshots-v1"
    "+u0-200-support-v1"
    "+preconditioner-ablation-v1"
)
STATE_DEGREE = 1
DEFAULT_PARAMETER_POINTS = 120


def find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    candidates = [
        script_path.parent,
        *script_path.parents,
        Path.cwd().resolve(),
        *Path.cwd().resolve().parents,
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "reaction_core").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate reaction_emulator_project_matern_only.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reaction_core import (  # noqa: E402
    ReactionConfig,
    ReactionSolver2D,
    RegularGrid2D,
    StructuredCoefficientGridMap,
    StructuredLagrangeStateGridMap,
)
from reaction_core.inversion import (  # noqa: E402
    IdentityCovariance2D,
    ScalarVertexObservationAdjoint,
    StationaryMaternCovariance2D,
    interior_vertex_mask,
    trapezoid_weights,
    weighted_inner,
    weighted_norm,
)
from reaction_core.synthetic_archive import (  # noqa: E402
    FineSyntheticArchive,
    load_fine_synthetic_archive,
)
from reaction_core.discrepancy_snapshots import (  # noqa: E402
    DiscrepancyCrossingSnapshots,
)
from reaction_core.projection_diagnostics import (  # noqa: E402
    ProjectionDiagnostics,
    summarize_projection_diagnostics,
)
from reaction_core.trajectory_states import TrajectoryStateWriter  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Exact P1 Reaction Landweber with a fixed parameter grid and an "
            "independently refined state/observation FEM grid."
        ),
    )
    source = parser.add_argument_group("synthetic truth/data")
    source.add_argument("--synthetic-file", type=Path, required=True)
    source.add_argument(
        "--data-kind", choices=("noisy", "noiseless"), default="noisy"
    )

    parameter = parser.add_argument_group("fixed inverse parameter grid")
    parameter.add_argument(
        "--parameter-nx-points", type=int, default=DEFAULT_PARAMETER_POINTS
    )
    parameter.add_argument(
        "--parameter-ny-points", type=int, default=DEFAULT_PARAMETER_POINTS
    )

    state = parser.add_argument_group("internal P1 state/observation mesh")
    state.add_argument(
        "--fem-nx",
        type=int,
        default=119,
        help="P1 state-mesh CELL count in x (119/238/476 for 1x/2x/4x).",
    )
    state.add_argument("--fem-ny", type=int, default=119)
    state.add_argument(
        "--linear-solver", choices=("cg-hypre", "lu"), default="cg-hypre"
    )
    state.add_argument("--ksp-rtol", type=float, default=1.0e-11)
    state.add_argument("--ksp-max-it", type=int, default=10000)

    inv = parser.add_argument_group("covariance-preconditioned Landweber")
    inv.add_argument("--iterations", type=int, default=1500)
    inv.add_argument("--initial-u", type=float, default=100.0)
    inv.add_argument("--u-min", type=float, default=0.0)
    inv.add_argument("--u-max", type=float, default=200.0)
    inv.add_argument(
        "--preconditioner",
        choices=("matern", "identity"),
        default="matern",
        help=(
            "Search-direction preconditioner in d = K q. The identity option "
            "is the fixed-grid coordinate-space ablation arm."
        ),
    )
    inv.add_argument("--matern-nu", type=float, default=1.5)
    inv.add_argument("--matern-length-scale", type=float, default=0.1)
    inv.add_argument("--matern-variance", type=float, default=1.0)
    inv.add_argument("--fft-workers", type=int, default=None)
    inv.add_argument(
        "--alpha-mode",
        choices=("derivative", "adjoint-step-norm", "minimum-residual"),
        default="derivative",
    )
    inv.add_argument("--damping", type=float, default=0.95)
    inv.add_argument("--max-physical-change", type=float, default=5.0)
    inv.add_argument(
        "--backtracking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use projection-aware Armijo backtracking. Default is "
            "--no-backtracking, which accepts the capped/projected proposal "
            "directly. If enabled and no Armijo trial succeeds, the final "
            "(smallest) usable proposal is accepted."
        ),
    )
    inv.add_argument("--backtracking-factor", type=float, default=0.5)
    inv.add_argument("--max-backtracks", type=int, default=8)
    inv.add_argument("--armijo", type=float, default=1.0e-4)
    inv.add_argument("--relative-residual-tolerance", type=float, default=0.0)
    inv.add_argument(
        "--discrepancy-principle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    inv.add_argument("--discrepancy-tau", type=float, default=1.05)
    inv.add_argument("--model-discrepancy-scale", type=float, default=1.0)
    inv.add_argument("--adjoint-check-seed", type=int, default=20260818)
    inv.add_argument("--adjoint-check-tolerance", type=float, default=5.0e-8)
    inv.add_argument(
        "--crossing-snapshots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Retain the states immediately before and at the first crossings "
            "of delta, tau*delta, and the model-adjusted discrepancy target. "
            "This never activates early stopping."
        ),
    )
    inv.add_argument(
        "--projection-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record accepted-update box-projection diagnostics and cumulative "
            "lower/upper-bound hit maps. This is observational only and does "
            "not alter the Landweber update."
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=Path, default=None)
    output.add_argument("--overwrite", action="store_true")
    output.add_argument("--print-every", type=int, default=1)
    output.add_argument(
        "--trajectory-state-every",
        type=int,
        default=0,
        help=(
            "Save every Nth physical-u iterate to trajectory_states.h5; "
            "zero disables this diagnostic."
        ),
    )
    output.add_argument("--plots", action="store_true")
    output.add_argument("--dpi", type=int, default=180)
    return parser.parse_args(argv)


def canonical_alpha_mode(value: str) -> str:
    return "adjoint-step-norm" if value == "minimum-residual" else value


def solver_options(args: argparse.Namespace) -> dict[str, Any]:
    if args.linear_solver == "lu":
        return {"ksp_type": "preonly", "pc_type": "lu"}
    return {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": float(args.ksp_rtol),
        "ksp_atol": 0.0,
        "ksp_max_it": int(args.ksp_max_it),
    }


def validate_args(args: argparse.Namespace, archive: FineSyntheticArchive) -> None:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Run this exact inversion with exactly one MPI rank.")
    if args.parameter_nx_points < 2 or args.parameter_ny_points < 2:
        raise ValueError("Parameter grid must contain at least two points per axis.")
    if args.fem_nx < 2 or args.fem_ny < 2:
        raise ValueError("State FEM cell counts must be at least two.")

    # The paper benchmark intentionally uses nested grids.  This makes truth
    # data restriction exact at pressure vertices and makes the parameter truth
    # diagnostic an exact restriction of the fine coefficient field.
    parameter_intervals_x = args.parameter_nx_points - 1
    parameter_intervals_y = args.parameter_ny_points - 1
    if archive.fine_fem_nx % parameter_intervals_x:
        raise ValueError(
            "Fine truth cells in x must be divisible by parameter intervals: "
            f"{archive.fine_fem_nx} is not divisible by {parameter_intervals_x}."
        )
    if archive.fine_fem_ny % parameter_intervals_y:
        raise ValueError(
            "Fine truth cells in y must be divisible by parameter intervals: "
            f"{archive.fine_fem_ny} is not divisible by {parameter_intervals_y}."
        )
    if archive.fine_fem_nx % args.fem_nx or archive.fine_fem_ny % args.fem_ny:
        raise ValueError(
            "State FEM cells must divide the saved fine truth mesh exactly. "
            f"fine={archive.fine_fem_nx}x{archive.fine_fem_ny}, "
            f"state={args.fem_nx}x{args.fem_ny}."
        )
    if args.fem_nx >= archive.fine_fem_nx or args.fem_ny >= archive.fine_fem_ny:
        raise ValueError("State FEM must be strictly coarser than the truth FEM.")
    if args.fem_nx % parameter_intervals_x or args.fem_ny % parameter_intervals_y:
        raise ValueError(
            "For the benchmark, state FEM cells must be an integer multiple of "
            "the fixed parameter-grid intervals. With 120 parameter points use "
            "119, 238, or 476 cells."
        )
    if args.iterations < 0:
        raise ValueError("--iterations cannot be negative.")
    if int(args.trajectory_state_every) < 0:
        raise ValueError("--trajectory-state-every cannot be negative.")
    if not 0.0 <= args.u_min < args.u_max:
        raise ValueError("Require 0 <= --u-min < --u-max.")
    if not args.u_min <= args.initial_u <= args.u_max:
        raise ValueError("--initial-u must lie inside the physical box.")
    if args.matern_nu <= 0.0 or args.matern_length_scale <= 0.0:
        raise ValueError("Matérn nu and length scale must be positive.")
    if args.matern_variance <= 0.0:
        raise ValueError("Matérn variance must be positive.")
    if args.fft_workers == 0:
        raise ValueError("--fft-workers cannot be zero.")
    if args.damping <= 0.0 or args.max_physical_change <= 0.0:
        raise ValueError("Damping and maximum physical change must be positive.")
    if not 0.0 < args.backtracking_factor < 1.0:
        raise ValueError("--backtracking-factor must be in (0,1).")
    if args.max_backtracks < 0:
        raise ValueError("--max-backtracks cannot be negative.")
    if args.armijo < 0.0:
        raise ValueError("--armijo cannot be negative.")
    if args.relative_residual_tolerance < 0.0:
        raise ValueError("--relative-residual-tolerance cannot be negative.")
    if args.discrepancy_tau <= 0.0 or args.model_discrepancy_scale < 0.0:
        raise ValueError("Invalid discrepancy settings.")
    if args.adjoint_check_tolerance <= 0.0:
        raise ValueError("Adjoint-check tolerance must be positive.")
    if args.ksp_rtol <= 0.0 or args.ksp_max_it < 1:
        raise ValueError("Invalid KSP settings.")
    if args.print_every < 1 or args.dpi < 40:
        raise ValueError("Invalid output frequency/DPI.")


class BilinearGridTransfer:
    """Bilinear grid interpolation B and its exact Euclidean transpose B^T."""

    def __init__(self, source: RegularGrid2D, target: RegularGrid2D) -> None:
        self.source = source
        self.target = target
        if not (
            np.isclose(source.x[0], target.x[0])
            and np.isclose(source.x[-1], target.x[-1])
            and np.isclose(source.y[0], target.y[0])
            and np.isclose(source.y[-1], target.y[-1])
        ):
            raise ValueError("Source and target grids must cover the same rectangle.")
        self._ix0, self._ix1, self._wx0, self._wx1 = self._axis_weights(
            source.x, target.x
        )
        self._iy0, self._iy1, self._wy0, self._wy1 = self._axis_weights(
            source.y, target.y
        )

        ny = source.ny
        ix0 = self._ix0[:, None]
        ix1 = self._ix1[:, None]
        iy0 = self._iy0[None, :]
        iy1 = self._iy1[None, :]
        wx0 = self._wx0[:, None]
        wx1 = self._wx1[:, None]
        wy0 = self._wy0[None, :]
        wy1 = self._wy1[None, :]

        self._index00 = (ix0 * ny + iy0).astype(np.int64)
        self._index01 = (ix0 * ny + iy1).astype(np.int64)
        self._index10 = (ix1 * ny + iy0).astype(np.int64)
        self._index11 = (ix1 * ny + iy1).astype(np.int64)
        self._weight00 = wx0 * wy0
        self._weight01 = wx0 * wy1
        self._weight10 = wx1 * wy0
        self._weight11 = wx1 * wy1

    @staticmethod
    def _axis_weights(
        source_coordinates: np.ndarray, target_coordinates: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        source = np.asarray(source_coordinates, dtype=np.float64)
        target = np.asarray(target_coordinates, dtype=np.float64)
        upper = np.searchsorted(source, target, side="right")
        upper = np.clip(upper, 1, source.size - 1)
        lower = upper - 1
        denominator = source[upper] - source[lower]
        t = (target - source[lower]) / denominator
        t = np.clip(t, 0.0, 1.0)
        return lower, upper, 1.0 - t, t

    def apply(self, field: np.ndarray) -> np.ndarray:
        values = np.asarray(field, dtype=np.float64)
        if values.shape != self.source.shape:
            raise ValueError(
                f"Transfer source shape must be {self.source.shape}, got {values.shape}."
            )
        flat = values.reshape(-1)
        result = (
            self._weight00 * flat[self._index00]
            + self._weight01 * flat[self._index01]
            + self._weight10 * flat[self._index10]
            + self._weight11 * flat[self._index11]
        )
        return np.asarray(result, dtype=np.float64)

    def transpose(self, target_covector: np.ndarray) -> np.ndarray:
        covector = np.asarray(target_covector, dtype=np.float64)
        if covector.shape != self.target.shape:
            raise ValueError(
                f"Transfer target shape must be {self.target.shape}, got {covector.shape}."
            )
        values = covector.reshape(-1)
        result = np.zeros(self.source.size, dtype=np.float64)
        for index, weight in (
            (self._index00, self._weight00),
            (self._index01, self._weight01),
            (self._index10, self._weight10),
            (self._index11, self._weight11),
        ):
            result += np.bincount(
                index.reshape(-1),
                weights=(weight.reshape(-1) * values),
                minlength=self.source.size,
            )
        return result.reshape(self.source.shape)

    def adjoint_test(self, seed: int = 12345) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(self.source.shape)
        y = rng.standard_normal(self.target.shape)
        lhs = float(np.sum(self.apply(x) * y, dtype=np.float64))
        rhs = float(np.sum(x * self.transpose(y), dtype=np.float64))
        absolute = abs(lhs - rhs)
        relative = absolute / max(abs(lhs), abs(rhs), 1.0e-30)
        return {"lhs": lhs, "rhs": rhs, "absolute_error": absolute, "relative_error": relative}


class BenchmarkData:
    def __init__(
        self,
        archive: FineSyntheticArchive,
        parameter_grid: RegularGrid2D,
        state_grid: RegularGrid2D,
        u_truth_parameter: np.ndarray,
        data_noiseless: np.ndarray,
        data_observed: np.ndarray,
        noise: np.ndarray,
        observation_mask: np.ndarray,
        observation_weights: np.ndarray,
    ) -> None:
        self.archive = archive
        self.parameter_grid = parameter_grid
        self.state_grid = state_grid
        self.u_truth = u_truth_parameter
        self.data_noiseless = data_noiseless
        self.data_observed = data_observed
        self.noise = noise
        self.observation_mask = observation_mask
        self.observation_weights = observation_weights


def exact_nested_restrict(
    values: np.ndarray,
    fine_grid: RegularGrid2D,
    coarse_grid: RegularGrid2D,
) -> np.ndarray:
    values = np.asarray(values)
    ratio_x, rem_x = divmod(fine_grid.nx - 1, coarse_grid.nx - 1)
    ratio_y, rem_y = divmod(fine_grid.ny - 1, coarse_grid.ny - 1)
    if rem_x or rem_y or ratio_x < 1 or ratio_y < 1:
        raise ValueError("Fine and coarse grids are not exactly nested.")
    if not np.allclose(fine_grid.x[::ratio_x], coarse_grid.x, atol=2.0e-13):
        raise ValueError("Nested x vertices do not coincide.")
    if not np.allclose(fine_grid.y[::ratio_y], coarse_grid.y, atol=2.0e-13):
        raise ValueError("Nested y vertices do not coincide.")
    restricted = values[::ratio_x, ::ratio_y]
    if restricted.shape != coarse_grid.shape:
        raise RuntimeError("Exact nested restriction returned the wrong shape.")
    return restricted.copy()


def build_benchmark_data(args: argparse.Namespace, archive: FineSyntheticArchive) -> BenchmarkData:
    parameter_grid = RegularGrid2D.from_domain(
        1.0, 1.0, args.parameter_nx_points, args.parameter_ny_points
    )
    state_grid = RegularGrid2D.from_domain(
        1.0, 1.0, args.fem_nx + 1, args.fem_ny + 1
    )
    u_truth = exact_nested_restrict(archive.u_fine, archive.fine_grid, parameter_grid)
    clean = exact_nested_restrict(
        archive.data_noiseless_fine, archive.fine_grid, state_grid
    )
    noisy = exact_nested_restrict(
        archive.data_noisy_fine, archive.fine_grid, state_grid
    )
    noise = exact_nested_restrict(archive.noise_fine, archive.fine_grid, state_grid)
    mask = exact_nested_restrict(
        archive.observation_mask_fine.astype(np.float64), archive.fine_grid, state_grid
    ) > 0.5
    mask &= interior_vertex_mask(state_grid)
    weights = trapezoid_weights(state_grid)
    observed = noisy if args.data_kind == "noisy" else clean
    effective_noise = noise if args.data_kind == "noisy" else np.zeros_like(clean)
    return BenchmarkData(
        archive=archive,
        parameter_grid=parameter_grid,
        state_grid=state_grid,
        u_truth_parameter=u_truth,
        data_noiseless=clean,
        data_observed=observed,
        noise=effective_noise,
        observation_mask=mask,
        observation_weights=weights,
    )


def forward_parameter(
    solver: ReactionSolver2D,
    coefficient_map: StructuredCoefficientGridMap,
    state_map: StructuredLagrangeStateGridMap,
    transfer: BilinearGridTransfer,
    u_parameter: np.ndarray,
):
    u_fem_vertices = transfer.apply(u_parameter)
    solver.set_coefficient(coefficient_map.grid_to_array(u_fem_vertices))
    state = solver.forward()
    return state, state_map.state_to_grid(state)


def tangent_parameter(
    solver: ReactionSolver2D,
    coefficient_map: StructuredCoefficientGridMap,
    state_map: StructuredLagrangeStateGridMap,
    transfer: BilinearGridTransfer,
    state,
    direction_parameter: np.ndarray,
) -> np.ndarray:
    direction_fem = transfer.apply(direction_parameter)
    direction_function = coefficient_map.grid_to_function(
        direction_fem, name="physical_u_parameter_direction_prolongated"
    )
    tangent = solver.tangent(state, direction_function)
    return state_map.state_to_grid(tangent)


def vjp_parameter(
    solver: ReactionSolver2D,
    coefficient_map: StructuredCoefficientGridMap,
    observation_adjoint: ScalarVertexObservationAdjoint,
    transfer: BilinearGridTransfer,
    state,
    residual: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    *,
    iteration: int,
) -> np.ndarray:
    source = observation_adjoint.lift_weighted_residual(
        residual,
        weights,
        mask=mask,
        name=f"weighted_residual_source_{iteration}",
    )
    adjoint = solver.adjoint(source)
    native_covector = solver.parameter_coordinate_adjoint(state, adjoint)
    fem_covector = coefficient_map.array_to_grid(native_covector)
    return transfer.transpose(fem_covector)


def jvp_vjp_check(
    *,
    solver: ReactionSolver2D,
    coefficient_map: StructuredCoefficientGridMap,
    state_map: StructuredLagrangeStateGridMap,
    observation_adjoint: ScalarVertexObservationAdjoint,
    transfer: BilinearGridTransfer,
    state,
    residual: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    parameter_covector: np.ndarray,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(transfer.source.shape)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-30)
    tangent = tangent_parameter(
        solver, coefficient_map, state_map, transfer, state, direction
    )
    lhs = weighted_inner(residual, tangent, weights, mask=mask)
    rhs = float(np.sum(parameter_covector * direction, dtype=np.float64))
    absolute = abs(lhs - rhs)
    relative = absolute / max(abs(lhs), abs(rhs), 1.0e-30)
    return {
        "lhs_weighted_output_pairing": float(lhs),
        "rhs_fixed_parameter_coordinate_pairing": float(rhs),
        "absolute_error": float(absolute),
        "relative_error": float(relative),
    }


def resolve_output_dir(args: argparse.Namespace, archive: FineSyntheticArchive) -> Path:
    if args.output_dir is None:
        multiplier_x = args.fem_nx // (args.parameter_nx_points - 1)
        path = PROJECT_ROOT / "results" / (
            f"reaction_landweber_exact_param{args.parameter_nx_points}x{args.parameter_ny_points}_"
            f"fem{args.fem_nx}x{args.fem_ny}_{multiplier_x}x_{args.data_kind}_"
            f"{canonical_alpha_mode(args.alpha_mode)}"
        )
    else:
        path = args.output_dir.expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / "results" / path
    path = path.resolve()
    if path.exists() and any(path.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing non-empty output directory {path}; use --overwrite or another path."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    """Hash a diagnostic input without loading a second full copy in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _projection_alpha_summary(
    history: dict[str, list[float | int | bool]],
) -> dict[str, Any]:
    """Summarise step-size controls aligned with accepted projection steps."""

    cap_applied = np.asarray(history["physical_change_cap_applied"], dtype=bool)
    backtracking_applied = np.asarray(history["backtracking_applied"], dtype=bool)
    count = int(cap_applied.size)

    def finite_minimum(values: list[float | int | bool]) -> float | None:
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        return None if finite.size == 0 else float(np.min(finite))

    def finite_maximum(values: list[float | int | bool]) -> float | None:
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        return None if finite.size == 0 else float(np.max(finite))

    def finite_mean(values: list[float | int | bool]) -> float | None:
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        return None if finite.size == 0 else float(np.mean(finite))

    return {
        "accepted_update_count": count,
        "physical_change_cap_applied_count": int(np.count_nonzero(cap_applied)),
        "physical_change_cap_applied_fraction": (
            float(np.mean(cap_applied)) if count else None
        ),
        "backtracking_applied_count": int(np.count_nonzero(backtracking_applied)),
        "backtracking_applied_fraction": (
            float(np.mean(backtracking_applied)) if count else None
        ),
        "formula_alpha_before_cap_min": finite_minimum(
            history["formula_alpha_before_cap"]
        ),
        "formula_alpha_before_cap_max": finite_maximum(
            history["formula_alpha_before_cap"]
        ),
        "capped_alpha_min": finite_minimum(history["capped_alpha"]),
        "capped_alpha_max": finite_maximum(history["capped_alpha"]),
        "accepted_alpha_min": finite_minimum(history["accepted_alpha"]),
        "accepted_alpha_max": finite_maximum(history["accepted_alpha"]),
        "physical_change_cap_ratio_min": finite_minimum(
            history["physical_change_cap_ratio"]
        ),
        "physical_change_cap_ratio_mean": finite_mean(
            history["physical_change_cap_ratio"]
        ),
        "accepted_to_capped_alpha_ratio_min": finite_minimum(
            history["accepted_to_capped_alpha_ratio"]
        ),
        "accepted_to_capped_alpha_ratio_mean": finite_mean(
            history["accepted_to_capped_alpha_ratio"]
        ),
    }


def save_plots(
    output_dir: Path,
    *,
    data: BenchmarkData,
    initial_u: np.ndarray,
    recovered_u: np.ndarray,
    prediction: np.ndarray,
    history: dict[str, list[float]],
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    error = recovered_u - data.u_truth
    error_lim = max(float(np.max(np.abs(error))), 1.0e-14)
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)
    panels = (
        (data.u_truth, "Truth u restricted to 120x120", "viridis", None, None),
        (initial_u, "Initial u (fixed parameter grid)", "viridis", None, None),
        (recovered_u, "Recovered u (fixed parameter grid)", "viridis", None, None),
        (error, "Recovered - truth", "coolwarm", -error_lim, error_lim),
        (data.data_observed, f"Observed p on full state grid ({data.data_observed.shape[0]}x{data.data_observed.shape[1]})", "viridis", None, None),
        (prediction - data.data_observed, "Final pressure residual", "coolwarm", None, None),
    )
    for axis, (field, title, cmap, vmin, vmax) in zip(axes.reshape(-1), panels):
        image = axis.imshow(
            np.asarray(field).T,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            aspect="equal",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.savefig(output_dir / "recovery_final.png", dpi=dpi)
    plt.close(figure)

    iteration = np.asarray(history["iteration"])
    residual = np.asarray(history["residual_norm"])
    threshold = np.asarray(history["discrepancy_threshold"])
    objective = np.asarray(history["objective"])
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), constrained_layout=True)
    axes[0].semilogy(iteration, np.maximum(residual, 1.0e-300), label="residual")
    finite = np.isfinite(threshold) & (threshold > 0)
    if np.any(finite):
        axes[0].semilogy(iteration[finite], threshold[finite], "--", label="target")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("weighted L2 norm")
    axes[0].legend()
    axes[1].semilogy(iteration, np.maximum(objective, 1.0e-300))
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("objective")
    figure.savefig(output_dir / "history.png", dpi=dpi)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Exactly one MPI rank is required.")
    archive = load_fine_synthetic_archive(
        args.synthetic_file,
        project_root=PROJECT_ROOT,
    )
    validate_args(args, archive)
    data = build_benchmark_data(args, archive)
    output_dir = resolve_output_dir(args, archive)
    alpha_mode = canonical_alpha_mode(args.alpha_mode)
    options = solver_options(args)

    transfer = BilinearGridTransfer(data.parameter_grid, data.state_grid)
    transfer_check = transfer.adjoint_test(seed=1729)
    if transfer_check["relative_error"] > 5.0e-13:
        raise RuntimeError(
            "Bilinear transfer transpose check failed: "
            f"{transfer_check['relative_error']:.3e}."
        )

    parameter_weights = trapezoid_weights(data.parameter_grid)
    initial_u = np.full(data.parameter_grid.shape, args.initial_u, dtype=np.float64)
    projection_tracker = (
        ProjectionDiagnostics(
            data.parameter_grid.shape,
            float(args.u_min),
            float(args.u_max),
        )
        if args.projection_diagnostics
        else None
    )
    projection_alpha_history: dict[str, list[float | int | bool]] | None = (
        {
            key: []
            for key in (
                "update_iteration",
                "formula_alpha_before_cap",
                "capped_alpha",
                "accepted_alpha",
                "physical_change_cap_alpha",
                "physical_change_cap_applied",
                "physical_change_cap_ratio",
                "accepted_to_capped_alpha_ratio",
                "backtracking_applied",
            )
        }
        if args.projection_diagnostics
        else None
    )

    print(f"SCRIPT PATCH VERSION: {PATCH_VERSION}", flush=True)
    print("Exact multiresolution Reaction Landweber", flush=True)
    print(f"  synthetic file            : {archive.path}", flush=True)
    print(
        f"  truth source/fine FEM     : {archive.truth_kind} / "
        f"{archive.fine_fem_nx}x{archive.fine_fem_ny} cells",
        flush=True,
    )
    print(
        f"  fixed parameter grid      : {data.parameter_grid.nx}x{data.parameter_grid.ny} points",
        flush=True,
    )
    print(
        f"  state FEM / p vertices    : {args.fem_nx}x{args.fem_ny} cells / "
        f"{data.state_grid.nx}x{data.state_grid.ny} vertices",
        flush=True,
    )
    print(
        f"  state refinement factor   : {args.fem_nx // (args.parameter_nx_points - 1)}x",
        flush=True,
    )
    print(
        f"  pressure observations     : {np.count_nonzero(data.observation_mask)} "
        "interior vertices (all interior state vertices)",
        flush=True,
    )
    print(f"  data kind                 : {args.data_kind}", flush=True)
    print(f"  alpha mode                : {alpha_mode}", flush=True)
    print(f"  preconditioner            : {args.preconditioner}", flush=True)
    if args.projection_diagnostics:
        print("  projection diagnostics    : ON (accepted updates)", flush=True)
    print(
        "  backtracking             : "
        + (
            f"ON (factor={args.backtracking_factor:g}, max={args.max_backtracks}; "
            "final proposal accepted if Armijo never passes)"
            if args.backtracking
            else "OFF (default; capped/projected proposal accepted directly)"
        ),
        flush=True,
    )
    print(f"  transfer B^T check        : {transfer_check['relative_error']:.3e}", flush=True)

    config = ReactionConfig(
        fem_nx=args.fem_nx,
        fem_ny=args.fem_ny,
        state_degree=STATE_DEGREE,
        rhs=float(archive.rhs),
        u_min=float(args.u_min),
    )
    solver = ReactionSolver2D(
        config,
        comm=MPI.COMM_WORLD,
        minimum_u=float(args.u_min),
        initial_u=float(args.initial_u),
        solver_options=options,
    )
    observation_adjoint = None
    covariance = None
    trajectory_writer: TrajectoryStateWriter | None = None
    started = perf_counter()
    try:
        coefficient_map = StructuredCoefficientGridMap(solver, data.state_grid)
        state_map = StructuredLagrangeStateGridMap(solver, data.state_grid)
        try:
            observation_adjoint = ScalarVertexObservationAdjoint(
                solver, state_map, solver_options=options
            )
        except TypeError as exc:
            if "solver_options" not in str(exc):
                raise
            # Compatibility with older cores used for small local tests.
            observation_adjoint = ScalarVertexObservationAdjoint(solver, state_map)
        if args.preconditioner == "matern":
            covariance = StationaryMaternCovariance2D(
                data.parameter_grid,
                nu=args.matern_nu,
                length_scale=args.matern_length_scale,
                variance=args.matern_variance,
                fft_workers=args.fft_workers,
            )
            resolved_preconditioner: dict[str, Any] = {
                "kind": "matern",
                "coordinate_space": "Euclidean nodal covector-to-vector map",
                "nu": float(args.matern_nu),
                "length_scale": float(args.matern_length_scale),
                "variance": float(args.matern_variance),
            }
        else:
            covariance = IdentityCovariance2D(data.parameter_grid)
            resolved_preconditioner = {
                "kind": "identity",
                "coordinate_space": "Euclidean nodal covector-to-vector map",
                "nu": None,
                "length_scale": None,
                "variance": None,
            }

        # Measured total model discrepancy for this particular state resolution:
        # parameter representation error + state discretisation error.
        _, prediction_at_parameter_truth = forward_parameter(
            solver, coefficient_map, state_map, transfer, data.u_truth
        )
        model_mismatch = prediction_at_parameter_truth - data.data_noiseless
        model_discrepancy_norm = weighted_norm(
            model_mismatch, data.observation_weights, mask=data.observation_mask
        )
        data_norm = weighted_norm(
            data.data_observed, data.observation_weights, mask=data.observation_mask
        )
        clean_data_norm = weighted_norm(
            data.data_noiseless, data.observation_weights, mask=data.observation_mask
        )
        noise_norm = weighted_norm(
            data.noise, data.observation_weights, mask=data.observation_mask
        )
        discrepancy_target = args.discrepancy_tau * (
            noise_norm + args.model_discrepancy_scale * model_discrepancy_norm
        )
        morozov_noise_target = args.discrepancy_tau * noise_norm
        crossing_snapshots = (
            DiscrepancyCrossingSnapshots(
                {
                    "noise_norm": noise_norm,
                    "morozov_noise": morozov_noise_target,
                    "model_adjusted": discrepancy_target,
                }
            )
            if args.crossing_snapshots
            else None
        )
        print(f"  clean data norm           : {clean_data_norm:.8e}", flush=True)
        print(f"  observed data norm        : {data_norm:.8e}", flush=True)
        print(f"  restricted noise norm     : {noise_norm:.8e}", flush=True)
        print(
            f"  relative restricted noise : {noise_norm / max(clean_data_norm,1e-30):.8e}",
            flush=True,
        )
        print(f"  total model discrepancy   : {model_discrepancy_norm:.8e}", flush=True)
        print(
            f"  relative model discrepancy: {model_discrepancy_norm / max(clean_data_norm,1e-30):.8e}",
            flush=True,
        )
        print(f"  discrepancy target        : {discrepancy_target:.8e}", flush=True)
        if crossing_snapshots is not None:
            print(
                "  crossing snapshots        : ON "
                "(delta, tau*delta, model-adjusted; no early stop)",
                flush=True,
            )

        current_u = initial_u.copy()
        state, prediction = forward_parameter(
            solver, coefficient_map, state_map, transfer, current_u
        )
        if int(args.trajectory_state_every) > 0:
            trajectory_writer = TrajectoryStateWriter(
                output_dir,
                state_shape=data.parameter_grid.shape,
                save_every=int(args.trajectory_state_every),
            )
            print(
                "  trajectory states        : every "
                f"{int(args.trajectory_state_every)} iterations",
                flush=True,
            )
        landweber_started = perf_counter()
        history: dict[str, list[float]] = {
            key: []
            for key in (
                "iteration", "objective", "residual_norm", "relative_residual",
                "relative_u_error", "proposed_alpha", "accepted_alpha",
                "gradient_direction_inner", "j_direction_norm",
                "j_direction_pairing_relative_error", "backtracks",
                "armijo_satisfied", "forced_final_backtrack_accept",
                "max_proposed_change", "active_bound_fraction",
                "discrepancy_threshold", "discrepancy_ratio",
                "operator_assemblies_factorizations", "landweber_elapsed_seconds",
            )
        }
        minimum_residual_norm = float("inf")
        minimum_residual_iteration = 0
        # Oracle synthetic diagnostic: retain the iterate with the
        # smallest relative coefficient error. This is NOT a stopping rule.
        minimum_u_error_tracked = float("inf")
        minimum_u_error_u = current_u.copy()
        minimum_u_error_prediction = prediction.copy()
        minimum_u_error_iteration_tracked = 0
        minimum_u_error_projection_maps = (
            projection_tracker.hit_count_arrays()
            if projection_tracker is not None
            else None
        )
        minimum_residual_u = current_u.copy()
        minimum_residual_prediction = prediction.copy()
        latest_covector = None
        latest_covector_iteration = None
        latest_direction = None
        latest_direction_iteration = None
        latest_jvp = None
        latest_jvp_iteration = None
        adjoint_check = None
        stop_reason = "maximum iterations reached"

        for iteration in range(args.iterations + 1):
            if trajectory_writer is not None:
                trajectory_writer.observe(iteration, current_u)
            residual = prediction - data.data_observed
            residual_norm = weighted_norm(
                residual, data.observation_weights, mask=data.observation_mask
            )
            residual_sq = residual_norm * residual_norm
            objective = 0.5 * residual_sq
            relative_residual = residual_norm / max(data_norm, 1.0e-30)
            relative_u_error = weighted_norm(
                current_u - data.u_truth, parameter_weights
            ) / max(weighted_norm(data.u_truth, parameter_weights), 1.0e-30)
            if residual_norm < minimum_residual_norm:
                minimum_residual_norm = residual_norm
                minimum_residual_iteration = iteration
                minimum_residual_u = current_u.copy()
                minimum_residual_prediction = prediction.copy()

            coordinate_covector = vjp_parameter(
                solver,
                coefficient_map,
                observation_adjoint,
                transfer,
                state,
                residual,
                data.observation_weights,
                data.observation_mask,
                iteration=iteration,
            )
            latest_covector = coordinate_covector.copy()
            latest_covector_iteration = iteration
            if iteration == 0:
                adjoint_check = jvp_vjp_check(
                    solver=solver,
                    coefficient_map=coefficient_map,
                    state_map=state_map,
                    observation_adjoint=observation_adjoint,
                    transfer=transfer,
                    state=state,
                    residual=residual,
                    weights=data.observation_weights,
                    mask=data.observation_mask,
                    parameter_covector=coordinate_covector,
                    seed=args.adjoint_check_seed,
                )
                print(
                    f"  iteration-0 JVP/VJP check : {adjoint_check['relative_error']:.3e}",
                    flush=True,
                )
                if adjoint_check["relative_error"] > args.adjoint_check_tolerance:
                    raise AssertionError(
                        "Fixed-parameter JVP/VJP test failed: "
                        f"{adjoint_check['relative_error']:.6e} > "
                        f"{args.adjoint_check_tolerance:.6e}."
                    )

            if relative_u_error < minimum_u_error_tracked:
                minimum_u_error_tracked = float(relative_u_error)
                minimum_u_error_iteration_tracked = int(iteration)
                minimum_u_error_u = current_u.copy()
                minimum_u_error_prediction = prediction.copy()
                if projection_tracker is not None:
                    minimum_u_error_projection_maps = (
                        projection_tracker.hit_count_arrays()
                    )
            if crossing_snapshots is not None:
                crossing_snapshots.observe(
                    iteration=iteration,
                    parameter=current_u,
                    prediction=prediction,
                    residual_norm=residual_norm,
                    relative_u_error=relative_u_error,
                )

            history["iteration"].append(float(iteration))
            history["objective"].append(float(objective))
            history["residual_norm"].append(float(residual_norm))
            history["relative_residual"].append(float(relative_residual))
            history["relative_u_error"].append(float(relative_u_error))
            for key in (
                "proposed_alpha", "accepted_alpha", "gradient_direction_inner",
                "j_direction_norm", "j_direction_pairing_relative_error",
                "backtracks", "armijo_satisfied", "forced_final_backtrack_accept",
                "max_proposed_change", "active_bound_fraction",
            ):
                history[key].append(float("nan"))
            history["discrepancy_threshold"].append(
                float(discrepancy_target) if args.discrepancy_principle else float("nan")
            )
            history["discrepancy_ratio"].append(
                residual_norm / discrepancy_target
                if args.discrepancy_principle and discrepancy_target > 0
                else float("nan")
            )
            history["operator_assemblies_factorizations"].append(
                float(solver.cache_diagnostics()["operator_assemblies_factorizations"])
            )
            history["landweber_elapsed_seconds"].append(
                float(perf_counter() - landweber_started)
            )

            if iteration % args.print_every == 0:
                print(
                    f"iter={iteration:4d}  rel_res={relative_residual:.6e}  "
                    f"rel_u={relative_u_error:.6e}  res={residual_norm:.6e}  "
                    f"phi={objective:.6e}",
                    flush=True,
                )

            if (
                args.discrepancy_principle
                and discrepancy_target > 0
                and residual_norm <= discrepancy_target
            ):
                stop_reason = "discrepancy target reached"
                break
            if (
                args.relative_residual_tolerance > 0
                and relative_residual <= args.relative_residual_tolerance
            ):
                stop_reason = "relative residual tolerance reached"
                break
            if iteration >= args.iterations:
                stop_reason = "maximum iterations reached"
                break

            direction = covariance.apply_coordinate_covector(coordinate_covector)
            q_dot_d = float(np.sum(coordinate_covector * direction, dtype=np.float64))
            if not np.isfinite(q_dot_d) or q_dot_d <= 0:
                stop_reason = "non-positive covariance direction pairing"
                break
            latest_direction = direction.copy()
            latest_direction_iteration = iteration
            history["gradient_direction_inner"][-1] = q_dot_d

            if alpha_mode == "derivative":
                j_direction = tangent_parameter(
                    solver, coefficient_map, state_map, transfer, state, direction
                )
                latest_jvp = j_direction.copy()
                latest_jvp_iteration = iteration
                j_norm = weighted_norm(
                    j_direction, data.observation_weights, mask=data.observation_mask
                )
                denominator = j_norm * j_norm
                if not np.isfinite(denominator) or denominator <= 0:
                    stop_reason = "non-positive tangent norm"
                    break
                formula_alpha = args.damping * q_dot_d / denominator
                pairing = weighted_inner(
                    residual, j_direction, data.observation_weights,
                    mask=data.observation_mask
                )
                history["j_direction_norm"][-1] = j_norm
                history["j_direction_pairing_relative_error"][-1] = (
                    abs(pairing - q_dot_d) / max(abs(pairing), abs(q_dot_d), 1.0e-30)
                )
            else:
                formula_alpha = args.damping * residual_sq / q_dot_d

            if not np.isfinite(formula_alpha) or formula_alpha <= 0:
                stop_reason = "non-positive proposed alpha"
                break
            max_direction = float(np.max(np.abs(direction)))
            if max_direction > 0:
                physical_change_cap_alpha = (
                    args.max_physical_change / max_direction
                )
                proposed_alpha = min(formula_alpha, physical_change_cap_alpha)
            else:
                physical_change_cap_alpha = formula_alpha
                proposed_alpha = formula_alpha
            physical_change_cap_applied = proposed_alpha < formula_alpha
            physical_change_cap_ratio = proposed_alpha / formula_alpha
            history["proposed_alpha"][-1] = float(proposed_alpha)
            history["max_proposed_change"][-1] = float(proposed_alpha * max_direction)

            accepted = False
            accepted_u = None
            accepted_state = None
            accepted_prediction = None

            if not args.backtracking:
                # Default behaviour: accept the capped/projected Landweber
                # proposal directly. No Armijo test and no line search.
                trial_alpha = proposed_alpha
                candidate_u = np.clip(
                    current_u - trial_alpha * direction, args.u_min, args.u_max
                )
                delta = candidate_u - current_u
                if not np.any(delta):
                    solver.set_coefficient(
                        coefficient_map.grid_to_array(transfer.apply(current_u))
                    )
                    stop_reason = "projected Landweber proposal produced no change"
                    break
                candidate_state, candidate_prediction = forward_parameter(
                    solver, coefficient_map, state_map, transfer, candidate_u
                )
                accepted = True
                accepted_u = candidate_u
                accepted_state = candidate_state
                accepted_prediction = candidate_prediction
                history["accepted_alpha"][-1] = float(trial_alpha)
                history["backtracks"][-1] = 0.0
                history["armijo_satisfied"][-1] = float("nan")
                history["forced_final_backtrack_accept"][-1] = 0.0
                history["active_bound_fraction"][-1] = float(
                    np.mean(
                        (candidate_u <= args.u_min + 1.0e-12)
                        | (candidate_u >= args.u_max - 1.0e-12)
                    )
                )
            else:
                # Optional Armijo backtracking. If no trial passes Armijo,
                # accept the final/smallest usable proposal instead of
                # terminating the Landweber iteration.
                last_trial = None
                for backtracks in range(args.max_backtracks + 1):
                    trial_alpha = proposed_alpha * args.backtracking_factor**backtracks
                    candidate_u = np.clip(
                        current_u - trial_alpha * direction, args.u_min, args.u_max
                    )
                    delta = candidate_u - current_u
                    if not np.any(delta):
                        continue
                    directional_derivative = float(
                        np.sum(coordinate_covector * delta, dtype=np.float64)
                    )
                    if not np.isfinite(directional_derivative):
                        continue

                    candidate_state, candidate_prediction = forward_parameter(
                        solver, coefficient_map, state_map, transfer, candidate_u
                    )
                    candidate_residual = candidate_prediction - data.data_observed
                    candidate_objective = 0.5 * weighted_inner(
                        candidate_residual,
                        candidate_residual,
                        data.observation_weights,
                        mask=data.observation_mask,
                    )
                    last_trial = (
                        float(trial_alpha),
                        int(backtracks),
                        candidate_u,
                        candidate_state,
                        candidate_prediction,
                        float(directional_derivative),
                        float(candidate_objective),
                    )

                    armijo_satisfied = (
                        directional_derivative < 0.0
                        and candidate_objective
                        <= objective + args.armijo * directional_derivative
                    )
                    if armijo_satisfied:
                        accepted = True
                        accepted_u = candidate_u
                        accepted_state = candidate_state
                        accepted_prediction = candidate_prediction
                        history["accepted_alpha"][-1] = float(trial_alpha)
                        history["backtracks"][-1] = float(backtracks)
                        history["armijo_satisfied"][-1] = 1.0
                        history["forced_final_backtrack_accept"][-1] = 0.0
                        history["active_bound_fraction"][-1] = float(
                            np.mean(
                                (candidate_u <= args.u_min + 1.0e-12)
                                | (candidate_u >= args.u_max - 1.0e-12)
                            )
                        )
                        break

                if not accepted:
                    if last_trial is None:
                        solver.set_coefficient(
                            coefficient_map.grid_to_array(transfer.apply(current_u))
                        )
                        stop_reason = "backtracking produced no usable projected proposal"
                        break

                    (
                        trial_alpha,
                        backtracks,
                        candidate_u,
                        candidate_state,
                        candidate_prediction,
                        directional_derivative,
                        candidate_objective,
                    ) = last_trial
                    accepted = True
                    accepted_u = candidate_u
                    accepted_state = candidate_state
                    accepted_prediction = candidate_prediction
                    history["accepted_alpha"][-1] = float(trial_alpha)
                    history["backtracks"][-1] = float(backtracks)
                    history["armijo_satisfied"][-1] = 0.0
                    history["forced_final_backtrack_accept"][-1] = 1.0
                    history["active_bound_fraction"][-1] = float(
                        np.mean(
                            (candidate_u <= args.u_min + 1.0e-12)
                            | (candidate_u >= args.u_max - 1.0e-12)
                        )
                    )
                    print(
                        "  Armijo not satisfied after "
                        f"{backtracks} backtracks; accepting final proposal "
                        f"alpha={trial_alpha:.6e}",
                        flush=True,
                    )

            assert (
                accepted_u is not None
                and accepted_state is not None
                and accepted_prediction is not None
            )
            if projection_tracker is not None:
                assert projection_alpha_history is not None
                accepted_alpha = float(history["accepted_alpha"][-1])
                raw_accepted_candidate = current_u - accepted_alpha * direction
                projection_tracker.record(
                    current_u,
                    raw_accepted_candidate,
                    accepted_u,
                )
                # This records the source-state index: iteration 0 is u_0 -> u_1.
                projection_alpha_history["update_iteration"].append(iteration)
                projection_alpha_history["formula_alpha_before_cap"].append(
                    float(formula_alpha)
                )
                projection_alpha_history["capped_alpha"].append(
                    float(proposed_alpha)
                )
                projection_alpha_history["accepted_alpha"].append(accepted_alpha)
                projection_alpha_history["physical_change_cap_alpha"].append(
                    float(physical_change_cap_alpha)
                )
                projection_alpha_history["physical_change_cap_applied"].append(
                    bool(physical_change_cap_applied)
                )
                projection_alpha_history["physical_change_cap_ratio"].append(
                    float(physical_change_cap_ratio)
                )
                projection_alpha_history["accepted_to_capped_alpha_ratio"].append(
                    accepted_alpha / proposed_alpha
                )
                projection_alpha_history["backtracking_applied"].append(
                    bool(history["backtracks"][-1] > 0.0)
                )

            current_u = accepted_u
            state = accepted_state
            prediction = accepted_prediction

        assert adjoint_check is not None
        final_residual = prediction - data.data_observed
        final_residual_norm = weighted_norm(
            final_residual, data.observation_weights, mask=data.observation_mask
        )
        final_relative_residual = final_residual_norm / max(data_norm, 1.0e-30)
        final_relative_u_error = weighted_norm(
            current_u - data.u_truth, parameter_weights
        ) / max(weighted_norm(data.u_truth, parameter_weights), 1.0e-30)
        landweber_seconds = float(perf_counter() - landweber_started)
        total_seconds = float(perf_counter() - started)

        arrays: dict[str, np.ndarray] = {
            "truth_u_parameter_grid": data.u_truth,
            "initial_u_parameter_grid": initial_u,
            "recovered_u_final": current_u,
            "recovered_u_minimum_residual": minimum_residual_u,
            "recovered_u_minimum_u_error": minimum_u_error_u,
            "minimum_u_error_prediction": minimum_u_error_prediction,
            "minimum_u_error": np.asarray(minimum_u_error_tracked),
            "minimum_u_error_iteration": np.asarray(
                minimum_u_error_iteration_tracked, dtype=np.int64
            ),
            "minimum_residual_prediction": minimum_residual_prediction,
            "minimum_residual_iteration": np.asarray(minimum_residual_iteration),
            "parameter_x_grid": data.parameter_grid.x,
            "parameter_y_grid": data.parameter_grid.y,
            "state_x_grid": data.state_grid.x,
            "state_y_grid": data.state_grid.y,
            "data_noiseless": data.data_noiseless,
            "data_observed": data.data_observed,
            "noise_effective": data.noise,
            "observation_mask": data.observation_mask,
            "observation_weights": data.observation_weights,
            "model_prediction_at_parameter_truth": prediction_at_parameter_truth,
            "model_discrepancy_at_parameter_truth": model_mismatch,
            "final_prediction": prediction,
            "final_residual": final_residual,
            "evaluated_parameter_covector": np.asarray([]) if latest_covector is None else latest_covector,
            "evaluated_covector_iteration": np.asarray(-1 if latest_covector_iteration is None else latest_covector_iteration),
            "latest_parameter_matern_direction": np.asarray([]) if latest_direction is None else latest_direction,
            "latest_direction_iteration": np.asarray(-1 if latest_direction_iteration is None else latest_direction_iteration),
            "latest_full_state_tangent_jd": np.asarray([]) if latest_jvp is None else latest_jvp,
            "latest_jvp_iteration": np.asarray(-1 if latest_jvp_iteration is None else latest_jvp_iteration),
        }
        for key, values in history.items():
            arrays[f"history_{key}"] = np.asarray(values, dtype=np.float64)
        if crossing_snapshots is not None:
            arrays.update(crossing_snapshots.archive_arrays())
        for key, value in adjoint_check.items():
            arrays[f"adjoint_check_{key}"] = np.asarray(value)
        projection_summary = None
        if projection_tracker is not None:
            assert projection_alpha_history is not None
            assert minimum_u_error_projection_maps is not None
            for key, values in projection_tracker.history_arrays().items():
                arrays[f"projection_diagnostics_{key}"] = values
            for key, values in projection_tracker.hit_count_arrays().items():
                arrays[f"projection_diagnostics_{key}"] = values
            for key, values in minimum_u_error_projection_maps.items():
                arrays[
                    "projection_diagnostics_through_minimum_u_error_" + key
                ] = values
            for key, values in projection_alpha_history.items():
                if key == "update_iteration":
                    dtype = np.int64
                elif key in (
                    "physical_change_cap_applied",
                    "backtracking_applied",
                ):
                    dtype = np.bool_
                else:
                    dtype = np.float64
                arrays[f"projection_diagnostics_{key}"] = np.asarray(
                    values, dtype=dtype
                )

            projection_summary = projection_tracker.summary()
            projection_summary["update_iteration_definition"] = (
                "zero-based source-state/update index; 0 denotes u_0 -> u_1"
            )
            projection_summary["input_sha256"] = {
                "synthetic_file": _sha256_file(archive.path),
            }
            projection_summary["step_size_controls"] = _projection_alpha_summary(
                projection_alpha_history
            )
            selected_update_mask = (
                np.asarray(
                    projection_alpha_history["update_iteration"],
                    dtype=np.int64,
                )
                < minimum_u_error_iteration_tracked
            )
            selected_projection_summary = summarize_projection_diagnostics(
                {
                    name: values[selected_update_mask]
                    for name, values in projection_tracker.history_arrays().items()
                },
                minimum_u_error_projection_maps,
            )
            selected_projection_summary["state_iteration"] = int(
                minimum_u_error_iteration_tracked
            )
            projection_summary["through_minimum_u_error_state"] = (
                selected_projection_summary
            )
        np.savez_compressed(output_dir / "landweber_exact_multires_results.npz", **arrays)

        summary = {
            "patch_version": PATCH_VERSION,
            "stop_reason": stop_reason,
            "alpha_mode": alpha_mode,
            "preconditioner": resolved_preconditioner,
            "backtracking_enabled": bool(args.backtracking),
            "backtracking_factor": float(args.backtracking_factor),
            "max_backtracks": int(args.max_backtracks),
            "final_backtrack_proposal_is_accepted_if_armijo_fails": True,
            "parameter_grid_points": [data.parameter_grid.nx, data.parameter_grid.ny],
            "state_fem_cells": [args.fem_nx, args.fem_ny],
            "state_pressure_vertices": [data.state_grid.nx, data.state_grid.ny],
            "truth_fem_cells": [archive.fine_fem_nx, archive.fine_fem_ny],
            "number_of_pressure_observations": int(np.count_nonzero(data.observation_mask)),
            "clean_data_norm": float(clean_data_norm),
            "noise_norm": float(noise_norm),
            "relative_noise_norm": float(noise_norm / max(clean_data_norm, 1.0e-30)),
            "model_discrepancy_norm": float(model_discrepancy_norm),
            "relative_model_discrepancy": float(model_discrepancy_norm / max(clean_data_norm, 1.0e-30)),
            "discrepancy_target": float(discrepancy_target),
            "iterations_evaluated": int(history["iteration"][-1]),
            "final_relative_residual": float(final_relative_residual),
            "final_relative_u_error": float(final_relative_u_error),
            "minimum_residual_norm": float(minimum_residual_norm),
            "minimum_residual_iteration": int(minimum_residual_iteration),
            "minimum_u_error": float(minimum_u_error_tracked),
            "minimum_u_error_iteration": int(
                minimum_u_error_iteration_tracked
            ),
            "transfer_adjoint_test": transfer_check,
            "landweber_adjoint_test": adjoint_check,
            "landweber_elapsed_seconds": landweber_seconds,
            "total_driver_elapsed_seconds": total_seconds,
            "solver_cache": solver.cache_diagnostics(),
            "observation_adjoint": observation_adjoint.diagnostics(),
        }
        if crossing_snapshots is not None:
            crossing_summary = crossing_snapshots.summary()
            crossing_summary["threshold_definitions"] = {
                "noise_norm": "delta = ||effective observation noise||_W",
                "morozov_noise": "tau * delta",
                "model_adjusted": (
                    "tau * (delta + model_discrepancy_scale * exact-grid "
                    "discretisation discrepancy at u_truth)"
                ),
            }
            summary["discrepancy_crossing_snapshots"] = crossing_summary
        if projection_summary is not None:
            summary["projection_diagnostics"] = projection_summary
        command_arguments = vars(args).copy()
        if not args.projection_diagnostics:
            # Preserve the historical settings schema in the default-off mode.
            command_arguments.pop("projection_diagnostics")
        if crossing_snapshots is None:
            command_arguments.pop("crossing_snapshots")
        if trajectory_writer is None:
            command_arguments.pop("trajectory_state_every")
        else:
            summary["trajectory_states"] = {
                "path": "trajectory_states.h5",
                "save_every": int(args.trajectory_state_every),
                "saved_state_count": int(trajectory_writer.count),
            }
        settings = {
            "patch_version": PATCH_VERSION,
            "command_arguments": command_arguments,
            "preconditioner": resolved_preconditioner,
            "synthetic_file": archive.path,
            "mathematics": {
                "parameter_space": "fixed regular nodal grid",
                "forward": "F_h(U)=S_h(B_h U)",
                "adjoint": "F_h'(U)^*=B_h^T S_h'(B_h U)^*",
                "objective": "0.5 * full-interior-pressure residual^T W residual",
                "direction": (
                    "d=K_Matern q on the fixed parameter grid"
                    if args.preconditioner == "matern"
                    else "d=q under the fixed-grid Euclidean coordinate identity"
                ),
                "step_acceptance": (
                    "default: accept capped/projected proposal directly; "
                    "optional backtracking: Armijo search, but accept final "
                    "usable proposal if Armijo never passes"
                ),
            },
            "result_summary": summary,
        }
        save_json(output_dir / "results.json", summary)
        save_json(output_dir / "settings.json", settings)
        if args.plots:
            save_plots(
                output_dir,
                data=data,
                initial_u=initial_u,
                recovered_u=current_u,
                prediction=prediction,
                history=history,
                dpi=args.dpi,
            )
        if trajectory_writer is not None:
            trajectory_writer.finalize()

        print("", flush=True)
        print("Exact multiresolution Reaction Landweber completed", flush=True)
        print(f"reason:                   {stop_reason}", flush=True)
        print(f"final relative residual:  {final_relative_residual:.6e}", flush=True)
        print(f"final relative u error:   {final_relative_u_error:.6e}", flush=True)
        print(f"minimum residual iter:    {minimum_residual_iteration}", flush=True)
        print(f"Landweber time:           {landweber_seconds:.3f} s ({landweber_seconds/60:.3f} min)", flush=True)
        print(f"saved in:                 {output_dir}", flush=True)
    finally:
        if trajectory_writer is not None:
            trajectory_writer.abort()
        if covariance is not None:
            covariance.close()
        if observation_adjoint is not None:
            observation_adjoint.close()
        solver.close()


if __name__ == "__main__":
    main()
