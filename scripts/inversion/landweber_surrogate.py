#!/usr/bin/env python3
"""Data-only Landweber inversion with a trained surrogate.

This script inverts the direct physical reaction coefficient ``u`` from an
archive produced by :mod:`generate_reaction_fine_truth_data`.  It deliberately
does *not* import FEniCS or generate a new truth/data set.  The high-resolution
FEM pressure and coefficient stored in the archive are bilinearly resampled to
the single production grid used by both trained emulators:

    120 x 120 points for u, p, observations, and the inverse variable.

Keeping this grid fixed is essential for a like-for-like FNO/DeepONet
comparison.  In particular, the Darcy-style DeepONet branch encoder is
architecturally tied to a 120 x 120 coefficient field.  The continuous trunk
is evaluated at the same 120 x 120 observation coordinates, in chunks, so its
output pairing is identical to the FNO pairing.

The two step proposals are:

``derivative``
    ``alpha = damping * (q dot d) / ||J d||_W^2``.
``adjoint-step-norm``
    ``alpha = damping * ||r||_W^2 / (q dot d)``.

``minimum-residual`` is retained only as a command-line alias for
``adjoint-step-norm``.  By default the capped/projected Landweber proposal
is accepted directly with no Armijo line search.  Optional projected Armijo
backtracking can be enabled; if no Armijo trial succeeds, the final/smallest
usable proposal is accepted.  The production default uses the non-periodic
stationary Matérn covariance preconditioner; an explicit coordinate-space
identity option is retained only for controlled preconditioner ablations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import (  # noqa: E402
    ReactionDeepONetEmulator,
    ReactionFNOEmulator,
    load_reaction_emulator,
)
from reaction_core.grid import RegularGrid2D  # noqa: E402
from reaction_core.inversion import (  # noqa: E402
    IdentityCovariance2D,
    StationaryMaternCovariance2D,
    interior_vertex_mask,
    resample_regular_grid_field,
    trapezoid_weights,
    weighted_inner,
    weighted_norm,
    weighted_residual_covector,
)
from reaction_core.discrepancy_snapshots import (  # noqa: E402
    DiscrepancyCrossingSnapshots,
)
from reaction_core.projection_diagnostics import (  # noqa: E402
    ProjectionDiagnostics,
    summarize_projection_diagnostics,
)
from reaction_core.synthetic_archive import (  # noqa: E402
    SYNTHETIC_FILE_FORMAT,
    FineSyntheticArchive,
    load_fine_synthetic_archive,
)
from reaction_core.trajectory_states import TrajectoryStateWriter  # noqa: E402


RESULT_FILE_FORMAT = "reaction-surrogate-landweber-v1"
PATCH_VERSION = (
    "2026-08-22-reaction-surrogate-landweber-semiconv-v1"
    "+projection-diagnostics-v1"
    "+discrepancy-crossing-snapshots-v1"
    "+u0-200-support-v1"
    "+preconditioner-ablation-v1"
)
PRODUCTION_GRID_SIZE = 120
TRAINING_U_BOUNDS = (0.0, 200.0)


@dataclass(frozen=True)
class SurrogateInversionData:
    """One source archive resampled to the shared production grid."""

    archive: FineSyntheticArchive
    grid: RegularGrid2D
    u_truth: np.ndarray
    data_noiseless: np.ndarray
    data_observed: np.ndarray
    stored_noise: np.ndarray
    effective_noise: np.ndarray
    observation_mask: np.ndarray
    observation_weights: np.ndarray
    data_kind: str


def _resolve_input(path: Path) -> Path:
    """Resolve a project-relative input path without requiring it to exist."""

    expanded = path.expanduser()
    return (expanded if expanded.is_absolute() else PROJECT_ROOT / expanded).resolve()


def _json_ready(value: Any) -> Any:
    """Convert project values to the small JSON types accepted by ``json``."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    # Keep this module free of a direct torch import, while still allowing
    # checkpoints that happened to store normalisation statistics as tensors.
    if (
        hasattr(value, "detach")
        and hasattr(value, "cpu")
        and hasattr(value, "tolist")
    ):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def write_json(path: Path, values: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(values), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sha256_file(path: Path) -> str:
    """Hash a diagnostic input without loading a second full copy in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resample_synthetic_data(
    archive: FineSyntheticArchive,
    *,
    grid_size: int,
    data_kind: str,
) -> SurrogateInversionData:
    """Bilinearly resample fine FEM quantities to the common model grid.

    This intentionally does not require a nested grid ratio.  A 1025-point
    fine P1 grid therefore maps fairly to the 120-point FNO grid and to the
    120-point DeepONet branch/observation grid.
    """

    if int(grid_size) != PRODUCTION_GRID_SIZE:
        raise ValueError(
            f"Only the shared production {PRODUCTION_GRID_SIZE}x"
            f"{PRODUCTION_GRID_SIZE} grid is supported; received {grid_size}."
        )
    grid = RegularGrid2D.from_domain(1.0, 1.0, int(grid_size), int(grid_size))
    u_truth = resample_regular_grid_field(archive.u_fine, archive.fine_grid, grid)
    clean = resample_regular_grid_field(
        archive.data_noiseless_fine, archive.fine_grid, grid
    )
    noisy = resample_regular_grid_field(
        archive.data_noisy_fine, archive.fine_grid, grid
    )
    stored_noise = resample_regular_grid_field(
        archive.noise_fine, archive.fine_grid, grid
    )
    # Preserve the source observation geometry under resampling, then exclude
    # the target boundary explicitly because both models are trained with
    # Dirichlet p=0 boundaries rather than boundary observations.
    mask_values = resample_regular_grid_field(
        archive.observation_mask_fine.astype(np.float64), archive.fine_grid, grid
    )
    mask = (mask_values > 0.5) & interior_vertex_mask(grid)
    if not np.any(mask):
        raise ValueError("No interior observations remain after resampling.")
    weights = trapezoid_weights(grid)

    if data_kind == "noisy":
        observed = noisy
        # Defining the effective noise by the *two resampled data arrays*
        # makes the stored discrepancy exactly consistent with the inverse
        # objective even if roundoff differs from a separately resampled noise.
        effective_noise = observed - clean
    elif data_kind == "noiseless":
        observed = clean
        effective_noise = np.zeros_like(clean)
    else:  # Defensive: argparse already restricts this.
        raise ValueError(f"Unknown data kind {data_kind!r}.")

    if not np.allclose(
        noisy - clean, stored_noise, rtol=0.0, atol=2.0e-11
    ):
        raise RuntimeError("Bilinear resampling did not preserve data_noisy-clean.")
    if np.min(u_truth) < TRAINING_U_BOUNDS[0] - 1.0e-8 or np.max(u_truth) > TRAINING_U_BOUNDS[1] + 1.0e-8:
        raise RuntimeError("Resampled truth left the direct-u training support.")

    return SurrogateInversionData(
        archive=archive,
        grid=grid,
        u_truth=u_truth,
        data_noiseless=clean,
        data_observed=observed,
        stored_noise=stored_noise,
        effective_noise=effective_noise,
        observation_mask=mask,
        observation_weights=weights,
        data_kind=data_kind,
    )


class SurrogateReactionOperator:
    """One physical-u model with 2-D grid forward/JVP/VJP methods.

    The checkpoint wrappers use Euclidean VJPs.  Consequently, this class
    leaves observation quadrature outside the wrapper: the caller passes
    ``W * residual`` to :meth:`vjp`, exactly as required by the weighted
    least-squares objective.
    """

    def __init__(
        self,
        checkpoint: Path,
        *,
        grid: RegularGrid2D,
        device: str,
        deeponet_query_chunk: int,
    ) -> None:
        if grid.shape != (PRODUCTION_GRID_SIZE, PRODUCTION_GRID_SIZE):
            raise ValueError("Surrogate operator requires the production 120x120 grid.")
        if deeponet_query_chunk < 1:
            raise ValueError("--deeponet-query-chunk must be at least one.")

        self.checkpoint_path = _resolve_input(checkpoint)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {self.checkpoint_path}")
        self.emulator = load_reaction_emulator(self.checkpoint_path, device=device)
        self.grid = grid
        self.deeponet_query_chunk = int(deeponet_query_chunk)
        self.coordinates = np.asarray(grid.points(), dtype=np.float32)

        if isinstance(self.emulator, ReactionFNOEmulator):
            self.model_type = "fno"
            if (
                int(self.emulator.model.in_channels) != 1
                or int(self.emulator.model.out_channels) != 1
            ):
                raise ValueError(
                    "Reaction FNO inversion requires one direct-u input and one p output."
                )
        elif isinstance(self.emulator, ReactionDeepONetEmulator):
            self.model_type = "deeponet"
            if tuple(self.emulator.x_grid.shape) != (PRODUCTION_GRID_SIZE,) or tuple(
                self.emulator.y_grid.shape
            ) != (PRODUCTION_GRID_SIZE,):
                raise ValueError("DeepONet checkpoint does not have a 120x120 branch grid.")
        else:  # Future wrapper types should not silently enter this workflow.
            raise TypeError(
                "models.load_reaction_emulator returned an unsupported Reaction wrapper "
                f"{type(self.emulator).__name__}."
            )

    @property
    def device(self) -> str:
        return str(self.emulator.device)

    @property
    def epoch(self) -> int:
        return int(self.emulator.epoch)

    def _field(self, values: Any, *, name: str) -> np.ndarray:
        field = np.asarray(values, dtype=np.float64)
        if field.shape != self.grid.shape:
            raise ValueError(
                f"{name} must have shape {self.grid.shape}; got {field.shape}."
            )
        if not np.all(np.isfinite(field)):
            raise ValueError(f"{name} contains non-finite values.")
        return field

    def _chunks(self):
        for first in range(0, self.coordinates.shape[0], self.deeponet_query_chunk):
            yield first, min(first + self.deeponet_query_chunk, self.coordinates.shape[0])

    def forward(self, physical_u: np.ndarray) -> np.ndarray:
        u = self._field(physical_u, name="physical_u")
        if self.model_type == "fno":
            output = self.emulator.forward(u.astype(np.float32, copy=False))
            return self._field(output, name="FNO output")

        values: list[np.ndarray] = []
        for first, last in self._chunks():
            chunk = self.emulator.forward(
                u.astype(np.float32, copy=False),
                coordinates=self.coordinates[first:last],
            )
            values.append(np.asarray(chunk, dtype=np.float64).reshape(-1))
        return self._field(
            np.concatenate(values, axis=0).reshape(self.grid.shape),
            name="DeepONet output",
        )

    def jvp(self, physical_u: np.ndarray, direction: np.ndarray) -> np.ndarray:
        u = self._field(physical_u, name="physical_u")
        h = self._field(direction, name="direction")
        if self.model_type == "fno":
            output = self.emulator.jvp(
                u.astype(np.float32, copy=False), h.astype(np.float32, copy=False)
            )
            return self._field(output, name="FNO JVP")

        values: list[np.ndarray] = []
        for first, last in self._chunks():
            chunk = self.emulator.jvp(
                u.astype(np.float32, copy=False),
                h.astype(np.float32, copy=False),
                coordinates=self.coordinates[first:last],
            )
            values.append(np.asarray(chunk, dtype=np.float64).reshape(-1))
        return self._field(
            np.concatenate(values, axis=0).reshape(self.grid.shape),
            name="DeepONet JVP",
        )

    def vjp(self, physical_u: np.ndarray, output_dual: np.ndarray) -> np.ndarray:
        u = self._field(physical_u, name="physical_u")
        dual = self._field(output_dual, name="output_dual")
        if self.model_type == "fno":
            output = self.emulator.vjp(
                u.astype(np.float32, copy=False), dual.astype(np.float32, copy=False)
            )
            return self._field(output, name="FNO VJP")

        gradient = np.zeros(self.grid.shape, dtype=np.float64)
        dual_flat = dual.reshape(-1)
        for first, last in self._chunks():
            # VJP is linear in the output dual.  Summing chunk VJPs gives the
            # exact full-grid VJP without materialising all DeepONet queries.
            chunk_gradient = self.emulator.vjp(
                u.astype(np.float32, copy=False),
                dual_flat[first:last].astype(np.float32, copy=False),
                coordinates=self.coordinates[first:last],
            )
            gradient += self._field(chunk_gradient, name="DeepONet chunk VJP")
        return gradient


def _canonical_alpha_mode(value: str) -> str:
    return "adjoint-step-norm" if value == "minimum-residual" else value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run data-only direct-u Reaction Landweber with either a trained "
            "FNO or a trained DeepONet on their common 120x120 grid."
        ),
    )
    source = parser.add_argument_group("required source and checkpoint")
    source.add_argument(
        "--synthetic-file",
        type=Path,
        required=True,
        help="reaction-fine-truth-data-v1 NPZ archive from the fine truth generator.",
    )
    source.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Canonical reaction-fno-v1 or reaction-deeponet-v1 checkpoint.",
    )
    source.add_argument(
        "--data-kind",
        choices=("noisy", "noiseless"),
        default="noisy",
        help="Select saved noisy data or saved clean data from the archive.",
    )

    model = parser.add_argument_group("fixed production surrogate grid")
    model.add_argument(
        "--grid-size",
        type=int,
        default=PRODUCTION_GRID_SIZE,
        help=(
            "Must remain 120: it is the FNO output grid and the fixed Darcy "
            "DeepONet branch-input grid."
        ),
    )
    model.add_argument(
        "--device",
        default="auto",
        help="PyTorch device passed to models.load_reaction_emulator.",
    )
    model.add_argument(
        "--deeponet-query-chunk",
        type=int,
        default=4096,
        help="Number of 120-grid DeepONet coordinates processed per wrapper call.",
    )

    inv = parser.add_argument_group("covariance-preconditioned Landweber")
    inv.add_argument("--iterations", type=int, default=1500)
    inv.add_argument("--initial-u", type=float, default=100.0)
    inv.add_argument("--u-min", type=float, default=TRAINING_U_BOUNDS[0])
    inv.add_argument("--u-max", type=float, default=TRAINING_U_BOUNDS[1])
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
    inv.add_argument(
        "--fft-workers",
        type=int,
        default=None,
        help="SciPy FFT workers for the non-periodic BTTB Matérn action.",
    )
    inv.add_argument(
        "--alpha-mode",
        choices=("derivative", "adjoint-step-norm", "minimum-residual"),
        default="derivative",
        help=(
            "derivative: damping*(q dot d)/||Jd||_W^2; "
            "adjoint-step-norm (minimum-residual alias): "
            "damping*||r||_W^2/(q dot d)."
        ),
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
    inv.add_argument(
        "--relative-residual-tolerance",
        type=float,
        default=0.0,
        help="Additional relative-residual stopping threshold; zero disables it.",
    )
    inv.add_argument(
        "--discrepancy-principle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Stop at tau*(resampled noise norm + model-discrepancy-scale * "
            "surrogate mismatch at the resampled loaded truth)."
        ),
    )
    inv.add_argument("--discrepancy-tau", type=float, default=1.05)
    inv.add_argument("--model-discrepancy-scale", type=float, default=1.0)
    inv.add_argument(
        "--adjoint-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check the surrogate JVP/VJP Euclidean pairing once before iterating.",
    )
    inv.add_argument("--adjoint-check-seed", type=int, default=20260820)
    inv.add_argument("--adjoint-check-tolerance", type=float, default=2.0e-3)

    output = parser.add_argument_group("output")
    output.add_argument("--output-dir", type=Path, default=None)
    output.add_argument("--overwrite", action="store_true")
    output.add_argument("--print-every", type=int, default=10)
    output.add_argument(
        "--trajectory-state-every",
        type=int,
        default=0,
        help=(
            "Save every Nth physical-u iterate to trajectory_states.h5; "
            "zero disables this diagnostic."
        ),
    )
    output.add_argument(
        "--crossing-snapshots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Retain the states immediately before and at the first crossings "
            "of delta, tau*delta, and the model-adjusted discrepancy target. "
            "This never activates early stopping."
        ),
    )
    output.add_argument(
        "--projection-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record accepted-step lower/upper projection diagnostics and "
            "cumulative spatial hit-count maps without changing the iteration."
        ),
    )
    output.add_argument(
        "--plots",
        action="store_true",
        help="Save optional PNGs; matplotlib is imported only for this flag.",
    )
    output.add_argument("--dpi", type=int, default=180)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if int(args.grid_size) != PRODUCTION_GRID_SIZE:
        raise ValueError(
            f"--grid-size must be {PRODUCTION_GRID_SIZE}; the production FNO "
            "and DeepONet cannot be compared safely on another grid."
        )
    if int(args.deeponet_query_chunk) < 1:
        raise ValueError("--deeponet-query-chunk must be at least one.")
    if int(args.iterations) < 0:
        raise ValueError("--iterations cannot be negative.")
    if int(args.trajectory_state_every) < 0:
        raise ValueError("--trajectory-state-every cannot be negative.")
    if not (
        TRAINING_U_BOUNDS[0]
        <= float(args.u_min)
        < float(args.u_max)
        <= TRAINING_U_BOUNDS[1]
    ):
        raise ValueError(
            f"Require {TRAINING_U_BOUNDS[0]:g} <= --u-min < --u-max <= "
            f"{TRAINING_U_BOUNDS[1]:g} so iterations remain inside the "
            "supported physical-u interval."
        )
    if not float(args.u_min) <= float(args.initial_u) <= float(args.u_max):
        raise ValueError("--initial-u must lie within [--u-min, --u-max].")
    if args.matern_nu <= 0.0 or args.matern_length_scale <= 0.0:
        raise ValueError("Matérn nu and length scale must be positive.")
    if args.matern_variance <= 0.0:
        raise ValueError("--matern-variance must be positive.")
    if args.fft_workers == 0:
        raise ValueError("--fft-workers cannot be zero.")
    if args.damping <= 0.0 or args.max_physical_change <= 0.0:
        raise ValueError("Damping and --max-physical-change must be positive.")
    if not 0.0 < args.backtracking_factor < 1.0:
        raise ValueError("--backtracking-factor must lie in (0, 1).")
    if args.max_backtracks < 0:
        raise ValueError("--max-backtracks cannot be negative.")
    if args.armijo < 0.0:
        raise ValueError("--armijo cannot be negative.")
    if args.relative_residual_tolerance < 0.0:
        raise ValueError("--relative-residual-tolerance cannot be negative.")
    if args.discrepancy_tau <= 0.0:
        raise ValueError("--discrepancy-tau must be positive.")
    if args.model_discrepancy_scale < 0.0:
        raise ValueError("--model-discrepancy-scale cannot be negative.")
    if args.adjoint_check_tolerance <= 0.0:
        raise ValueError("--adjoint-check-tolerance must be positive.")
    if args.print_every < 1 or args.dpi < 40:
        raise ValueError("--print-every must be >= 1 and --dpi must be >= 40.")


def _output_directory(
    args: argparse.Namespace,
    *,
    data: SurrogateInversionData,
    checkpoint_path: Path,
) -> Path:
    if args.output_dir is None:
        path = PROJECT_ROOT / "results" / (
            f"reaction_landweber_surrogate_{checkpoint_path.stem}_"
            f"{data.archive.path.stem}_{data.data_kind}_"
            f"{_canonical_alpha_mode(args.alpha_mode)}_grid{data.grid.nx}"
        )
    else:
        expanded = args.output_dir.expanduser()
        path = expanded if expanded.is_absolute() else PROJECT_ROOT / "results" / expanded
    path = path.resolve()
    if path.exists() and any(path.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to write into non-empty output directory {path}. "
            "Use --overwrite deliberately or choose a new --output-dir."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_surrogate_adjoint(
    operator: SurrogateReactionOperator,
    *,
    physical_u: np.ndarray,
    seed: int,
) -> dict[str, float]:
    """Robust Euclidean JVP/VJP pairing check in physical units.

    The previous diagnostic paired ``J h`` with an independent random output
    dual.  For float32 neural operators that scalar can be small because of
    cancellation, making an otherwise harmless absolute roundoff discrepancy
    look large in relative terms.  Here the output dual is the normalized
    ``J h`` itself, so the left pairing is positive and well-scaled:

        <J h, v> = <h, J^T v>,   v = J h / ||J h||_2.

    This checks the same physical-u wrapper JVP/VJP without a cancellation-
    dominated denominator.
    """

    rng = np.random.default_rng(int(seed))
    direction = rng.standard_normal(operator.grid.shape)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-30)

    j_direction = np.asarray(
        operator.jvp(physical_u, direction), dtype=np.float64
    )
    j_norm = float(np.linalg.norm(j_direction))
    if not np.isfinite(j_norm) or j_norm <= 1.0e-30:
        raise RuntimeError(
            "Surrogate adjoint check produced a zero/non-finite JVP norm."
        )

    output_dual = j_direction / j_norm
    left = float(np.sum(j_direction * output_dual, dtype=np.float64))
    coordinate_covector = np.asarray(
        operator.vjp(physical_u, output_dual), dtype=np.float64
    )
    right = float(np.sum(direction * coordinate_covector, dtype=np.float64))
    absolute_error = abs(left - right)
    relative_error = absolute_error / max(abs(left), abs(right), 1.0e-30)
    return {
        "jvp_norm": j_norm,
        "jvp_dual_pairing": left,
        "direction_vjp_pairing": right,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
    }


def _lazy_pyplot():
    """Load matplotlib only after the user expressly requests plots."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def save_plots(
    output_dir: Path,
    *,
    data: SurrogateInversionData,
    initial_u: np.ndarray,
    recovered_u: np.ndarray,
    prediction: np.ndarray,
    history: dict[str, list[float]],
    dpi: int,
) -> None:
    plt = _lazy_pyplot()
    error = recovered_u - data.u_truth
    error_maximum = max(float(np.max(np.abs(error))), 1.0e-14)
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.1), constrained_layout=True)
    panels = (
        (data.u_truth, "Loaded truth u (resampled)", "viridis", None, None),
        (initial_u, "Initial u", "viridis", None, None),
        (recovered_u, "Recovered u", "viridis", None, None),
        (error, "Recovered - loaded truth", "coolwarm", -error_maximum, error_maximum),
        (data.data_observed, f"Observed p ({data.data_kind})", "viridis", None, None),
        (prediction - data.data_observed, "Final p residual", "coolwarm", None, None),
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
    figure.savefig(output_dir / "recovery_final.png", dpi=int(dpi))
    plt.close(figure)

    iterations = np.asarray(history["iteration"], dtype=np.int64)
    residuals = np.asarray(history["residual_norm"], dtype=np.float64)
    objectives = np.asarray(history["objective"], dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), constrained_layout=True)
    axes[0].semilogy(iterations, np.maximum(residuals, 1.0e-300), label="residual")
    target = np.asarray(history["discrepancy_threshold"], dtype=np.float64)
    finite_target = np.isfinite(target) & (target > 0.0)
    if np.any(finite_target):
        axes[0].semilogy(iterations[finite_target], target[finite_target], "--", label="target")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("weighted L2 norm")
    axes[0].legend()
    axes[1].semilogy(iterations, np.maximum(objectives, 1.0e-300))
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("objective")
    figure.savefig(output_dir / "history.png", dpi=int(dpi))
    plt.close(figure)


def _history_arrays(history: dict[str, list[float]]) -> dict[str, np.ndarray]:
    return {
        f"history_{key}": np.asarray(values, dtype=np.float64)
        for key, values in history.items()
    }


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


def save_results(
    output_dir: Path,
    *,
    data: SurrogateInversionData,
    operator: SurrogateReactionOperator,
    initial_u: np.ndarray,
    recovered_u: np.ndarray,
    recovered_prediction: np.ndarray,
    minimum_residual_u: np.ndarray,
    minimum_residual_prediction: np.ndarray,
    minimum_residual_iteration: int,
    minimum_u_error_u: np.ndarray,
    minimum_u_error_prediction: np.ndarray,
    minimum_u_error_iteration: int,
    minimum_u_error: float,
    model_prediction_at_truth: np.ndarray,
    history: dict[str, list[float]],
    latest_covector: np.ndarray | None,
    latest_direction: np.ndarray | None,
    latest_jvp: np.ndarray | None,
    adjoint_check: dict[str, float] | None,
    landweber_elapsed_seconds: float,
    total_driver_elapsed_seconds: float,
    summary: dict[str, Any],
    settings: dict[str, Any],
    projection_diagnostics: ProjectionDiagnostics | None,
    projection_alpha_history: dict[str, list[float | int | bool]] | None,
    minimum_u_error_projection_maps: dict[str, np.ndarray] | None,
    crossing_snapshots: DiscrepancyCrossingSnapshots | None,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "format": np.asarray(RESULT_FILE_FORMAT),
        "truth_u_resampled_model_grid": np.asarray(data.u_truth, dtype=np.float64),
        "initial_u": np.asarray(initial_u, dtype=np.float64),
        "recovered_u_final": np.asarray(recovered_u, dtype=np.float64),
        "recovered_u_minimum_residual": np.asarray(minimum_residual_u, dtype=np.float64),
        "minimum_residual_prediction": np.asarray(
            minimum_residual_prediction, dtype=np.float64
        ),
        "minimum_residual_iteration": np.asarray(minimum_residual_iteration, dtype=np.int64),
        "recovered_u_minimum_u_error": np.asarray(minimum_u_error_u, dtype=np.float64),
        "minimum_u_error_prediction": np.asarray(minimum_u_error_prediction, dtype=np.float64),
        "minimum_u_error_iteration": np.asarray(minimum_u_error_iteration, dtype=np.int64),
        "minimum_u_error": np.asarray(minimum_u_error, dtype=np.float64),
        "data_noiseless_resampled": np.asarray(data.data_noiseless, dtype=np.float64),
        "data_observed_resampled": np.asarray(data.data_observed, dtype=np.float64),
        "noise_stored_resampled": np.asarray(data.stored_noise, dtype=np.float64),
        "noise_effective": np.asarray(data.effective_noise, dtype=np.float64),
        "observation_mask": np.asarray(data.observation_mask, dtype=bool),
        "observation_weights": np.asarray(data.observation_weights, dtype=np.float64),
        "model_x_grid": np.asarray(data.grid.x, dtype=np.float64),
        "model_y_grid": np.asarray(data.grid.y, dtype=np.float64),
        "fine_fem_x_grid": np.asarray(data.archive.fine_grid.x, dtype=np.float64),
        "fine_fem_y_grid": np.asarray(data.archive.fine_grid.y, dtype=np.float64),
        "surrogate_prediction_at_resampled_truth": np.asarray(
            model_prediction_at_truth, dtype=np.float64
        ),
        "surrogate_model_discrepancy_at_truth": np.asarray(
            model_prediction_at_truth - data.data_noiseless, dtype=np.float64
        ),
        "final_prediction": np.asarray(recovered_prediction, dtype=np.float64),
        "final_residual": np.asarray(
            recovered_prediction - data.data_observed, dtype=np.float64
        ),
        "final_residual_masked": np.where(
            data.observation_mask,
            recovered_prediction - data.data_observed,
            0.0,
        ),
        "latest_coordinate_covector": (
            np.asarray([], dtype=np.float64)
            if latest_covector is None
            else np.asarray(latest_covector, dtype=np.float64)
        ),
        "latest_matern_direction": (
            np.asarray([], dtype=np.float64)
            if latest_direction is None
            else np.asarray(latest_direction, dtype=np.float64)
        ),
        "latest_tangent_jd": (
            np.asarray([], dtype=np.float64)
            if latest_jvp is None
            else np.asarray(latest_jvp, dtype=np.float64)
        ),
        "landweber_elapsed_seconds": np.asarray(landweber_elapsed_seconds, dtype=np.float64),
        "landweber_elapsed_minutes": np.asarray(
            landweber_elapsed_seconds / 60.0, dtype=np.float64
        ),
        "total_driver_elapsed_seconds": np.asarray(
            total_driver_elapsed_seconds, dtype=np.float64
        ),
        "checkpoint_epoch": np.asarray(operator.epoch, dtype=np.int64),
    }
    arrays.update(_history_arrays(history))
    if crossing_snapshots is not None:
        arrays.update(crossing_snapshots.archive_arrays())
    if projection_diagnostics is not None:
        assert projection_alpha_history is not None
        assert minimum_u_error_projection_maps is not None
        arrays.update(
            {
                f"projection_diagnostics_{key}": np.asarray(values)
                for key, values in projection_diagnostics.history_arrays().items()
            }
        )
        arrays.update(
            {
                f"projection_diagnostics_{key}": np.asarray(
                    values,
                    dtype=(
                        np.int64
                        if key == "update_iteration"
                        else np.bool_
                        if key in {
                            "physical_change_cap_applied",
                            "backtracking_applied",
                        }
                        else np.float64
                    ),
                )
                for key, values in projection_alpha_history.items()
            }
        )
        arrays.update(
            {
                f"projection_diagnostics_{key}": values
                for key, values in projection_diagnostics.hit_count_arrays().items()
            }
        )
        arrays.update(
            {
                (
                    "projection_diagnostics_through_minimum_u_error_"
                    f"{key}"
                ): np.asarray(values)
                for key, values in minimum_u_error_projection_maps.items()
            }
        )
    if adjoint_check is not None:
        for key, value in adjoint_check.items():
            arrays[f"adjoint_check_{key}"] = np.asarray(value, dtype=np.float64)
    np.savez_compressed(output_dir / "result.npz", **arrays)
    write_json(output_dir / "results.json", summary)
    write_json(output_dir / "settings.json", settings)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    started = perf_counter()
    archive = load_fine_synthetic_archive(
        args.synthetic_file,
        project_root=PROJECT_ROOT,
        supported_u_bounds=TRAINING_U_BOUNDS,
    )
    data = resample_synthetic_data(
        archive,
        grid_size=int(args.grid_size),
        data_kind=str(args.data_kind),
    )
    operator = SurrogateReactionOperator(
        args.checkpoint,
        grid=data.grid,
        device=str(args.device),
        deeponet_query_chunk=int(args.deeponet_query_chunk),
    )
    output_dir = _output_directory(
        args, data=data, checkpoint_path=operator.checkpoint_path
    )
    alpha_mode = _canonical_alpha_mode(str(args.alpha_mode))
    covariance: StationaryMaternCovariance2D | IdentityCovariance2D | None = None
    trajectory_writer: TrajectoryStateWriter | None = None

    try:
        print(f"SCRIPT PATCH VERSION: {PATCH_VERSION}", flush=True)
        print("Reaction surrogate Landweber", flush=True)
        print(f"  checkpoint type          : {operator.model_type}", flush=True)
        print(f"  checkpoint epoch         : {operator.epoch}", flush=True)
        print(f"  PyTorch device           : {operator.device}", flush=True)
        print(f"  shared model grid        : {data.grid.nx} x {data.grid.ny}", flush=True)
        print("  fine-to-model mapping    : bilinear resampling", flush=True)
        print(f"  source truth kind        : {archive.truth_kind}", flush=True)
        print(f"  selected data kind       : {data.data_kind}", flush=True)
        print(f"  alpha mode               : {alpha_mode}", flush=True)
        print(f"  preconditioner           : {args.preconditioner}", flush=True)
        if args.projection_diagnostics:
            print("  projection diagnostics   : ON (accepted updates only)", flush=True)
        print(
            "  backtracking             : "
            + (
                f"ON (factor={args.backtracking_factor:g}, max={args.max_backtracks}; "
                "final trial accepted if Armijo never passes)"
                if args.backtracking
                else "OFF (default; capped/projected proposal accepted directly)"
            ),
            flush=True,
        )

        initial_u = np.full(data.grid.shape, float(args.initial_u), dtype=np.float64)
        # The loaded truth is only a diagnostic reference.  This call measures
        # surrogate-vs-resampled-exact model error; it never calls FEniCS and
        # is not used to construct synthetic data.
        model_prediction_at_truth = operator.forward(data.u_truth)
        model_mismatch = model_prediction_at_truth - data.data_noiseless
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
            data.effective_noise, data.observation_weights, mask=data.observation_mask
        )
        discrepancy_target = float(args.discrepancy_tau) * (
            noise_norm + float(args.model_discrepancy_scale) * model_discrepancy_norm
        )
        morozov_noise_target = float(args.discrepancy_tau) * noise_norm
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
        print(f"  resampled data norm      : {data_norm:.8e}", flush=True)
        print(f"  resampled noise norm     : {noise_norm:.8e}", flush=True)
        print(f"  surrogate mismatch norm  : {model_discrepancy_norm:.8e}", flush=True)
        print(f"  discrepancy target       : {discrepancy_target:.8e}", flush=True)
        if crossing_snapshots is not None:
            print(
                "  crossing snapshots       : ON "
                "(delta, tau*delta, model-adjusted; no early stop)",
                flush=True,
            )

        adjoint_check: dict[str, float] | None = None
        if args.adjoint_check:
            adjoint_check = check_surrogate_adjoint(
                operator, physical_u=initial_u, seed=int(args.adjoint_check_seed)
            )
            print(
                "  JVP/VJP check            : "
                f"lhs={adjoint_check['jvp_dual_pairing']:.8e}, "
                f"rhs={adjoint_check['direction_vjp_pairing']:.8e}, "
                f"abs={adjoint_check['absolute_error']:.3e}, "
                f"rel={adjoint_check['relative_error']:.3e}",
                flush=True,
            )
            if adjoint_check["relative_error"] > float(args.adjoint_check_tolerance):
                raise AssertionError(
                    "Surrogate JVP/VJP check failed: "
                    f"{adjoint_check['relative_error']:.6e} > "
                    f"{args.adjoint_check_tolerance:.6e}."
                )

        if args.preconditioner == "matern":
            covariance = StationaryMaternCovariance2D(
                data.grid,
                nu=float(args.matern_nu),
                length_scale=float(args.matern_length_scale),
                variance=float(args.matern_variance),
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
            covariance = IdentityCovariance2D(data.grid)
            resolved_preconditioner = {
                "kind": "identity",
                "coordinate_space": "Euclidean nodal covector-to-vector map",
                "nu": None,
                "length_scale": None,
                "variance": None,
            }
        current_u = initial_u.copy()
        prediction = operator.forward(current_u)
        if int(args.trajectory_state_every) > 0:
            trajectory_writer = TrajectoryStateWriter(
                output_dir,
                state_shape=data.grid.shape,
                save_every=int(args.trajectory_state_every),
            )
            print(
                "  trajectory states       : every "
                f"{int(args.trajectory_state_every)} iterations",
                flush=True,
            )
        # This interval deliberately excludes archive loading, model loading,
        # truth diagnostic, initial forward, and optional JVP/VJP check.
        landweber_started = perf_counter()
        history: dict[str, list[float]] = {
            "iteration": [],
            "objective": [],
            "residual_norm": [],
            "relative_residual": [],
            "relative_u_error": [],
            "proposed_alpha": [],
            "accepted_alpha": [],
            "gradient_direction_inner": [],
            "j_direction_norm": [],
            "j_direction_pairing_relative_error": [],
            "backtracks": [],
            "armijo_satisfied": [],
            "forced_final_backtrack_accept": [],
            "max_proposed_change": [],
            "active_bound_fraction": [],
            "discrepancy_threshold": [],
            "discrepancy_ratio": [],
            "elapsed_seconds": [],
            "landweber_elapsed_seconds": [],
        }
        projection_diagnostics = (
            ProjectionDiagnostics(
                data.grid.shape,
                float(args.u_min),
                float(args.u_max),
            )
            if args.projection_diagnostics
            else None
        )
        projection_alpha_history: dict[
            str, list[float | int | bool]
        ] | None = (
            {
                "update_iteration": [],
                "formula_alpha_before_cap": [],
                "capped_alpha": [],
                "accepted_alpha": [],
                "physical_change_cap_alpha": [],
                "physical_change_cap_applied": [],
                "physical_change_cap_ratio": [],
                "accepted_to_capped_alpha_ratio": [],
                "backtracking_applied": [],
            }
            if args.projection_diagnostics
            else None
        )
        latest_covector: np.ndarray | None = None
        latest_direction: np.ndarray | None = None
        latest_jvp: np.ndarray | None = None
        minimum_residual_norm = float("inf")
        minimum_residual_u = current_u.copy()
        minimum_residual_prediction = prediction.copy()
        minimum_residual_iteration = 0
        minimum_u_error = float("inf")
        minimum_u_error_u = current_u.copy()
        minimum_u_error_prediction = prediction.copy()
        minimum_u_error_iteration = 0
        minimum_u_error_projection_maps = (
            projection_diagnostics.hit_count_arrays()
            if projection_diagnostics is not None
            else None
        )
        stop_reason = "maximum iterations reached"

        truth_norm = weighted_norm(data.u_truth, data.observation_weights)
        for iteration in range(int(args.iterations) + 1):
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
                current_u - data.u_truth, data.observation_weights
            ) / max(truth_norm, 1.0e-30)
            if residual_norm < minimum_residual_norm:
                minimum_residual_norm = float(residual_norm)
                minimum_residual_u = current_u.copy()
                minimum_residual_prediction = prediction.copy()
                minimum_residual_iteration = int(iteration)
            if relative_u_error < minimum_u_error:
                minimum_u_error = float(relative_u_error)
                minimum_u_error_u = current_u.copy()
                minimum_u_error_prediction = prediction.copy()
                minimum_u_error_iteration = int(iteration)
                if projection_diagnostics is not None:
                    minimum_u_error_projection_maps = (
                        projection_diagnostics.hit_count_arrays()
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
            for name in (
                "proposed_alpha",
                "accepted_alpha",
                "gradient_direction_inner",
                "j_direction_norm",
                "j_direction_pairing_relative_error",
                "backtracks",
                "armijo_satisfied",
                "forced_final_backtrack_accept",
                "max_proposed_change",
                "active_bound_fraction",
            ):
                history[name].append(float("nan"))
            history["discrepancy_threshold"].append(
                float(discrepancy_target)
                if args.discrepancy_principle
                else float("nan")
            )
            history["discrepancy_ratio"].append(
                float(residual_norm / discrepancy_target)
                if args.discrepancy_principle and discrepancy_target > 0.0
                else float("nan")
            )
            history["elapsed_seconds"].append(float(perf_counter() - started))
            history["landweber_elapsed_seconds"].append(
                float(perf_counter() - landweber_started)
            )

            if iteration % int(args.print_every) == 0:
                print(
                    f"iter={iteration:4d}  rel_res={relative_residual:.6e}  "
                    f"res={residual_norm:.6e}  phi={objective:.6e}",
                    flush=True,
                )

            if (
                args.discrepancy_principle
                and discrepancy_target > 0.0
                and residual_norm <= discrepancy_target
            ):
                stop_reason = "discrepancy target reached"
                break
            if (
                args.relative_residual_tolerance > 0.0
                and relative_residual <= args.relative_residual_tolerance
            ):
                stop_reason = "relative residual tolerance reached"
                break
            if iteration >= int(args.iterations):
                stop_reason = "maximum iterations reached"
                break

            # The neural wrappers return an Euclidean VJP.  Passing W*r here
            # gives q=J^T W r, exactly matching the weighted objective.
            weighted_dual = weighted_residual_covector(
                residual, data.observation_weights, mask=data.observation_mask
            )
            coordinate_covector = operator.vjp(current_u, weighted_dual)
            latest_covector = coordinate_covector.copy()
            direction = covariance.apply_coordinate_covector(coordinate_covector)
            latest_direction = direction.copy()
            q_dot_d = float(np.sum(coordinate_covector * direction, dtype=np.float64))
            if not np.isfinite(q_dot_d) or q_dot_d <= 0.0:
                stop_reason = "non-positive or non-finite covariance direction pairing"
                break
            history["gradient_direction_inner"][-1] = q_dot_d

            j_direction: np.ndarray | None = None
            if alpha_mode == "derivative":
                j_direction = operator.jvp(current_u, direction)
                latest_jvp = j_direction.copy()
                j_direction_norm = weighted_norm(
                    j_direction, data.observation_weights, mask=data.observation_mask
                )
                denominator = j_direction_norm * j_direction_norm
                if not np.isfinite(denominator) or denominator <= 0.0:
                    stop_reason = "non-positive or non-finite tangent norm"
                    break
                formula_alpha = float(args.damping) * q_dot_d / denominator
                pairing = weighted_inner(
                    residual,
                    j_direction,
                    data.observation_weights,
                    mask=data.observation_mask,
                )
                history["j_direction_norm"][-1] = j_direction_norm
                history["j_direction_pairing_relative_error"][-1] = abs(
                    pairing - q_dot_d
                ) / max(abs(pairing), abs(q_dot_d), 1.0e-30)
                if iteration == 0:
                    weighted_pairing_error = history[
                        "j_direction_pairing_relative_error"
                    ][-1]
                    print(
                        "  weighted Landweber pairing: "
                        f"<r,Jd>_W={pairing:.8e}, q.d={q_dot_d:.8e}, "
                        f"rel={weighted_pairing_error:.3e}",
                        flush=True,
                    )
                    if (
                        args.adjoint_check
                        and weighted_pairing_error > args.adjoint_check_tolerance
                    ):
                        raise AssertionError(
                            "Weighted Landweber JVP/VJP pairing failed: "
                            f"{weighted_pairing_error:.6e} > "
                            f"{args.adjoint_check_tolerance:.6e}."
                        )
            elif alpha_mode == "adjoint-step-norm":
                # Latest TransientDarcy-style preconditioned minimal-error
                # proposal.  It intentionally avoids an extra JVP call.
                formula_alpha = float(args.damping) * residual_sq / q_dot_d
            else:  # pragma: no cover - protected by argparse/canonicalisation.
                raise RuntimeError(f"Unexpected alpha mode {alpha_mode!r}.")

            if not np.isfinite(formula_alpha) or formula_alpha <= 0.0:
                stop_reason = "non-positive or non-finite proposed step"
                break
            maximum_direction = float(np.max(np.abs(direction)))
            proposed_alpha = formula_alpha
            if maximum_direction > 0.0:
                physical_change_cap_alpha = (
                    float(args.max_physical_change) / maximum_direction
                )
                proposed_alpha = min(
                    formula_alpha,
                    physical_change_cap_alpha,
                )
            else:
                physical_change_cap_alpha = formula_alpha
            history["proposed_alpha"][-1] = proposed_alpha
            history["max_proposed_change"][-1] = proposed_alpha * maximum_direction

            accepted = False
            accepted_u: np.ndarray | None = None
            accepted_prediction: np.ndarray | None = None

            if not args.backtracking:
                # Default behaviour: accept the capped/projected Landweber
                # proposal directly. No Armijo test and no line search.
                trial_alpha = proposed_alpha
                candidate_u = np.clip(
                    current_u - trial_alpha * direction,
                    float(args.u_min),
                    float(args.u_max),
                )
                delta = candidate_u - current_u
                if not np.any(delta):
                    stop_reason = "projected Landweber proposal produced no change"
                    break
                candidate_prediction = operator.forward(candidate_u)
                accepted = True
                accepted_u = candidate_u
                accepted_prediction = candidate_prediction
                history["accepted_alpha"][-1] = float(trial_alpha)
                history["backtracks"][-1] = 0.0
                history["armijo_satisfied"][-1] = float("nan")
                history["forced_final_backtrack_accept"][-1] = 0.0
                history["active_bound_fraction"][-1] = float(
                    np.mean(
                        (candidate_u <= float(args.u_min) + 1.0e-12)
                        | (candidate_u >= float(args.u_max) - 1.0e-12)
                    )
                )
            else:
                # Optional Armijo backtracking. If no trial passes Armijo,
                # accept the final/smallest usable proposal instead of stopping.
                last_trial = None
                for backtracks in range(int(args.max_backtracks) + 1):
                    trial_alpha = (
                        proposed_alpha
                        * float(args.backtracking_factor) ** backtracks
                    )
                    candidate_u = np.clip(
                        current_u - trial_alpha * direction,
                        float(args.u_min),
                        float(args.u_max),
                    )
                    delta = candidate_u - current_u
                    if not np.any(delta):
                        continue
                    directional_derivative = float(
                        np.sum(coordinate_covector * delta, dtype=np.float64)
                    )
                    if not np.isfinite(directional_derivative):
                        continue

                    candidate_prediction = operator.forward(candidate_u)
                    candidate_residual = (
                        candidate_prediction - data.data_observed
                    )
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
                        candidate_prediction,
                        float(directional_derivative),
                        float(candidate_objective),
                    )

                    armijo_satisfied = (
                        directional_derivative < 0.0
                        and candidate_objective
                        <= objective
                        + float(args.armijo) * directional_derivative
                    )
                    if armijo_satisfied:
                        accepted = True
                        accepted_u = candidate_u
                        accepted_prediction = candidate_prediction
                        history["accepted_alpha"][-1] = float(trial_alpha)
                        history["backtracks"][-1] = float(backtracks)
                        history["armijo_satisfied"][-1] = 1.0
                        history["forced_final_backtrack_accept"][-1] = 0.0
                        history["active_bound_fraction"][-1] = float(
                            np.mean(
                                (candidate_u <= float(args.u_min) + 1.0e-12)
                                | (candidate_u >= float(args.u_max) - 1.0e-12)
                            )
                        )
                        break

                if not accepted:
                    if last_trial is None:
                        stop_reason = (
                            "backtracking produced no usable projected proposal"
                        )
                        break

                    (
                        trial_alpha,
                        backtracks,
                        candidate_u,
                        candidate_prediction,
                        directional_derivative,
                        candidate_objective,
                    ) = last_trial
                    accepted = True
                    accepted_u = candidate_u
                    accepted_prediction = candidate_prediction
                    history["accepted_alpha"][-1] = float(trial_alpha)
                    history["backtracks"][-1] = float(backtracks)
                    history["armijo_satisfied"][-1] = 0.0
                    history["forced_final_backtrack_accept"][-1] = 1.0
                    history["active_bound_fraction"][-1] = float(
                        np.mean(
                            (candidate_u <= float(args.u_min) + 1.0e-12)
                            | (candidate_u >= float(args.u_max) - 1.0e-12)
                        )
                    )
                    print(
                        "  Armijo not satisfied after "
                        f"{backtracks} backtracks; accepting final proposal "
                        f"alpha={trial_alpha:.6e}",
                        flush=True,
                    )

            assert accepted_u is not None and accepted_prediction is not None
            if projection_diagnostics is not None:
                assert projection_alpha_history is not None
                accepted_alpha = float(history["accepted_alpha"][-1])
                raw_candidate = current_u - accepted_alpha * direction
                projection_diagnostics.record(current_u, raw_candidate, accepted_u)
                projection_alpha_history["update_iteration"].append(
                    int(iteration)
                )
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
                    bool(proposed_alpha < formula_alpha)
                )
                projection_alpha_history["physical_change_cap_ratio"].append(
                    float(proposed_alpha / formula_alpha)
                )
                projection_alpha_history[
                    "accepted_to_capped_alpha_ratio"
                ].append(float(accepted_alpha / proposed_alpha))
                projection_alpha_history["backtracking_applied"].append(
                    bool(history["backtracks"][-1] > 0.0)
                )
            current_u = accepted_u
            prediction = accepted_prediction

        final_residual = prediction - data.data_observed
        final_residual_norm = weighted_norm(
            final_residual, data.observation_weights, mask=data.observation_mask
        )
        final_relative_residual = final_residual_norm / max(data_norm, 1.0e-30)
        final_relative_u_error = weighted_norm(
            current_u - data.u_truth, data.observation_weights
        ) / max(truth_norm, 1.0e-30)
        landweber_elapsed_seconds = float(perf_counter() - landweber_started)
        total_driver_elapsed_seconds = float(perf_counter() - started)

        summary: dict[str, Any] = {
            "format": RESULT_FILE_FORMAT,
            "patch_version": PATCH_VERSION,
            "stop_reason": stop_reason,
            "model_type": operator.model_type,
            "checkpoint": operator.checkpoint_path,
            "checkpoint_epoch": operator.epoch,
            "device": operator.device,
            "alpha_mode": alpha_mode,
            "preconditioner": resolved_preconditioner,
            "iterations_evaluated": int(history["iteration"][-1]),
            "final_residual_norm": float(final_residual_norm),
            "final_relative_residual": float(final_relative_residual),
            "final_relative_u_error_diagnostic": float(final_relative_u_error),
            "minimum_residual_norm": float(minimum_residual_norm),
            "minimum_residual_iteration": int(minimum_residual_iteration),
            "minimum_u_error": float(minimum_u_error),
            "minimum_u_error_iteration": int(minimum_u_error_iteration),
            "noise_norm_resampled": float(noise_norm),
            "surrogate_model_discrepancy_norm": float(model_discrepancy_norm),
            "discrepancy_target": float(discrepancy_target),
            "adjoint_check": adjoint_check,
            "landweber_elapsed_seconds": landweber_elapsed_seconds,
            "landweber_elapsed_minutes": landweber_elapsed_seconds / 60.0,
            "total_driver_elapsed_seconds": total_driver_elapsed_seconds,
        }
        if crossing_snapshots is not None:
            crossing_summary = crossing_snapshots.summary()
            crossing_summary["threshold_definitions"] = {
                "noise_norm": "delta = ||effective observation noise||_W",
                "morozov_noise": "tau * delta",
                "model_adjusted": (
                    "tau * (delta + model_discrepancy_scale * "
                    "||G_theta(u_truth)-G(u_truth)||_W)"
                ),
            }
            summary["discrepancy_crossing_snapshots"] = crossing_summary
        command_arguments = vars(args).copy()
        if projection_diagnostics is not None:
            assert projection_alpha_history is not None
            assert minimum_u_error_projection_maps is not None
            projection_summary: dict[str, Any] = projection_diagnostics.summary()
            projection_summary["update_iteration_definition"] = (
                "zero-based source-state/update index; 0 denotes u_0 -> u_1"
            )
            projection_summary["input_sha256"] = {
                "synthetic_file": _sha256_file(archive.path),
                "checkpoint": _sha256_file(operator.checkpoint_path),
            }
            projection_summary["step_size_controls"] = _projection_alpha_summary(
                projection_alpha_history
            )
            selected_update_mask = (
                np.asarray(
                    projection_alpha_history["update_iteration"],
                    dtype=np.int64,
                )
                < minimum_u_error_iteration
            )
            selected_projection_summary = summarize_projection_diagnostics(
                {
                    name: values[selected_update_mask]
                    for name, values in projection_diagnostics.history_arrays().items()
                },
                minimum_u_error_projection_maps,
            )
            selected_projection_summary["state_iteration"] = int(
                minimum_u_error_iteration
            )
            projection_summary["through_minimum_u_error_state"] = (
                selected_projection_summary
            )
            summary["projection_diagnostics"] = projection_summary
        else:
            # Preserve the historical settings schema for default runs.  The
            # optional flag is recorded only when diagnostics are requested.
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
        settings: dict[str, Any] = {
            "format": RESULT_FILE_FORMAT,
            "patch_version": PATCH_VERSION,
            "script": Path(__file__).resolve(),
            "command_arguments": command_arguments,
            "canonical_alpha_mode": alpha_mode,
            "preconditioner": resolved_preconditioner,
            "mathematics": {
                "parameterisation": "direct physical u",
                "model_grid": [PRODUCTION_GRID_SIZE, PRODUCTION_GRID_SIZE],
                "fine_to_model_interpolation": "bilinear regular-grid resampling",
                "objective": "0.5 * ||F_surrogate(u)-data||_W^2 on interior 120-grid vertices",
                "covector": "q = J_surrogate(u)^T W (F_surrogate(u)-data)",
                "direction": (
                    "d = K_Matern q"
                    if args.preconditioner == "matern"
                    else "d = q (fixed-grid Euclidean coordinate identity)"
                ),
                "derivative_alpha": "damping*(q dot d)/||Jd||_W^2",
                "adjoint_step_norm_alpha": "damping*||F_surrogate(u)-data||_W^2/(q dot d)",
                "acceptance": ("direct capped/projected proposal" if not args.backtracking else "projected Armijo; final usable proposal accepted if no Armijo trial passes"),
            },
            "synthetic_source": {
                "path": archive.path,
                "format": SYNTHETIC_FILE_FORMAT,
                "truth_kind": archive.truth_kind,
                "truth_seed": archive.truth_seed,
                "noise_seed": archive.noise_seed,
                "fine_fem_cells": [archive.fine_fem_nx, archive.fine_fem_ny],
                "data_kind": data.data_kind,
                "metadata": archive.metadata,
            },
            "emulator": {
                "checkpoint": operator.checkpoint_path,
                "checkpoint_format": operator.emulator.checkpoint.get(
                    "checkpoint_format", "unknown"
                ),
                "model_type": operator.model_type,
                "epoch": operator.epoch,
                "device": operator.device,
                "deeponet_query_chunk": int(args.deeponet_query_chunk),
                "model_config": dict(operator.emulator.model_config),
                "normalisation_stats": dict(
                    operator.emulator.checkpoint.get("stats", {})
                ),
            },
            "effective_data_diagnostics": {
                "noiseless_data_norm_resampled": clean_data_norm,
                "observed_data_norm_resampled": data_norm,
                "noise_norm_resampled": noise_norm,
                "surrogate_model_discrepancy_norm": model_discrepancy_norm,
                "discrepancy_target": discrepancy_target,
                "morozov_noise_target": morozov_noise_target,
            },
            "timing": {
                "definition": (
                    "landweber_elapsed_seconds starts immediately before iteration 0 "
                    "and excludes archive/model loading, the resampled-truth surrogate "
                    "diagnostic, initial forward evaluation, and optional JVP/VJP check."
                ),
                "landweber_elapsed_seconds": landweber_elapsed_seconds,
                "total_driver_elapsed_seconds": total_driver_elapsed_seconds,
            },
        }
        save_results(
            output_dir,
            data=data,
            operator=operator,
            initial_u=initial_u,
            recovered_u=current_u,
            recovered_prediction=prediction,
            minimum_residual_u=minimum_residual_u,
            minimum_residual_prediction=minimum_residual_prediction,
            minimum_residual_iteration=minimum_residual_iteration,
            minimum_u_error_u=minimum_u_error_u,
            minimum_u_error_prediction=minimum_u_error_prediction,
            minimum_u_error_iteration=minimum_u_error_iteration,
            minimum_u_error=minimum_u_error,
            model_prediction_at_truth=model_prediction_at_truth,
            history=history,
            latest_covector=latest_covector,
            latest_direction=latest_direction,
            latest_jvp=latest_jvp,
            adjoint_check=adjoint_check,
            landweber_elapsed_seconds=landweber_elapsed_seconds,
            total_driver_elapsed_seconds=total_driver_elapsed_seconds,
            summary=summary,
            settings=settings,
            projection_diagnostics=projection_diagnostics,
            projection_alpha_history=projection_alpha_history,
            minimum_u_error_projection_maps=minimum_u_error_projection_maps,
            crossing_snapshots=crossing_snapshots,
        )
        if args.plots:
            save_plots(
                output_dir,
                data=data,
                initial_u=initial_u,
                recovered_u=current_u,
                prediction=prediction,
                history=history,
                dpi=int(args.dpi),
            )
        if trajectory_writer is not None:
            trajectory_writer.finalize()

        print()
        print("Reaction surrogate Landweber completed", flush=True)
        print(f"reason:                  {stop_reason}", flush=True)
        print(f"final relative residual: {final_relative_residual:.6e}", flush=True)
        print(f"final relative u error:  {final_relative_u_error:.6e}", flush=True)
        print(f"minimum u error:         {minimum_u_error:.6e} at iter {minimum_u_error_iteration}", flush=True)
        print(f"minimum residual iter:   {minimum_residual_iteration}", flush=True)
        print(
            f"Landweber time:          {landweber_elapsed_seconds:.3f} s "
            f"({landweber_elapsed_seconds / 60.0:.3f} min)",
            flush=True,
        )
        print(f"saved in:                {output_dir}", flush=True)
    finally:
        if trajectory_writer is not None:
            trajectory_writer.abort()
        if covariance is not None:
            covariance.close()


if __name__ == "__main__":
    main()
