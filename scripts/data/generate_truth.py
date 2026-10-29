#!/usr/bin/env python3
"""Generate one reproducible fine-mesh truth and data set.

The forward problem is

    -Delta p + u p = 1  in (0, 1)^2,       p = 0 on the boundary.

This is deliberately a *truth/data* utility, not a training-shard generator.
Its two resolutions have different roles:

* ``u`` is first defined on a regular ``120 x 120`` POINT grid by default.
  It is interpreted as a continuous bilinear field.
* The forward solve uses a P1 ``952 x 952``-CELL triangular mesh by
  default. Consequently its P1 coefficient and pressure vertex grids are
  ``953 x 953``. Since 952 = 8 x 119, this truth mesh nests exactly with
  the 120-point surrogate/inversion grid.

The coefficient is sampled/interpreted on the source grid and then bilinearly
evaluated at the fine FEM vertices.  This avoids making the coefficient itself
artificially finer merely because the forward truth is fine.

The script has three named truth families:

``matern``
    One Matérn realisation with one explicit smoothness and length scale.
``three_gaussian_blobs``
    Three smooth positive/negative Gaussian features.
``rectangle_two_circles``
    One rectangle and two circles with discontinuous interfaces.

Every raw pattern is centred, standardised, then mapped to the direct physical
reaction coefficient using a requested mean and standard deviation. In this
generic base utility, a single global fluctuation scale keeps it inside the
legacy default support
``mean in [40,60]``, ``std in [5,15]``, and ``10 <= u <= 100`` without
pointwise clipping distortion. The frozen T1/T2 wrappers replace these bounds
with the paper distribution where appropriate.

For a requested ``r`` percent noise level, the saved Gaussian noise is scaled
globally so that, on the saved interior observation mask and trapezoidal
weights,

    ||p_noisy - p_noiseless||_W = (r / 100) ||p_noiseless||_W.

Thus ``noise_norm_realized`` in the JSON file is the exact synthetic
discrepancy-principle level (up to floating-point rounding), rather than a
nominal per-point standard deviation.

Run with exactly one MPI rank.  The default 952-cell mesh has about 0.91
million P1 state degrees of freedom and is intended for an HPC node.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

from mpi4py import MPI
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reaction_core.config import DistributionConfig, ReactionConfig  # noqa: E402
from reaction_core.grid import (  # noqa: E402
    RegularGrid2D,
    StructuredCoefficientGridMap,
    StructuredLagrangeStateGridMap,
)
from reaction_core.inversion import (  # noqa: E402
    interior_vertex_mask,
    resample_regular_grid_field,
    trapezoid_weights,
    weighted_norm,
)
from reaction_core.random_fields import MaternCirculantEmbedding2D  # noqa: E402
from reaction_core.solver import ReactionSolver2D  # noqa: E402


FORMAT = "reaction-fine-truth-data-v1"
STATE_DEGREE = 1
TRAINING_MEAN_RANGE = (40.0, 60.0)
TRAINING_STD_RANGE = (5.0, 15.0)
TRAINING_U_BOUNDS = (10.0, 100.0)


@dataclass(frozen=True)
class PhysicalFieldMetadata:
    """Metadata for the bounded affine direct-u construction."""

    requested_mean_u: float
    requested_std_u: float
    actual_mean_u: float
    actual_std_u: float
    minimum_u: float
    maximum_u: float
    fluctuation_scale: float
    globally_rescaled_to_bounds: bool


@dataclass(frozen=True)
class NoiseMetadata:
    """Exact weighted norm information for one Gaussian noise realisation."""

    seed: int
    requested_noise_percent: float
    requested_relative_noise_fraction: float
    noiseless_data_norm: float
    noise_norm_target: float
    noise_norm_realized: float
    relative_noise_fraction_realized: float
    relative_noise_percent_realized: float
    gaussian_direction_weighted_norm: float
    gaussian_scale: float
    number_of_noisy_observations: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Create one direct-u Reaction truth on a regular coefficient "
            "grid and a fine P1 FEM pressure grid, with an exactly scaled "
            "Gaussian-noise realisation."
        ),
    )
    mesh = parser.add_argument_group("source grid and fine P1 forward mesh")
    mesh.add_argument(
        "--u-input-nx",
        type=int,
        default=120,
        help="Number of x POINTS for the source/input u grid.",
    )
    mesh.add_argument(
        "--u-input-ny",
        type=int,
        default=120,
        help="Number of y POINTS for the source/input u grid.",
    )
    mesh.add_argument(
        "--fine-fem-nx",
        type=int,
        default=952,
        help="Fine P1 FEM CELL count in x; output has nx+1 pressure vertices.",
    )
    mesh.add_argument(
        "--fine-fem-ny",
        type=int,
        default=952,
        help="Fine P1 FEM CELL count in y; output has ny+1 pressure vertices.",
    )
    mesh.add_argument("--rhs", type=float, default=1.0)

    truth = parser.add_argument_group("named truth family")
    truth.add_argument(
        "--truth-kind",
        choices=("matern", "three_gaussian_blobs", "rectangle_two_circles"),
        default="matern",
        help="Named direct-u truth family.",
    )
    truth.add_argument(
        "--truth-seed",
        type=int,
        default=20_260_820,
        help=(
            "Matérn field seed. It is recorded for all families, although "
            "the two analytic families are deterministic."
        ),
    )
    truth.add_argument(
        "--matern-nu",
        type=float,
        default=1.5,
        help="One fixed Matérn smoothness, used only by --truth-kind matern.",
    )
    truth.add_argument(
        "--matern-length-scale",
        type=float,
        default=0.15,
        help="One fixed Matérn length scale in the unit-square coordinates.",
    )
    truth.add_argument(
        "--mean-u",
        type=float,
        default=50.0,
        help="Requested physical-u spatial mean; must be inside training [40,60].",
    )
    truth.add_argument(
        "--std-u",
        type=float,
        default=10.0,
        help="Requested physical-u spatial std; must be inside training [5,15].",
    )
    truth.add_argument(
        "--fft-workers",
        type=int,
        default=None,
        help="Optional SciPy FFT worker count for the Matérn sampler.",
    )

    noise = parser.add_argument_group("exactly scaled Gaussian observation noise")
    noise.add_argument(
        "--noise-level-percent",
        type=float,
        default=0.0,
        help=(
            "Relative W-L2 noise level in percent. Use 0 for noise-free "
            "data; noiseless and noisy arrays are always both saved."
        ),
    )
    noise.add_argument(
        "--noise-seed",
        type=int,
        default=20_260_821,
        help="Seed for the saved Gaussian noise realisation.",
    )

    solve = parser.add_argument_group("serial fine forward solve")
    solve.add_argument(
        "--linear-solver",
        choices=("cg-hypre", "lu"),
        default="cg-hypre",
        help="State solver. CG+HYPRE is the practical 1024-cell default.",
    )
    solve.add_argument("--ksp-rtol", type=float, default=1.0e-11)
    solve.add_argument("--ksp-max-it", type=int, default=10_000)

    output = parser.add_argument_group("output")
    output.add_argument(
        "--case-name",
        type=str,
        default="T1",
        help="Short case identifier used for compact output paths, e.g. T1 or blob1.",
    )
    output.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; a descriptive directory is used by default.",
    )
    output.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _filename_number(value: float) -> str:
    """Format a numeric setting safely and readably in an output filename."""

    rendered = f"{float(value):.10g}"
    return rendered.replace("-", "m").replace(".", "p").replace("+", "")


def _training_distribution() -> DistributionConfig:
    """Return the project's training-support contract for metadata/checks."""

    return DistributionConfig(
        mean_u_range=TRAINING_MEAN_RANGE,
        std_u_range=TRAINING_STD_RANGE,
        u_min=TRAINING_U_BOUNDS[0],
        u_max=TRAINING_U_BOUNDS[1],
    )


def _validate_args(args: argparse.Namespace) -> Path:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError(
            "This structured-grid truth generator requires exactly one MPI rank."
        )
    for label, value in (
        ("--u-input-nx", args.u_input_nx),
        ("--u-input-ny", args.u_input_ny),
        ("--fine-fem-nx", args.fine_fem_nx),
        ("--fine-fem-ny", args.fine_fem_ny),
    ):
        if int(value) < 2:
            raise ValueError(f"{label} is too small: {value}.")
    if int(args.ksp_max_it) < 1:
        raise ValueError(f"--ksp-max-it is too small: {args.ksp_max_it}.")
    # Main paper benchmark: the fine truth FEM should nest exactly with the
    # source/inverse parameter grid. This makes the 120-point truth an exact
    # subset of the 953 fine FEM vertices, eliminating gratuitous interpolation
    # from the truth diagnostic.
    if int(args.fine_fem_nx) % (int(args.u_input_nx) - 1) != 0 or int(args.fine_fem_ny) % (int(args.u_input_ny) - 1) != 0:
        raise ValueError(
            "Fine FEM cell counts must be integer multiples of the source-grid "
            "interval counts. For the paper default use 120 source points and "
            "952 fine FEM cells (= 8 x 119)."
        )
    if int(args.truth_seed) < 0 or int(args.noise_seed) < 0:
        raise ValueError("--truth-seed and --noise-seed must be non-negative.")
    if not np.isfinite(args.rhs) or args.rhs <= 0.0:
        raise ValueError("--rhs must be finite and positive.")
    if not np.isfinite(args.matern_nu) or args.matern_nu <= 0.0:
        raise ValueError("--matern-nu must be finite and positive.")
    if not np.isfinite(args.matern_length_scale) or args.matern_length_scale <= 0.0:
        raise ValueError("--matern-length-scale must be finite and positive.")
    if args.fft_workers is not None and int(args.fft_workers) < 1:
        raise ValueError("--fft-workers must be positive when supplied.")
    if not np.isfinite(args.ksp_rtol) or args.ksp_rtol <= 0.0:
        raise ValueError("--ksp-rtol must be finite and positive.")
    if not np.isfinite(args.noise_level_percent) or args.noise_level_percent < 0.0:
        raise ValueError("--noise-level-percent must be finite and non-negative.")

    distribution = _training_distribution()
    distribution.validate()
    if not TRAINING_MEAN_RANGE[0] <= float(args.mean_u) <= TRAINING_MEAN_RANGE[1]:
        raise ValueError(
            "--mean-u must remain within the training distribution "
            f"{TRAINING_MEAN_RANGE}; got {args.mean_u:g}."
        )
    if not TRAINING_STD_RANGE[0] <= float(args.std_u) <= TRAINING_STD_RANGE[1]:
        raise ValueError(
            "--std-u must remain within the training distribution "
            f"{TRAINING_STD_RANGE}; got {args.std_u:g}."
        )

    if not str(args.case_name).strip():
        raise ValueError("--case-name cannot be empty.")
    if any(ch in str(args.case_name) for ch in ("/", "\\")):
        raise ValueError("--case-name must be a short name, not a path.")
    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "truths" / str(args.case_name)
    else:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is non-empty: {output_dir}. "
            "Choose a new --output-dir or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _stem(args: argparse.Namespace) -> str:
    if args.truth_kind == "matern":
        truth_label = (
            f"matern_nu{_filename_number(args.matern_nu)}"
            f"_ell{_filename_number(args.matern_length_scale)}"
        )
    else:
        truth_label = str(args.truth_kind)
    return (
        f"reaction_truth_{truth_label}"
        f"_u{int(args.u_input_nx)}x{int(args.u_input_ny)}points"
        f"_pfem{int(args.fine_fem_nx)}x{int(args.fine_fem_ny)}cells"
        f"_mean{_filename_number(args.mean_u)}"
        f"_std{_filename_number(args.std_u)}"
        f"_truthseed{int(args.truth_seed)}"
        f"_noise{_filename_number(args.noise_level_percent)}pct"
        f"_noiseseed{int(args.noise_seed)}"
    )


def _make_grid(nx: int, ny: int) -> RegularGrid2D:
    return RegularGrid2D.from_domain(1.0, 1.0, int(nx), int(ny))


def _standardise_pattern(pattern: np.ndarray) -> np.ndarray:
    array = np.asarray(pattern, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("Truth pattern must be a finite two-dimensional array.")
    centred = array - float(np.mean(array))
    standard_deviation = float(np.std(centred))
    if standard_deviation <= 1.0e-13:
        raise RuntimeError("Truth pattern has essentially zero spatial variation.")
    return centred / standard_deviation


def _bounded_affine_direct_u(
    pattern: np.ndarray,
    *,
    mean_u: float,
    std_u: float,
) -> tuple[np.ndarray, PhysicalFieldMetadata]:
    """Map a pattern to direct u with one globally bounded fluctuation scale."""

    z = _standardise_pattern(pattern)
    lower, upper = TRAINING_U_BOUNDS
    fluctuation = float(std_u) * z
    scale = 1.0
    negative = float(np.min(fluctuation))
    positive = float(np.max(fluctuation))
    if negative < 0.0:
        scale = min(scale, (float(mean_u) - lower) / (-negative))
    if positive > 0.0:
        scale = min(scale, (upper - float(mean_u)) / positive)
    scale = float(np.clip(scale, 0.0, 1.0))

    field = float(mean_u) + scale * fluctuation
    tolerance = 512.0 * np.finfo(np.float64).eps * max(1.0, abs(lower), abs(upper))
    if float(np.min(field)) < lower - tolerance or float(np.max(field)) > upper + tolerance:
        raise RuntimeError("Global direct-u bound scaling failed.")
    field = np.clip(field, lower, upper)
    metadata = PhysicalFieldMetadata(
        requested_mean_u=float(mean_u),
        requested_std_u=float(std_u),
        actual_mean_u=float(np.mean(field)),
        actual_std_u=float(np.std(field)),
        minimum_u=float(np.min(field)),
        maximum_u=float(np.max(field)),
        fluctuation_scale=scale,
        globally_rescaled_to_bounds=bool(
            scale < 1.0 - 64.0 * np.finfo(np.float64).eps
        ),
    )
    return np.asarray(field, dtype=np.float64), metadata


def _gaussian_blob_pattern(grid: RegularGrid2D) -> tuple[np.ndarray, dict[str, Any]]:
    """Return three resolved Gaussian features before direct-u normalisation."""

    x, y = grid.mesh()
    # (centre x, centre y, x width, y width, signed pattern amplitude)
    blobs = (
        (0.25, 0.30, 0.090, 0.110, 1.00),
        (0.70, 0.29, 0.125, 0.090, -0.90),
        (0.55, 0.72, 0.085, 0.105, 1.10),
    )
    pattern = np.zeros(grid.shape, dtype=np.float64)
    for center_x, center_y, sigma_x, sigma_y, amplitude in blobs:
        exponent = (
            ((x - center_x) / sigma_x) ** 2
            + ((y - center_y) / sigma_y) ** 2
        )
        pattern += amplitude * np.exp(-0.5 * exponent)
    return pattern, {
        "construction": "sum of three anisotropic Gaussian blobs before global direct-u normalisation",
        "blobs": [
            {
                "center": [center_x, center_y],
                "sigma": [sigma_x, sigma_y],
                "pattern_amplitude": amplitude,
            }
            for center_x, center_y, sigma_x, sigma_y, amplitude in blobs
        ],
    }


def _rectangle_two_circles_pattern(
    grid: RegularGrid2D,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return one rectangle plus two discontinuous circles before normalisation."""

    x, y = grid.mesh()
    # The features are deliberately disjoint, so all interfaces remain clear.
    rectangle = (x >= 0.12) & (x <= 0.40) & (y >= 0.55) & (y <= 0.80)
    circle_one = (x - 0.70) ** 2 + (y - 0.30) ** 2 <= 0.14**2
    circle_two = (x - 0.66) ** 2 + (y - 0.72) ** 2 <= 0.12**2
    pattern = np.zeros(grid.shape, dtype=np.float64)
    pattern[rectangle] = 1.0
    pattern[circle_one] = -0.90
    pattern[circle_two] = 1.20
    return pattern, {
        "construction": (
            "one rectangle and two circles with discontinuous interfaces "
            "before global direct-u normalisation"
        ),
        "rectangle": {"x_bounds": [0.12, 0.40], "y_bounds": [0.55, 0.80], "pattern_value": 1.0},
        "circles": [
            {"center": [0.70, 0.30], "radius": 0.14, "pattern_value": -0.90},
            {"center": [0.66, 0.72], "radius": 0.12, "pattern_value": 1.20},
        ],
    }


def _matern_pattern(
    grid: RegularGrid2D,
    *,
    nu: float,
    length_scale: float,
    seed: int,
    fft_workers: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    sampler = MaternCirculantEmbedding2D(
        grid.x,
        grid.y,
        nu=float(nu),
        length_scale=float(length_scale),
        variance=1.0,
        mean=0.0,
        fft_workers=fft_workers,
    )
    pattern = sampler.sample(seed=int(seed), dtype=np.float64)
    return pattern, {
        "construction": "one non-periodic Matérn circulant-embedding realisation before global direct-u normalisation",
        "truth_seed": int(seed),
        "nu": float(nu),
        "length_scale": float(length_scale),
        "embedding_diagnostics": asdict(sampler.diagnostics),
    }


def build_source_truth(
    args: argparse.Namespace,
    grid: RegularGrid2D,
) -> tuple[np.ndarray, dict[str, Any]]:
    if args.truth_kind == "matern":
        pattern, family = _matern_pattern(
            grid,
            nu=float(args.matern_nu),
            length_scale=float(args.matern_length_scale),
            seed=int(args.truth_seed),
            fft_workers=args.fft_workers,
        )
    elif args.truth_kind == "three_gaussian_blobs":
        pattern, family = _gaussian_blob_pattern(grid)
        family["truth_seed"] = int(args.truth_seed)
        family["truth_seed_note"] = "recorded for naming/reproducibility; analytic construction is deterministic"
    elif args.truth_kind == "rectangle_two_circles":
        pattern, family = _rectangle_two_circles_pattern(grid)
        family["truth_seed"] = int(args.truth_seed)
        family["truth_seed_note"] = "recorded for naming/reproducibility; analytic construction is deterministic"
    else:  # argparse choices make this defensive branch unreachable.
        raise ValueError(f"Unsupported truth kind {args.truth_kind!r}.")

    field, physical = _bounded_affine_direct_u(
        pattern,
        mean_u=float(args.mean_u),
        std_u=float(args.std_u),
    )
    metadata: dict[str, Any] = {
        "truth_kind": str(args.truth_kind),
        "direct_parameterisation": "physical u (not log u)",
        "source_grid_points": [int(grid.nx), int(grid.ny)],
        "family": family,
        "physical_field": asdict(physical),
    }
    _validate_field_in_training_support(field, label="source u input field")
    return field, metadata


def _validate_field_in_training_support(field: np.ndarray, *, label: str) -> None:
    array = np.asarray(field, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array))
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    tolerance = 5.0e-11
    if not TRAINING_MEAN_RANGE[0] - tolerance <= mean <= TRAINING_MEAN_RANGE[1] + tolerance:
        raise RuntimeError(f"{label} mean {mean:g} lies outside training support.")
    if not TRAINING_STD_RANGE[0] - tolerance <= std <= TRAINING_STD_RANGE[1] + tolerance:
        raise RuntimeError(f"{label} std {std:g} lies outside training support.")
    if minimum < TRAINING_U_BOUNDS[0] - tolerance or maximum > TRAINING_U_BOUNDS[1] + tolerance:
        raise RuntimeError(
            f"{label} range [{minimum:g}, {maximum:g}] lies outside training support."
        )


def _solver_options(args: argparse.Namespace) -> dict[str, Any]:
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


def _solve_fine_pressure(
    args: argparse.Namespace,
    fine_grid: RegularGrid2D,
    fine_u: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], float]:
    config = ReactionConfig(
        fem_nx=int(args.fine_fem_nx),
        fem_ny=int(args.fine_fem_ny),
        state_degree=STATE_DEGREE,
        rhs=float(args.rhs),
        u_min=TRAINING_U_BOUNDS[0],
    )
    options = _solver_options(args)
    start = perf_counter()
    with ReactionSolver2D(
        config,
        minimum_u=TRAINING_U_BOUNDS[0],
        initial_u=max(50.0, TRAINING_U_BOUNDS[0]),
        solver_options=options,
    ) as solver:
        coefficient_map = StructuredCoefficientGridMap(solver, fine_grid)
        state_map = StructuredLagrangeStateGridMap(solver, fine_grid)
        solver.set_coefficient(coefficient_map.grid_to_array(fine_u))
        state = solver.forward()
        pressure = state_map.state_to_grid(state)
        diagnostics: dict[str, Any] = {
            "state_dofs": solver.num_state_dofs(),
            "parameter_dofs": solver.num_parameter_dofs(),
            "solver_options": options,
            "solver_cache": solver.cache_diagnostics(),
        }
    pressure = np.asarray(pressure, dtype=np.float64)
    if not np.all(np.isfinite(pressure)):
        raise RuntimeError("Fine forward pressure contains non-finite values.")
    boundary = np.concatenate(
        (pressure[0, :], pressure[-1, :], pressure[:, 0], pressure[:, -1])
    )
    boundary_linf = float(np.max(np.abs(boundary)))
    if boundary_linf > 1.0e-10:
        raise RuntimeError(
            "Fine forward pressure violates the homogeneous Dirichlet boundary: "
            f"Linf={boundary_linf:.3e}."
        )
    diagnostics["pressure_boundary_linf"] = boundary_linf
    return pressure, diagnostics, float(perf_counter() - start)


def _globally_scaled_gaussian_noise(
    noiseless: np.ndarray,
    *,
    weights: np.ndarray,
    observation_mask: np.ndarray,
    noise_percent: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, NoiseMetadata]:
    """Make a Gaussian vector with its *realised* W-norm set exactly."""

    clean = np.asarray(noiseless, dtype=np.float64)
    mask = np.asarray(observation_mask, dtype=bool)
    if clean.shape != weights.shape or clean.shape != mask.shape:
        raise ValueError("Noiseless data, weights, and mask must share one grid.")
    if not np.any(mask):
        raise ValueError("Observation mask is empty.")
    clean_norm = weighted_norm(clean, weights, mask=mask)
    if not np.isfinite(clean_norm) or clean_norm <= 0.0:
        raise RuntimeError("Noiseless observation norm must be finite and positive.")
    fraction = float(noise_percent) / 100.0
    target_norm = fraction * clean_norm

    raw = np.zeros_like(clean)
    rng = np.random.default_rng(int(seed))
    raw[mask] = rng.standard_normal(int(np.count_nonzero(mask)))
    raw_norm = weighted_norm(raw, weights, mask=mask)
    if not np.isfinite(raw_norm) or raw_norm <= 0.0:
        raise RuntimeError("Generated Gaussian direction has invalid weighted norm.")
    scale = 0.0 if target_norm == 0.0 else target_norm / raw_norm
    noise = scale * raw
    noisy = clean + noise
    realized_norm = weighted_norm(noisy - clean, weights, mask=mask)
    metadata = NoiseMetadata(
        seed=int(seed),
        requested_noise_percent=float(noise_percent),
        requested_relative_noise_fraction=fraction,
        noiseless_data_norm=float(clean_norm),
        noise_norm_target=float(target_norm),
        noise_norm_realized=float(realized_norm),
        relative_noise_fraction_realized=float(realized_norm / clean_norm),
        relative_noise_percent_realized=float(100.0 * realized_norm / clean_norm),
        gaussian_direction_weighted_norm=float(raw_norm),
        gaussian_scale=float(scale),
        number_of_noisy_observations=int(np.count_nonzero(mask)),
    )
    tolerance = 5.0e-13 * max(1.0, target_norm)
    if abs(realized_norm - target_norm) > tolerance:
        raise RuntimeError(
            "Gaussian noise did not attain the requested weighted norm: "
            f"target={target_norm:.16e}, realised={realized_norm:.16e}."
        )
    if np.any(noise[~mask] != 0.0):
        raise RuntimeError("Noise was generated outside the observation mask.")
    return noise, noisy, metadata


def _make_report(
    *,
    args: argparse.Namespace,
    stem: str,
    source_grid: RegularGrid2D,
    fine_grid: RegularGrid2D,
    source_u: np.ndarray,
    fine_u: np.ndarray,
    truth_metadata: dict[str, Any],
    pressure: np.ndarray,
    noise_metadata: NoiseMetadata,
    solver_diagnostics: dict[str, Any],
    solve_seconds: float,
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "file_stem": stem,
        "equation": "-Delta p + u p = rhs on (0,1)^2; p=0 on the full boundary",
        "parameterisation": "direct physical u (not log u)",
        "state_degree": STATE_DEGREE,
        "serial_only": True,
        "grid_convention": {
            "u_input_grid": {
                "points": [source_grid.nx, source_grid.ny],
                "description": (
                    "The source truth field. It has 120x120 points by default, matching "
                    "the inverse parameter grid (119 intervals per axis)."
                ),
            },
            "fine_fem_mesh": {
                "cells": [int(args.fine_fem_nx), int(args.fine_fem_ny)],
                "p1_vertices": [fine_grid.nx, fine_grid.ny],
                "description": (
                    "The pressure p and P1 FEM coefficient u are represented at "
                    "953x953 vertices by default (952 = 8 x 119 cells)."
                ),
            },
            "coefficient_transfer": (
                "u_input is interpreted as a bilinear field and evaluated at "
                "the fine P1 FEM vertices before the forward solve."
            ),
        },
        "training_distribution_support": _training_distribution().to_dict(),
        "truth": truth_metadata,
        "source_u_input_statistics": _field_statistics(source_u),
        "fine_u_fem_statistics": _field_statistics(fine_u),
        "pressure_noiseless_statistics": _field_statistics(pressure),
        "noise": asdict(noise_metadata),
        "solver": solver_diagnostics,
        "forward_solve_seconds": float(solve_seconds),
        "command_arguments": vars(args),
    }


def _field_statistics(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "l2_unweighted": float(np.sqrt(np.mean(array * array))),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = _validate_args(args)
    stem = str(args.case_name)
    npz_path = output_dir / "truth.npz"
    json_path = output_dir / "truth.json"
    if (npz_path.exists() or json_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {npz_path.name} or {json_path.name}. "
            "Use --overwrite deliberately."
        )

    source_grid = _make_grid(args.u_input_nx, args.u_input_ny)
    fine_grid = _make_grid(args.fine_fem_nx + 1, args.fine_fem_ny + 1)
    source_u, truth_metadata = build_source_truth(args, source_grid)
    fine_u = resample_regular_grid_field(source_u, source_grid, fine_grid)
    _validate_field_in_training_support(fine_u, label="fine P1 u field")

    print("Reaction fine truth/data generation", flush=True)
    print(f"  truth kind              : {args.truth_kind}", flush=True)
    print(
        f"  u source grid           : {source_grid.nx} x {source_grid.ny} points",
        flush=True,
    )
    print(
        f"  fine P1 FEM mesh        : {args.fine_fem_nx} x {args.fine_fem_ny} cells "
        f"({fine_grid.nx} x {fine_grid.ny} vertices)",
        flush=True,
    )
    print(
        "  u source stats          : "
        f"mean={np.mean(source_u):.6g}, std={np.std(source_u):.6g}, "
        f"range=[{np.min(source_u):.6g}, {np.max(source_u):.6g}]",
        flush=True,
    )
    print(f"  truth seed              : {args.truth_seed}", flush=True)
    if args.truth_kind == "matern":
        print(
            f"  Matérn nu, length scale : {args.matern_nu:g}, "
            f"{args.matern_length_scale:g}",
            flush=True,
        )
    print("Solving the fine P1 forward model...", flush=True)
    pressure, solver_diagnostics, solve_seconds = _solve_fine_pressure(
        args, fine_grid, fine_u
    )

    weights = trapezoid_weights(fine_grid)
    mask = interior_vertex_mask(fine_grid)
    noise, noisy_pressure, noise_metadata = _globally_scaled_gaussian_noise(
        pressure,
        weights=weights,
        observation_mask=mask,
        noise_percent=float(args.noise_level_percent),
        seed=int(args.noise_seed),
    )
    report = _make_report(
        args=args,
        stem=stem,
        source_grid=source_grid,
        fine_grid=fine_grid,
        source_u=source_u,
        fine_u=fine_u,
        truth_metadata=truth_metadata,
        pressure=pressure,
        noise_metadata=noise_metadata,
        solver_diagnostics=solver_diagnostics,
        solve_seconds=solve_seconds,
    )

    # Keep generic names for downstream code and dimensions in dedicated,
    # dynamically named aliases.  The latter are truthful even for reduced
    # smoke tests instead of misleadingly claiming a 512/1025 grid.
    arrays: dict[str, np.ndarray] = {
        "format": np.asarray(FORMAT),
        "parameterisation": np.asarray("direct physical u"),
        "state_degree": np.asarray(STATE_DEGREE, dtype=np.int64),
        "rhs": np.asarray(args.rhs, dtype=np.float64),
        "truth_kind": np.asarray(args.truth_kind),
        "truth_seed": np.asarray(args.truth_seed, dtype=np.int64),
        "noise_seed": np.asarray(args.noise_seed, dtype=np.int64),
        "noise_level_percent_requested": np.asarray(
            args.noise_level_percent, dtype=np.float64
        ),
        "u_input_x_grid": np.asarray(source_grid.x, dtype=np.float64),
        "u_input_y_grid": np.asarray(source_grid.y, dtype=np.float64),
        "u_input_grid": np.asarray(source_u, dtype=np.float64),
        "fine_fem_x_grid": np.asarray(fine_grid.x, dtype=np.float64),
        "fine_fem_y_grid": np.asarray(fine_grid.y, dtype=np.float64),
        "fine_fem_nx_cells": np.asarray(args.fine_fem_nx, dtype=np.int64),
        "fine_fem_ny_cells": np.asarray(args.fine_fem_ny, dtype=np.int64),
        "u_fem_P1_vertices": np.asarray(fine_u, dtype=np.float64),
        "p_noiseless_fem_P1_vertices": np.asarray(pressure, dtype=np.float64),
        "p_noisy_fem_P1_vertices": np.asarray(noisy_pressure, dtype=np.float64),
        "noise_fem_P1_vertices": np.asarray(noise, dtype=np.float64),
        "data_noiseless": np.asarray(pressure, dtype=np.float64),
        "data_noisy": np.asarray(noisy_pressure, dtype=np.float64),
        "observation_mask": np.asarray(mask, dtype=bool),
        "observation_weights_trapezoid": np.asarray(weights, dtype=np.float64),
        "noiseless_data_norm": np.asarray(
            noise_metadata.noiseless_data_norm, dtype=np.float64
        ),
        "noise_norm_target": np.asarray(
            noise_metadata.noise_norm_target, dtype=np.float64
        ),
        "noise_norm_realized": np.asarray(
            noise_metadata.noise_norm_realized, dtype=np.float64
        ),
        "relative_noise_percent_realized": np.asarray(
            noise_metadata.relative_noise_percent_realized, dtype=np.float64
        ),
        "truth_metadata_json": np.asarray(
            json.dumps(_json_ready(truth_metadata), sort_keys=True)
        ),
        "noise_metadata_json": np.asarray(
            json.dumps(_json_ready(asdict(noise_metadata)), sort_keys=True)
        ),
        "report_json": np.asarray(json.dumps(_json_ready(report), sort_keys=True)),
    }
    source_shape = f"{source_grid.nx}x{source_grid.ny}"
    fine_shape = f"{fine_grid.nx}x{fine_grid.ny}"
    arrays[f"u_input_{source_shape}"] = np.asarray(source_u, dtype=np.float64)
    arrays[f"u_fem_P1_vertices_{fine_shape}"] = np.asarray(
        fine_u, dtype=np.float64
    )
    arrays[f"p_noiseless_fem_P1_vertices_{fine_shape}"] = np.asarray(
        pressure, dtype=np.float64
    )
    arrays[f"p_noisy_fem_P1_vertices_{fine_shape}"] = np.asarray(
        noisy_pressure, dtype=np.float64
    )
    arrays[f"noise_fem_P1_vertices_{fine_shape}"] = np.asarray(
        noise, dtype=np.float64
    )
    np.savez_compressed(npz_path, **arrays)
    _write_json(json_path, report)

    print(
        "  weighted data norm      : "
        f"{noise_metadata.noiseless_data_norm:.16e}",
        flush=True,
    )
    print(
        "  noise norm target/actual: "
        f"{noise_metadata.noise_norm_target:.16e} / "
        f"{noise_metadata.noise_norm_realized:.16e}",
        flush=True,
    )
    print(
        "  realised relative noise : "
        f"{noise_metadata.relative_noise_percent_realized:.12g}% "
        f"(seed {args.noise_seed})",
        flush=True,
    )
    print(f"Saved data:    {npz_path}", flush=True)
    print(f"Saved metadata:{json_path}", flush=True)


if __name__ == "__main__":
    main()
