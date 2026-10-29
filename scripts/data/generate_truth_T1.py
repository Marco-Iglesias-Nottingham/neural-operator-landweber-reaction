#!/usr/bin/env python3
"""Generate the frozen in-distribution T1 paper truth.

T1 is one held-out draw from exactly the broad Matérn mixture used for the
paper training shards.  Its seed and the resulting random hyperparameters are
frozen before any T1 inversion is run.  This wrapper delegates the fine FEM
solve and archive writing to ``generate_truth.py``.
"""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "generate_truth.py"

spec = importlib.util.spec_from_file_location("_reaction_truth_T1_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import {BASE_PATH}.")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

from reaction_core.config import DistributionConfig  # noqa: E402
from reaction_core.sampling import MaternUSampler  # noqa: E402


FROZEN_TRUTH_SEED = 4_240_000
FROZEN_NOISE_SEED = 20_260_821
PRODUCTION_DISTRIBUTION = DistributionConfig(
    nu_values=(1.0, 1.5, 2.0, 2.5),
    length_scales=(0.075, 0.10, 0.15, 0.20, 0.25),
    mean_u_range=(95.0, 105.0),
    std_u_range=(10.0, 30.0),
    u_min=0.0,
    u_max=200.0,
)
EXPECTED_SAMPLE_METADATA = {
    "field_seed": 1_657_906_990,
    "nu": 1.5,
    "length_scale": 0.075,
    "target_mean_u": 97.82987628024206,
    "target_std_u": 19.504740726669073,
    "rescaled": False,
}

# The shared writer uses these contracts when validating both the 120-point
# source field and its fine-grid interpolation.
base.TRAINING_MEAN_RANGE = PRODUCTION_DISTRIBUTION.mean_u_range
base.TRAINING_STD_RANGE = PRODUCTION_DISTRIBUTION.std_u_range
base.TRAINING_U_BOUNDS = (
    PRODUCTION_DISTRIBUTION.u_min,
    PRODUCTION_DISTRIBUTION.u_max,
)
base._training_distribution = lambda: PRODUCTION_DISTRIBUTION


def build_frozen_t1(args, grid):
    """Return T1 and fail if its frozen sampler identity has drifted."""

    if args.truth_kind != "matern":
        raise ValueError("T1 requires --truth-kind matern.")
    if int(args.truth_seed) != FROZEN_TRUTH_SEED:
        raise ValueError(
            f"T1 truth seed is frozen at {FROZEN_TRUTH_SEED}; "
            f"received {args.truth_seed}."
        )

    sampled = MaternUSampler(
        grid,
        PRODUCTION_DISTRIBUTION,
        max_cached_samplers=1,
        fft_workers=args.fft_workers,
    ).sample(FROZEN_TRUTH_SEED)
    metadata = asdict(sampled.metadata)

    for key in ("field_seed", "rescaled"):
        if metadata[key] != EXPECTED_SAMPLE_METADATA[key]:
            raise RuntimeError(
                f"Frozen T1 metadata changed for {key}: "
                f"expected {EXPECTED_SAMPLE_METADATA[key]!r}, "
                f"received {metadata[key]!r}."
            )
    for key in ("nu", "length_scale", "target_mean_u", "target_std_u"):
        if not np.isclose(
            float(metadata[key]),
            float(EXPECTED_SAMPLE_METADATA[key]),
            rtol=0.0,
            atol=1.0e-13,
        ):
            raise RuntimeError(
                f"Frozen T1 metadata changed for {key}: "
                f"expected {EXPECTED_SAMPLE_METADATA[key]!r}, "
                f"received {metadata[key]!r}."
            )

    field = np.asarray(sampled.u, dtype=np.float64)
    truth_metadata = {
        "truth_kind": "held_out_training_distribution_matern",
        "direct_parameterisation": "physical u (not log u)",
        "source_grid_points": [int(grid.nx), int(grid.ny)],
        "selection_protocol": (
            "seed fixed before any T1 inversion; disjoint from training, "
            "validation, and final-test seed ranges"
        ),
        "production_distribution": PRODUCTION_DISTRIBUTION.to_dict(),
        "sample": metadata,
    }
    base._validate_field_in_training_support(field, label="T1 source u field")
    return field, truth_metadata


base.build_source_truth = build_frozen_t1


def _has_option(argv: list[str], option: str) -> bool:
    return option in argv or any(value.startswith(option + "=") for value in argv)


def _validate_frozen_arguments(arguments: list[str]) -> None:
    parsed = base.parse_args(arguments)
    expected_exact = {
        "case_name": "T1",
        "truth_kind": "matern",
        "truth_seed": FROZEN_TRUTH_SEED,
        "noise_seed": FROZEN_NOISE_SEED,
        "u_input_nx": 120,
        "u_input_ny": 120,
        "fine_fem_nx": 952,
        "fine_fem_ny": 952,
    }
    for name, expected in expected_exact.items():
        received = getattr(parsed, name)
        if received != expected:
            raise ValueError(
                f"Frozen T1 requires --{name.replace('_', '-')}={expected}; "
                f"received {received}."
            )
    expected_float = {
        "matern_nu": EXPECTED_SAMPLE_METADATA["nu"],
        "matern_length_scale": EXPECTED_SAMPLE_METADATA["length_scale"],
        "mean_u": EXPECTED_SAMPLE_METADATA["target_mean_u"],
        "std_u": EXPECTED_SAMPLE_METADATA["target_std_u"],
        "noise_level_percent": 0.0,
    }
    for name, expected in expected_float.items():
        received = float(getattr(parsed, name))
        if not np.isclose(received, float(expected), rtol=0.0, atol=1.0e-13):
            raise ValueError(
                f"Frozen T1 requires --{name.replace('_', '-')}={expected}; "
                f"received {received}."
            )


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    defaults = {
        "--case-name": "T1",
        "--truth-kind": "matern",
        "--truth-seed": str(FROZEN_TRUTH_SEED),
        "--matern-nu": str(EXPECTED_SAMPLE_METADATA["nu"]),
        "--matern-length-scale": str(EXPECTED_SAMPLE_METADATA["length_scale"]),
        "--mean-u": str(EXPECTED_SAMPLE_METADATA["target_mean_u"]),
        "--std-u": str(EXPECTED_SAMPLE_METADATA["target_std_u"]),
        "--noise-level-percent": "0",
        "--noise-seed": str(FROZEN_NOISE_SEED),
    }
    for option, value in defaults.items():
        if not _has_option(arguments, option):
            arguments.extend((option, value))
    _validate_frozen_arguments(arguments)
    base.main(arguments)


if __name__ == "__main__":
    main()
