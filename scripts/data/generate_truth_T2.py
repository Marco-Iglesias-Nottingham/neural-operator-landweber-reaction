#!/usr/bin/env python3
"""
Generate the broad-amplitude T2 synthetic truth/data set.

This is a thin wrapper around scripts/data/generate_truth.py.
It keeps the original T2 geometry (one rectangle and two circles) but uses
fixed broad-regime physical levels:

    background          = 100
    rectangle           = 150
    negative circle     = 20
    positive circle     = 180

On the default 120 x 120 source grid this gives approximately
mean(u)=102.154 and std(u)=28.947, so the truth remains within the broad
training contracts

    mean in [95,105], std in [10,30], 0 <= u <= 200,

while attaining the desired min/max 20/180.

The fine synthetic pressure is generated with the same clean truth generator:
120 x 120 source u grid -> bilinear transfer -> P1 FEM on 952 x 952 cells
(953 x 953 vertices), with exactly scaled optional observation noise.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR / "generate_truth.py"

if not BASE_PATH.exists():
    raise FileNotFoundError(
        f"Expected base truth generator at {BASE_PATH}. "
        "Put this file in the same scripts/ directory as "
        "generate_truth.py."
    )

spec = importlib.util.spec_from_file_location(
    "_reaction_truth_clean_base", BASE_PATH
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import {BASE_PATH}.")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

# Broad training-support contract.
base.TRAINING_MEAN_RANGE = (95.0, 105.0)
base.TRAINING_STD_RANGE = (10.0, 30.0)
base.TRAINING_U_BOUNDS = (0.0, 200.0)

BACKGROUND = 100.0
RECTANGLE_VALUE = 150.0
NEGATIVE_CIRCLE_VALUE = 20.0
POSITIVE_CIRCLE_VALUE = 180.0


def build_broad_t2(args, grid):
    """Build the fixed broad-regime T2 field on the source grid."""

    x, y = grid.mesh()

    rectangle = (
        (x >= 0.12) & (x <= 0.40) &
        (y >= 0.55) & (y <= 0.80)
    )
    circle_one = (x - 0.70) ** 2 + (y - 0.30) ** 2 <= 0.14**2
    circle_two = (x - 0.66) ** 2 + (y - 0.72) ** 2 <= 0.12**2

    u = np.full(grid.shape, BACKGROUND, dtype=np.float64)
    u[rectangle] = RECTANGLE_VALUE
    u[circle_one] = NEGATIVE_CIRCLE_VALUE
    u[circle_two] = POSITIVE_CIRCLE_VALUE

    mean_u = float(np.mean(u))
    std_u = float(np.std(u))
    min_u = float(np.min(u))
    max_u = float(np.max(u))

    # Make failure explicit if the grid/geometry is ever changed enough to
    # leave the intended broad training regime.
    tol = 5.0e-11
    if not (base.TRAINING_MEAN_RANGE[0] - tol <= mean_u <=
            base.TRAINING_MEAN_RANGE[1] + tol):
        raise RuntimeError(
            f"T2 broad mean {mean_u:g} is outside "
            f"{base.TRAINING_MEAN_RANGE}."
        )
    if not (base.TRAINING_STD_RANGE[0] - tol <= std_u <=
            base.TRAINING_STD_RANGE[1] + tol):
        raise RuntimeError(
            f"T2 broad std {std_u:g} is outside "
            f"{base.TRAINING_STD_RANGE}."
        )
    if min_u < base.TRAINING_U_BOUNDS[0] - tol or \
       max_u > base.TRAINING_U_BOUNDS[1] + tol:
        raise RuntimeError(
            f"T2 broad range [{min_u:g}, {max_u:g}] is outside "
            f"{base.TRAINING_U_BOUNDS}."
        )

    metadata = {
        "truth_kind": "rectangle_two_circles_broad",
        "direct_parameterisation": "physical u (not log u)",
        "source_grid_points": [int(grid.nx), int(grid.ny)],
        "family": {
            "construction": (
                "fixed broad-regime T2: one rectangle and two circles "
                "with discontinuous interfaces"
            ),
            "background_u": BACKGROUND,
            "rectangle": {
                "x_bounds": [0.12, 0.40],
                "y_bounds": [0.55, 0.80],
                "u_value": RECTANGLE_VALUE,
            },
            "circles": [
                {
                    "center": [0.70, 0.30],
                    "radius": 0.14,
                    "u_value": NEGATIVE_CIRCLE_VALUE,
                },
                {
                    "center": [0.66, 0.72],
                    "radius": 0.12,
                    "u_value": POSITIVE_CIRCLE_VALUE,
                },
            ],
            "truth_seed": int(args.truth_seed),
            "truth_seed_note": (
                "recorded for naming/reproducibility; "
                "this analytic truth is deterministic"
            ),
        },
        "physical_field": {
            "requested_background_u": BACKGROUND,
            "actual_mean_u": mean_u,
            "actual_std_u": std_u,
            "minimum_u": min_u,
            "maximum_u": max_u,
            "hard_training_bounds": list(base.TRAINING_U_BOUNDS),
        },
    }

    base._validate_field_in_training_support(
        u, label="broad T2 source u field"
    )
    return u, metadata


# Replace only the truth construction. Everything else (fine-grid transfer,
# PDE solve, noise construction, NPZ/JSON format) remains the tested clean
# generator implementation.
base.build_source_truth = build_broad_t2


if __name__ == "__main__":
    # Use a parser-accepted analytic truth label. build_source_truth above
    # replaces its construction with the fixed broad T2 field.
    argv = sys.argv[1:]

    # These are bookkeeping values required by the base parser/validator.
    # They agree with the actual broad T2 statistics to sufficient accuracy.
    # If the user supplied them explicitly, leave their values untouched.
    if "--truth-kind" not in argv:
        argv += ["--truth-kind", "rectangle_two_circles"]
    if "--mean-u" not in argv:
        argv += ["--mean-u", "102.15416666666667"]
    if "--std-u" not in argv:
        argv += ["--std-u", "28.94652367105399"]

    base.main(argv)
