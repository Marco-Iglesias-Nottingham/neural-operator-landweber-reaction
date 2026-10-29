"""Pure helpers shared by CPU and GPU revision trajectory diagnostics."""

from __future__ import annotations

import numpy as np


def sampled_state_plan(
    saved_iterations: np.ndarray,
    oracle_iteration: int,
) -> list[tuple[int, int | None, str]]:
    """Return sorted saved states plus the exact oracle state when necessary."""

    iterations = np.asarray(saved_iterations, dtype=np.int64)
    if iterations.ndim != 1 or iterations.size < 2:
        raise ValueError("The trajectory must contain at least two saved states.")
    if np.any(np.diff(iterations) <= 0):
        raise ValueError("Saved trajectory iterations must be strictly increasing.")
    plan = [(int(value), int(index), "trajectory") for index, value in enumerate(iterations)]
    if oracle_iteration not in set(int(value) for value in iterations):
        plan.append((int(oracle_iteration), None, "oracle_archive"))
        plan.sort(key=lambda item: item[0])
    else:
        plan = [
            (iteration, index, "trajectory_oracle" if iteration == oracle_iteration else source)
            for iteration, index, source in plan
        ]
    return plan


def monotonicity_summary(
    iterations: np.ndarray,
    values: np.ndarray,
    *,
    through_iteration: int,
    relative_tolerance: float = 1.0e-10,
) -> dict[str, float | int | bool]:
    """Assess non-increase at diagnostic states through the oracle iteration."""

    mask = np.asarray(iterations, dtype=np.int64) <= int(through_iteration)
    selected = np.asarray(values, dtype=np.float64)[mask]
    if selected.size < 2:
        raise ValueError("At least two diagnostic states are required for monotonicity.")
    changes = np.diff(selected)
    scale = max(float(np.max(np.abs(selected))), 1.0)
    tolerance = relative_tolerance * scale
    return {
        "states_through_oracle": int(selected.size),
        "maximum_positive_change_through_oracle": float(max(np.max(changes), 0.0)),
        "monotone_nonincreasing_through_oracle": bool(np.all(changes <= tolerance)),
        "monotonicity_absolute_tolerance": float(tolerance),
    }
