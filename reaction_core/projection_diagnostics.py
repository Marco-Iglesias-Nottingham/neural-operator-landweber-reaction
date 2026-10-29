"""Diagnostics for box projection in Landweber-type iterations.

The helpers in this module observe an already-computed update; they do not
construct, alter, accept, or reject a candidate.  This separation keeps the
diagnostic path from changing the numerical algorithm.  All calculations are
NumPy-only so the same definitions can be used by exact-FEM and surrogate
drivers without importing either FEniCSx or PyTorch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TypeAlias

import numpy as np


MetricValue: TypeAlias = float | int
FloatArray: TypeAlias = np.ndarray
BoolArray: TypeAlias = np.ndarray


PROJECTION_METRIC_NAMES = (
    "raw_candidate_min",
    "raw_candidate_max",
    "lower_projection_count",
    "lower_projection_fraction",
    "lower_projection_max_overshoot",
    "lower_projection_correction_l2",
    "lower_projection_correction_l2_over_raw_step_l2",
    "upper_projection_count",
    "upper_projection_fraction",
    "upper_projection_max_overshoot",
    "upper_projection_correction_l2",
    "upper_projection_correction_l2_over_raw_step_l2",
    "raw_step_l2",
    "projected_step_l2",
    "projection_correction_l2",
    "projection_correction_max_abs",
    "projection_correction_l2_over_raw_step_l2",
    "accepted_state_min",
    "accepted_state_max",
    "accepted_lower_active_count",
    "accepted_lower_active_fraction",
    "accepted_upper_active_count",
    "accepted_upper_active_fraction",
)

PROJECTION_COUNT_METRIC_NAMES = frozenset(
    {
        "lower_projection_count",
        "upper_projection_count",
        "accepted_lower_active_count",
        "accepted_upper_active_count",
    }
)

PROJECTION_MAP_NAMES = (
    "lower_projection_hit_count_map",
    "upper_projection_hit_count_map",
    "accepted_lower_active_count_map",
    "accepted_upper_active_count_map",
)


def _finite_scalar(value: float, *, name: str) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must be a finite scalar.")
    try:
        result = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar.")
    return result


def _finite_array(values: np.ndarray, *, name: str) -> FloatArray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numerical array.") from exc
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _validate_bounds(
    lower_bound: float,
    upper_bound: float,
    active_tolerance: float,
    projection_tolerance: float,
) -> tuple[float, float, float, float]:
    lower = _finite_scalar(lower_bound, name="lower_bound")
    upper = _finite_scalar(upper_bound, name="upper_bound")
    active = _finite_scalar(active_tolerance, name="active_tolerance")
    projection = _finite_scalar(
        projection_tolerance,
        name="projection_tolerance",
    )
    if lower >= upper:
        raise ValueError("lower_bound must be strictly smaller than upper_bound.")
    if active < 0.0:
        raise ValueError("active_tolerance must be non-negative.")
    if 2.0 * active >= upper - lower:
        raise ValueError(
            "active_tolerance must be less than half the bound interval."
        )
    if projection < 0.0:
        raise ValueError("projection_tolerance must be non-negative.")
    return lower, upper, active, projection


def _stable_l2(values: FloatArray) -> float:
    absolute = np.abs(np.ravel(values))
    scale = float(np.max(absolute))
    if scale == 0.0:
        return 0.0
    result = scale * float(np.sqrt(np.sum((absolute / scale) ** 2)))
    if not np.isfinite(result):
        raise ValueError("A diagnostic L2 norm is not finite.")
    return result


def _readonly_mask(values: np.ndarray) -> BoolArray:
    result = np.array(values, dtype=bool, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ProjectionStepDiagnostics:
    """Metrics and masks for one accepted projected update.

    A *projection hit* means that the raw proposal lies strictly outside its
    bound.  An *active* point means that the accepted/projected state is on a
    bound to within ``active_tolerance``.  Keeping these concepts separate
    distinguishes clipping caused by the current update from a state that was
    already on, or arrived exactly at, a bound.
    """

    raw_candidate_min: float
    raw_candidate_max: float
    lower_projection_count: int
    lower_projection_fraction: float
    lower_projection_max_overshoot: float
    lower_projection_correction_l2: float
    lower_projection_correction_l2_over_raw_step_l2: float
    upper_projection_count: int
    upper_projection_fraction: float
    upper_projection_max_overshoot: float
    upper_projection_correction_l2: float
    upper_projection_correction_l2_over_raw_step_l2: float
    raw_step_l2: float
    projected_step_l2: float
    projection_correction_l2: float
    projection_correction_max_abs: float
    projection_correction_l2_over_raw_step_l2: float
    accepted_state_min: float
    accepted_state_max: float
    accepted_lower_active_count: int
    accepted_lower_active_fraction: float
    accepted_upper_active_count: int
    accepted_upper_active_fraction: float
    lower_projection_mask: BoolArray
    upper_projection_mask: BoolArray
    accepted_lower_active_mask: BoolArray
    accepted_upper_active_mask: BoolArray

    def as_record(self) -> dict[str, MetricValue]:
        """Return the scalar schema as Python ``int``/``float`` values."""

        return {
            name: getattr(self, name)
            for name in PROJECTION_METRIC_NAMES
        }

    def hit_masks(self) -> dict[str, BoolArray]:
        """Return the four immutable per-step masks under stable names."""

        return {
            "lower_projection_hit_count_map": self.lower_projection_mask,
            "upper_projection_hit_count_map": self.upper_projection_mask,
            "accepted_lower_active_count_map": (
                self.accepted_lower_active_mask
            ),
            "accepted_upper_active_count_map": (
                self.accepted_upper_active_mask
            ),
        }


def compute_projection_step_diagnostics(
    current_u: np.ndarray,
    raw_candidate: np.ndarray,
    projected_candidate: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    *,
    active_tolerance: float = 1.0e-12,
    projection_tolerance: float = 0.0,
) -> ProjectionStepDiagnostics:
    """Measure how box projection changed one accepted proposal.

    ``projected_candidate`` is checked against
    ``clip(raw_candidate, lower_bound, upper_bound)``.  This catches accidental
    instrumentation of a pre-backtracking proposal or a state modified by a
    second operation.  Set ``projection_tolerance`` only when candidates have
    passed through a lower-precision representation.

    The inputs are never written to and the returned masks own immutable
    storage.  Norms are unweighted Euclidean norms over all supplied entries.
    """

    lower, upper, active, projection = _validate_bounds(
        lower_bound,
        upper_bound,
        active_tolerance,
        projection_tolerance,
    )
    current = _finite_array(current_u, name="current_u")
    raw = _finite_array(raw_candidate, name="raw_candidate")
    accepted = _finite_array(projected_candidate, name="projected_candidate")
    if raw.shape != current.shape or accepted.shape != current.shape:
        raise ValueError(
            "current_u, raw_candidate, and projected_candidate must have "
            f"the same shape; got {current.shape}, {raw.shape}, and "
            f"{accepted.shape}."
        )

    if np.any(current < lower) or np.any(current > upper):
        raise ValueError("current_u lies outside the supplied bounds.")
    expected = np.clip(raw, lower, upper)
    if not np.allclose(
        accepted,
        expected,
        rtol=0.0,
        atol=projection,
        equal_nan=False,
    ):
        maximum_error = float(np.max(np.abs(accepted - expected)))
        raise ValueError(
            "projected_candidate is not the box projection of raw_candidate; "
            f"maximum absolute error is {maximum_error:.6e}."
        )
    if np.any(accepted < lower - projection) or np.any(
        accepted > upper + projection
    ):
        raise ValueError("projected_candidate lies outside the supplied bounds.")

    lower_projection = raw < lower
    upper_projection = raw > upper
    accepted_lower_active = accepted <= lower + active
    accepted_upper_active = accepted >= upper - active

    lower_count = int(np.count_nonzero(lower_projection))
    upper_count = int(np.count_nonzero(upper_projection))
    lower_active_count = int(np.count_nonzero(accepted_lower_active))
    upper_active_count = int(np.count_nonzero(accepted_upper_active))
    number_of_points = int(current.size)

    raw_step = raw - current
    projected_step = accepted - current
    correction = accepted - raw
    raw_step_l2 = _stable_l2(raw_step)
    correction_l2 = _stable_l2(correction)
    lower_correction_l2 = _stable_l2(
        np.where(lower_projection, lower - raw, 0.0)
    )
    upper_correction_l2 = _stable_l2(
        np.where(upper_projection, raw - upper, 0.0)
    )
    if raw_step_l2 == 0.0 and correction_l2 != 0.0:
        raise ValueError(
            "A zero raw step has a non-zero projection correction, so the "
            "correction-to-step ratio is undefined."
        )
    correction_ratio = (
        correction_l2 / raw_step_l2 if raw_step_l2 > 0.0 else 0.0
    )
    lower_correction_ratio = (
        lower_correction_l2 / raw_step_l2 if raw_step_l2 > 0.0 else 0.0
    )
    upper_correction_ratio = (
        upper_correction_l2 / raw_step_l2 if raw_step_l2 > 0.0 else 0.0
    )

    lower_overshoot = (
        float(np.max(lower - raw[lower_projection]))
        if lower_count
        else 0.0
    )
    upper_overshoot = (
        float(np.max(raw[upper_projection] - upper))
        if upper_count
        else 0.0
    )

    return ProjectionStepDiagnostics(
        raw_candidate_min=float(np.min(raw)),
        raw_candidate_max=float(np.max(raw)),
        lower_projection_count=lower_count,
        lower_projection_fraction=lower_count / number_of_points,
        lower_projection_max_overshoot=lower_overshoot,
        lower_projection_correction_l2=lower_correction_l2,
        lower_projection_correction_l2_over_raw_step_l2=(
            lower_correction_ratio
        ),
        upper_projection_count=upper_count,
        upper_projection_fraction=upper_count / number_of_points,
        upper_projection_max_overshoot=upper_overshoot,
        upper_projection_correction_l2=upper_correction_l2,
        upper_projection_correction_l2_over_raw_step_l2=(
            upper_correction_ratio
        ),
        raw_step_l2=raw_step_l2,
        projected_step_l2=_stable_l2(projected_step),
        projection_correction_l2=correction_l2,
        projection_correction_max_abs=float(np.max(np.abs(correction))),
        projection_correction_l2_over_raw_step_l2=correction_ratio,
        accepted_state_min=float(np.min(accepted)),
        accepted_state_max=float(np.max(accepted)),
        accepted_lower_active_count=lower_active_count,
        accepted_lower_active_fraction=lower_active_count / number_of_points,
        accepted_upper_active_count=upper_active_count,
        accepted_upper_active_fraction=upper_active_count / number_of_points,
        lower_projection_mask=_readonly_mask(lower_projection),
        upper_projection_mask=_readonly_mask(upper_projection),
        accepted_lower_active_mask=_readonly_mask(accepted_lower_active),
        accepted_upper_active_mask=_readonly_mask(accepted_upper_active),
    )


def _history_arrays(
    history: Mapping[str, Sequence[MetricValue] | np.ndarray],
) -> tuple[dict[str, np.ndarray], int]:
    missing = [name for name in PROJECTION_METRIC_NAMES if name not in history]
    if missing:
        raise ValueError(
            "Projection history is missing metrics: " + ", ".join(missing)
        )
    arrays: dict[str, np.ndarray] = {}
    lengths: set[int] = set()
    for name in PROJECTION_METRIC_NAMES:
        try:
            numeric = np.asarray(history[name], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Projection history metric {name!r} is invalid.") from exc
        if numeric.ndim != 1:
            raise ValueError(f"Projection history metric {name!r} must be 1D.")
        if not np.all(np.isfinite(numeric)):
            raise ValueError(
                f"Projection history metric {name!r} contains non-finite values."
            )
        if name in PROJECTION_COUNT_METRIC_NAMES:
            if np.any(numeric < 0) or np.any(numeric != np.floor(numeric)):
                raise ValueError(
                    f"Projection history count {name!r} must contain "
                    "non-negative integers."
                )
            if np.any(numeric > np.iinfo(np.int64).max):
                raise ValueError(
                    f"Projection history count {name!r} exceeds int64 range."
                )
            array = numeric.astype(np.int64)
        else:
            array = numeric
        arrays[name] = array
        lengths.add(int(array.size))
    if len(lengths) != 1:
        raise ValueError("All projection history metrics must have equal length.")
    return arrays, lengths.pop()


def _validated_maps(
    hit_count_maps: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    missing = [name for name in PROJECTION_MAP_NAMES if name not in hit_count_maps]
    if missing:
        raise ValueError(
            "Projection maps are missing entries: " + ", ".join(missing)
        )
    arrays: dict[str, np.ndarray] = {}
    shapes: set[tuple[int, ...]] = set()
    for name in PROJECTION_MAP_NAMES:
        original = np.asarray(hit_count_maps[name])
        if not np.issubdtype(original.dtype, np.integer):
            raise ValueError(f"Projection map {name!r} must contain integers.")
        array = np.asarray(original, dtype=np.int64)
        if array.size == 0 or np.any(array < 0):
            raise ValueError(
                f"Projection map {name!r} must be non-empty and non-negative."
            )
        arrays[name] = array
        shapes.add(array.shape)
    if len(shapes) != 1:
        raise ValueError("All projection maps must have the same shape.")
    return arrays


def summarize_projection_diagnostics(
    history: Mapping[str, Sequence[MetricValue] | np.ndarray],
    hit_count_maps: Mapping[str, np.ndarray] | None = None,
) -> dict[str, float | int | None]:
    """Return a compact JSON-safe summary of projection history and maps."""

    arrays, steps = _history_arrays(history)
    maps = _validated_maps(hit_count_maps) if hit_count_maps is not None else None

    lower_counts = arrays["lower_projection_count"]
    upper_counts = arrays["upper_projection_count"]
    any_projection = (lower_counts > 0) | (upper_counts > 0)

    def maximum(name: str) -> float | int | None:
        if steps == 0:
            return None
        result = np.max(arrays[name])
        if name in PROJECTION_COUNT_METRIC_NAMES:
            return int(result)
        return float(result)

    def final(name: str) -> float | int | None:
        if steps == 0:
            return None
        result = arrays[name][-1]
        if name in PROJECTION_COUNT_METRIC_NAMES:
            return int(result)
        return float(result)

    summary: dict[str, float | int | None] = {
        "steps_recorded": steps,
        "steps_with_lower_projection": int(np.count_nonzero(lower_counts)),
        "steps_with_upper_projection": int(np.count_nonzero(upper_counts)),
        "steps_with_any_projection": int(np.count_nonzero(any_projection)),
        "fraction_steps_with_lower_projection": (
            float(np.mean(lower_counts > 0)) if steps else 0.0
        ),
        "fraction_steps_with_upper_projection": (
            float(np.mean(upper_counts > 0)) if steps else 0.0
        ),
        "fraction_steps_with_any_projection": (
            float(np.mean(any_projection)) if steps else 0.0
        ),
        "total_lower_projection_point_hits": int(np.sum(lower_counts)),
        "total_upper_projection_point_hits": int(np.sum(upper_counts)),
        "maximum_lower_projection_count": maximum("lower_projection_count"),
        "maximum_upper_projection_count": maximum("upper_projection_count"),
        "maximum_lower_projection_fraction": maximum(
            "lower_projection_fraction"
        ),
        "maximum_upper_projection_fraction": maximum(
            "upper_projection_fraction"
        ),
        "maximum_lower_projection_overshoot": maximum(
            "lower_projection_max_overshoot"
        ),
        "maximum_lower_projection_correction_l2": maximum(
            "lower_projection_correction_l2"
        ),
        "maximum_lower_projection_correction_l2_over_raw_step_l2": maximum(
            "lower_projection_correction_l2_over_raw_step_l2"
        ),
        "maximum_upper_projection_overshoot": maximum(
            "upper_projection_max_overshoot"
        ),
        "maximum_upper_projection_correction_l2": maximum(
            "upper_projection_correction_l2"
        ),
        "maximum_upper_projection_correction_l2_over_raw_step_l2": maximum(
            "upper_projection_correction_l2_over_raw_step_l2"
        ),
        (
            "mean_lower_projection_correction_l2_over_raw_step_l2_"
            "on_lower_projection_steps"
        ): (
            float(
                np.mean(
                    arrays["lower_projection_correction_l2_over_raw_step_l2"]
                    [lower_counts > 0]
                )
            )
            if np.any(lower_counts > 0)
            else 0.0
        ),
        (
            "mean_upper_projection_correction_l2_over_raw_step_l2_"
            "on_upper_projection_steps"
        ): (
            float(
                np.mean(
                    arrays["upper_projection_correction_l2_over_raw_step_l2"]
                    [upper_counts > 0]
                )
            )
            if np.any(upper_counts > 0)
            else 0.0
        ),
        "maximum_projection_correction_l2": maximum(
            "projection_correction_l2"
        ),
        "maximum_projection_correction_max_abs": maximum(
            "projection_correction_max_abs"
        ),
        "maximum_projection_correction_l2_over_raw_step_l2": maximum(
            "projection_correction_l2_over_raw_step_l2"
        ),
        "mean_projection_correction_l2_over_raw_step_l2": (
            float(
                np.mean(
                    arrays["projection_correction_l2_over_raw_step_l2"]
                )
            )
            if steps
            else None
        ),
        "final_accepted_state_min": final("accepted_state_min"),
        "final_accepted_state_max": final("accepted_state_max"),
        "final_accepted_lower_active_count": final(
            "accepted_lower_active_count"
        ),
        "final_accepted_lower_active_fraction": final(
            "accepted_lower_active_fraction"
        ),
        "final_accepted_upper_active_count": final(
            "accepted_upper_active_count"
        ),
        "final_accepted_upper_active_fraction": final(
            "accepted_upper_active_fraction"
        ),
        "maximum_accepted_lower_active_count": maximum(
            "accepted_lower_active_count"
        ),
        "maximum_accepted_lower_active_fraction": maximum(
            "accepted_lower_active_fraction"
        ),
        "maximum_accepted_upper_active_count": maximum(
            "accepted_upper_active_count"
        ),
        "maximum_accepted_upper_active_fraction": maximum(
            "accepted_upper_active_fraction"
        ),
    }

    if maps is not None:
        map_to_history = {
            "lower_projection_hit_count_map": "lower_projection_count",
            "upper_projection_hit_count_map": "upper_projection_count",
            "accepted_lower_active_count_map": "accepted_lower_active_count",
            "accepted_upper_active_count_map": "accepted_upper_active_count",
        }
        for map_name, history_name in map_to_history.items():
            map_total = int(np.sum(maps[map_name], dtype=np.int64))
            history_total = int(np.sum(arrays[history_name], dtype=np.int64))
            if map_total != history_total:
                raise ValueError(
                    f"Projection map {map_name!r} totals {map_total}, but "
                    f"history {history_name!r} totals {history_total}."
                )
        number_of_points = int(next(iter(maps.values())).size)
        unique_lower_projection = int(
            np.count_nonzero(maps["lower_projection_hit_count_map"])
        )
        unique_upper_projection = int(
            np.count_nonzero(maps["upper_projection_hit_count_map"])
        )
        unique_any_projection = int(
            np.count_nonzero(
                (maps["lower_projection_hit_count_map"] > 0)
                | (maps["upper_projection_hit_count_map"] > 0)
            )
        )
        unique_lower_active = int(
            np.count_nonzero(maps["accepted_lower_active_count_map"])
        )
        unique_upper_active = int(
            np.count_nonzero(maps["accepted_upper_active_count_map"])
        )
        summary.update(
            {
                "unique_lower_projection_points": unique_lower_projection,
                "unique_upper_projection_points": unique_upper_projection,
                "unique_any_projection_points": unique_any_projection,
                "fraction_points_with_lower_projection": (
                    unique_lower_projection / number_of_points
                ),
                "fraction_points_with_upper_projection": (
                    unique_upper_projection / number_of_points
                ),
                "fraction_points_with_any_projection": (
                    unique_any_projection / number_of_points
                ),
                "unique_accepted_lower_active_points": unique_lower_active,
                "unique_accepted_upper_active_points": unique_upper_active,
                "fraction_points_ever_accepted_lower_active": (
                    unique_lower_active / number_of_points
                ),
                "fraction_points_ever_accepted_upper_active": (
                    unique_upper_active / number_of_points
                ),
                "maximum_lower_projection_hits_at_one_point": int(
                    np.max(maps["lower_projection_hit_count_map"])
                ),
                "maximum_upper_projection_hits_at_one_point": int(
                    np.max(maps["upper_projection_hit_count_map"])
                ),
                "maximum_accepted_lower_active_steps_at_one_point": int(
                    np.max(maps["accepted_lower_active_count_map"])
                ),
                "maximum_accepted_upper_active_steps_at_one_point": int(
                    np.max(maps["accepted_upper_active_count_map"])
                ),
            }
        )
    return summary


class ProjectionDiagnostics:
    """Online history and hit-map accumulator for accepted projections."""

    def __init__(
        self,
        shape: Sequence[int],
        lower_bound: float,
        upper_bound: float,
        *,
        active_tolerance: float = 1.0e-12,
        projection_tolerance: float = 0.0,
    ) -> None:
        try:
            original_shape = tuple(shape)
        except TypeError as exc:
            raise ValueError("shape must be a non-empty sequence of integers.") from exc
        if (
            not original_shape
            or any(
                isinstance(entry, (bool, np.bool_))
                or not isinstance(entry, Integral)
                or entry <= 0
                for entry in original_shape
            )
        ):
            raise ValueError("shape must be a non-empty sequence of positive integers.")
        shape_tuple = tuple(int(entry) for entry in original_shape)
        lower, upper, active, projection = _validate_bounds(
            lower_bound,
            upper_bound,
            active_tolerance,
            projection_tolerance,
        )
        self.shape = shape_tuple
        self.lower_bound = lower
        self.upper_bound = upper
        self.active_tolerance = active
        self.projection_tolerance = projection
        self._history: dict[str, list[MetricValue]] = {
            name: [] for name in PROJECTION_METRIC_NAMES
        }
        self._maps: dict[str, np.ndarray] = {
            name: np.zeros(self.shape, dtype=np.int64)
            for name in PROJECTION_MAP_NAMES
        }

    @property
    def steps_recorded(self) -> int:
        """Number of accepted proposals recorded so far."""

        return len(self._history["raw_candidate_min"])

    def record(
        self,
        current_u: np.ndarray,
        raw_candidate: np.ndarray,
        projected_candidate: np.ndarray,
    ) -> ProjectionStepDiagnostics:
        """Diagnose and append one accepted proposal, returning its details."""

        step = compute_projection_step_diagnostics(
            current_u,
            raw_candidate,
            projected_candidate,
            self.lower_bound,
            self.upper_bound,
            active_tolerance=self.active_tolerance,
            projection_tolerance=self.projection_tolerance,
        )
        if step.lower_projection_mask.shape != self.shape:
            raise ValueError(
                f"Recorded arrays must have shape {self.shape}; got "
                f"{step.lower_projection_mask.shape}."
            )
        for name, value in step.as_record().items():
            self._history[name].append(value)
        for name, mask in step.hit_masks().items():
            self._maps[name] += mask
        return step

    def history_arrays(self) -> dict[str, np.ndarray]:
        """Return independent immutable arrays for every scalar metric."""

        output: dict[str, np.ndarray] = {}
        for name in PROJECTION_METRIC_NAMES:
            dtype = (
                np.int64
                if name in PROJECTION_COUNT_METRIC_NAMES
                else np.float64
            )
            array = np.asarray(self._history[name], dtype=dtype)
            array.setflags(write=False)
            output[name] = array
        return output

    def hit_count_arrays(self) -> dict[str, np.ndarray]:
        """Return independent immutable cumulative spatial count maps."""

        output: dict[str, np.ndarray] = {}
        for name in PROJECTION_MAP_NAMES:
            array = self._maps[name].copy()
            array.setflags(write=False)
            output[name] = array
        return output

    def summary(self) -> dict[str, float | int | None]:
        """Return a JSON-safe summary of the accumulated diagnostics."""

        summary = summarize_projection_diagnostics(
            self._history,
            self._maps,
        )
        summary.update(
            {
                "lower_bound": self.lower_bound,
                "upper_bound": self.upper_bound,
                "active_tolerance": self.active_tolerance,
                "projection_validation_tolerance": self.projection_tolerance,
            }
        )
        return summary


__all__ = [
    "PROJECTION_COUNT_METRIC_NAMES",
    "PROJECTION_MAP_NAMES",
    "PROJECTION_METRIC_NAMES",
    "ProjectionDiagnostics",
    "ProjectionStepDiagnostics",
    "compute_projection_step_diagnostics",
    "summarize_projection_diagnostics",
]
