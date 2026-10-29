"""Retain inverse iterates immediately around residual-threshold crossings.

The tracker is observational: it never changes an iteration or requests that
the caller stop.  It keeps references only to the most recently observed
state, so enabling it has constant memory cost until a threshold is crossed.
At a first crossing it copies the state immediately before the crossing and
the first state at or below the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class _ObservedState:
    iteration: int
    parameter: np.ndarray
    prediction: np.ndarray
    residual_norm: float
    relative_u_error: float


class DiscrepancyCrossingSnapshots:
    """Record the first crossing of each fixed positive residual threshold."""

    def __init__(self, thresholds: Mapping[str, float]) -> None:
        if not thresholds:
            raise ValueError("At least one crossing threshold is required.")
        self.thresholds: dict[str, float] = {}
        for name, value in thresholds.items():
            if not name or not name.replace("_", "").isalnum():
                raise ValueError(f"Unsafe crossing-threshold name: {name!r}.")
            threshold = float(value)
            if not np.isfinite(threshold) or threshold < 0.0:
                raise ValueError(
                    f"Crossing threshold {name!r} must be finite and non-negative."
                )
            self.thresholds[name] = threshold
        self._previous: _ObservedState | None = None
        self._crossings: dict[
            str, tuple[_ObservedState | None, _ObservedState]
        ] = {}

    @staticmethod
    def _state(
        iteration: int,
        parameter: np.ndarray,
        prediction: np.ndarray,
        residual_norm: float,
        relative_u_error: float,
        *,
        copy: bool,
    ) -> _ObservedState:
        parameter_array = np.asarray(parameter, dtype=np.float64)
        prediction_array = np.asarray(prediction, dtype=np.float64)
        if parameter_array.ndim != 2 or prediction_array.ndim != 2:
            raise ValueError("Crossing snapshots require two-dimensional fields.")
        # Full-field validation is deliberately limited to the few states that
        # are actually retained. Normal observation therefore remains O(1)
        # per iteration rather than scanning both grids at every step.
        if copy and not np.all(np.isfinite(parameter_array)):
            raise ValueError("Crossing-snapshot parameter field is not finite.")
        if copy and not np.all(np.isfinite(prediction_array)):
            raise ValueError("Crossing-snapshot prediction field is not finite.")
        residual = float(residual_norm)
        u_error = float(relative_u_error)
        if not np.isfinite(residual) or residual < 0.0:
            raise ValueError("Crossing-snapshot residual norm is invalid.")
        if not np.isfinite(u_error) or u_error < 0.0:
            raise ValueError("Crossing-snapshot relative u error is invalid.")
        return _ObservedState(
            iteration=int(iteration),
            parameter=(parameter_array.copy() if copy else parameter_array),
            prediction=(prediction_array.copy() if copy else prediction_array),
            residual_norm=residual,
            relative_u_error=u_error,
        )

    def observe(
        self,
        *,
        iteration: int,
        parameter: np.ndarray,
        prediction: np.ndarray,
        residual_norm: float,
        relative_u_error: float,
    ) -> None:
        """Observe one state without changing it or the caller's iteration."""

        current = self._state(
            iteration,
            parameter,
            prediction,
            residual_norm,
            relative_u_error,
            copy=False,
        )
        if self._previous is not None and current.iteration <= self._previous.iteration:
            raise ValueError("Crossing-snapshot iterations must be strictly increasing.")

        for name, threshold in self.thresholds.items():
            if name in self._crossings or threshold <= 0.0:
                continue
            if current.residual_norm <= threshold:
                before = self._previous
                if before is not None:
                    before = self._state(
                        before.iteration,
                        before.parameter,
                        before.prediction,
                        before.residual_norm,
                        before.relative_u_error,
                        copy=True,
                    )
                at = self._state(
                    current.iteration,
                    current.parameter,
                    current.prediction,
                    current.residual_norm,
                    current.relative_u_error,
                    copy=True,
                )
                self._crossings[name] = (before, at)

        # The drivers replace, rather than mutate, current_u and prediction
        # after every accepted update. Keeping references therefore avoids two
        # full-grid copies per iteration while preserving the previous state.
        self._previous = current

    @staticmethod
    def _state_arrays(
        prefix: str,
        state: _ObservedState | None,
    ) -> dict[str, np.ndarray]:
        if state is None:
            return {
                prefix + "iteration": np.asarray(-1, dtype=np.int64),
                prefix + "residual_norm": np.asarray(np.nan, dtype=np.float64),
                prefix + "relative_u_error": np.asarray(np.nan, dtype=np.float64),
                prefix + "parameter": np.asarray([], dtype=np.float64),
                prefix + "prediction": np.asarray([], dtype=np.float64),
            }
        return {
            prefix + "iteration": np.asarray(state.iteration, dtype=np.int64),
            prefix + "residual_norm": np.asarray(
                state.residual_norm, dtype=np.float64
            ),
            prefix + "relative_u_error": np.asarray(
                state.relative_u_error, dtype=np.float64
            ),
            prefix + "parameter": np.asarray(state.parameter, dtype=np.float64),
            prefix + "prediction": np.asarray(state.prediction, dtype=np.float64),
        }

    def archive_arrays(
        self,
        *,
        prefix: str = "discrepancy_crossing_",
    ) -> dict[str, np.ndarray]:
        """Return consistently named arrays for an NPZ result archive."""

        arrays: dict[str, np.ndarray] = {}
        for name, threshold in self.thresholds.items():
            base = prefix + name + "_"
            crossing = self._crossings.get(name)
            before, at = (None, None) if crossing is None else crossing
            arrays[base + "threshold"] = np.asarray(threshold, dtype=np.float64)
            arrays[base + "reached"] = np.asarray(
                crossing is not None, dtype=np.bool_
            )
            arrays.update(self._state_arrays(base + "before_", before))
            arrays.update(self._state_arrays(base + "at_", at))
        return arrays

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready description of all threshold crossings."""

        result: dict[str, Any] = {
            "state_definition": (
                "before is the preceding stored Landweber state; at is the "
                "first state whose residual norm is <= the fixed threshold"
            ),
            "thresholds": {},
        }
        for name, threshold in self.thresholds.items():
            crossing = self._crossings.get(name)
            before, at = (None, None) if crossing is None else crossing
            result["thresholds"][name] = {
                "threshold": threshold,
                "reached": crossing is not None,
                "before_iteration": None if before is None else before.iteration,
                "before_residual_norm": (
                    None if before is None else before.residual_norm
                ),
                "before_relative_u_error": (
                    None if before is None else before.relative_u_error
                ),
                "at_iteration": None if at is None else at.iteration,
                "at_residual_norm": None if at is None else at.residual_norm,
                "at_relative_u_error": (
                    None if at is None else at.relative_u_error
                ),
            }
        return result
