#!/usr/bin/env python3
"""Summarize opt-in box-projection diagnostics from Landweber reruns.

The Landweber drivers store one diagnostic record for each accepted update.
A projection event means that the *raw* accepted proposal was strictly below
``u_min`` or above ``u_max`` before ``numpy.clip`` was applied.  Bound-active
accepted states and the independent maximum-physical-change cap are reported
separately.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


RESULT_NAMES = (
    "result.npz",
    "landweber_exact_multires_results.npz",
)
PREFIX = "projection_diagnostics_"
MINIMUM_U_ERROR_MAP_PREFIX = (
    PREFIX + "through_minimum_u_error_"
)
MAP_TO_HISTORY = {
    "lower_projection_hit_count_map": "lower_projection_count",
    "upper_projection_hit_count_map": "upper_projection_count",
    "accepted_lower_active_count_map": "accepted_lower_active_count",
    "accepted_upper_active_count_map": "accepted_upper_active_count",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Summarize lower/upper box projection, accepted bound activity, "
            "and step-cap use in diagnostic Landweber reruns."
        ),
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Diagnostic run directories or their result NPZ files.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optionally write the complete summary as CSV.",
    )
    parser.add_argument(
        "--overwrite-csv",
        action="store_true",
        help="Allow replacement of an existing --csv file.",
    )
    return parser.parse_args(argv)


def locate_result(path: Path) -> Path:
    path = path.expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Run path does not exist: {path}")
    matches = [path / name for name in RESULT_NAMES if (path / name).is_file()]
    if len(matches) != 1:
        expected = ", ".join(RESULT_NAMES)
        raise FileNotFoundError(
            f"Expected exactly one of [{expected}] in {path}; found {len(matches)}."
        )
    return matches[0].resolve()


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _metadata(run_dir: Path) -> dict[str, Any]:
    for name in ("headline_selection_metadata.json", "results.json"):
        value = _read_json_mapping(run_dir / name)
        if value:
            if name == "headline_selection_metadata.json":
                return value
            return {"results": value}
    return {}


def _required_vector(
    archive: np.lib.npyio.NpzFile,
    name: str,
    *,
    dtype: Any,
    length: int | None = None,
) -> np.ndarray:
    key = PREFIX + name
    if key not in archive.files:
        raise KeyError(
            f"Result archive lacks {key!r}; was the run made with "
            "--projection-diagnostics?"
        )
    values = np.asarray(archive[key], dtype=dtype)
    if values.ndim != 1:
        raise ValueError(f"{key} must be one-dimensional, got {values.shape}.")
    if length is not None and values.size != length:
        raise ValueError(
            f"{key} has {values.size} entries; expected {length} accepted updates."
        )
    if np.issubdtype(values.dtype, np.floating) and not np.all(np.isfinite(values)):
        raise ValueError(f"{key} contains non-finite values.")
    return values


def _required_map(
    archive: np.lib.npyio.NpzFile,
    name: str,
    *,
    prefix: str = PREFIX,
) -> np.ndarray:
    key = prefix + name
    if key not in archive.files:
        raise KeyError(f"Result archive lacks required map {key!r}.")
    values = np.asarray(archive[key])
    if values.ndim < 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{key} must be a non-empty integer array.")
    if values.size == 0 or np.any(values < 0):
        raise ValueError(f"{key} must be non-empty and non-negative.")
    return values


def _required_integer_scalar(
    archive: np.lib.npyio.NpzFile,
    name: str,
) -> int:
    if name not in archive.files:
        raise KeyError(f"Result archive lacks required scalar {name!r}.")
    values = np.asarray(archive[name])
    if values.size != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must be a scalar integer.")
    value = int(values.reshape(()))
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _first_last(
    update_iterations: np.ndarray,
    mask: np.ndarray,
) -> tuple[int | None, int | None]:
    selected = update_iterations[mask]
    if selected.size == 0:
        return None, None
    return int(selected[0]), int(selected[-1])


def _maximum(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def _field_minimum(archive: np.lib.npyio.NpzFile, key: str) -> float | None:
    if key not in archive.files:
        return None
    values = np.asarray(archive[key], dtype=np.float64)
    return float(np.min(values)) if values.size else None


def _accepted_update_iterations(
    archive: np.lib.npyio.NpzFile,
) -> np.ndarray:
    """Recover accepted-update indices from the legacy Landweber history."""

    required = ("history_iteration", "history_accepted_alpha")
    missing = [name for name in required if name not in archive.files]
    if missing:
        raise KeyError(
            "Result archive lacks legacy update histories: " + ", ".join(missing)
        )
    iterations = np.asarray(archive["history_iteration"], dtype=np.float64)
    accepted_alpha = np.asarray(
        archive["history_accepted_alpha"], dtype=np.float64
    )
    if iterations.ndim != 1 or accepted_alpha.ndim != 1:
        raise ValueError("Legacy iteration and accepted-alpha histories must be 1D.")
    if iterations.shape != accepted_alpha.shape:
        raise ValueError("Legacy iteration and accepted-alpha histories disagree.")
    if not np.all(np.isfinite(iterations)):
        raise ValueError("Legacy iteration history contains non-finite values.")
    accepted_iterations = iterations[np.isfinite(accepted_alpha)]
    if np.any(accepted_iterations != np.floor(accepted_iterations)):
        raise ValueError("Legacy accepted-update iteration labels are not integers.")
    return accepted_iterations.astype(np.int64)


def summarize_result(path: Path) -> dict[str, Any]:
    result_path = locate_result(path)
    run_dir = result_path.parent
    metadata = _metadata(run_dir)
    architecture = metadata.get("architecture")
    run_id = metadata.get("run_id")
    settings = _read_json_mapping(run_dir / "settings.json")
    command_arguments = settings.get("command_arguments", {})
    if not isinstance(command_arguments, dict):
        command_arguments = {}
    results_payload = _read_json_mapping(run_dir / "results.json")
    result_projection = results_payload.get("projection_diagnostics", {})
    if not isinstance(result_projection, dict):
        result_projection = {}
    input_sha256 = result_projection.get("input_sha256", {})
    if not isinstance(input_sha256, dict):
        input_sha256 = {}
    label = (
        f"{architecture}: {run_id}"
        if architecture and run_id
        else run_dir.name
    )

    with np.load(result_path, allow_pickle=False) as archive:
        update = _required_vector(
            archive, "update_iteration", dtype=np.int64
        )
        number_of_updates = int(update.size)
        expected_updates = np.arange(number_of_updates, dtype=np.int64)
        if not np.array_equal(update, expected_updates):
            raise ValueError(
                "Diagnostic update indices must be consecutive from zero in "
                f"{result_path}."
            )
        legacy_accepted_updates = _accepted_update_iterations(archive)
        if not np.array_equal(update, legacy_accepted_updates):
            raise ValueError(
                "Projection diagnostics do not cover exactly every accepted "
                f"Landweber update in {result_path}."
            )
        minimum_u_error_iteration = _required_integer_scalar(
            archive, "minimum_u_error_iteration"
        )
        if minimum_u_error_iteration > number_of_updates:
            raise ValueError(
                "minimum_u_error_iteration exceeds the number of accepted "
                f"updates in {result_path}."
            )
        through_minimum_u_error = update < minimum_u_error_iteration
        selected_update_count = int(np.count_nonzero(through_minimum_u_error))

        lower_count = _required_vector(
            archive,
            "lower_projection_count",
            dtype=np.int64,
            length=number_of_updates,
        )
        upper_count = _required_vector(
            archive,
            "upper_projection_count",
            dtype=np.int64,
            length=number_of_updates,
        )
        lower_fraction = _required_vector(
            archive,
            "lower_projection_fraction",
            dtype=np.float64,
            length=number_of_updates,
        )
        upper_fraction = _required_vector(
            archive,
            "upper_projection_fraction",
            dtype=np.float64,
            length=number_of_updates,
        )
        lower_overshoot = _required_vector(
            archive,
            "lower_projection_max_overshoot",
            dtype=np.float64,
            length=number_of_updates,
        )
        lower_correction_ratio = _required_vector(
            archive,
            "lower_projection_correction_l2_over_raw_step_l2",
            dtype=np.float64,
            length=number_of_updates,
        )
        upper_overshoot = _required_vector(
            archive,
            "upper_projection_max_overshoot",
            dtype=np.float64,
            length=number_of_updates,
        )
        upper_correction_ratio = _required_vector(
            archive,
            "upper_projection_correction_l2_over_raw_step_l2",
            dtype=np.float64,
            length=number_of_updates,
        )
        correction_ratio = _required_vector(
            archive,
            "projection_correction_l2_over_raw_step_l2",
            dtype=np.float64,
            length=number_of_updates,
        )
        correction_l2 = _required_vector(
            archive,
            "projection_correction_l2",
            dtype=np.float64,
            length=number_of_updates,
        )
        correction_max_abs = _required_vector(
            archive,
            "projection_correction_max_abs",
            dtype=np.float64,
            length=number_of_updates,
        )
        lower_active_count = _required_vector(
            archive,
            "accepted_lower_active_count",
            dtype=np.int64,
            length=number_of_updates,
        )
        lower_active_fraction = _required_vector(
            archive,
            "accepted_lower_active_fraction",
            dtype=np.float64,
            length=number_of_updates,
        )
        upper_active_count = _required_vector(
            archive,
            "accepted_upper_active_count",
            dtype=np.int64,
            length=number_of_updates,
        )
        upper_active_fraction = _required_vector(
            archive,
            "accepted_upper_active_fraction",
            dtype=np.float64,
            length=number_of_updates,
        )
        cap_applied = _required_vector(
            archive,
            "physical_change_cap_applied",
            dtype=bool,
            length=number_of_updates,
        )
        backtracking_applied = _required_vector(
            archive,
            "backtracking_applied",
            dtype=bool,
            length=number_of_updates,
        )

        histories_by_name = {
            "lower_projection_count": lower_count,
            "upper_projection_count": upper_count,
            "accepted_lower_active_count": lower_active_count,
            "accepted_upper_active_count": upper_active_count,
        }
        full_maps = {
            name: _required_map(archive, name)
            for name in MAP_TO_HISTORY
        }
        selected_maps = {
            name: _required_map(
                archive,
                name,
                prefix=MINIMUM_U_ERROR_MAP_PREFIX,
            )
            for name in MAP_TO_HISTORY
        }
        map_shapes = {
            values.shape for values in (*full_maps.values(), *selected_maps.values())
        }
        if len(map_shapes) != 1:
            raise ValueError(
                "Full-run and minimum-u-error projection maps have different shapes."
            )
        for map_name, history_name in MAP_TO_HISTORY.items():
            history_values = histories_by_name[history_name]
            full_total = int(np.sum(full_maps[map_name], dtype=np.int64))
            expected_full_total = int(np.sum(history_values, dtype=np.int64))
            if full_total != expected_full_total:
                raise ValueError(
                    f"Full-run map {map_name!r} totals {full_total}, but its "
                    f"step history totals {expected_full_total}."
                )
            selected_total = int(
                np.sum(selected_maps[map_name], dtype=np.int64)
            )
            expected_selected_total = int(
                np.sum(
                    history_values[through_minimum_u_error],
                    dtype=np.int64,
                )
            )
            if selected_total != expected_selected_total:
                raise ValueError(
                    f"Minimum-u-error snapshot map {map_name!r} totals "
                    f"{selected_total}, but the history for updates with "
                    "update_iteration < minimum_u_error_iteration totals "
                    f"{expected_selected_total}."
                )
            if np.any(selected_maps[map_name] > full_maps[map_name]):
                raise ValueError(
                    f"Minimum-u-error snapshot map {map_name!r} exceeds its "
                    "full-run cumulative map."
                )

        lower_map = full_maps["lower_projection_hit_count_map"]
        upper_map = full_maps["upper_projection_hit_count_map"]
        selected_lower_map = selected_maps["lower_projection_hit_count_map"]
        selected_upper_map = selected_maps["upper_projection_hit_count_map"]

        lower_steps = lower_count > 0
        upper_steps = upper_count > 0
        any_steps = lower_steps | upper_steps
        first_lower, last_lower = _first_last(update, lower_steps)
        first_upper, last_upper = _first_last(update, upper_steps)
        projected_ratios = correction_ratio[any_steps]
        grid_points = int(lower_map.size)
        selected_lower_steps = lower_steps & through_minimum_u_error
        selected_upper_steps = upper_steps & through_minimum_u_error
        selected_any_steps = any_steps & through_minimum_u_error
        selected_projected_ratios = correction_ratio[selected_any_steps]

        row: dict[str, Any] = {
            "label": label,
            "result_path": str(result_path),
            "u_min": command_arguments.get("u_min"),
            "u_max": command_arguments.get("u_max"),
            "max_physical_change": command_arguments.get(
                "max_physical_change"
            ),
            "synthetic_file_sha256": input_sha256.get("synthetic_file"),
            "checkpoint_sha256": input_sha256.get("checkpoint"),
            "minimum_u_error_iteration": minimum_u_error_iteration,
            "accepted_updates_through_minimum_u_error": selected_update_count,
            "steps_with_any_projection_through_minimum_u_error": int(
                np.count_nonzero(selected_any_steps)
            ),
            "fraction_steps_with_any_projection_through_minimum_u_error": (
                float(
                    np.count_nonzero(selected_any_steps)
                    / selected_update_count
                )
                if selected_update_count
                else 0.0
            ),
            "steps_with_lower_projection_through_minimum_u_error": int(
                np.count_nonzero(selected_lower_steps)
            ),
            "fraction_steps_with_lower_projection_through_minimum_u_error": (
                float(
                    np.count_nonzero(selected_lower_steps)
                    / selected_update_count
                )
                if selected_update_count
                else 0.0
            ),
            "total_lower_projection_point_hits_through_minimum_u_error": int(
                np.sum(lower_count[through_minimum_u_error], dtype=np.int64)
            ),
            "maximum_lower_projection_fraction_through_minimum_u_error": (
                _maximum(lower_fraction[through_minimum_u_error])
            ),
            "maximum_lower_projection_overshoot_through_minimum_u_error": (
                _maximum(lower_overshoot[through_minimum_u_error])
            ),
            "unique_lower_projection_points_through_minimum_u_error": int(
                np.count_nonzero(selected_lower_map)
            ),
            "unique_lower_projection_point_fraction_through_minimum_u_error": (
                float(np.count_nonzero(selected_lower_map) / grid_points)
            ),
            "steps_with_upper_projection_through_minimum_u_error": int(
                np.count_nonzero(selected_upper_steps)
            ),
            "fraction_steps_with_upper_projection_through_minimum_u_error": (
                float(
                    np.count_nonzero(selected_upper_steps)
                    / selected_update_count
                )
                if selected_update_count
                else 0.0
            ),
            "total_upper_projection_point_hits_through_minimum_u_error": int(
                np.sum(upper_count[through_minimum_u_error], dtype=np.int64)
            ),
            "maximum_upper_projection_fraction_through_minimum_u_error": (
                _maximum(upper_fraction[through_minimum_u_error])
            ),
            "maximum_upper_projection_overshoot_through_minimum_u_error": (
                _maximum(upper_overshoot[through_minimum_u_error])
            ),
            "unique_upper_projection_points_through_minimum_u_error": int(
                np.count_nonzero(selected_upper_map)
            ),
            "unique_upper_projection_point_fraction_through_minimum_u_error": (
                float(np.count_nonzero(selected_upper_map) / grid_points)
            ),
            "mean_projection_correction_ratio_on_projected_updates_through_minimum_u_error": (
                float(np.mean(selected_projected_ratios))
                if selected_projected_ratios.size
                else 0.0
            ),
            "maximum_projection_correction_ratio_through_minimum_u_error": (
                _maximum(correction_ratio[through_minimum_u_error])
            ),
            "maximum_lower_active_fraction_through_minimum_u_error": (
                _maximum(lower_active_fraction[through_minimum_u_error])
            ),
            "maximum_upper_active_fraction_through_minimum_u_error": (
                _maximum(upper_active_fraction[through_minimum_u_error])
            ),
            "unique_lower_active_points_through_minimum_u_error": int(
                np.count_nonzero(
                    selected_maps["accepted_lower_active_count_map"]
                )
            ),
            "unique_upper_active_points_through_minimum_u_error": int(
                np.count_nonzero(
                    selected_maps["accepted_upper_active_count_map"]
                )
            ),
            "physical_change_cap_steps_through_minimum_u_error": int(
                np.count_nonzero(cap_applied[through_minimum_u_error])
            ),
            "physical_change_cap_step_fraction_through_minimum_u_error": (
                float(np.mean(cap_applied[through_minimum_u_error]))
                if selected_update_count
                else 0.0
            ),
            "accepted_updates": number_of_updates,
            "steps_with_any_projection": int(np.count_nonzero(any_steps)),
            "fraction_steps_with_any_projection": (
                float(np.mean(any_steps)) if number_of_updates else 0.0
            ),
            "steps_with_lower_projection": int(np.count_nonzero(lower_steps)),
            "fraction_steps_with_lower_projection": (
                float(np.mean(lower_steps)) if number_of_updates else 0.0
            ),
            "first_lower_projection_update": first_lower,
            "last_lower_projection_update": last_lower,
            "total_lower_projection_point_hits": int(np.sum(lower_count)),
            "maximum_lower_projection_count": (
                int(np.max(lower_count)) if number_of_updates else 0
            ),
            "maximum_lower_projection_fraction": _maximum(lower_fraction),
            "maximum_lower_projection_overshoot": _maximum(lower_overshoot),
            "mean_lower_projection_correction_ratio_on_lower_projected_updates": (
                float(np.mean(lower_correction_ratio[lower_steps]))
                if np.any(lower_steps)
                else 0.0
            ),
            "maximum_lower_projection_correction_ratio": _maximum(
                lower_correction_ratio
            ),
            "unique_lower_projection_points": int(np.count_nonzero(lower_map)),
            "unique_lower_projection_point_fraction": (
                float(np.count_nonzero(lower_map) / grid_points)
            ),
            "steps_with_upper_projection": int(np.count_nonzero(upper_steps)),
            "fraction_steps_with_upper_projection": (
                float(np.mean(upper_steps)) if number_of_updates else 0.0
            ),
            "first_upper_projection_update": first_upper,
            "last_upper_projection_update": last_upper,
            "total_upper_projection_point_hits": int(np.sum(upper_count)),
            "maximum_upper_projection_count": (
                int(np.max(upper_count)) if number_of_updates else 0
            ),
            "maximum_upper_projection_fraction": _maximum(upper_fraction),
            "maximum_upper_projection_overshoot": _maximum(upper_overshoot),
            "mean_upper_projection_correction_ratio_on_upper_projected_updates": (
                float(np.mean(upper_correction_ratio[upper_steps]))
                if np.any(upper_steps)
                else 0.0
            ),
            "maximum_upper_projection_correction_ratio": _maximum(
                upper_correction_ratio
            ),
            "unique_upper_projection_points": int(np.count_nonzero(upper_map)),
            "unique_upper_projection_point_fraction": (
                float(np.count_nonzero(upper_map) / grid_points)
            ),
            "mean_projection_correction_ratio_on_projected_updates": (
                float(np.mean(projected_ratios))
                if projected_ratios.size
                else 0.0
            ),
            "maximum_projection_correction_ratio": _maximum(correction_ratio),
            "maximum_projection_correction_l2": _maximum(correction_l2),
            "maximum_projection_correction_abs": _maximum(correction_max_abs),
            "final_lower_active_count": (
                int(lower_active_count[-1]) if number_of_updates else 0
            ),
            "final_lower_active_fraction": (
                float(lower_active_fraction[-1]) if number_of_updates else 0.0
            ),
            "maximum_lower_active_fraction": _maximum(lower_active_fraction),
            "final_upper_active_count": (
                int(upper_active_count[-1]) if number_of_updates else 0
            ),
            "final_upper_active_fraction": (
                float(upper_active_fraction[-1]) if number_of_updates else 0.0
            ),
            "maximum_upper_active_fraction": _maximum(upper_active_fraction),
            "physical_change_cap_steps": int(np.count_nonzero(cap_applied)),
            "physical_change_cap_step_fraction": (
                float(np.mean(cap_applied)) if number_of_updates else 0.0
            ),
            "backtracking_steps": int(np.count_nonzero(backtracking_applied)),
            "recovered_u_final_min": _field_minimum(
                archive, "recovered_u_final"
            ),
            "recovered_u_minimum_error_field_min": _field_minimum(
                archive, "recovered_u_minimum_u_error"
            ),
        }
    return row


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.3f}%"


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


def print_report(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(row["label"])
        print(f"  result: {row['result_path']}")
        print(
            "  input hashes: "
            f"truth={row['synthetic_file_sha256'] or 'not recorded'}, "
            f"checkpoint={row['checkpoint_sha256'] or 'n/a'}"
        )
        print(
            "  selected minimum-u-error state: "
            f"u_{row['minimum_u_error_iteration']} after "
            f"{row['accepted_updates_through_minimum_u_error']} accepted updates"
        )
        print("  through selected state:")
        print(
            "    any raw box crossing: "
            f"{row['steps_with_any_projection_through_minimum_u_error']}/"
            f"{row['accepted_updates_through_minimum_u_error']} updates "
            f"({_percent(row['fraction_steps_with_any_projection_through_minimum_u_error'])})"
        )
        print(
            "    lower-bound crossings: "
            f"{row['steps_with_lower_projection_through_minimum_u_error']}/"
            f"{row['accepted_updates_through_minimum_u_error']} updates "
            f"({_percent(row['fraction_steps_with_lower_projection_through_minimum_u_error'])}); "
            f"total node hits="
            f"{row['total_lower_projection_point_hits_through_minimum_u_error']}, "
            f"unique nodes="
            f"{row['unique_lower_projection_points_through_minimum_u_error']} "
            f"({_percent(row['unique_lower_projection_point_fraction_through_minimum_u_error'])})"
        )
        print(
            "    upper-bound crossings: "
            f"{row['steps_with_upper_projection_through_minimum_u_error']}/"
            f"{row['accepted_updates_through_minimum_u_error']} updates "
            f"({_percent(row['fraction_steps_with_upper_projection_through_minimum_u_error'])}); "
            f"total node hits="
            f"{row['total_upper_projection_point_hits_through_minimum_u_error']}, "
            f"unique nodes="
            f"{row['unique_upper_projection_points_through_minimum_u_error']} "
            f"({_percent(row['unique_upper_projection_point_fraction_through_minimum_u_error'])})"
        )
        print(
            "    peak crossing fraction/overshoot: "
            "lower="
            f"{_percent(row['maximum_lower_projection_fraction_through_minimum_u_error'])}/"
            f"{_number(row['maximum_lower_projection_overshoot_through_minimum_u_error'])}, "
            "upper="
            f"{_percent(row['maximum_upper_projection_fraction_through_minimum_u_error'])}/"
            f"{_number(row['maximum_upper_projection_overshoot_through_minimum_u_error'])}"
        )
        print(
            "    projection correction/raw-step L2: "
            "mean on projected updates="
            f"{_number(row['mean_projection_correction_ratio_on_projected_updates_through_minimum_u_error'])}, "
            "maximum="
            f"{_number(row['maximum_projection_correction_ratio_through_minimum_u_error'])}"
        )
        print(
            "    accepted bound activity through selected state: "
            "peak lower/upper fractions="
            f"{_percent(row['maximum_lower_active_fraction_through_minimum_u_error'])}/"
            f"{_percent(row['maximum_upper_active_fraction_through_minimum_u_error'])}, "
            "unique lower/upper nodes="
            f"{row['unique_lower_active_points_through_minimum_u_error']}/"
            f"{row['unique_upper_active_points_through_minimum_u_error']}"
        )
        print(
            "    independent max-physical-change cap: "
            f"{row['physical_change_cap_steps_through_minimum_u_error']}/"
            f"{row['accepted_updates_through_minimum_u_error']} updates "
            f"({_percent(row['physical_change_cap_step_fraction_through_minimum_u_error'])})"
        )
        print("  full accepted trajectory:")
        print(
            "    any raw box crossing: "
            f"{row['steps_with_any_projection']}/{row['accepted_updates']} updates "
            f"({_percent(row['fraction_steps_with_any_projection'])})"
        )
        print(
            "    raw lower-bound crossings: "
            f"{row['steps_with_lower_projection']}/{row['accepted_updates']} updates "
            f"({_percent(row['fraction_steps_with_lower_projection'])}); "
            f"first={row['first_lower_projection_update']}, "
            f"last={row['last_lower_projection_update']}"
        )
        print(
            "    lower-bound severity: "
            f"peak nodes={row['maximum_lower_projection_count']} "
            f"({_percent(row['maximum_lower_projection_fraction'])}), "
            f"total node hits={row['total_lower_projection_point_hits']}, "
            f"max undershoot={_number(row['maximum_lower_projection_overshoot'])}, "
            f"unique nodes={row['unique_lower_projection_points']} "
            f"({_percent(row['unique_lower_projection_point_fraction'])})"
        )
        print(
            "    lower correction/raw-step L2: "
            f"mean on lower-projected updates="
            f"{_number(row['mean_lower_projection_correction_ratio_on_lower_projected_updates'])}, "
            f"maximum={_number(row['maximum_lower_projection_correction_ratio'])}"
        )
        print(
            "    accepted lower-bound activity: "
            f"final nodes={row['final_lower_active_count']} "
            f"({_percent(row['final_lower_active_fraction'])}), "
            f"peak fraction={_percent(row['maximum_lower_active_fraction'])}"
        )
        print(
            "    raw upper-bound crossings: "
            f"{row['steps_with_upper_projection']}/{row['accepted_updates']} updates "
            f"({_percent(row['fraction_steps_with_upper_projection'])}); "
            f"first={row['first_upper_projection_update']}, "
            f"last={row['last_upper_projection_update']}"
        )
        print(
            "    upper-bound severity: "
            f"peak nodes={row['maximum_upper_projection_count']} "
            f"({_percent(row['maximum_upper_projection_fraction'])}), "
            f"total node hits={row['total_upper_projection_point_hits']}, "
            f"max overshoot={_number(row['maximum_upper_projection_overshoot'])}, "
            f"unique nodes={row['unique_upper_projection_points']} "
            f"({_percent(row['unique_upper_projection_point_fraction'])})"
        )
        print(
            "    upper correction/raw-step L2: "
            f"mean on upper-projected updates="
            f"{_number(row['mean_upper_projection_correction_ratio_on_upper_projected_updates'])}, "
            f"maximum={_number(row['maximum_upper_projection_correction_ratio'])}"
        )
        print(
            "    accepted upper-bound activity: "
            f"final nodes={row['final_upper_active_count']} "
            f"({_percent(row['final_upper_active_fraction'])}), "
            f"peak fraction={_percent(row['maximum_upper_active_fraction'])}"
        )
        print(
            "    projection correction/raw-step L2: "
            f"mean on projected updates="
            f"{_number(row['mean_projection_correction_ratio_on_projected_updates'])}, "
            f"maximum={_number(row['maximum_projection_correction_ratio'])}"
        )
        print(
            "    independent max-physical-change cap "
            f"({_number(row['max_physical_change'])}): "
            f"{row['physical_change_cap_steps']}/{row['accepted_updates']} updates "
            f"({_percent(row['physical_change_cap_step_fraction'])})"
        )
        print(
            "  recovered minima: "
            f"final={_number(row['recovered_u_final_min'])}, "
            f"minimum-error field="
            f"{_number(row['recovered_u_minimum_error_field_min'])}"
        )
        print()


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    overwrite: bool = False,
) -> None:
    if not rows:
        raise ValueError("No summaries to write.")
    path = path.expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace existing CSV {path}; use --overwrite-csv deliberately."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        rows = [summarize_result(path) for path in args.runs]
        print_report(rows)
        if args.csv is not None:
            write_csv(args.csv, rows, overwrite=bool(args.overwrite_csv))
            print(f"Wrote {args.csv}")
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
