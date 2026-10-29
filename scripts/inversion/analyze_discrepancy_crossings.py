#!/usr/bin/env python3
"""Compare empirical emulator-error thresholds with T2 semiconvergence.

This is a retrospective, post-selection diagnostic.  It never changes a
Landweber result and must not be used to reselect the frozen checkpoints.

Two threshold families are evaluated:

1. fixed empirical forward thresholds Q_q(e_F);
2. theory-inspired thresholds

       tau * (delta + 3 Q_q(e_F) + D Q_q(max_j e_J) / (1 + eta)).

For the second family, ``tau=1`` records the unmultiplied effective scale.
The ``theory_min_margin`` policy uses a fixed margin above the strict lower
bound tau > 2(1+eta)/(1-2eta) in the supplied convergence analysis.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CORE_MODELS = 32
QUANTILES = {
    "median": (
        "forward_absolute_l2_error_median",
        "jvp_fixed_direction_envelope_median",
    ),
    "p95": (
        "forward_absolute_l2_error_p95",
        "jvp_fixed_direction_envelope_p95",
    ),
    "p99": (
        "forward_absolute_l2_error_p99",
        "jvp_fixed_direction_envelope_p99",
    ),
    "max": (
        "forward_absolute_l2_error_max",
        "jvp_fixed_direction_envelope_max",
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--test-table",
        type=Path,
        default=PROJECT_ROOT
        / "tables/paper_T2_broad/test/core_paper_models_test.csv",
    )
    parser.add_argument(
        "--inversion-table",
        type=Path,
        default=PROJECT_ROOT
        / "tables/paper_T2_broad/inversion/core_inversion_diagnostics.csv",
    )
    parser.add_argument(
        "--landweber-roots",
        type=Path,
        nargs="+",
        default=[
            PROJECT_ROOT / "results/lw_noiseless_headline_min_u/T2_broad",
            PROJECT_ROOT / "results/lw_noiseless_headline_J0/T2_broad",
            PROJECT_ROOT / "results/lw_noiseless_ablation_J10/T2_broad",
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tables/paper_T2_broad/semiconvergence",
    )
    parser.add_argument(
        "--eta",
        type=float,
        nargs="+",
        default=[0.0, 0.25],
        help="Fixed tangential-cone sensitivity values; each must lie in [0, 0.5).",
    )
    parser.add_argument(
        "--tau-margin",
        type=float,
        default=1.01,
        help="Multiplier above the strict theoretical lower bound for tau.",
    )
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--u-min", type=float, default=0.0)
    parser.add_argument("--u-max", type=float, default=200.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def unique_by_run_id(
    rows: list[dict[str, str]], *, label: str
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        run_id = row.get("run_id", "")
        if not run_id or run_id in result:
            raise ValueError(
                f"{label} contains an empty or duplicate run_id: {run_id!r}."
            )
        result[run_id] = row
    return result


def nonnegative_finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative; found {result}.")
    return result


def theory_tau_lower_bound(eta: float) -> float:
    eta = float(eta)
    if not 0.0 <= eta < 0.5:
        raise ValueError("eta must lie in [0, 0.5).")
    return 2.0 * (1.0 + eta) / (1.0 - 2.0 * eta)


def combined_effective_scale(
    epsilon_f: float,
    epsilon_j: float,
    distance_bound: float,
    eta: float,
    *,
    noise_level: float = 0.0,
) -> float:
    if not 0.0 <= eta < 0.5:
        raise ValueError("eta must lie in [0, 0.5).")
    values = {
        "epsilon_f": epsilon_f,
        "epsilon_j": epsilon_j,
        "distance_bound": distance_bound,
        "noise_level": noise_level,
    }
    checked = {
        name: nonnegative_finite(value, label=name)
        for name, value in values.items()
    }
    return (
        checked["noise_level"]
        + 3.0 * checked["epsilon_f"]
        + checked["distance_bound"] * checked["epsilon_j"] / (1.0 + eta)
    )


def first_crossing_index(residual: np.ndarray, threshold: float) -> int | None:
    residual = np.asarray(residual, dtype=np.float64)
    threshold = nonnegative_finite(threshold, label="threshold")
    crossings = np.flatnonzero(residual <= threshold)
    return int(crossings[0]) if crossings.size else None


def weighted_norm(field: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError(
            f"Weighted-norm shape mismatch: field={values.shape}, weights={weights.shape}."
        )
    return float(np.sqrt(np.sum(weights * values * values)))


def discover_results(roots: list[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.rglob("result.npz"):
            stem = path.parent.name
            if stem in found:
                raise ValueError(
                    f"Duplicate Landweber result directory {stem}: "
                    f"{found[stem]} and {path}."
                )
            found[stem] = path
    return found


def load_trajectory(
    path: Path,
    *,
    expected_kstar: int,
    expected_minimum: float,
    expected_final: float,
    u_min: float,
    u_max: float,
) -> dict[str, Any]:
    required = {
        "history_iteration",
        "history_residual_norm",
        "history_relative_residual",
        "history_relative_u_error",
        "truth_u_resampled_model_grid",
        "data_observed_resampled",
        "noise_effective",
        "observation_mask",
        "observation_weights",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}.")
        iteration = np.asarray(archive["history_iteration"], dtype=np.float64)
        residual = np.asarray(archive["history_residual_norm"], dtype=np.float64)
        relative_residual = np.asarray(
            archive["history_relative_residual"], dtype=np.float64
        )
        u_error = np.asarray(archive["history_relative_u_error"], dtype=np.float64)
        truth = np.asarray(
            archive["truth_u_resampled_model_grid"], dtype=np.float64
        )
        observed = np.asarray(archive["data_observed_resampled"], dtype=np.float64)
        noise = np.asarray(archive["noise_effective"], dtype=np.float64)
        mask = np.asarray(archive["observation_mask"], dtype=bool)
        weights = np.asarray(archive["observation_weights"], dtype=np.float64)

    size = iteration.size
    if size == 0 or any(array.size != size for array in (residual, relative_residual, u_error)):
        raise ValueError(f"{path}: inconsistent or empty history arrays.")
    if not np.all(np.isfinite(iteration)) or not np.array_equal(
        iteration, np.arange(size, dtype=np.float64)
    ):
        raise ValueError(f"{path}: iterations must be contiguous from zero.")
    for name, array in (
        ("residual", residual),
        ("relative residual", relative_residual),
        ("relative u error", u_error),
    ):
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{path}: {name} history is invalid.")
    if truth.shape != weights.shape or observed.shape != weights.shape:
        raise ValueError(f"{path}: stored field/weight shapes disagree.")
    if mask.shape != weights.shape:
        raise ValueError(f"{path}: observation-mask shape disagrees with weights.")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(observed)):
        raise ValueError(f"{path}: truth or observations contain non-finite values.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{path}: invalid quadrature weights.")
    if np.max(np.abs(noise)) > 1.0e-14:
        raise ValueError(f"{path}: expected a noiseless trajectory.")

    data_norm = weighted_norm(np.where(mask, observed, 0.0), weights)
    observation_weight_fraction = float(
        np.sum(weights[mask]) / max(np.sum(weights), 1.0e-300)
    )
    if not np.allclose(
        relative_residual,
        residual / max(data_norm, 1.0e-300),
        rtol=1.0e-10,
        atol=1.0e-14,
    ):
        raise ValueError(f"{path}: absolute and relative residual histories disagree.")

    kstar_index = int(np.argmin(u_error))
    kstar = int(iteration[kstar_index])
    minimum = float(u_error[kstar_index])
    final = float(u_error[-1])
    if kstar != expected_kstar:
        raise ValueError(f"{path}: k*={kstar}, expected {expected_kstar}.")
    if not math.isclose(minimum, expected_minimum, rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError(f"{path}: minimum reconstruction error disagrees with table.")
    if not math.isclose(final, expected_final, rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError(f"{path}: final reconstruction error disagrees with table.")

    truth_norm = weighted_norm(truth, weights)
    observed_distance_bound = float(np.max(u_error) * truth_norm)
    initial_distance = float(u_error[0] * truth_norm)
    box_distance_bound = float((u_max - u_min) * np.sqrt(np.sum(weights)))
    if observed_distance_bound > box_distance_bound * (1.0 + 1.0e-10):
        raise ValueError(f"{path}: observed trajectory exceeds the box-bound diameter.")

    return {
        "iteration": iteration.astype(np.int64),
        "residual": residual,
        "relative_residual": relative_residual,
        "u_error": u_error,
        "data_norm": data_norm,
        "observation_weight_fraction": observation_weight_fraction,
        "truth_norm": truth_norm,
        "kstar_index": kstar_index,
        "kstar": kstar,
        "last_iteration": int(iteration[-1]),
        "minimum_u_error": minimum,
        "final_u_error": final,
        "semiconvergence_ratio": final / max(minimum, 1.0e-300),
        "residual_at_kstar": float(residual[kstar_index]),
        "observed_distance_bound": observed_distance_bound,
        "initial_distance": initial_distance,
        "box_distance_bound": box_distance_bound,
    }


def crossing_record(
    *,
    test_row: dict[str, str],
    trajectory: dict[str, Any],
    threshold_id: str,
    threshold_family: str,
    quantile: str,
    epsilon_f: float,
    epsilon_j: float | None,
    distance_policy: str,
    distance_bound: float | None,
    eta: float | None,
    tau_policy: str,
    tau: float,
    threshold: float,
) -> dict[str, Any]:
    residual = trajectory["residual"]
    u_error = trajectory["u_error"]
    iteration = trajectory["iteration"]
    kstar = int(trajectory["kstar"])
    minimum = float(trajectory["minimum_u_error"])
    crossing = first_crossing_index(residual, threshold)
    if crossing is None:
        crossing_iteration: int | str = ""
        offset: int | str = ""
        relation = "never"
        error_at_crossing: float | str = ""
        penalty_ratio: float | str = ""
        penalty_percent: float | str = ""
    else:
        crossing_iteration = int(iteration[crossing])
        offset = crossing_iteration - kstar
        relation = "before" if offset < 0 else "after" if offset > 0 else "at"
        error_at_crossing = float(u_error[crossing])
        penalty_ratio = error_at_crossing / max(minimum, 1.0e-300)
        penalty_percent = 100.0 * (penalty_ratio - 1.0)

    return {
        "threshold_id": threshold_id,
        "threshold_family": threshold_family,
        "quantile": quantile,
        "distance_policy": distance_policy,
        "distance_bound": "" if distance_bound is None else distance_bound,
        "eta": "" if eta is None else eta,
        "tau_policy": tau_policy,
        "tau": tau,
        "epsilon_F": epsilon_f,
        "epsilon_J": "" if epsilon_j is None else epsilon_j,
        "threshold_absolute": threshold,
        "threshold_relative": threshold / trajectory["data_norm"],
        "emulator_error_norm_scope": "full_grid_trapezoidal_L2",
        "landweber_residual_norm_scope": "interior_masked_trapezoidal_L2",
        "observation_weight_fraction": trajectory["observation_weight_fraction"],
        "architecture": test_row["architecture"],
        "study": test_row["study"],
        "ablation_axis": test_row["ablation_axis"],
        "run_id": test_row["run_id"],
        "checkpoint_epoch": test_row["checkpoint_epoch"],
        "k_star": kstar,
        "last_iteration": trajectory["last_iteration"],
        "minimum_relative_u_error": minimum,
        "final_relative_u_error": trajectory["final_u_error"],
        "semiconvergence_ratio": trajectory["semiconvergence_ratio"],
        "residual_at_kstar_absolute": trajectory["residual_at_kstar"],
        "residual_at_kstar_over_threshold": trajectory["residual_at_kstar"]
        / max(threshold, 1.0e-300),
        "crossing_reached": int(crossing is not None),
        "crossing_relation_to_kstar": relation,
        "crossing_iteration": crossing_iteration,
        "crossing_minus_kstar": offset,
        "relative_u_error_at_crossing": error_at_crossing,
        "reconstruction_penalty_ratio": penalty_ratio,
        "reconstruction_penalty_percent": penalty_percent,
        "data_norm": trajectory["data_norm"],
        "truth_norm": trajectory["truth_norm"],
        "observed_trajectory_distance_bound": trajectory[
            "observed_distance_bound"
        ],
        "initial_distance_to_truth": trajectory["initial_distance"],
        "box_distance_bound": trajectory["box_distance_bound"],
    }


def percentile(values: list[float], probability: float) -> float | str:
    if not values:
        return ""
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def summarize_thresholds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cohort_definitions: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("all", lambda row: True),
        (
            "kstar_before_end",
            lambda row: int(row["k_star"]) < int(row["last_iteration"]),
        ),
        (
            "final_min_excess_gt_1pct",
            lambda row: float(row["semiconvergence_ratio"]) > 1.01,
        ),
    ]
    by_threshold: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_threshold.setdefault(str(row["threshold_id"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for threshold_id, threshold_rows in by_threshold.items():
        reference = threshold_rows[0]
        for cohort_name, predicate in cohort_definitions:
            selected = [row for row in threshold_rows if predicate(row)]
            crossed = [row for row in selected if int(row["crossing_reached"])]
            crossing_iterations = [
                float(row["crossing_iteration"]) for row in crossed
            ]
            offsets = [float(row["crossing_minus_kstar"]) for row in crossed]
            penalties = [
                float(row["reconstruction_penalty_ratio"]) for row in crossed
            ]
            residual_ratios = [
                float(row["residual_at_kstar_over_threshold"])
                for row in selected
            ]
            summaries.append(
                {
                    "threshold_id": threshold_id,
                    "threshold_family": reference["threshold_family"],
                    "quantile": reference["quantile"],
                    "distance_policy": reference["distance_policy"],
                    "eta": reference["eta"],
                    "tau_policy": reference["tau_policy"],
                    "tau": reference["tau"],
                    "emulator_error_norm_scope": reference[
                        "emulator_error_norm_scope"
                    ],
                    "landweber_residual_norm_scope": reference[
                        "landweber_residual_norm_scope"
                    ],
                    "observation_weight_fraction": reference[
                        "observation_weight_fraction"
                    ],
                    "cohort": cohort_name,
                    "n_models": len(selected),
                    "n_crossed": len(crossed),
                    "n_before_kstar": sum(
                        row["crossing_relation_to_kstar"] == "before"
                        for row in selected
                    ),
                    "n_at_kstar": sum(
                        row["crossing_relation_to_kstar"] == "at"
                        for row in selected
                    ),
                    "n_after_kstar": sum(
                        row["crossing_relation_to_kstar"] == "after"
                        for row in selected
                    ),
                    "n_never_crossed": sum(
                        row["crossing_relation_to_kstar"] == "never"
                        for row in selected
                    ),
                    "n_crossing_at_initial": sum(
                        int(row["crossing_reached"])
                        and int(row["crossing_iteration"]) == 0
                        for row in selected
                    ),
                    "crossing_rate": len(crossed) / len(selected) if selected else "",
                    "threshold_absolute_median": percentile(
                        [float(row["threshold_absolute"]) for row in selected], 0.5
                    ),
                    "crossing_iteration_median": percentile(
                        crossing_iterations, 0.5
                    ),
                    "crossing_iteration_p25": percentile(
                        crossing_iterations, 0.25
                    ),
                    "crossing_iteration_p75": percentile(
                        crossing_iterations, 0.75
                    ),
                    "crossing_minus_kstar_median": percentile(offsets, 0.5),
                    "crossing_minus_kstar_p25": percentile(offsets, 0.25),
                    "crossing_minus_kstar_p75": percentile(offsets, 0.75),
                    "reconstruction_penalty_ratio_median": percentile(
                        penalties, 0.5
                    ),
                    "reconstruction_penalty_ratio_p95": percentile(
                        penalties, 0.95
                    ),
                    "reconstruction_penalty_ratio_max": max(penalties)
                    if penalties
                    else "",
                    "residual_at_kstar_over_threshold_median": percentile(
                        residual_ratios, 0.5
                    ),
                    "residual_at_kstar_over_threshold_p25": percentile(
                        residual_ratios, 0.25
                    ),
                    "residual_at_kstar_over_threshold_p75": percentile(
                        residual_ratios, 0.75
                    ),
                    "residual_at_kstar_over_threshold_min": min(residual_ratios)
                    if residual_ratios
                    else "",
                    "residual_at_kstar_over_threshold_max": max(residual_ratios)
                    if residual_ratios
                    else "",
                }
            )
    return summaries


def write_csv(
    path: Path, rows: list[dict[str, Any]], *, overwrite: bool
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace {path}; pass --overwrite.")
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.u_max <= args.u_min:
        raise ValueError("u_max must be greater than u_min.")
    if args.tau_margin <= 1.0:
        raise ValueError("tau_margin must be greater than one.")
    eta_values = [float(value) for value in args.eta]
    if len(set(eta_values)) != len(eta_values):
        raise ValueError("Duplicate eta values were supplied.")
    for eta in eta_values:
        theory_tau_lower_bound(eta)

    test_rows = read_csv(args.test_table.resolve())
    inversion_rows = read_csv(args.inversion_table.resolve())
    tests = unique_by_run_id(test_rows, label="test table")
    inversions = unique_by_run_id(inversion_rows, label="inversion table")
    if len(tests) != EXPECTED_CORE_MODELS or len(inversions) != EXPECTED_CORE_MODELS:
        raise ValueError(
            "Expected exactly 32 core models in each table; found "
            f"test={len(tests)}, inversion={len(inversions)}."
        )
    if tests.keys() != inversions.keys():
        raise ValueError("Test and inversion run sets differ.")

    available = discover_results([path.resolve() for path in args.landweber_roots])
    detailed: list[dict[str, Any]] = []
    for test_row in test_rows:
        run_id = test_row["run_id"]
        epoch = int(float(test_row["checkpoint_epoch"]))
        expected_stem = f"{run_id}_e{epoch}_minu"
        if expected_stem not in available:
            raise FileNotFoundError(
                f"Missing frozen Landweber result directory {expected_stem}."
            )
        inversion = inversions[run_id]
        trajectory = load_trajectory(
            available[expected_stem],
            expected_kstar=int(float(inversion["k_star"])),
            expected_minimum=float(inversion["minimum_relative_u_error"]),
            expected_final=float(inversion["final_relative_u_error"]),
            u_min=float(args.u_min),
            u_max=float(args.u_max),
        )

        for quantile, (forward_field, jvp_field) in QUANTILES.items():
            epsilon_f = nonnegative_finite(
                test_row[forward_field], label=f"{run_id}.{forward_field}"
            )
            epsilon_j = nonnegative_finite(
                test_row[jvp_field], label=f"{run_id}.{jvp_field}"
            )
            detailed.append(
                crossing_record(
                    test_row=test_row,
                    trajectory=trajectory,
                    threshold_id=f"forward_{quantile}",
                    threshold_family="forward_only",
                    quantile=quantile,
                    epsilon_f=epsilon_f,
                    epsilon_j=None,
                    distance_policy="none",
                    distance_bound=None,
                    eta=None,
                    tau_policy="fixed_1",
                    tau=1.0,
                    threshold=epsilon_f,
                )
            )

            distance_policies = {
                "observed_trajectory": trajectory["observed_distance_bound"],
                "box_diameter": trajectory["box_distance_bound"],
            }
            for distance_policy, distance_bound in distance_policies.items():
                for eta in eta_values:
                    effective_scale = combined_effective_scale(
                        epsilon_f,
                        epsilon_j,
                        distance_bound,
                        eta,
                        noise_level=float(args.noise_level),
                    )
                    tau_policies = {
                        "effective_scale": 1.0,
                        "theory_min_margin": args.tau_margin
                        * theory_tau_lower_bound(eta),
                    }
                    eta_label = str(eta).replace(".", "p")
                    for tau_policy, tau in tau_policies.items():
                        threshold_id = (
                            f"combined_{quantile}_{distance_policy}_"
                            f"eta{eta_label}_{tau_policy}"
                        )
                        detailed.append(
                            crossing_record(
                                test_row=test_row,
                                trajectory=trajectory,
                                threshold_id=threshold_id,
                                threshold_family="theory_inspired_combined",
                                quantile=quantile,
                                epsilon_f=epsilon_f,
                                epsilon_j=epsilon_j,
                                distance_policy=distance_policy,
                                distance_bound=distance_bound,
                                eta=eta,
                                tau_policy=tau_policy,
                                tau=tau,
                                threshold=tau * effective_scale,
                            )
                        )

    expected_thresholds_per_model = len(QUANTILES) * (
        1 + 2 * len(eta_values) * 2
    )
    expected_rows = EXPECTED_CORE_MODELS * expected_thresholds_per_model
    if len(detailed) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, built {len(detailed)}.")
    summaries = summarize_thresholds(detailed)

    output_dir = args.output_dir.resolve()
    detail_path = output_dir / "discrepancy_threshold_crossings.csv"
    summary_path = output_dir / "discrepancy_threshold_summary.csv"
    write_csv(detail_path, detailed, overwrite=args.overwrite)
    write_csv(summary_path, summaries, overwrite=args.overwrite)

    observed_bounds = {
        round(float(row["observed_trajectory_distance_bound"]), 12)
        for row in detailed
    }
    box_bounds = {
        round(float(row["box_distance_bound"]), 12) for row in detailed
    }
    print(f"Core models                         : {EXPECTED_CORE_MODELS}")
    print(f"Threshold definitions per model    : {expected_thresholds_per_model}")
    print(f"Detailed model/threshold rows       : {len(detailed)}")
    print(f"Summary rows                        : {len(summaries)}")
    print(f"Observed trajectory distance bounds: {sorted(observed_bounds)}")
    print(f"Box distance bounds                : {sorted(box_bounds)}")
    print(f"Detailed table                      : {detail_path}")
    print(f"Summary table                       : {summary_path}")


if __name__ == "__main__":
    main()
