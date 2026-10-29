#!/usr/bin/env python3
"""Join independent test errors to T2 semiconvergence diagnostics.

This is a post-selection diagnostic, not a stopping-rule calibration tool.
It quantifies whether held-out emulator errors co-vary with the observed T2
turnover iteration, reconstruction error, and semiconvergence severity.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CORE_MODELS = 32


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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tables/paper_T2_broad/semiconvergence",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def unique_by_run_id(rows: list[dict[str, str]], *, label: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        run_id = row.get("run_id", "")
        if not run_id or run_id in result:
            raise ValueError(f"{label} contains an empty or duplicate run_id: {run_id!r}.")
        result[run_id] = row
    return result


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = 0.5 * (start + stop - 1)
        for index in order[start:stop]:
            result[index] = rank
        start = stop
    return result


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    x_centered = [value - x_mean for value in x]
    y_centered = [value - y_mean for value in y]
    denominator = math.sqrt(
        sum(value * value for value in x_centered)
        * sum(value * value for value in y_centered)
    )
    if denominator == 0.0:
        return float("nan")
    return sum(a * b for a, b in zip(x_centered, y_centered)) / denominator


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace {path}; pass --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    test_rows = read_csv(args.test_table.resolve())
    inversion_rows = read_csv(args.inversion_table.resolve())
    tests = unique_by_run_id(test_rows, label="test table")
    inversions = unique_by_run_id(inversion_rows, label="inversion table")
    if len(tests) != EXPECTED_CORE_MODELS or len(inversions) != EXPECTED_CORE_MODELS:
        raise ValueError(
            "Expected exactly 32 core models in each input; found "
            f"test={len(tests)}, inversion={len(inversions)}."
        )
    if tests.keys() != inversions.keys():
        missing_test = sorted(inversions.keys() - tests.keys())
        missing_inversion = sorted(tests.keys() - inversions.keys())
        raise ValueError(
            f"Run sets differ; missing test={missing_test}, "
            f"missing inversion={missing_inversion}."
        )

    inversion_fields = [
        "k_star",
        "last_iteration",
        "minimum_relative_u_error",
        "final_relative_u_error",
        "semiconvergence_ratio",
        "surrogate_relative_residual_at_kstar",
        "fem_relative_residual_at_same_field",
        "emulator_relative_error_at_same_field",
    ]
    joined: list[dict[str, Any]] = []
    for test_row in test_rows:
        inversion = inversions[test_row["run_id"]]
        row: dict[str, Any] = dict(test_row)
        row.update({name: inversion[name] for name in inversion_fields})
        k_star = int(float(inversion["k_star"]))
        last = int(float(inversion["last_iteration"]))
        ratio = float(inversion["semiconvergence_ratio"])
        row["k_star_fraction"] = k_star / last
        row["semiconvergence_excess"] = max(0.0, ratio - 1.0)
        row["semiconvergence_observed"] = int(k_star < last)
        joined.append(row)

    joined_fields = list(test_rows[0]) + inversion_fields + [
        "k_star_fraction",
        "semiconvergence_excess",
        "semiconvergence_observed",
    ]
    x_metrics = [
        "E_F_test",
        "E_J_test",
        "S_test",
        "forward_absolute_l2_error_p95",
        "forward_absolute_l2_error_max",
        "jvp_fixed_direction_envelope_p95",
        "jvp_fixed_direction_envelope_max",
    ]
    y_metrics = [
        "k_star_fraction",
        "semiconvergence_excess",
        "minimum_relative_u_error",
        "surrogate_relative_residual_at_kstar",
    ]
    correlations: list[dict[str, Any]] = []
    for x_name in x_metrics:
        x = [float(row[x_name]) for row in joined]
        for y_name in y_metrics:
            y = [float(row[y_name]) for row in joined]
            correlations.append(
                {
                    "test_error_metric": x_name,
                    "T2_outcome": y_name,
                    "n_models": len(joined),
                    "spearman_rank_correlation": pearson(ranks(x), ranks(y)),
                    "pearson_correlation": pearson(x, y),
                }
            )

    output_dir = args.output_dir.resolve()
    write_csv(
        output_dir / "test_error_vs_T2_semiconvergence.csv",
        joined,
        joined_fields,
        overwrite=args.overwrite,
    )
    write_csv(
        output_dir / "test_error_semiconvergence_correlations.csv",
        correlations,
        list(correlations[0]),
        overwrite=args.overwrite,
    )
    print(f"Core models                 : {len(joined)}")
    print(
        "Semiconvergence before end  : "
        f"{sum(int(row['semiconvergence_observed']) for row in joined)}"
    )
    print(f"Joined diagnostics          : {output_dir / 'test_error_vs_T2_semiconvergence.csv'}")
    print(f"Correlation diagnostics     : {output_dir / 'test_error_semiconvergence_correlations.csv'}")


if __name__ == "__main__":
    main()
