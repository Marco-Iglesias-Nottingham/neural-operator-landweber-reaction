#!/usr/bin/env python3
"""Audit and collect the frozen ``paper_T2_broad`` final-test campaign.

The collector deliberately treats the three Landweber registries as the
scientific source of truth.  It accepts an evaluation only when the result
uses the exact registered checkpoint and the reserved four-shard final-test
pool.  Test scores are reported; they are never used to select a checkpoint.

The supported set contains 32 models: four optimized headline models, four
no-DINO headline models, and 24 one-axis capacity/training-size ablations.
Nine completed cross-grid checkpoints remain in historical registries but are
deliberately excluded from this collector.

Partial collections are useful while jobs are running.  Pass
``--require-complete`` when producing paper tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_NAMES = (
    "headline_opt.tsv",
    "headline_J0.tsv",
    "ablations_J10.tsv",
)
EXPECTED_REGISTRY_COUNTS = {
    "headline_opt.tsv": {4},
    "headline_J0.tsv": {4},
    # Accept the historical 33-row file as an input provenance format; main()
    # filters it to the same 24 supported one-axis rows as the current registry.
    "ablations_J10.tsv": {24, 33},
}
ARCHITECTURE_ORDER = {
    "FNO": 0,
    "DeepONet": 1,
    "GenFNO": 2,
    "GenDeepONet": 3,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    base = PROJECT_ROOT / "results/paper_T2_broad"
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=base / "budget2500_validation/landweber_registries",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=base / "final_test_evaluation_theory_diagnostics/raw",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tables/paper_T2_broad/test",
    )
    parser.add_argument("--expected-base-seed", type=int, default=3240000)
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--expected-samples-per-shard", type=int, default=625)
    parser.add_argument("--expected-jvp-samples", type=int, default=256)
    parser.add_argument("--expected-modes", type=int, default=16)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every result in the requested paper set is present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated CSV files.",
    )
    return parser.parse_args(argv)


def read_delimited(path: Path, *, delimiter: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def read_registries(directory: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in REGISTRY_NAMES:
        path = directory / name
        current = read_delimited(path, delimiter="\t")
        expected = EXPECTED_REGISTRY_COUNTS[name]
        if len(current) not in expected:
            raise ValueError(
                f"{path}: expected one of {sorted(expected)} rows, "
                f"found {len(current)}."
            )
        for row in current:
            row = dict(row)
            row["registry"] = name.removesuffix(".tsv")
            rows.append(row)

    run_ids = [row["run_id"] for row in rows]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Frozen registries contain duplicate run_id values.")
    checkpoints = [Path(row["checkpoint"]).name for row in rows]
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Frozen registries contain duplicate checkpoints.")
    return rows


def is_core_paper_row(row: dict[str, str]) -> bool:
    """Accept current registries and filter the historical 33-row format."""
    if row["registry"] != "ablations_J10":
        return True
    axes = set(row["ablation_axis"].split("+"))
    return bool(axes & {"capacity", "training_size"})


def output_stem(row: dict[str, str]) -> str:
    epoch = int(float(row["checkpoint_epoch"]))
    return f"{row['registry']}__{row['run_id']}__e{epoch}"


def _integer(value: Any, *, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid integer for {label}: {value!r}.") from error


def _nonnegative_finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid floating-point value for {label}: {value!r}.") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative; found {result}.")
    return result


def validate_error_summary(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is missing or is not an object.")
    summary = {
        name: _nonnegative_finite(value.get(name), label=f"{label}.{name}")
        for name in ("mean", "rms", "median", "p95", "p99", "max")
    }
    if not (
        summary["median"] <= summary["p95"]
        <= summary["p99"]
        <= summary["max"]
    ):
        raise ValueError(f"{label} quantiles are not ordered.")
    return summary


def summarize_recorded_values(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty diagnostic record.")
    ordered = sorted(values)

    def quantile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "mean": math.fsum(values) / len(values),
        "rms": math.sqrt(math.fsum(value * value for value in values) / len(values)),
        "median": quantile(0.5),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": ordered[-1],
    }


def assert_summary_matches_records(
    summary: dict[str, float], values: list[float], *, label: str
) -> None:
    recomputed = summarize_recorded_values(values)
    for name, expected in recomputed.items():
        if not math.isclose(
            summary[name], expected, rel_tol=1.0e-10, abs_tol=1.0e-15
        ):
            raise ValueError(
                f"{label}.{name}={summary[name]} disagrees with "
                f"per-sample records ({expected})."
            )


def validate_test_payload(
    row: dict[str, str],
    payload: dict[str, Any],
    *,
    expected_base_seed: int,
    expected_shards: int,
    expected_samples_per_shard: int,
    expected_jvp_samples: int,
    expected_modes: int,
) -> dict[str, Any]:
    if payload.get("format") != "reaction-full-grid-checkpoint-evaluation-v1":
        raise ValueError(f"{row['run_id']}: unexpected evaluation format.")
    if payload.get("error_diagnostics") != (
        "absolute-and-fixed-direction-normalized-v1"
    ):
        raise ValueError(
            f"{row['run_id']}: missing theory-aligned error diagnostics."
        )
    if payload.get("label") != "test":
        raise ValueError(f"{row['run_id']}: result is not labelled as test.")

    audit = payload.get("shard_audit")
    if not isinstance(audit, dict):
        raise ValueError(f"{row['run_id']}: missing shard audit.")
    expected_total = expected_shards * expected_samples_per_shard
    if _integer(audit.get("n_shards"), label="n_shards") != expected_shards:
        raise ValueError(f"{row['run_id']}: wrong number of final-test shards.")
    if _integer(audit.get("n_samples"), label="n_samples") != expected_total:
        raise ValueError(f"{row['run_id']}: wrong final-test sample count.")
    sample_counts = [_integer(v, label="sample_counts") for v in audit.get("sample_counts", [])]
    if sample_counts != [expected_samples_per_shard] * expected_shards:
        raise ValueError(f"{row['run_id']}: unexpected per-shard sample counts.")

    shard_names = [Path(value).name for value in payload.get("shards", [])]
    expected_names = [f"reaction_shard_{index:03d}.h5" for index in range(expected_shards)]
    if shard_names != expected_names:
        raise ValueError(
            f"{row['run_id']}: expected final-test shards {expected_names}, "
            f"found {shard_names}."
        )

    records = sorted(
        audit.get("seed_records", []), key=lambda value: int(value["sample_offset"])
    )
    if len(records) != expected_shards:
        raise ValueError(f"{row['run_id']}: incomplete seed provenance.")
    for index, record in enumerate(records):
        offset = index * expected_samples_per_shard
        expected = {
            "base_seed": expected_base_seed,
            "sample_offset": offset,
            "first_seed": expected_base_seed + offset,
            "last_seed": expected_base_seed + offset + expected_samples_per_shard - 1,
        }
        actual = {key: _integer(record.get(key), label=key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"{row['run_id']}: wrong seed record {index}: {actual}; "
                f"expected {expected}."
            )

    forward_indices = [_integer(v, label="forward_indices") for v in payload.get("forward_indices", [])]
    if forward_indices != list(range(expected_total)):
        raise ValueError(
            f"{row['run_id']}: final-test E_F was not evaluated on all samples."
        )
    jvp_indices = [_integer(v, label="jvp_indices") for v in payload.get("jvp_indices", [])]
    if len(jvp_indices) != expected_jvp_samples or len(set(jvp_indices)) != len(jvp_indices):
        raise ValueError(f"{row['run_id']}: unexpected JVP sample subset.")
    if any(value < 0 or value >= expected_total for value in jvp_indices):
        raise ValueError(f"{row['run_id']}: JVP sample index is out of range.")

    results = payload.get("results_sorted_by_joint_score")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"{row['run_id']}: expected exactly one checkpoint result.")
    result = results[0]
    checkpoint_name = Path(str(result.get("checkpoint", ""))).name
    registered_name = Path(row["checkpoint"]).name
    if checkpoint_name != registered_name:
        raise ValueError(
            f"{row['run_id']}: evaluated {checkpoint_name}, expected {registered_name}."
        )
    expected_epoch = int(float(row["checkpoint_epoch"]))
    if _integer(result.get("epoch"), label="epoch") != expected_epoch:
        raise ValueError(f"{row['run_id']}: evaluated checkpoint epoch is wrong.")
    modes = [_integer(v, label="jvp_modes") for v in result.get("jvp_modes", [])]
    if modes != list(range(expected_modes)):
        raise ValueError(f"{row['run_id']}: E_J does not use all expected modes.")
    if _integer(result.get("n_forward_samples"), label="n_forward_samples") != expected_total:
        raise ValueError(f"{row['run_id']}: result reports the wrong E_F sample count.")
    if _integer(result.get("n_jvp_samples"), label="n_jvp_samples") != expected_jvp_samples:
        raise ValueError(f"{row['run_id']}: result reports the wrong E_J sample count.")

    ef = float(result["forward"]["pooled_relative_l2"])
    ej = float(result["jvp"]["pooled_relative_l2"])
    score = float(result["joint_score"])
    if not math.isclose(score, math.hypot(ef, ej), rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError(f"{row['run_id']}: inconsistent final-test joint score.")

    forward_absolute = validate_error_summary(
        result["forward"].get("absolute_l2_error"),
        label=f"{row['run_id']}.forward.absolute_l2_error",
    )
    jvp_unit = validate_error_summary(
        result["jvp"].get("error_per_unit_direction"),
        label=f"{row['run_id']}.jvp.error_per_unit_direction",
    )
    jvp_envelope = validate_error_summary(
        result["jvp"].get("fixed_direction_envelope_per_input"),
        label=f"{row['run_id']}.jvp.fixed_direction_envelope_per_input",
    )
    if not math.isclose(
        jvp_unit["max"], jvp_envelope["max"], rel_tol=1.0e-12, abs_tol=1.0e-15
    ):
        raise ValueError(
            f"{row['run_id']}: inconsistent fixed-direction maximum error."
        )

    forward_records = result.get("per_sample_forward")
    if not isinstance(forward_records, list) or len(forward_records) != expected_total:
        raise ValueError(
            f"{row['run_id']}: expected {expected_total} per-sample forward records."
        )
    if [_integer(item.get("index"), label="forward record index") for item in forward_records] != forward_indices:
        raise ValueError(f"{row['run_id']}: forward diagnostic indices are inconsistent.")
    recorded_forward_absolute: list[float] = []
    for item in forward_records:
        recorded_forward_absolute.append(
            _nonnegative_finite(
                item.get("absolute_l2_error"), label="forward absolute error"
            )
        )
        _nonnegative_finite(item.get("relative_l2_error"), label="forward relative error")
    assert_summary_matches_records(
        forward_absolute,
        recorded_forward_absolute,
        label=f"{row['run_id']}.forward.absolute_l2_error",
    )

    jvp_records = result.get("per_sample_jvp")
    expected_jvp_record_count = expected_jvp_samples * expected_modes
    if not isinstance(jvp_records, list) or len(jvp_records) != expected_jvp_record_count:
        raise ValueError(
            f"{row['run_id']}: expected {expected_jvp_record_count} per-direction JVP records."
        )
    expected_pairs = [
        (sample_index, mode)
        for sample_index in jvp_indices
        for mode in range(expected_modes)
    ]
    actual_pairs = [
        (
            _integer(item.get("index"), label="JVP record index"),
            _integer(item.get("mode"), label="JVP record mode"),
        )
        for item in jvp_records
    ]
    if actual_pairs != expected_pairs:
        raise ValueError(f"{row['run_id']}: JVP diagnostic index/mode pairs are inconsistent.")
    recorded_unit_errors: list[float] = []
    for item in jvp_records:
        _nonnegative_finite(item.get("absolute_l2_error"), label="JVP absolute error")
        direction_norm = _nonnegative_finite(
            item.get("direction_l2_norm"), label="JVP direction norm"
        )
        if direction_norm == 0.0:
            raise ValueError(f"{row['run_id']}: a stored JVP direction has zero norm.")
        recorded_unit_errors.append(
            _nonnegative_finite(
                item.get("error_per_unit_direction"),
                label="JVP unit-direction error",
            )
        )
        _nonnegative_finite(item.get("relative_l2_error"), label="JVP relative error")
    assert_summary_matches_records(
        jvp_unit,
        recorded_unit_errors,
        label=f"{row['run_id']}.jvp.error_per_unit_direction",
    )
    recorded_envelopes = [
        max(recorded_unit_errors[start : start + expected_modes])
        for start in range(0, len(recorded_unit_errors), expected_modes)
    ]
    assert_summary_matches_records(
        jvp_envelope,
        recorded_envelopes,
        label=f"{row['run_id']}.jvp.fixed_direction_envelope_per_input",
    )

    return {
        "E_F_test": ef,
        "E_J_test": ej,
        "S_test": score,
        "E_F_test_mean": float(result["forward"]["mean"]),
        "E_F_test_p95": float(result["forward"]["p95"]),
        "E_J_test_mean": float(result["jvp"]["mean"]),
        "E_J_test_p95": float(result["jvp"]["p95"]),
        "forward_absolute_l2_error_median": forward_absolute["median"],
        "forward_absolute_l2_error_p95": forward_absolute["p95"],
        "forward_absolute_l2_error_p99": forward_absolute["p99"],
        "forward_absolute_l2_error_max": forward_absolute["max"],
        "jvp_unit_direction_error_median": jvp_unit["median"],
        "jvp_unit_direction_error_p95": jvp_unit["p95"],
        "jvp_unit_direction_error_p99": jvp_unit["p99"],
        "jvp_unit_direction_error_max": jvp_unit["max"],
        "jvp_fixed_direction_envelope_median": jvp_envelope["median"],
        "jvp_fixed_direction_envelope_p95": jvp_envelope["p95"],
        "jvp_fixed_direction_envelope_p99": jvp_envelope["p99"],
        "jvp_fixed_direction_envelope_max": jvp_envelope["max"],
        "test_seconds": float(result["seconds"]),
        "test_subset_seed": _integer(payload.get("subset_seed"), label="subset_seed"),
        "n_forward_test": expected_total,
        "n_jvp_test": len(jvp_indices),
        "test_jvp_indices": jvp_indices,
    }


OUTPUT_FIELDS = [
    "paper_set",
    "study",
    "ablation_axis",
    "architecture",
    "train_samples",
    "capacity",
    "lambda_J",
    "checkpoint_epoch",
    "E_F_val",
    "E_J_val",
    "S_val",
    "E_F_test",
    "E_J_test",
    "S_test",
    "E_F_test_mean",
    "E_F_test_p95",
    "E_J_test_mean",
    "E_J_test_p95",
    "forward_absolute_l2_error_median",
    "forward_absolute_l2_error_p95",
    "forward_absolute_l2_error_p99",
    "forward_absolute_l2_error_max",
    "jvp_unit_direction_error_median",
    "jvp_unit_direction_error_p95",
    "jvp_unit_direction_error_p99",
    "jvp_unit_direction_error_max",
    "jvp_fixed_direction_envelope_median",
    "jvp_fixed_direction_envelope_p95",
    "jvp_fixed_direction_envelope_p99",
    "jvp_fixed_direction_envelope_max",
    "n_forward_test",
    "n_jvp_test",
    "test_subset_seed",
    "trainable_parameters",
    "run_id",
    "checkpoint",
    "evaluation_json",
]


def table_row(
    registry_row: dict[str, str], metrics: dict[str, Any], evaluation_json: Path
) -> dict[str, Any]:
    def portable_path(path: Path) -> str:
        """Prefer a repository-relative or project-artifact path in tables."""
        resolved = path.resolve()
        try:
            return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            pass
        for anchor in ("checkpoints", "results", "data"):
            if anchor in path.parts:
                return Path(*path.parts[path.parts.index(anchor) :]).as_posix()
        return path.as_posix()

    return {
        "paper_set": "core",
        "study": registry_row["study"],
        "ablation_axis": registry_row["ablation_axis"],
        "architecture": registry_row["architecture"],
        "train_samples": int(float(registry_row["train_samples"])),
        "capacity": registry_row["capacity"],
        "lambda_J": float(registry_row["dino_weight"]),
        "checkpoint_epoch": int(float(registry_row["checkpoint_epoch"])),
        "E_F_val": float(registry_row["E_F_val"]),
        "E_J_val": float(registry_row["E_J_val"]),
        "S_val": float(registry_row["S_val"]),
        "E_F_test": metrics["E_F_test"],
        "E_J_test": metrics["E_J_test"],
        "S_test": metrics["S_test"],
        "E_F_test_mean": metrics["E_F_test_mean"],
        "E_F_test_p95": metrics["E_F_test_p95"],
        "E_J_test_mean": metrics["E_J_test_mean"],
        "E_J_test_p95": metrics["E_J_test_p95"],
        "forward_absolute_l2_error_median": metrics["forward_absolute_l2_error_median"],
        "forward_absolute_l2_error_p95": metrics["forward_absolute_l2_error_p95"],
        "forward_absolute_l2_error_p99": metrics["forward_absolute_l2_error_p99"],
        "forward_absolute_l2_error_max": metrics["forward_absolute_l2_error_max"],
        "jvp_unit_direction_error_median": metrics["jvp_unit_direction_error_median"],
        "jvp_unit_direction_error_p95": metrics["jvp_unit_direction_error_p95"],
        "jvp_unit_direction_error_p99": metrics["jvp_unit_direction_error_p99"],
        "jvp_unit_direction_error_max": metrics["jvp_unit_direction_error_max"],
        "jvp_fixed_direction_envelope_median": metrics["jvp_fixed_direction_envelope_median"],
        "jvp_fixed_direction_envelope_p95": metrics["jvp_fixed_direction_envelope_p95"],
        "jvp_fixed_direction_envelope_p99": metrics["jvp_fixed_direction_envelope_p99"],
        "jvp_fixed_direction_envelope_max": metrics[
            "jvp_fixed_direction_envelope_max"
        ],
        "n_forward_test": metrics["n_forward_test"],
        "n_jvp_test": metrics["n_jvp_test"],
        "test_subset_seed": metrics["test_subset_seed"],
        "trainable_parameters": registry_row["trainable_parameters"],
        "run_id": registry_row["run_id"],
        "checkpoint": portable_path(Path(registry_row["checkpoint"])),
        "evaluation_json": portable_path(evaluation_json),
    }


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    study_order = {"headline_opt": 0, "headline_J0": 1, "ablations_J10": 2}
    return (
        study_order.get(str(row["study"]), 99),
        ARCHITECTURE_ORDER.get(str(row["architecture"]), 99),
        int(row["train_samples"]),
        str(row["capacity"]),
        str(row["run_id"]),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace {path}; pass --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    all_registry_rows = read_registries(args.registry_dir.resolve())
    registry_rows = [row for row in all_registry_rows if is_core_paper_row(row)]
    collected: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    reference_jvp_indices: list[int] | None = None

    for row in registry_rows:
        json_path = args.evaluation_dir.resolve() / f"{output_stem(row)}.test.json"
        csv_path = args.evaluation_dir.resolve() / f"{output_stem(row)}.test.csv"
        status = "missing"
        detail = ""
        if json_path.is_file() and csv_path.is_file():
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
                metrics = validate_test_payload(
                    row,
                    payload,
                    expected_base_seed=int(args.expected_base_seed),
                    expected_shards=int(args.expected_shards),
                    expected_samples_per_shard=int(args.expected_samples_per_shard),
                    expected_jvp_samples=int(args.expected_jvp_samples),
                    expected_modes=int(args.expected_modes),
                )
                csv_rows = read_delimited(csv_path, delimiter=",")
                if len(csv_rows) != 1:
                    raise ValueError(f"{csv_path}: expected exactly one data row.")
                csv_checkpoint = Path(csv_rows[0]["checkpoint"]).name
                if csv_checkpoint != Path(row["checkpoint"]).name:
                    raise ValueError(f"{csv_path}: checkpoint disagrees with registry.")
                for csv_name, metric_name in (
                    ("forward_pooled_relative_l2", "E_F_test"),
                    ("jvp_pooled_relative_l2", "E_J_test"),
                    ("joint_score", "S_test"),
                    (
                        "forward_absolute_l2_error_p95",
                        "forward_absolute_l2_error_p95",
                    ),
                    (
                        "forward_absolute_l2_error_max",
                        "forward_absolute_l2_error_max",
                    ),
                    (
                        "jvp_unit_direction_error_p95",
                        "jvp_unit_direction_error_p95",
                    ),
                    (
                        "jvp_fixed_direction_envelope_p95",
                        "jvp_fixed_direction_envelope_p95",
                    ),
                    (
                        "jvp_fixed_direction_envelope_max",
                        "jvp_fixed_direction_envelope_max",
                    ),
                ):
                    if not math.isclose(
                        float(csv_rows[0][csv_name]),
                        float(metrics[metric_name]),
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-15,
                    ):
                        raise ValueError(f"{csv_path}: {csv_name} disagrees with JSON.")
                if reference_jvp_indices is None:
                    reference_jvp_indices = metrics["test_jvp_indices"]
                elif reference_jvp_indices != metrics["test_jvp_indices"]:
                    raise ValueError(
                        f"{row['run_id']}: JVP sample subset differs across models."
                    )
                collected.append(table_row(row, metrics, json_path))
                status = "validated"
            except Exception as error:
                status = "invalid"
                detail = str(error)
        elif json_path.exists() or csv_path.exists():
            status = "incomplete_pair"
            detail = "Both CSV and JSON are required."
        coverage.append(
            {
                "paper_set": "core",
                "study": row["study"],
                "architecture": row["architecture"],
                "run_id": row["run_id"],
                "checkpoint_epoch": int(float(row["checkpoint_epoch"])),
                "status": status,
                "detail": detail,
                "evaluation_stem": output_stem(row),
            }
        )

    invalid = [row for row in coverage if row["status"] not in {"missing", "validated"}]
    if invalid:
        messages = [f"{row['run_id']}: {row['status']}: {row['detail']}" for row in invalid]
        raise ValueError("Invalid final-test artifacts:\n" + "\n".join(messages))
    missing = [row for row in coverage if row["status"] == "missing"]
    if args.require_complete and missing:
        raise RuntimeError(
            f"Final-test campaign is incomplete: {len(missing)} of "
            f"{len(registry_rows)} registered evaluations are missing."
        )

    collected.sort(key=sort_key)
    subsets = {
        "all_selected_models_test.csv": collected,
        "core_paper_models_test.csv": [row for row in collected if row["paper_set"] == "core"],
        "supplementary_cross_grid_test.csv": [
            row for row in collected if row["paper_set"] == "supplementary_cross_grid"
        ],
        "headline_opt_test.csv": [row for row in collected if row["study"] == "headline_opt"],
        "headline_J0_test.csv": [row for row in collected if row["study"] == "headline_J0"],
        "ablations_J10_test.csv": [row for row in collected if row["study"] == "ablations_J10"],
    }
    for name, rows in subsets.items():
        write_csv(args.output_dir.resolve() / name, rows, overwrite=args.overwrite)

    coverage_path = args.output_dir.resolve() / "coverage.csv"
    if coverage_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {coverage_path}; pass --overwrite.")
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    with coverage_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(coverage[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(coverage)

    print("Requested set     : supported paper set")
    print(f"Registered models : {len(registry_rows)}")
    print(f"Validated results : {len(collected)}")
    print(f"Missing results   : {len(missing)}")
    print(f"Core paper set    : {len(registry_rows)}")
    print(f"Tables            : {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
