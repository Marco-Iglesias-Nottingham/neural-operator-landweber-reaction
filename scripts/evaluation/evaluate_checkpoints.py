#!/usr/bin/env python3
"""Full-grid validation/test of FNO and DeepONet checkpoints.

This evaluator is deliberately separate from training.  It evaluates every
selected pressure/JVP field at all 120x120 spatial points and reports physical
trapezoidal-L2 errors.  It can evaluate many epoch snapshots on the same fixed
validation subset, or one chosen checkpoint on the untouched test pool.

Recommended use:
  1. During training save epoch snapshots (e.g. every 50 epochs).
  2. Run this script on *validation* shards to select a checkpoint.
  3. Run the chosen checkpoint once on *test* shards.

The headline pooled relative errors are

  E_F = sqrt(sum_i ||F_theta(u_i)-F(u_i)||_L2^2 /
             sum_i ||F(u_i)||_L2^2)

and

  E_J = sqrt(sum_{i,j} ||F'_theta(u_i)h_j-F'(u_i)h_j||_L2^2 /
             sum_{i,j} ||F'(u_i)h_j||_L2^2).

All stored KL/JVP modes are used unless --modes is supplied.

In addition to the relative headline errors, the evaluator reports absolute
forward-error quantiles and L2-output error per unit L2-input KL direction.
The maximum across the fixed directions is an empirical directional envelope,
not a full derivative operator norm.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = PROJECT_ROOT / "scripts" / "training"
for search_path in (PROJECT_ROOT, SCRIPT_ROOT, TRAINING_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from models.emulator import load_reaction_emulator  # noqa: E402
from reaction_core.datasets import ReactionH5Dataset  # noqa: E402
from training_utils import audit_shards, expand_shard_patterns  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Full-grid Reaction surrogate validation/test from stored shards.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint paths or glob patterns. Quote globs on the shell.",
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        required=True,
        help="Validation/test shard paths or glob patterns.",
    )
    parser.add_argument(
        "--label",
        choices=("validation", "test"),
        default="validation",
        help="Only labels the output; test data should never be used for selection.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-forward-samples",
        type=int,
        default=0,
        help="0 means all samples; otherwise use a fixed random subset.",
    )
    parser.add_argument(
        "--max-jvp-samples",
        type=int,
        default=128,
        help="Number of samples for expensive all-mode JVP evaluation; 0 means all.",
    )
    parser.add_argument(
        "--modes",
        type=int,
        nargs="*",
        default=None,
        help="Zero-based JVP mode indices. Omit to evaluate every stored mode.",
    )
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=20260821,
        help="Fixed subset seed, shared by every checkpoint.",
    )
    parser.add_argument(
        "--expected-shards",
        type=int,
        default=0,
        help="If positive, require exactly this many input shards.",
    )
    parser.add_argument(
        "--expected-samples-per-shard",
        type=int,
        default=0,
        help="If positive, require this sample count in every shard.",
    )
    parser.add_argument(
        "--expected-base-seed",
        type=int,
        default=-1,
        help=(
            "If non-negative, require every shard to declare this base seed "
            "and consecutive offsets starting at zero."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=PROJECT_ROOT / "results/reaction_checkpoint_evaluation",
    )
    parser.add_argument(
        "--save-per-sample",
        action="store_true",
        help="Save per-sample/per-mode relative errors in the JSON output.",
    )
    return parser.parse_args()


def expand_checkpoints(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        if matches:
            paths.extend(Path(match).expanduser().resolve() for match in matches)
        else:
            candidate = Path(pattern).expanduser().resolve()
            if candidate.is_file():
                paths.append(candidate)
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise FileNotFoundError(f"No checkpoints matched {list(patterns)!r}.")
    return unique


def trapezoid_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(y) < 2:
        raise ValueError("x/y grids must be one-dimensional with at least two nodes.")
    dx = np.diff(x)
    dy = np.diff(y)
    if not np.allclose(dx, dx[0], rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("x grid is not uniform.")
    if not np.allclose(dy, dy[0], rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("y grid is not uniform.")
    wx = np.full(len(x), float(dx[0]), dtype=np.float64)
    wy = np.full(len(y), float(dy[0]), dtype=np.float64)
    wx[[0, -1]] *= 0.5
    wy[[0, -1]] *= 0.5
    return wx[:, None] * wy[None, :]


def weighted_squared_norm(field: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(field, dtype=np.float64)
    return float(np.sum(weights * values * values))


def summarize_values(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "mean": float("nan"),
            "rms": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(array)),
        "rms": float(np.sqrt(np.mean(array * array))),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def choose_indices(n_total: int, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if maximum < 0:
        raise ValueError("Sample limits cannot be negative.")
    if maximum == 0 or maximum >= n_total:
        return np.arange(n_total, dtype=np.int64)
    return np.sort(rng.choice(n_total, size=maximum, replace=False).astype(np.int64))


def validate_expected_pool(
    audit: dict[str, Any],
    *,
    expected_shards: int,
    expected_samples_per_shard: int,
    expected_base_seed: int,
) -> None:
    """Apply optional exact-pool requirements after the general shard audit."""
    if expected_shards < 0:
        raise ValueError("--expected-shards cannot be negative.")
    if expected_samples_per_shard < 0:
        raise ValueError("--expected-samples-per-shard cannot be negative.")
    if expected_base_seed < -1:
        raise ValueError("--expected-base-seed must be -1 or non-negative.")

    n_shards = int(audit["n_shards"])
    if expected_shards and n_shards != expected_shards:
        raise ValueError(
            f"Expected exactly {expected_shards} shards, found {n_shards}."
        )
    counts = [int(value) for value in audit["sample_counts"]]
    if expected_samples_per_shard and counts != [expected_samples_per_shard] * n_shards:
        raise ValueError(
            f"Expected {expected_samples_per_shard} samples in every shard, "
            f"found {counts}."
        )
    if expected_base_seed >= 0:
        records = sorted(
            audit["seed_records"], key=lambda value: int(value["sample_offset"])
        )
        offset = 0
        for index, record in enumerate(records):
            base_seed = int(record["base_seed"])
            sample_offset = int(record["sample_offset"])
            first_seed = int(record["first_seed"])
            last_seed = int(record["last_seed"])
            if base_seed != expected_base_seed or sample_offset != offset:
                raise ValueError(
                    f"Shard {index} does not belong to the expected base-seed/"
                    f"offset sequence: base_seed={base_seed}, "
                    f"sample_offset={sample_offset}; expected "
                    f"base_seed={expected_base_seed}, sample_offset={offset}."
                )
            count = last_seed - first_seed + 1
            if first_seed != expected_base_seed + offset or count != counts[index]:
                raise ValueError(f"Shard {index} has inconsistent expected seed coverage.")
            offset += counts[index]


def reshape_output(values: Any, shape: tuple[int, int], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == shape:
        return array
    if array.size == shape[0] * shape[1]:
        return array.reshape(shape)
    raise ValueError(f"{name} has shape {array.shape}; expected {shape} or flattened equivalent.")


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    dataset: ReactionH5Dataset,
    forward_indices: np.ndarray,
    jvp_indices: np.ndarray,
    mode_indices: list[int],
    weights: np.ndarray,
    direction_norms: np.ndarray,
    device: str,
    save_per_sample: bool,
) -> dict[str, Any]:
    started = perf_counter()
    emulator = load_reaction_emulator(checkpoint, device=device)
    model_type = str(emulator.model_type)
    epoch = int(emulator.epoch)
    shape = tuple(int(value) for value in weights.shape)

    forward_num = 0.0
    forward_den = 0.0
    forward_ratios: list[float] = []
    forward_absolute_errors: list[float] = []
    forward_records: list[dict[str, float | int]] = []

    print(f"\n[{checkpoint.name}] {model_type}, epoch={epoch}", flush=True)
    print(f"  forward samples: {len(forward_indices)}", flush=True)
    for count, index in enumerate(forward_indices, start=1):
        u_t, p_t, _ = dataset[int(index)]
        u = np.asarray(u_t.numpy(), dtype=np.float32)
        exact = np.asarray(p_t.numpy()[0], dtype=np.float64)
        prediction = reshape_output(emulator.forward(u), shape, name="forward prediction")
        num = weighted_squared_norm(prediction - exact, weights)
        den = weighted_squared_norm(exact, weights)
        forward_num += num
        forward_den += den
        absolute_error = float(np.sqrt(num))
        ratio = float(np.sqrt(num / max(den, 1.0e-300)))
        forward_absolute_errors.append(absolute_error)
        forward_ratios.append(ratio)
        if save_per_sample:
            forward_records.append(
                {
                    "index": int(index),
                    "absolute_l2_error": absolute_error,
                    "relative_l2_error": ratio,
                }
            )
        if count % 250 == 0 or count == len(forward_indices):
            print(f"    forward {count}/{len(forward_indices)}", flush=True)

    n_modes = int(dataset.directions.shape[0])
    mode_num = np.zeros(n_modes, dtype=np.float64)
    mode_den = np.zeros(n_modes, dtype=np.float64)
    mode_ratios: list[list[float]] = [[] for _ in range(n_modes)]
    mode_unit_direction_errors: list[list[float]] = [
        [] for _ in range(n_modes)
    ]
    per_input_directional_envelopes: list[float] = []
    jvp_records: list[dict[str, float | int]] = []

    print(f"  JVP samples/modes: {len(jvp_indices)} x {len(mode_indices)}", flush=True)
    for sample_count, index in enumerate(jvp_indices, start=1):
        u_t, _, exact_all_t = dataset[int(index)]
        u = np.asarray(u_t.numpy(), dtype=np.float32)
        exact_all = np.asarray(exact_all_t.numpy(), dtype=np.float64)
        input_unit_errors: list[float] = []
        for mode in mode_indices:
            direction = np.asarray(dataset.directions[mode], dtype=np.float32)
            exact = exact_all[mode, 0]
            prediction = reshape_output(
                emulator.jvp(u, direction),
                shape,
                name=f"JVP prediction mode {mode}",
            )
            num = weighted_squared_norm(prediction - exact, weights)
            den = weighted_squared_norm(exact, weights)
            mode_num[mode] += num
            mode_den[mode] += den
            absolute_error = float(np.sqrt(num))
            direction_norm = float(direction_norms[mode])
            if not np.isfinite(direction_norm) or direction_norm <= 0.0:
                raise ValueError(f"Direction {mode} has invalid L2 norm {direction_norm}.")
            unit_direction_error = absolute_error / direction_norm
            ratio = float(np.sqrt(num / max(den, 1.0e-300)))
            mode_ratios[mode].append(ratio)
            mode_unit_direction_errors[mode].append(unit_direction_error)
            input_unit_errors.append(unit_direction_error)
            if save_per_sample:
                jvp_records.append(
                    {
                        "index": int(index),
                        "mode": int(mode),
                        "absolute_l2_error": absolute_error,
                        "direction_l2_norm": direction_norm,
                        "error_per_unit_direction": unit_direction_error,
                        "relative_l2_error": ratio,
                    }
                )
        per_input_directional_envelopes.append(max(input_unit_errors))
        if sample_count % 16 == 0 or sample_count == len(jvp_indices):
            print(f"    JVP {sample_count}/{len(jvp_indices)}", flush=True)

    selected_num = float(np.sum(mode_num[mode_indices]))
    selected_den = float(np.sum(mode_den[mode_indices]))
    forward_pooled = float(np.sqrt(forward_num / max(forward_den, 1.0e-300)))
    jvp_pooled = float(np.sqrt(selected_num / max(selected_den, 1.0e-300)))
    per_mode: dict[str, Any] = {}
    all_jvp_ratios: list[float] = []
    all_unit_direction_errors: list[float] = []
    for mode in mode_indices:
        pooled = float(np.sqrt(mode_num[mode] / max(mode_den[mode], 1.0e-300)))
        summary = summarize_values(mode_ratios[mode])
        unit_summary = summarize_values(mode_unit_direction_errors[mode])
        per_mode[f"{mode:02d}"] = {
            "pooled_relative_l2": pooled,
            **summary,
            "direction_l2_norm": float(direction_norms[mode]),
            "error_per_unit_direction": unit_summary,
        }
        all_jvp_ratios.extend(mode_ratios[mode])
        all_unit_direction_errors.extend(mode_unit_direction_errors[mode])

    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "model_type": model_type,
        "epoch": epoch,
        "n_forward_samples": int(len(forward_indices)),
        "n_jvp_samples": int(len(jvp_indices)),
        "jvp_modes": [int(value) for value in mode_indices],
        "forward": {
            "pooled_relative_l2": forward_pooled,
            **summarize_values(forward_ratios),
            "absolute_l2_error": summarize_values(forward_absolute_errors),
        },
        "jvp": {
            "pooled_relative_l2": jvp_pooled,
            **summarize_values(all_jvp_ratios),
            "error_per_unit_direction": summarize_values(
                all_unit_direction_errors
            ),
            "fixed_direction_envelope_per_input": summarize_values(
                per_input_directional_envelopes
            ),
            "per_mode": per_mode,
        },
        "joint_score": float(np.sqrt(forward_pooled**2 + jvp_pooled**2)),
        "seconds": float(perf_counter() - started),
    }
    if save_per_sample:
        result["per_sample_forward"] = forward_records
        result["per_sample_jvp"] = jvp_records

    print(f"  E_F pooled : {forward_pooled:.6e}", flush=True)
    print(f"  E_J pooled : {jvp_pooled:.6e}", flush=True)
    print(
        "  abs F error: "
        f"p95={result['forward']['absolute_l2_error']['p95']:.6e}, "
        f"max={result['forward']['absolute_l2_error']['max']:.6e}",
        flush=True,
    )
    print(
        "  KL J error : "
        f"envelope p95={result['jvp']['fixed_direction_envelope_per_input']['p95']:.6e}, "
        f"max={result['jvp']['fixed_direction_envelope_per_input']['max']:.6e}",
        flush=True,
    )
    print(f"  joint score: {result['joint_score']:.6e}", flush=True)
    return result


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def main() -> None:
    args = parse_args()
    checkpoints = expand_checkpoints(args.checkpoints)
    shard_paths = expand_shard_patterns(args.shards)
    audit = audit_shards(shard_paths, label=args.label)
    validate_expected_pool(
        audit,
        expected_shards=int(args.expected_shards),
        expected_samples_per_shard=int(args.expected_samples_per_shard),
        expected_base_seed=int(args.expected_base_seed),
    )
    dataset = ReactionH5Dataset(shard_paths, include_jvp=True)
    try:
        n_total = len(dataset)
        rng = np.random.default_rng(int(args.subset_seed))
        forward_indices = choose_indices(n_total, int(args.max_forward_samples), rng)
        # Use a fresh deterministic stream while keeping the JVP subset independent
        # of checkpoint/model type.  It need not be a subset of the forward subset.
        jvp_rng = np.random.default_rng(int(args.subset_seed) + 1)
        jvp_indices = choose_indices(n_total, int(args.max_jvp_samples), jvp_rng)
        n_modes = int(dataset.directions.shape[0])
        if args.modes is None or len(args.modes) == 0:
            mode_indices = list(range(n_modes))
        else:
            mode_indices = [int(value) for value in args.modes]
            if len(set(mode_indices)) != len(mode_indices):
                raise ValueError("Duplicate --modes were supplied.")
            if any(value < 0 or value >= n_modes for value in mode_indices):
                raise ValueError(f"--modes must lie in [0,{n_modes}).")

        weights = trapezoid_weights(dataset.x_grid, dataset.y_grid)
        direction_norms = np.asarray(
            [
                np.sqrt(
                    weighted_squared_norm(
                        np.asarray(dataset.directions[mode], dtype=np.float64),
                        weights,
                    )
                )
                for mode in range(n_modes)
            ],
            dtype=np.float64,
        )
        results = [
            evaluate_checkpoint(
                checkpoint,
                dataset=dataset,
                forward_indices=forward_indices,
                jvp_indices=jvp_indices,
                mode_indices=mode_indices,
                weights=weights,
                direction_norms=direction_norms,
                device=str(args.device),
                save_per_sample=bool(args.save_per_sample),
            )
            for checkpoint in checkpoints
        ]
    finally:
        dataset.close()

    results.sort(key=lambda item: float(item["joint_score"]))
    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = Path(f"{prefix}.{args.label}.json")
    csv_path = Path(f"{prefix}.{args.label}.csv")

    payload = {
        "format": "reaction-full-grid-checkpoint-evaluation-v1",
        "error_diagnostics": "absolute-and-fixed-direction-normalized-v1",
        "label": args.label,
        "shards": [str(path) for path in shard_paths],
        "shard_audit": {
            "n_shards": int(audit["n_shards"]),
            "n_samples": int(audit["n_samples"]),
            "sample_counts": [int(v) for v in audit["sample_counts"]],
            "shape": list(audit["shape"]),
            "seed_records": list(audit["seed_records"]),
        },
        "subset_seed": int(args.subset_seed),
        "forward_indices": forward_indices.tolist(),
        "jvp_indices": jvp_indices.tolist(),
        "results_sorted_by_joint_score": results,
    }
    json_path.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "rank",
        "checkpoint",
        "model_type",
        "epoch",
        "n_forward_samples",
        "n_jvp_samples",
        "forward_pooled_relative_l2",
        "forward_mean_relative_l2",
        "forward_p95_relative_l2",
        "forward_absolute_l2_error_median",
        "forward_absolute_l2_error_p95",
        "forward_absolute_l2_error_p99",
        "forward_absolute_l2_error_max",
        "jvp_pooled_relative_l2",
        "jvp_mean_relative_l2",
        "jvp_p95_relative_l2",
        "jvp_unit_direction_error_median",
        "jvp_unit_direction_error_p95",
        "jvp_unit_direction_error_p99",
        "jvp_unit_direction_error_max",
        "jvp_fixed_direction_envelope_median",
        "jvp_fixed_direction_envelope_p95",
        "jvp_fixed_direction_envelope_p99",
        "jvp_fixed_direction_envelope_max",
        "joint_score",
        "seconds",
    ] + [f"jvp_mode_{mode:02d}_pooled_relative_l2" for mode in mode_indices]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            row = {
                "rank": rank,
                "checkpoint": result["checkpoint"],
                "model_type": result["model_type"],
                "epoch": result["epoch"],
                "n_forward_samples": result["n_forward_samples"],
                "n_jvp_samples": result["n_jvp_samples"],
                "forward_pooled_relative_l2": result["forward"]["pooled_relative_l2"],
                "forward_mean_relative_l2": result["forward"]["mean"],
                "forward_p95_relative_l2": result["forward"]["p95"],
                "forward_absolute_l2_error_median": result["forward"]["absolute_l2_error"]["median"],
                "forward_absolute_l2_error_p95": result["forward"]["absolute_l2_error"]["p95"],
                "forward_absolute_l2_error_p99": result["forward"]["absolute_l2_error"]["p99"],
                "forward_absolute_l2_error_max": result["forward"]["absolute_l2_error"]["max"],
                "jvp_pooled_relative_l2": result["jvp"]["pooled_relative_l2"],
                "jvp_mean_relative_l2": result["jvp"]["mean"],
                "jvp_p95_relative_l2": result["jvp"]["p95"],
                "jvp_unit_direction_error_median": result["jvp"]["error_per_unit_direction"]["median"],
                "jvp_unit_direction_error_p95": result["jvp"]["error_per_unit_direction"]["p95"],
                "jvp_unit_direction_error_p99": result["jvp"]["error_per_unit_direction"]["p99"],
                "jvp_unit_direction_error_max": result["jvp"]["error_per_unit_direction"]["max"],
                "jvp_fixed_direction_envelope_median": result["jvp"]["fixed_direction_envelope_per_input"]["median"],
                "jvp_fixed_direction_envelope_p95": result["jvp"]["fixed_direction_envelope_per_input"]["p95"],
                "jvp_fixed_direction_envelope_p99": result["jvp"]["fixed_direction_envelope_per_input"]["p99"],
                "jvp_fixed_direction_envelope_max": result["jvp"]["fixed_direction_envelope_per_input"]["max"],
                "joint_score": result["joint_score"],
                "seconds": result["seconds"],
            }
            for mode in mode_indices:
                row[f"jvp_mode_{mode:02d}_pooled_relative_l2"] = (
                    result["jvp"]["per_mode"][f"{mode:02d}"]["pooled_relative_l2"]
                )
            writer.writerow(row)

    print("\nRanking by validation/test joint score sqrt(E_F^2 + E_J^2)")
    for rank, result in enumerate(results, start=1):
        print(
            f"  {rank:2d}. {result['checkpoint_name']} "
            f"epoch={result['epoch']:4d} "
            f"E_F={result['forward']['pooled_relative_l2']:.6e} "
            f"E_J={result['jvp']['pooled_relative_l2']:.6e} "
            f"score={result['joint_score']:.6e}"
        )
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
