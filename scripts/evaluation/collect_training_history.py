#!/usr/bin/env python3
"""Collect checkpoint histories for paper training-convergence summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def first_number(mapping: Any, keys: tuple[str, ...]) -> float | None:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("checkpoints/paper_T2_broad"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/paper_T2_broad/training_convergence"))
    parser.add_argument("--dino-validation-every", type=int, default=50)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history_paths = sorted(args.checkpoint_dir.glob("*.history.json"))
    if not history_paths:
        raise FileNotFoundError(f"No *.history.json files found in {args.checkpoint_dir}")

    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for path in history_paths:
        run_id = path.name[:-len(".history.json")]
        history = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(history, list) or not history:
            print(f"SKIP: {path}")
            continue

        cumulative = 0.0
        run_rows = []

        for item in history:
            if not isinstance(item, dict):
                continue

            epoch = int(item.get("epoch", 0))
            sec = item.get("seconds")
            seconds = float(sec) if isinstance(sec, (int, float)) and math.isfinite(float(sec)) else None
            if seconds is not None:
                cumulative += seconds

            train = item.get("train", {})
            val = item.get("val", {})

            row = {
                "run_id": run_id,
                "epoch": epoch,
                "learning_rate": first_number(item, ("learning_rate", "lr")),
                "seconds": seconds,
                "cumulative_loop_seconds": cumulative,
                "train_total": first_number(train, ("loss", "total", "forward_objective", "forward", "field")),
                "train_forward": first_number(train, ("forward_objective", "forward", "field", "loss")),
                "train_jvp": first_number(train, ("dino", "jvp", "derivative")),
                "val_forward_internal": first_number(val, ("forward_objective", "forward", "field", "loss")),
                "val_jvp_internal": first_number(val, ("dino", "jvp", "derivative")),
                "val_composite_internal": first_number(val, ("composite",)),
                "val_relative_l2_internal": first_number(val, ("relative_l2", "rel_l2", "relL2")),
            }
            run_rows.append(row)

        if not run_rows:
            continue

        long_rows.extend(run_rows)

        valid_seconds = [float(r["seconds"]) for r in run_rows if r["seconds"] is not None]
        ordinary_seconds = [
            float(r["seconds"]) for r in run_rows
            if r["seconds"] is not None
            and int(r["epoch"]) > 1
            and (args.dino_validation_every <= 0 or int(r["epoch"]) % args.dino_validation_every != 0)
        ]
        lrs = [float(r["learning_rate"]) for r in run_rows if r["learning_rate"] is not None]
        min_lr = min(lrs) if lrs else None
        first_min_lr_epoch = None
        if min_lr is not None:
            for r in run_rows:
                if r["learning_rate"] is not None and math.isclose(float(r["learning_rate"]), min_lr, rel_tol=1e-12):
                    first_min_lr_epoch = int(r["epoch"])
                    break

        final = run_rows[-1]
        summary_rows.append({
            "run_id": run_id,
            "history_file": str(path),
            "final_epoch": int(final["epoch"]),
            "training_loop_seconds": sum(valid_seconds),
            "training_loop_hours": sum(valid_seconds) / 3600.0,
            "median_epoch_seconds_all": median(valid_seconds) if valid_seconds else None,
            "median_epoch_seconds_ordinary": median(ordinary_seconds) if ordinary_seconds else None,
            "min_learning_rate": min_lr,
            "first_min_lr_epoch": first_min_lr_epoch,
            "final_train_total": final["train_total"],
            "final_val_forward_internal": final["val_forward_internal"],
            "final_val_jvp_internal": final["val_jvp_internal"],
            "final_val_composite_internal": final["val_composite_internal"],
        })

    long_fields = [
        "run_id","epoch","learning_rate","seconds","cumulative_loop_seconds",
        "train_total","train_forward","train_jvp","val_forward_internal",
        "val_jvp_internal","val_composite_internal","val_relative_l2_internal"
    ]
    summary_fields = [
        "run_id","history_file","final_epoch","training_loop_seconds",
        "training_loop_hours","median_epoch_seconds_all",
        "median_epoch_seconds_ordinary","min_learning_rate",
        "first_min_lr_epoch","final_train_total","final_val_forward_internal",
        "final_val_jvp_internal","final_val_composite_internal"
    ]

    write_csv(args.output_dir / "training_history_long.csv", long_rows, long_fields)
    write_csv(args.output_dir / "training_summary.csv", summary_rows, summary_fields)

    payload = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "dino_validation_every": args.dino_validation_every,
        "n_runs": len(summary_rows),
        "n_epoch_rows": len(long_rows),
        "summary": summary_rows,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print(f"Collected {len(summary_rows)} runs and {len(long_rows)} epoch rows.")
    print(f"Saved {args.output_dir / 'training_history_long.csv'}")
    print(f"Saved {args.output_dir / 'training_summary.csv'}")
    print(f"Saved {args.output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
