#!/usr/bin/env python3
"""
Count trainable parameters for each unique surrogate architecture
represented in the fixed-budget validation table.

Input:
  results/paper_T2_broad/budget2500_validation/all_runs_at2500.csv

Output:
  results/paper_T2_broad/budget2500_validation/
      trainable_parameters_at2500.csv

The script loads one exact-epoch-2500 checkpoint per unique base architecture
and counts:
    sum(p.numel() for p in model.parameters() if p.requires_grad)

It does NOT use test data.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.emulator import load_reaction_emulator


RUN_PATTERNS = [
    re.compile(r"^(fno_N\d+_w\d+_m\d+_L\d+)_J[0-9p.]+$"),
    re.compile(r"^(do_N\d+_Dp\d+_L\d+)_J[0-9p.]+$"),
    re.compile(r"^(gfno_N\d+_im\d+_om\d+_h\d+_rb\d+)_J[0-9p.]+$"),
    re.compile(r"^(gdo_N\d+_M\d+_N\d+_h\d+_rb\d+)_J[0-9p.]+$"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/paper_T2_broad/budget2500_validation/all_runs_at2500.csv"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/paper_T2_broad/budget2500_validation/"
            "trainable_parameters_at2500.csv"
        ),
    )
    return p.parse_args()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def base_architecture_id(run_id: str) -> str:
    for pat in RUN_PATTERNS:
        m = pat.match(run_id)
        if m:
            return m.group(1)
    raise ValueError(f"Cannot parse run_id: {run_id}")


def exact_budget(row):
    return str(row.get("has_exact_budget_epoch", "")).strip().lower() == "true"


def main():
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    rows = read_csv(args.input)

    # One checkpoint per unique architecture. Prefer the first exact-budget row.
    reps = {}
    for r in rows:
        if not exact_budget(r):
            continue
        ckpt = r.get("checkpoint_at_budget", "").strip()
        if not ckpt:
            continue
        base = base_architecture_id(r["run_id"])
        reps.setdefault(base, r)

    if not reps:
        raise RuntimeError(
            "No exact epoch-2500 checkpoints were found in all_runs_at2500.csv"
        )

    output = []

    print("TRAINABLE PARAMETER COUNTS")
    print("=" * 90)

    for base in sorted(reps):
        r = reps[base]
        ckpt = Path(r["checkpoint_at_budget"])

        if not ckpt.is_file():
            raise FileNotFoundError(
                f"Checkpoint for {base} does not exist: {ckpt}"
            )

        print()
        print(base)
        print("  checkpoint:", ckpt)

        emulator = load_reaction_emulator(ckpt, device="cpu")

        # Current emulator wrappers expose the underlying torch module as .model.
        model = getattr(emulator, "model", None)
        if model is None:
            # Fallback in case the loader itself returns an nn.Module.
            if hasattr(emulator, "parameters"):
                model = emulator
            else:
                raise AttributeError(
                    f"Loaded emulator for {base} has neither .model nor .parameters()"
                )

        nparams = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        print(f"  trainable parameters: {nparams:,}")

        output.append(
            {
                "base_architecture_id": base,
                "architecture": r["architecture"],
                "capacity": r["capacity"],
                "trainable_parameters": nparams,
                "parameter_millions": nparams / 1.0e6,
                "source_checkpoint": str(ckpt),
            }
        )

        del emulator
        del model

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "base_architecture_id",
        "architecture",
        "capacity",
        "trainable_parameters",
        "parameter_millions",
        "source_checkpoint",
    ]

    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(output)

    print()
    print("=" * 90)
    print(f"Counted {len(output)} unique architectures.")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
