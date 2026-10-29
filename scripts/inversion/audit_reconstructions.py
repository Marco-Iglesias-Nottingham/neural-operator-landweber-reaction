#!/usr/bin/env python3
"""
Audit completed surrogate Landweber campaigns after the min-u patch.

The Landweber driver writes:
    result.npz
    results.json
    settings.json

This script verifies that result.npz contains the paper diagnostics:
    recovered_u_minimum_u_error
    minimum_u_error
    minimum_u_error_iteration
    history_relative_u_error
    history_relative_residual
    history_landweber_elapsed_seconds

It does not modify run directories.

Outputs
-------
results/paper_T2_broad/landweber_min_u_audit.csv
results/paper_T2_broad/landweber_min_u_audit.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_ROOTS = [
    Path("results/lw_noiseless_headline_min_u/T2_broad"),
    Path("results/lw_noiseless_headline_J0/T2_broad"),
    Path("results/lw_noiseless_ablation_J10/T2_broad"),
]

REQUIRED = {
    "recovered_u_minimum_u_error",
    "minimum_u_error",
    "minimum_u_error_iteration",
    "history_relative_u_error",
    "history_relative_residual",
    "history_landweber_elapsed_seconds",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="*", type=Path, default=DEFAULT_ROOTS)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/paper_T2_broad/landweber_min_u_audit.csv"),
    )
    return p.parse_args()


def scalar(z, key, cast=float):
    return cast(np.asarray(z[key]).reshape(()))


def inspect(run_dir: Path):
    result = run_dir / "result.npz"
    results_json = run_dir / "results.json"
    settings_json = run_dir / "settings.json"

    row = {
        "campaign": run_dir.parent.name,
        "run_dir": str(run_dir),
        "result_npz_exists": result.is_file(),
        "results_json_exists": results_json.is_file(),
        "settings_json_exists": settings_json.is_file(),
        "status": "",
        "missing_keys": "",
        "minimum_u_error": "",
        "minimum_u_error_iteration": "",
        "history_length": "",
        "final_iteration": "",
        "landweber_elapsed_seconds": "",
    }

    if not result.is_file():
        row["status"] = "MISSING_RESULT"
        return row

    try:
        with np.load(result, allow_pickle=False) as z:
            missing = sorted(REQUIRED.difference(z.files))
            if missing:
                row["status"] = "MISSING_KEYS"
                row["missing_keys"] = ";".join(missing)
                return row

            hist_iter = np.asarray(z["history_iteration"], dtype=np.int64)
            hist_u = np.asarray(z["history_relative_u_error"], dtype=float)
            hist_time = np.asarray(
                z["history_landweber_elapsed_seconds"], dtype=float
            )

            kmin = scalar(z, "minimum_u_error_iteration", int)
            emin = scalar(z, "minimum_u_error", float)
            saved_field = np.asarray(z["recovered_u_minimum_u_error"])

            # Strong consistency checks.
            if hist_iter.size == 0:
                raise RuntimeError("empty history")
            j = int(np.argmin(hist_u))
            if int(hist_iter[j]) != kmin:
                raise RuntimeError(
                    f"saved min-u iteration {kmin} != history argmin {int(hist_iter[j])}"
                )
            if not np.isclose(float(hist_u[j]), emin, rtol=1e-10, atol=1e-12):
                raise RuntimeError(
                    f"saved minimum_u_error {emin} != history minimum {float(hist_u[j])}"
                )
            if saved_field.ndim != 2:
                raise RuntimeError(
                    f"saved min-u field is not 2-D: {saved_field.shape}"
                )

            row["status"] = "OK"
            row["minimum_u_error"] = emin
            row["minimum_u_error_iteration"] = kmin
            row["history_length"] = hist_iter.size
            row["final_iteration"] = int(hist_iter[-1])
            row["landweber_elapsed_seconds"] = float(hist_time[-1])

    except Exception as exc:
        row["status"] = "INVALID"
        row["missing_keys"] = str(exc)

    return row


def main():
    args = parse_args()
    rows = []

    for root in args.roots:
        if not root.is_dir():
            print(f"WARNING: root not found: {root}")
            continue
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            rows.append(inspect(run_dir))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "campaign",
        "run_dir",
        "result_npz_exists",
        "results_json_exists",
        "settings_json_exists",
        "status",
        "missing_keys",
        "minimum_u_error",
        "minimum_u_error_iteration",
        "history_length",
        "final_iteration",
        "landweber_elapsed_seconds",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    lines = [
        "Reaction Landweber minimum-u audit",
        "=" * 72,
        f"Run directories inspected: {len(rows)}",
    ]
    for status in sorted(counts):
        lines.append(f"{status:16s}: {counts[status]}")
    lines.append("")
    bad = [r for r in rows if r["status"] != "OK"]
    if bad:
        lines.append("Non-OK runs:")
        for r in bad:
            lines.append(
                f"  {r['status']:14s} {r['run_dir']} {r['missing_keys']}"
            )
    else:
        lines.append("All inspected runs contain consistent minimum-u diagnostics.")

    txt = args.output.with_suffix(".txt")
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print()
    print("CSV:", args.output)
    print("TXT:", txt)


if __name__ == "__main__":
    main()
