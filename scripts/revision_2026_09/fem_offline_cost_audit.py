#!/usr/bin/env python3
"""Build the E9 FEM reuse and offline-cost audit from frozen metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HEADLINES = (
    ("FNO", "fno_N10000_w24_m10_L4_J300", "fno_N10000_w24_m10_L4_J100"),
    ("DeepONet", "do_N10000_Dp400_L6_J100", "do_N10000_Dp400_L6_J100"),
    (
        "GenFNO",
        "gfno_N10000_im16_om24_h512_rb4_J100",
        "gfno_N10000_im16_om24_h512_rb4_J100",
    ),
    (
        "GenDeepONet",
        "gdo_N10000_M400_N300_h1024_rb4_J1000",
        "gdo_N10000_M400_N300_h1024_rb4_J100",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "paper_products/generated/revision/E9",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def break_even_inversions(
    offline_hours: float,
    exact_inversion_hours: float,
    surrogate_inversion_hours: float,
) -> float:
    saving = exact_inversion_hours - surrogate_inversion_hours
    if saving <= 0.0:
        return float("inf")
    return offline_hours / saving


def data_generation_hours() -> tuple[float, list[Path]]:
    logs = sorted(
        (
            ROOT
            / "sulis_snapshot_2026-09-06/provenance/data/"
            "shards_broad_amp_final_test"
        ).glob("reaction_shard_*.log")
    )
    if len(logs) != 4:
        raise ValueError("Expected four final-test generation logs.")
    hours: list[float] = []
    for path in logs:
        matches = re.findall(r"Total generation time:\s*([0-9.]+)\s*h", path.read_text())
        if len(matches) != 1:
            raise ValueError(f"{path}: missing/duplicate total generation time.")
        hours.append(float(matches[0]))
    # Four logs cover 2,500 fields.  Extrapolate the same measured generator
    # throughput to 10,000 train + 2,500 validation + 2,500 final-test fields.
    return sum(hours) * (15_000 / 2_500), logs


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exact_dir = ROOT / "results/lw_noiseless_headline/T2_broad/exact_fem1x"
    exact_results = json.loads((exact_dir / "results.json").read_text())
    exact_settings = json.loads((exact_dir / "settings.json").read_text())
    summary = exact_results["solver_cache"]
    solver = exact_settings["command_arguments"]["linear_solver"]
    if solver != "cg-hypre":
        raise ValueError(f"Unexpected frozen exact solver: {solver}.")
    if int(summary["state_linear_solves"]) != (
        int(summary["forward_solves"])
        + int(summary["tangent_solves"])
        + int(summary["adjoint_solves"])
    ):
        raise ValueError("Frozen solver-cache counters do not balance.")

    runtime_rows = {
        row["method"]: row
        for row in read_csv(
            ROOT
            / "paper_products/generated/figure_data/"
            "noiseless_headline_runtime_T2_summary.csv"
        )
    }
    training_rows = {
        row["run_id"]: row
        for row in read_csv(
            ROOT / "results/paper_T2_broad/training_convergence/training_summary.csv"
        )
    }
    generation_hours, generation_logs = data_generation_hours()
    exact_hours = float(runtime_rows["Exact FEM"]["total_seconds"]) / 3600.0

    rows: list[dict[str, Any]] = []
    for architecture, frozen_run_id, timing_run_id in HEADLINES:
        timing = training_rows[timing_run_id]
        training_hours = float(timing["training_loop_hours"])
        training_kind = "measured_frozen_run"
        if timing_run_id != frozen_run_id:
            training_kind = "architecture_matched_lambda_proxy"
        surrogate_hours = float(runtime_rows[architecture]["total_seconds"]) / 3600.0
        offline_hours = generation_hours + training_hours
        continuous = break_even_inversions(offline_hours, exact_hours, surrogate_hours)
        rows.append(
            {
                "architecture": architecture,
                "frozen_run_id": frozen_run_id,
                "training_timing_run_id": timing_run_id,
                "training_timing_kind": training_kind,
                "data_generation_hours_extrapolated": generation_hours,
                "training_loop_hours": training_hours,
                "offline_hours_data_plus_training": offline_hours,
                "exact_10000_iteration_hours": exact_hours,
                "surrogate_10000_iteration_hours": surrogate_hours,
                "break_even_inversions_continuous": continuous,
                "break_even_inversions_ceiling": math.ceil(continuous),
                "hardware_normalised": 0,
            }
        )

    csv_path = args.output_dir / "offline_cost_break_even.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    accounting = {
        "frozen_exact_solver": solver,
        "operator_assemblies_or_ksp_setups": int(
            summary["operator_assemblies_factorizations"]
        ),
        "state_linear_solves": int(summary["state_linear_solves"]),
        "forward_solves": int(summary["forward_solves"]),
        "tangent_solves": int(summary["tangent_solves"]),
        "adjoint_solves": int(summary["adjoint_solves"]),
        "production_parameter_fields": 15_000,
        "jvps_per_field": 16,
        "data_generation_operator_assemblies": 15_000,
        "data_generation_state_solves": 255_000,
        "covariance_ensemble_parameter_fields": 10_000,
        "covariance_ensemble_pde_solves": 0,
        "data_generation_hours_extrapolated": generation_hours,
        "data_generation_time_sources": [str(path.relative_to(ROOT)) for path in generation_logs],
    }
    (args.output_dir / "fem_solve_accounting.json").write_text(
        json.dumps(accounting, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# E9 FEM timing and offline-cost audit",
        "",
        "## Operator reuse",
        "",
        "The frozen exact T2 trajectory used PETSc CG with Hypre BoomerAMG, not",
        "a sparse direct factorisation. At each coefficient, the DOLFINx operator",
        "is assembled and the KSP/preconditioner is configured once; the same",
        "operator/KSP is then reused for the forward, tangent and adjoint solves",
        "at that coefficient. The 10,000-iteration archive records 10,002 operator",
        f"assemblies/KSP setups and {summary['state_linear_solves']:,} state solves:",
        f"{summary['forward_solves']:,} forward, {summary['tangent_solves']:,} tangent and",
        f"{summary['adjoint_solves']:,} adjoint solves. Because reuse is already",
        "implemented, E9's conditional 20-iteration reuse benchmark is unnecessary.",
        "",
        "## Data generation",
        "",
        "The production split contains 10,000 training, 2,500 validation and",
        "2,500 independent-test fields. Each field uses one operator assembly,",
        "one forward solve and 16 tangent solves: 15,000 assemblies and 255,000",
        "state solves in total. The separate 10,000-field covariance ensemble",
        "does not call the PDE solver. Four retained final-test logs report 0.044 h",
        "per 625-field shard; extrapolating that measured throughput gives",
        f"{generation_hours:.3f} h of serial-equivalent generation time for 15,000 fields.",
        "",
        "## Operational break-even estimate",
        "",
        "The following comparison uses observed wall-clock times and is not",
        "hardware-normalised. The exact solve ran on a Sulis compute allocation",
        "with eight requested CPUs; the processor model and original node were not",
        "embedded in the archive. The historical model training and surrogate",
        "timings used the Sulis GPU stack associated with A100 nodes, but the exact",
        "node for every historical job is not retained locally.",
        "",
        "| Architecture | Data (h) | Training (h) | Surrogate inversion (h) | Break-even inversions | Timing status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['architecture']} | {row['data_generation_hours_extrapolated']:.3f} | "
            f"{row['training_loop_hours']:.3f} | {row['surrogate_10000_iteration_hours']:.4f} | "
            f"{row['break_even_inversions_ceiling']} | {row['training_timing_kind']} |"
        )
    lines.extend(
        [
            "",
            f"The exact 10,000-iteration inversion took {exact_hours:.3f} h. Break-even",
            "solves `(data + training + N*surrogate = N*exact)` separately for each",
            "architecture. FNO J300 and GenDeepONet J1000 lack retained per-model",
            "history timings, so their entries use the same-architecture J100 run as",
            "a compute-equivalent lambda proxy; changing lambda changes the loss",
            "weight but not the network or number of derivative directions.",
            "",
            "These counts exclude hyperparameter search and model-selection campaigns.",
            "They answer the narrower amortisation question for one frozen model.",
            "",
        ]
    )
    (args.output_dir / "NOTE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved E9 audit in {args.output_dir}")


if __name__ == "__main__":
    main()
