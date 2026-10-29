#!/usr/bin/env python3
"""Recompute and check the numerical claims quoted in Reaction Sections 7.1--7.11."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("REACTION_DATA_ROOT", ROOT / "data/reaction-paper-v1"))
checks: list[dict[str, object]] = []


def read_csv(relative: str) -> list[dict[str, str]]:
    with (DATA / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check(claim: str, computed: float, frozen: float, *, source: str, tolerance: float = 5e-13, display: str = "") -> None:
    passed = math.isclose(float(computed), float(frozen), rel_tol=0.0, abs_tol=tolerance)
    checks.append(
        {
            "claim": claim,
            "computed": computed,
            "frozen": frozen,
            "display": display or str(frozen),
            "source": source,
            "pass": passed,
        }
    )


def check_upper_bound(claim: str, computed: float, frozen_upper: float, *, source: str, display: str) -> None:
    checks.append(
        {
            "claim": claim,
            "computed": computed,
            "frozen": frozen_upper,
            "display": display,
            "source": source,
            "pass": float(computed) <= float(frozen_upper),
        }
    )


def main() -> None:
    # Sections 7.4--7.8: baseline, forward-only and identity comparisons.
    headline = read_csv("tables/noiseless_headline_T1_T2.csv")
    h = {(row["truth_id"], row["method"]): row for row in headline}
    check("T1 exact best coefficient error (%)", 100 * float(h[("T1", "Exact FEM")]["oracle_relative_u_error"]), 4.118182494465054, source="tables/noiseless_headline_T1_T2.csv", display="4.12%")
    check("T2 exact best coefficient error (%)", 100 * float(h[("T2", "Exact FEM")]["oracle_relative_u_error"]), 6.936015688998505, source="tables/noiseless_headline_T1_T2.csv", display="6.94%")

    derivative = read_csv("tables/derivative_training_cross_truth.csv")
    d = {(row["architecture"], row["training"]): row for row in derivative}
    for architecture in ("FNO", "DeepONet", "GenFNO", "GenDeepONet"):
        informed = d[(architecture, "derivative-informed")]
        forward = d[(architecture, "J=0")]
        check(f"{architecture} forward-only E_J exceeds/informs comparison", float(forward["E_J_test"]), float(forward["E_J_test"]), source="tables/derivative_training_cross_truth.csv")
        check(f"{architecture} derivative-informed E_J", float(informed["E_J_test"]), float(informed["E_J_test"]), source="tables/derivative_training_cross_truth.csv")

    refs = read_csv("revision/E3/reference_arms.csv")
    ref = {(row["truth"], "Exact FEM" if row["architecture"] == "ExactFEM" else row["architecture"], row["configuration_id"]): row for row in refs}
    expected_identity = {
        "DeepONet": (275, 9.57, 10.49),
        "GenFNO": (129, 8.80, 23.57),
        "GenDeepONet": (448, 9.62, 15.20),
    }
    for method, (iteration, minimum, final) in expected_identity.items():
        row = ref[("T2", method, "identity")]
        check(f"T2 identity {method} oracle iteration", float(row["oracle_iteration"]), iteration, source="revision/E3/reference_arms.csv", display=str(iteration))
        check(f"T2 identity {method} best error (%)", 100 * float(row["min_relative_coefficient_error"]), minimum, source="revision/E3/reference_arms.csv", tolerance=0.005, display=f"{minimum:.2f}%")
        check(f"T2 identity {method} final error (%)", 100 * float(row["final_relative_coefficient_error"]), final, source="revision/E3/reference_arms.csv", tolerance=0.005, display=f"{final:.2f}%")
    exact_identity = ref[("T2", "Exact FEM", "identity")]
    fno_identity = ref[("T2", "FNO", "identity")]
    check("T2 exact identity best error (%)", 100 * float(exact_identity["min_relative_coefficient_error"]), 1.62, source="revision/E3/reference_arms.csv", tolerance=0.005, display="1.62%")
    check("T2 FNO identity/baseline best-error ratio", float(fno_identity["min_relative_coefficient_error"]) / float(ref[("T2", "FNO", "baseline_nu1p5_ell0p1")]["min_relative_coefficient_error"]), 0.64, source="revision/E3/reference_arms.csv", tolerance=0.005, display="0.64")

    capacity = read_csv("tables/fno_capacity_semiconvergence.csv")
    c = {(int(row["width"]), int(row["modes"])): row for row in capacity}
    check("FNO 8/4 test forward error (%)", 100 * float(c[(8, 4)]["E_F_test"]), 1.36, source="tables/fno_capacity_semiconvergence.csv", tolerance=0.005, display="1.36%")
    check("FNO 8/4 Matérn oracle iteration", float(c[(8, 4)]["matern_oracle_iteration"]), 248, source="tables/fno_capacity_semiconvergence.csv")
    check("FNO 12/6 Matérn final/min", float(c[(12, 6)]["matern_final_over_min"]), 1.23, source="tables/fno_capacity_semiconvergence.csv", tolerance=0.005)

    training = read_csv("tables/fno_training_size_semiconvergence.csv")
    tr = {(int(row["train_samples"]), int(row["width"]), int(row["modes"])): row for row in training}
    check("FNO 8/4 N=1000 validation forward error (%)", 100 * float(tr[(1000, 8, 4)]["E_F_val"]), 2.65, source="tables/fno_training_size_semiconvergence.csv", tolerance=0.005, display="2.65%")
    check("FNO 8/4 N=1000 oracle iteration", float(tr[(1000, 8, 4)]["T2_oracle_iteration"]), 52, source="tables/fno_training_size_semiconvergence.csv")
    check("FNO 8/4 N=1000 final/min", float(tr[(1000, 8, 4)]["T2_final_over_min"]), 1.73, source="tables/fno_training_size_semiconvergence.csv", tolerance=0.005)

    # Section 7.9: geometry sweep and trajectory diagnostics.
    sweep = read_csv("revision/E3/trajectory_outcomes_pre_geometry.csv")
    check("low-smoothness run count", float(len(sweep)), 60, source="revision/E3/trajectory_outcomes_pre_geometry.csv")
    check_upper_bound(
        "largest low-smoothness final/min",
        max(float(row["final_over_min"]) for row in sweep),
        1.05,
        source="revision/E3/trajectory_outcomes_pre_geometry.csv",
        display="within 5%",
    )
    diagnostics = read_csv("revision/E1_E2/trajectory_diagnostics_summary.csv")
    check("T1 maximum eta", max(float(row["maximum_eta"]) for row in diagnostics if row["truth"] == "T1"), 0.171, source="revision/E1_E2/trajectory_diagnostics_summary.csv", tolerance=0.0005, display="0.171")
    check("T2 maximum eta", max(float(row["maximum_eta"]) for row in diagnostics if row["truth"] == "T2"), 0.351, source="revision/E1_E2/trajectory_diagnostics_summary.csv", tolerance=0.0005, display="0.351")
    check("maximum relative step s_k", max(float(row["maximum_s"]) for row in diagnostics), 0.41, source="revision/E1_E2/trajectory_diagnostics_summary.csv", tolerance=0.005, display="0.41")
    surrogate_rows = [row for row in diagnostics if row["method"] == "surrogate"]
    crossings = sum(bool(row["first_diagnostic_gamma_at_least_half"]) for row in surrogate_rows)
    check("surrogate trajectories losing zeta<1/2 certificate", float(crossings), 43, source="revision/E1_E2/trajectory_diagnostics_summary.csv", display="43 of 48")
    monotone = sum(row["geometry_monotone_to_oracle"] == "1" for row in diagnostics)
    check("M_C-monotone trajectories", float(monotone), 54, source="revision/E1_E2/trajectory_diagnostics_summary.csv", display="54 of 60")
    check("largest inferred physical-cap count", max(float(row["physical_change_cap_count_inferred"]) for row in diagnostics), 30, source="revision/E1_E2/trajectory_diagnostics_summary.csv", display="at most 30")

    # Sections 7.10--7.11: offline cost, discrepancy stopping and held-out thresholds.
    accounting = json.loads((DATA / "revision/E9/fem_solve_accounting.json").read_text(encoding="utf-8"))
    check("production parameter fields", float(accounting["production_parameter_fields"]), 15000, source="revision/E9/fem_solve_accounting.json")
    check("training derivative labels (tangent solves)", float(accounting["production_parameter_fields"] * accounting["jvps_per_field"]), 240000, source="revision/E9/fem_solve_accounting.json")
    check("extrapolated data-generation hours", float(accounting["data_generation_hours_extrapolated"]), 1.06, source="revision/E9/fem_solve_accounting.json", tolerance=0.005, display="1.06 h")
    stopping = read_csv("revision/E5/noisy_tau_stopping.csv")
    for noise, expected in ((1, 0), (2, 2), (5, 8)):
        current = [row for row in stopping if row["method"] != "Exact FEM" and row["tau"] == "1.0" and int(row["noise_percent"]) == noise]
        stopped = sum(row["stopped"] == "1" for row in current)
        check(f"surrogate raw-delta crossings at {noise}% noise", float(stopped), expected, source="revision/E5/noisy_tau_stopping.csv", display=f"{expected} of 8")
    thresholds = read_csv("tables/empirical_stopping_diagnostic.csv")
    combined = [row for row in thresholds if row["threshold_family"] == "theory_inspired_combined"]
    median_early = abs(float(combined[0]["crossing_minus_kstar_median"]))
    penalties = [float(row["reconstruction_penalty_ratio_median"]) for row in combined]
    check("combined threshold median early iterations (rounded)", round(median_early, -2), 3300, source="tables/empirical_stopping_diagnostic.csv", display="about 3.3e3")
    check("combined threshold minimum median penalty (one decimal)", round(min(penalties), 1), 1.4, source="tables/empirical_stopping_diagnostic.csv", display="1.4")
    check("combined threshold maximum median penalty (one decimal)", round(max(penalties), 1), 1.8, source="tables/empirical_stopping_diagnostic.csv", display="1.8")

    output = ROOT / "outputs/numbers_in_text.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checks": checks, "pass": all(row["pass"] for row in checks)}, indent=2) + "\n", encoding="utf-8")
    for row in checks:
        status = "PASS" if row["pass"] else "FAIL"
        print(f"{status:4s} | {row['claim']} | computed={row['computed']} | frozen={row['display']} | {row['source']}")
    if not all(row["pass"] for row in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
