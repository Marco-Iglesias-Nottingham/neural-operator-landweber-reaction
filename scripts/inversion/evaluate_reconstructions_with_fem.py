#!/usr/bin/env python3
"""
Evaluate emulator/model-error diagnostics at the oracle minimum-u Landweber
iterate for the noiseless campaigns.

For each surrogate Landweber run, let

    k* = argmin_k ||u_k - u^dagger|| / ||u^dagger||,
    u* = u_{k*},
    y  = fine-truth noiseless data resampled to the 120x120 observation grid.

The patched Landweber archives already contain
    u*                     = recovered_u_minimum_u_error
    G_theta(u*)            = minimum_u_error_prediction
    G_theta(u^dagger)      = surrogate_prediction_at_resampled_truth

This diagnostic performs only training-resolution P1 FEM forward solves,
G_h, on the 119x119-cell mesh (120x120 vertices). It therefore separates

  1) surrogate data misfit at the best reconstruction:
       r_theta* = ||G_theta(u*) - y||_W

  2) coarse-FEM data misfit at the same coefficient:
       r_h* = ||G_h(u*) - y||_W

  3) pure emulator error at the same coefficient:
       e_emu* = ||G_h(u*) - G_theta(u*)||_W

At the known truth it also separates

  4) the previously used total surrogate mismatch:
       e_total^dag = ||G_theta(u^dagger) - y||_W

  5) training-resolution FEM/model discrepancy:
       e_h^dag = ||G_h(u^dagger) - y||_W

  6) pure emulator error at the truth:
       e_emu^dag = ||G_h(u^dagger) - G_theta(u^dagger)||_W

This is a diagnostic only. k* uses the known synthetic truth and is not a
practical stopping rule.

By default the script scans:
    results/lw_noiseless_headline_min_u/T2_broad
    results/lw_noiseless_headline_J0/T2_broad
    results/lw_noiseless_ablation_J10/T2_broad

Outputs:
    results/paper_T2_broad/min_u_emulator_diagnostics/
        min_u_emulator_diagnostics.csv
        summary.txt
        fields/<run_name>.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from mpi4py import MPI
import numpy as np


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parent,
        *here.parents,
        Path.cwd().resolve(),
        *Path.cwd().resolve().parents,
    ]
    seen = set()
    for p in candidates:
        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        if (p / "reaction_core").is_dir() and (p / "scripts").is_dir():
            return p
    raise RuntimeError("Could not locate reaction_emulator_project_matern_only.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reaction_core import (  # noqa: E402
    ReactionConfig,
    ReactionSolver2D,
    RegularGrid2D,
    StructuredCoefficientGridMap,
    StructuredLagrangeStateGridMap,
)
from reaction_core.inversion import weighted_norm  # noqa: E402


DEFAULT_ROOTS = [
    Path("results/lw_noiseless_headline_min_u/T2_broad"),
    Path("results/lw_noiseless_headline_J0/T2_broad"),
    Path("results/lw_noiseless_ablation_J10/T2_broad"),
]

ARCH_NAMES = {
    "fno": "FNO",
    "do": "DeepONet",
    "gfno": "GenFNO",
    "gdo": "GenDeepONet",
}


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=DEFAULT_ROOTS,
        help="Campaign roots containing per-run result.npz directories.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/paper_T2_broad/min_u_emulator_diagnostics"
        ),
    )
    p.add_argument(
        "--fem-nx",
        type=int,
        default=119,
        help="Training-resolution P1 FEM cell count in x.",
    )
    p.add_argument("--fem-ny", type=int, default=119)
    p.add_argument(
        "--linear-solver",
        choices=("cg-hypre", "lu"),
        default="cg-hypre",
    )
    p.add_argument("--ksp-rtol", type=float, default=1.0e-11)
    p.add_argument("--ksp-max-it", type=int, default=10000)
    p.add_argument(
        "--u-min",
        type=float,
        default=0.0,
        help="Broad physical lower bound used to construct the FEM solver.",
    )
    return p.parse_args()


def solver_options(args):
    if args.linear_solver == "lu":
        return {"ksp_type": "preonly", "pc_type": "lu"}
    return {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": float(args.ksp_rtol),
        "ksp_atol": 0.0,
        "ksp_max_it": int(args.ksp_max_it),
    }


def resolve_project(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def discover_runs(roots):
    runs = []
    for root0 in roots:
        root = resolve_project(root0)
        if not root.is_dir():
            print(f"WARNING: campaign root not found: {root}", flush=True)
            continue
        campaign = root.parent.name
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            result = run_dir / "result.npz"
            if result.is_file():
                runs.append((campaign, run_dir, result))
    if not runs:
        raise RuntimeError("No surrogate Landweber result.npz files found.")
    return runs


def weighted(field, weights, mask):
    return float(weighted_norm(
        np.asarray(field, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        mask=np.asarray(mask, dtype=bool),
    ))


def parse_run_name(name: str):
    prefix = name.split("_", 1)[0]
    arch = ARCH_NAMES.get(prefix, prefix)

    mN = re.search(r"_N(\d+)", name)
    mJ = re.search(r"_J([0-9]+(?:p[0-9]+)?)", name)
    me = re.search(r"_e(\d+)_minu$", name)

    lambda_j = ""
    if mJ:
        lambda_j = float(mJ.group(1).replace("p", "."))

    return {
        "architecture": arch,
        "train_samples": int(mN.group(1)) if mN else "",
        "lambda_J": lambda_j,
        "checkpoint_epoch": int(me.group(1)) if me else "",
    }


def read_checkpoint(run_dir: Path):
    path = run_dir / "settings.json"
    if not path.is_file():
        return ""
    try:
        settings = json.loads(path.read_text())
    except Exception:
        return ""
    emulator = settings.get("emulator", {})
    if isinstance(emulator, dict):
        return str(emulator.get("checkpoint", ""))
    return ""


def load_result(path: Path):
    required = {
        "recovered_u_minimum_u_error",
        "minimum_u_error_prediction",
        "minimum_u_error_iteration",
        "minimum_u_error",
        "history_iteration",
        "history_residual_norm",
        "history_relative_residual",
        "history_relative_u_error",
        "truth_u_resampled_model_grid",
        "data_noiseless_resampled",
        "data_observed_resampled",
        "observation_mask",
        "observation_weights",
        "surrogate_prediction_at_resampled_truth",
    }

    with np.load(path, allow_pickle=False) as z:
        missing = sorted(required.difference(z.files))
        if missing:
            raise KeyError(f"{path}: missing keys: {', '.join(missing)}")

        out = {k: np.asarray(z[k]).copy() for k in required}

    return out


def coarse_forward(
    solver,
    coefficient_map,
    state_map,
    u_grid: np.ndarray,
) -> np.ndarray:
    u = np.asarray(u_grid, dtype=np.float64)
    coefficient = coefficient_map.grid_to_array(u)
    solver.set_coefficient(coefficient)
    state = solver.forward()
    return np.asarray(state_map.state_to_grid(state), dtype=np.float64)


def main():
    args = parse_args()

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Run this diagnostic with exactly one MPI rank.")
    if args.fem_nx != 119 or args.fem_ny != 119:
        print(
            "WARNING: you changed the default 119x119-cell FEM. "
            "For pure emulator error use the training-resolution mesh.",
            flush=True,
        )

    run_entries = discover_runs(args.roots)
    args.output_dir = resolve_project(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields_dir = args.output_dir / "fields"
    fields_dir.mkdir(parents=True, exist_ok=True)

    first = load_result(run_entries[0][2])
    grid_shape = first["truth_u_resampled_model_grid"].shape
    expected = (args.fem_nx + 1, args.fem_ny + 1)
    if grid_shape != expected:
        raise ValueError(
            f"Saved model grid is {grid_shape}, but FEM vertices would be "
            f"{expected}. Use the training-resolution 119x119-cell FEM."
        )

    grid = RegularGrid2D.from_domain(
        1.0, 1.0, expected[0], expected[1]
    )

    # rhs=1 is the Reaction benchmark PDE used by these experiments.
    config = ReactionConfig(
        fem_nx=int(args.fem_nx),
        fem_ny=int(args.fem_ny),
        state_degree=1,
        rhs=1.0,
        u_min=float(args.u_min),
    )
    solver = ReactionSolver2D(
        config,
        comm=MPI.COMM_WORLD,
        minimum_u=float(args.u_min),
        initial_u=100.0,
        solver_options=solver_options(args),
    )

    rows = []

    try:
        coefficient_map = StructuredCoefficientGridMap(solver, grid)
        state_map = StructuredLagrangeStateGridMap(solver, grid)

        reference_truth = np.asarray(
            first["truth_u_resampled_model_grid"], dtype=np.float64
        )
        reference_y = np.asarray(
            first["data_noiseless_resampled"], dtype=np.float64
        )
        reference_mask = np.asarray(first["observation_mask"], dtype=bool)
        reference_weights = np.asarray(
            first["observation_weights"], dtype=np.float64
        )

        # G_h(u^dagger) is common to every run.
        print("Computing training-resolution FEM at u^dagger ...", flush=True)
        p_h_truth = coarse_forward(
            solver, coefficient_map, state_map, reference_truth
        )
        data_norm = weighted(
            reference_y, reference_weights, reference_mask
        )
        coarse_model_discrepancy_truth = weighted(
            p_h_truth - reference_y,
            reference_weights,
            reference_mask,
        )

        for idx, (campaign, run_dir, result_path) in enumerate(
            run_entries, start=1
        ):
            z = load_result(result_path)

            truth = np.asarray(
                z["truth_u_resampled_model_grid"], dtype=np.float64
            )
            y = np.asarray(
                z["data_noiseless_resampled"], dtype=np.float64
            )
            observed = np.asarray(
                z["data_observed_resampled"], dtype=np.float64
            )
            mask = np.asarray(z["observation_mask"], dtype=bool)
            weights = np.asarray(
                z["observation_weights"], dtype=np.float64
            )

            if not np.allclose(
                truth, reference_truth, rtol=1e-10, atol=1e-11
            ):
                raise ValueError(
                    f"{result_path}: truth differs from the reference truth."
                )
            if not np.allclose(
                y, reference_y, rtol=1e-10, atol=1e-11
            ):
                raise ValueError(
                    f"{result_path}: noiseless data differ from reference."
                )
            if not np.array_equal(mask, reference_mask):
                raise ValueError(
                    f"{result_path}: observation mask differs from reference."
                )
            if not np.allclose(
                weights, reference_weights, rtol=0.0, atol=1e-14
            ):
                raise ValueError(
                    f"{result_path}: observation weights differ from reference."
                )
            if not np.allclose(
                observed, y, rtol=1e-12, atol=1e-13
            ):
                raise ValueError(
                    f"{result_path}: expected a noiseless run, but observed "
                    "data differ from noiseless data."
                )

            u_star = np.asarray(
                z["recovered_u_minimum_u_error"], dtype=np.float64
            )
            p_theta_star = np.asarray(
                z["minimum_u_error_prediction"], dtype=np.float64
            )
            p_theta_truth = np.asarray(
                z["surrogate_prediction_at_resampled_truth"],
                dtype=np.float64,
            )

            iterations = np.asarray(
                z["history_iteration"], dtype=np.int64
            )
            hist_residual = np.asarray(
                z["history_residual_norm"], dtype=np.float64
            )
            hist_rel_residual = np.asarray(
                z["history_relative_residual"], dtype=np.float64
            )
            hist_u_error = np.asarray(
                z["history_relative_u_error"], dtype=np.float64
            )

            j = int(np.nanargmin(hist_u_error))
            k_star = int(iterations[j])
            saved_k = int(
                np.asarray(z["minimum_u_error_iteration"]).reshape(())
            )
            saved_eu = float(
                np.asarray(z["minimum_u_error"]).reshape(())
            )

            if k_star != saved_k:
                raise ValueError(
                    f"{result_path}: saved k*={saved_k}, history gives {k_star}."
                )
            if not np.isclose(
                saved_eu, hist_u_error[j], rtol=1e-10, atol=1e-12
            ):
                raise ValueError(
                    f"{result_path}: minimum u-error metadata mismatch."
                )

            r_theta_star = weighted(
                p_theta_star - y, weights, mask
            )

            # Strong check against the Landweber history at k*.
            if not np.isclose(
                r_theta_star,
                hist_residual[j],
                rtol=5e-8,
                atol=1e-13,
            ):
                raise ValueError(
                    f"{result_path}: recomputed surrogate residual at k* "
                    f"{r_theta_star:.12e} != saved history "
                    f"{hist_residual[j]:.12e}."
                )

            print(
                f"[{idx:02d}/{len(run_entries):02d}] "
                f"{run_dir.name}: FEM forward at u* (k={k_star})",
                flush=True,
            )
            p_h_star = coarse_forward(
                solver, coefficient_map, state_map, u_star
            )

            r_h_star = weighted(
                p_h_star - y, weights, mask
            )
            e_emu_star = weighted(
                p_h_star - p_theta_star, weights, mask
            )

            total_surrogate_truth = weighted(
                p_theta_truth - y, weights, mask
            )
            e_emu_truth = weighted(
                p_h_truth - p_theta_truth, weights, mask
            )

            # Triangle-inequality consistency diagnostics.
            triangle_star_gap = (
                e_emu_star - abs(r_h_star - r_theta_star)
            )
            triangle_truth_gap = (
                e_emu_truth
                - abs(
                    coarse_model_discrepancy_truth
                    - total_surrogate_truth
                )
            )

            semiconvergent = bool(k_star < int(iterations[-1]))
            meta = parse_run_name(run_dir.name)

            field_file = fields_dir / f"{run_dir.name}.npz"
            np.savez_compressed(
                field_file,
                u_star=u_star,
                u_truth=truth,
                data_y=y,
                p_theta_star=p_theta_star,
                p_h_star=p_h_star,
                p_theta_truth=p_theta_truth,
                p_h_truth=p_h_truth,
                surrogate_residual_star=p_theta_star - y,
                coarse_fem_residual_star=p_h_star - y,
                emulator_error_star=p_h_star - p_theta_star,
                surrogate_total_error_truth=p_theta_truth - y,
                coarse_model_error_truth=p_h_truth - y,
                emulator_error_truth=p_h_truth - p_theta_truth,
                observation_mask=mask,
                observation_weights=weights,
            )

            row = {
                "campaign": campaign,
                "run_name": run_dir.name,
                "architecture": meta["architecture"],
                "train_samples": meta["train_samples"],
                "lambda_J": meta["lambda_J"],
                "checkpoint_epoch": meta["checkpoint_epoch"],
                "checkpoint": read_checkpoint(run_dir),
                "k_star": k_star,
                "last_iteration": int(iterations[-1]),
                "semiconvergence": int(semiconvergent),
                "min_relative_u_error": saved_eu,
                "final_relative_u_error": float(hist_u_error[-1]),
                "semiconvergence_ratio_final_over_best": (
                    float(hist_u_error[-1]) / max(saved_eu, 1e-30)
                ),
                "data_norm_W": data_norm,

                # At the oracle best reconstruction.
                "surrogate_data_misfit_at_kstar": r_theta_star,
                "surrogate_data_misfit_at_kstar_rel": (
                    r_theta_star / max(data_norm, 1e-30)
                ),
                "coarse_fem_data_misfit_at_kstar": r_h_star,
                "coarse_fem_data_misfit_at_kstar_rel": (
                    r_h_star / max(data_norm, 1e-30)
                ),
                "emulator_error_at_kstar": e_emu_star,
                "emulator_error_at_kstar_rel": (
                    e_emu_star / max(data_norm, 1e-30)
                ),

                # At the known truth.
                "surrogate_total_error_at_truth": total_surrogate_truth,
                "surrogate_total_error_at_truth_rel": (
                    total_surrogate_truth / max(data_norm, 1e-30)
                ),
                "coarse_model_discrepancy_at_truth": (
                    coarse_model_discrepancy_truth
                ),
                "coarse_model_discrepancy_at_truth_rel": (
                    coarse_model_discrepancy_truth
                    / max(data_norm, 1e-30)
                ),
                "emulator_error_at_truth": e_emu_truth,
                "emulator_error_at_truth_rel": (
                    e_emu_truth / max(data_norm, 1e-30)
                ),

                # Ratios that directly answer "how far from an emulator error?"
                "ratio_rtheta_star_to_emulator_error_star": (
                    r_theta_star / max(e_emu_star, 1e-30)
                ),
                "ratio_rtheta_star_to_surrogate_total_error_truth": (
                    r_theta_star / max(total_surrogate_truth, 1e-30)
                ),
                "ratio_rh_star_to_emulator_error_star": (
                    r_h_star / max(e_emu_star, 1e-30)
                ),
                "ratio_rtheta_star_to_rh_star": (
                    r_theta_star / max(r_h_star, 1e-30)
                ),

                "history_relative_residual_at_kstar": (
                    float(hist_rel_residual[j])
                ),
                "triangle_gap_at_kstar": triangle_star_gap,
                "triangle_gap_at_truth": triangle_truth_gap,
                "fields_npz": str(field_file),
            }
            rows.append(row)

    finally:
        # ReactionSolver2D owns PETSc/FEniCS objects; allow normal destruction.
        del solver

    csv_path = args.output_dir / "min_u_emulator_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    semiconv = [r for r in rows if int(r["semiconvergence"]) == 1]
    summary = [
        "Reaction minimum-u / emulator-error diagnostic",
        "=" * 78,
        f"Runs evaluated                         : {len(rows)}",
        f"Runs with k* before final iteration    : {len(semiconv)}",
        f"Training-resolution FEM                : {args.fem_nx} x {args.fem_ny} cells",
        "",
        "Definitions:",
        "  r_theta* = ||G_theta(u*) - y||_W",
        "  r_h*     = ||G_h(u*) - y||_W",
        "  e_emu*   = ||G_h(u*) - G_theta(u*)||_W",
        "  e_tot^d  = ||G_theta(u^dagger) - y||_W",
        "  e_h^d    = ||G_h(u^dagger) - y||_W",
        "  e_emu^d  = ||G_h(u^dagger) - G_theta(u^dagger)||_W",
        "",
        f"Common coarse model discrepancy at truth / ||y||: "
        f"{rows[0]['coarse_model_discrepancy_at_truth_rel']:.6e}",
        "",
        "Semiconvergent runs (k* < final):",
    ]

    for r in sorted(
        semiconv,
        key=lambda x: float(x["min_relative_u_error"])
    ):
        summary.append(
            f"  {r['run_name']}: "
            f"k*={r['k_star']}, "
            f"e_u*={float(r['min_relative_u_error']):.6e}, "
            f"r_theta*/||y||={float(r['surrogate_data_misfit_at_kstar_rel']):.6e}, "
            f"e_emu*/||y||={float(r['emulator_error_at_kstar_rel']):.6e}, "
            f"r_h*/||y||={float(r['coarse_fem_data_misfit_at_kstar_rel']):.6e}"
        )

    txt_path = args.output_dir / "summary.txt"
    txt_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print()
    print("\n".join(summary))
    print()
    print("CSV   :", csv_path)
    print("Fields:", fields_dir)
    print("TXT   :", txt_path)


if __name__ == "__main__":
    main()
