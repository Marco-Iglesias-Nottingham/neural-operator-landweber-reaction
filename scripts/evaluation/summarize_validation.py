#!/usr/bin/env python3
"""
Summarize paper validation results under a fixed training budget.

The intended paper protocol is:
    maximum training budget = B epochs (default B=2500)

This script DOES NOT run any network evaluations. It reuses the external-
validation CSVs already produced by the checkpoint evaluator.

For each run it:
  1. chooses the validation CSV reaching the largest epoch
     (canonical preferred on ties);
  2. retains the complete validation trajectory up to B;
  3. records metrics at EXACTLY epoch B when available;
  4. records the best validation checkpoint within e <= B as a diagnostic;
  5. builds the fixed-architecture lambda_J tables at epoch B;
  6. builds controlled capacity and training-size ablation tables at epoch B.

No test data are read.

Outputs:
  results/paper_T2_broad/budget2500_validation/
      all_validation_trajectories_through2500.csv
      all_runs_at2500.csv
      all_runs_best_within2500.csv
      missing_epoch2500.csv

      headline_lambda_sweep_at2500.csv
      headline_selected_lambdas_at2500.csv
      headline_lambda_FNO_at2500.csv
      headline_lambda_DeepONet_at2500.csv
      headline_lambda_GenFNO_at2500.csv
      headline_lambda_GenDeepONet_at2500.csv

      fno_capacity_at2500.csv
      fno_training_size_at2500.csv

      deeponet_capacity_at2500.csv
      deeponet_training_size_at2500.csv

      genfno_output_modes_at2500.csv
      genfno_training_size_at2500.csv

      gendeeponet_hidden_width_at2500.csv
      gendeeponet_training_size_at2500.csv

      low_data_representatives_at2500.csv
      summary.txt

IMPORTANT:
  * lambda_J selection in headline_selected_lambdas_at2500.csv uses ONLY
    S_val at the final budget epoch B, as requested.
  * all_runs_best_within2500.csv is provided only as a diagnostic / optional
    checkpoint-selection registry; it does not change the lambda_J table.
  * The supported run set comes from configs/paper/training_campaign.tsv.
    Recognized historical evaluations outside that manifest are ignored.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any


ARCH_ORDER = ["FNO", "DeepONet", "GenFNO", "GenDeepONet"]

HEADLINE_BASE = {
    "FNO": "fno_N10000_w24_m10_L4",
    "DeepONet": "do_N10000_Dp400_L6",
    "GenFNO": "gfno_N10000_im16_om24_h512_rb4",
    "GenDeepONet": "gdo_N10000_M400_N300_h1024_rb4",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path("results/checkpoint_evaluation"),
    )
    p.add_argument("--budget", type=int, default=2500)
    p.add_argument(
        "--campaign-manifest",
        type=Path,
        default=Path("configs/paper/training_campaign.tsv"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: results/paper_T2_broad/budget<B>_validation",
    )
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a partial diagnostic summary instead of requiring all runs at B.",
    )
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_campaign(path: Path, budget: int) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    run_ids = [row["run_id"] for row in rows]
    if len(run_ids) != 52 or len(set(run_ids)) != 52:
        raise RuntimeError(
            f"Expected 52 unique supported runs in {path}; found {len(run_ids)} rows."
        )
    wrong_budget = [
        row["run_id"] for row in rows
        if int(row["training_epochs"]) != budget
    ]
    if wrong_budget:
        raise RuntimeError(
            f"Campaign rows do not use budget {budget}: {', '.join(wrong_budget)}"
        )
    return run_ids


def write_csv(path: Path, rows: list[dict[str, Any]], fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def jval(text: str) -> float:
    return float(text.replace("p", "."))


def parse_run_id(run_id: str) -> dict[str, Any] | None:
    m = re.fullmatch(r"fno_N(\d+)_w(\d+)_m(\d+)_L(\d+)_J([0-9p.]+)", run_id)
    if m:
        return dict(
            architecture="FNO",
            train_samples=int(m.group(1)),
            dino_weight=jval(m.group(5)),
            width=int(m.group(2)),
            modes=int(m.group(3)),
            num_layers=int(m.group(4)),
            Dp="",
            input_modes="",
            output_modes="",
            hidden_width="",
            input_rank="",
            output_rank="",
            residual_blocks="",
            capacity=f"w={m.group(2)},m={m.group(3)}",
            base_architecture_id=(
                f"fno_N{m.group(1)}_w{m.group(2)}_m{m.group(3)}_L{m.group(4)}"
            ),
        )

    m = re.fullmatch(r"do_N(\d+)_Dp(\d+)_L(\d+)_J([0-9p.]+)", run_id)
    if m:
        return dict(
            architecture="DeepONet",
            train_samples=int(m.group(1)),
            dino_weight=jval(m.group(4)),
            width="",
            modes="",
            num_layers=int(m.group(3)),
            Dp=int(m.group(2)),
            input_modes="",
            output_modes="",
            hidden_width="",
            input_rank="",
            output_rank="",
            residual_blocks="",
            capacity=f"Dp={m.group(2)}",
            base_architecture_id=(
                f"do_N{m.group(1)}_Dp{m.group(2)}_L{m.group(3)}"
            ),
        )

    m = re.fullmatch(
        r"gfno_N(\d+)_im(\d+)_om(\d+)_h(\d+)_rb(\d+)_J([0-9p.]+)",
        run_id,
    )
    if m:
        return dict(
            architecture="GenFNO",
            train_samples=int(m.group(1)),
            dino_weight=jval(m.group(6)),
            width="",
            modes="",
            num_layers="",
            Dp="",
            input_modes=int(m.group(2)),
            output_modes=int(m.group(3)),
            hidden_width=int(m.group(4)),
            input_rank="",
            output_rank="",
            residual_blocks=int(m.group(5)),
            capacity=f"im={m.group(2)},om={m.group(3)},h={m.group(4)}",
            base_architecture_id=(
                f"gfno_N{m.group(1)}_im{m.group(2)}_om{m.group(3)}_"
                f"h{m.group(4)}_rb{m.group(5)}"
            ),
        )

    m = re.fullmatch(
        r"gdo_N(\d+)_M(\d+)_N(\d+)_h(\d+)_rb(\d+)_J([0-9p.]+)",
        run_id,
    )
    if m:
        return dict(
            architecture="GenDeepONet",
            train_samples=int(m.group(1)),
            dino_weight=jval(m.group(6)),
            width="",
            modes="",
            num_layers="",
            Dp="",
            input_modes="",
            output_modes="",
            hidden_width=int(m.group(4)),
            input_rank=int(m.group(2)),
            output_rank=int(m.group(3)),
            residual_blocks=int(m.group(5)),
            capacity=f"M={m.group(2)},N={m.group(3)},h={m.group(4)}",
            base_architecture_id=(
                f"gdo_N{m.group(1)}_M{m.group(2)}_N{m.group(3)}_"
                f"h{m.group(4)}_rb{m.group(5)}"
            ),
        )

    return None


RUN_PREFIX_PATTERNS = [
    re.compile(r"^(fno_N\d+_w\d+_m\d+_L\d+_J[0-9p.]+)"),
    re.compile(r"^(do_N\d+_Dp\d+_L\d+_J[0-9p.]+)"),
    re.compile(r"^(gfno_N\d+_im\d+_om\d+_h\d+_rb\d+_J[0-9p.]+)"),
    re.compile(r"^(gdo_N\d+_M\d+_N\d+_h\d+_rb\d+_J[0-9p.]+)"),
]


def run_id_from_validation_file(path: Path):
    suffix = ".validation.csv"
    if not path.name.endswith(suffix):
        return None
    stem = path.name[: -len(suffix)]

    for pat in RUN_PREFIX_PATTERNS:
        m = pat.match(stem)
        if m:
            run_id = m.group(1)
            if parse_run_id(run_id) is None:
                return None
            source = "canonical" if stem == run_id else stem[len(run_id):].lstrip("_")
            return run_id, source or "canonical"
    return None


def max_epoch_within(rows, budget):
    epochs = [int(float(r["epoch"])) for r in rows]
    return max((epoch for epoch in epochs if epoch <= budget), default=-1)


def metric_row(raw: dict[str, str]) -> dict[str, Any]:
    ef = float(raw["forward_pooled_relative_l2"])
    ej = float(raw["jvp_pooled_relative_l2"])
    s = float(raw.get("joint_score") or math.sqrt(ef * ef + ej * ej))
    return dict(
        epoch=int(float(raw["epoch"])),
        E_F_val=ef,
        E_J_val=ej,
        S_val=s,
        checkpoint=raw["checkpoint"],
    )


def at_budget_record(run_id, meta, source, path, trajectory, budget):
    exact = next((r for r in trajectory if r["epoch"] == budget), None)
    within = [r for r in trajectory if r["epoch"] <= budget]
    return dict(
        run_id=run_id,
        **meta,
        budget_epoch=budget,
        validated_through_epoch=(within[-1]["epoch"] if within else ""),
        has_exact_budget_epoch=bool(exact),
        E_F_val_at_budget="" if exact is None else exact["E_F_val"],
        E_J_val_at_budget="" if exact is None else exact["E_J_val"],
        S_val_at_budget="" if exact is None else exact["S_val"],
        checkpoint_at_budget="" if exact is None else exact["checkpoint"],
        validation_source=source,
        validation_csv=str(path),
    )


def best_within_record(run_id, meta, source, path, trajectory, budget):
    within = [r for r in trajectory if r["epoch"] <= budget]
    if not within:
        return None
    best = min(within, key=lambda r: r["S_val"])
    last = within[-1]
    return dict(
        run_id=run_id,
        **meta,
        budget_epoch=budget,
        last_available_epoch_within_budget=last["epoch"],
        best_epoch_within_budget=best["epoch"],
        E_F_val_best=best["E_F_val"],
        E_J_val_best=best["E_J_val"],
        S_val_best=best["S_val"],
        checkpoint_best=best["checkpoint"],
        validation_source=source,
        validation_csv=str(path),
    )


def exact_rows(rows):
    return [r for r in rows if r["has_exact_budget_epoch"]]


def select(rows, predicate, sort_key):
    x = [r for r in rows if predicate(r)]
    x.sort(key=sort_key)
    return x


def reduced_table(rows, x_name, x_fn, *, include_completion=False):
    out = []
    for r in rows:
        row = dict(
            architecture=r["architecture"],
            **{x_name: x_fn(r)},
            train_samples=r["train_samples"],
            dino_weight=r["dino_weight"],
        )
        if include_completion:
            complete = bool(r["has_exact_budget_epoch"])
            row.update(
                budget_epoch=r["budget_epoch"],
                has_exact_budget_epoch=complete,
                validated_through_epoch=r["validated_through_epoch"],
                completion_status=(
                    "available_at_budget" if complete else "pending_completion"
                ),
            )
        row.update(
            E_F_val=r["E_F_val_at_budget"],
            E_J_val=r["E_J_val_at_budget"],
            S_val=r["S_val_at_budget"],
            run_id=r["run_id"],
        )
        out.append(row)
    return out


def fmt(v):
    if v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.6e}"
    return str(v)


def main():
    args = parse_args()
    budget = args.budget
    campaign_run_ids = read_campaign(args.campaign_manifest, budget)
    campaign_set = set(campaign_run_ids)
    outdir = args.output_dir or Path(
        f"results/paper_T2_broad/budget{budget}_validation"
    )

    # ------------------------------------------------------------------
    # Find the most complete validation trajectory for every run.
    # ------------------------------------------------------------------
    candidates = {}
    ignored_run_ids = set()
    for path in sorted(args.evaluation_dir.glob("*.validation.csv")):
        parsed = run_id_from_validation_file(path)
        if parsed is None:
            continue
        run_id, source = parsed
        if run_id not in campaign_set:
            ignored_run_ids.add(run_id)
            continue
        raw = read_csv(path)
        if raw:
            candidates.setdefault(run_id, []).append((path, source, raw))

    chosen = {}
    for run_id, versions in candidates.items():
        versions.sort(
            key=lambda x: (
                max_epoch_within(x[2], budget),
                1 if x[1] == "canonical" else 0,
            ),
            reverse=True,
        )
        path, source, raw = versions[0]
        trajectory = sorted((metric_row(r) for r in raw), key=lambda r: r["epoch"])
        chosen[run_id] = (path, source, trajectory)

    missing_validations = [
        run_id for run_id in campaign_run_ids if run_id not in chosen
    ]
    if missing_validations and not args.allow_incomplete:
        raise RuntimeError(
            f"Missing validation trajectories for {len(missing_validations)} of "
            f"52 supported runs: {', '.join(missing_validations)}"
        )

    long_rows = []
    at_budget = []
    best_within = []

    for run_id, (path, source, trajectory) in chosen.items():
        meta = parse_run_id(run_id)
        assert meta is not None

        for r in trajectory:
            if r["epoch"] <= budget:
                long_rows.append(
                    dict(
                        run_id=run_id,
                        **meta,
                        validation_source=source,
                        validation_csv=str(path),
                        **r,
                    )
                )

        at_budget.append(
            at_budget_record(run_id, meta, source, path, trajectory, budget)
        )
        b = best_within_record(run_id, meta, source, path, trajectory, budget)
        if b is not None:
            best_within.append(b)

    def master_sort(r):
        return (
            ARCH_ORDER.index(r["architecture"]),
            int(r["train_samples"]),
            str(r["capacity"]),
            float(r["dino_weight"]),
        )

    long_rows.sort(
        key=lambda r: (
            ARCH_ORDER.index(r["architecture"]),
            r["run_id"],
            r["epoch"],
        )
    )
    at_budget.sort(key=master_sort)
    best_within.sort(key=master_sort)

    missing = [r for r in at_budget if not r["has_exact_budget_epoch"]]
    if missing and not args.allow_incomplete:
        missing_text = ", ".join(
            f"{row['run_id']} (through {row['validated_through_epoch']})"
            for row in missing
        )
        raise RuntimeError(
            f"Supported runs missing exact epoch {budget}: {missing_text}"
        )

    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(
        outdir / f"all_validation_trajectories_through{budget}.csv",
        long_rows,
    )
    write_csv(outdir / f"all_runs_at{budget}.csv", at_budget)
    write_csv(outdir / f"all_runs_best_within{budget}.csv", best_within)

    write_csv(outdir / f"missing_epoch{budget}.csv", missing)
    write_csv(
        outdir / "ignored_runs_outside_campaign.csv",
        [{"run_id": run_id} for run_id in sorted(ignored_run_ids)],
        fields=["run_id"],
    )

    exact = exact_rows(at_budget)

    # ------------------------------------------------------------------
    # 1) Lambda_J sweep: fixed representative architectures, Ntrain=10k.
    #    Selection uses S_val AT EXACTLY the budget epoch.
    # ------------------------------------------------------------------
    lambda_rows = []
    selected_lambda_rows = []

    for arch in ARCH_ORDER:
        base = HEADLINE_BASE[arch]
        group = [
            r for r in exact
            if r["base_architecture_id"] == base
        ]
        group.sort(key=lambda r: float(r["dino_weight"]))

        if group:
            best = min(group, key=lambda r: float(r["S_val_at_budget"]))
            for r in group:
                lambda_rows.append(
                    dict(
                        architecture=arch,
                        frozen_architecture=r["capacity"],
                        train_samples=r["train_samples"],
                        lambda_J=r["dino_weight"],
                        epoch=budget,
                        E_F_val=r["E_F_val_at_budget"],
                        E_J_val=r["E_J_val_at_budget"],
                        S_val=r["S_val_at_budget"],
                        selected_lambda=(
                            r["run_id"] == best["run_id"]
                        ),
                        run_id=r["run_id"],
                        checkpoint=r["checkpoint_at_budget"],
                    )
                )

            selected_lambda_rows.append(
                dict(
                    architecture=arch,
                    frozen_architecture=best["capacity"],
                    train_samples=best["train_samples"],
                    lambda_J_star=best["dino_weight"],
                    epoch=budget,
                    E_F_val=best["E_F_val_at_budget"],
                    E_J_val=best["E_J_val_at_budget"],
                    S_val=best["S_val_at_budget"],
                    run_id=best["run_id"],
                    checkpoint=best["checkpoint_at_budget"],
                )
            )

    write_csv(
        outdir / f"headline_lambda_sweep_at{budget}.csv",
        lambda_rows,
    )
    write_csv(
        outdir / f"headline_selected_lambdas_at{budget}.csv",
        selected_lambda_rows,
    )

    for arch in ARCH_ORDER:
        write_csv(
            outdir / f"headline_lambda_{arch}_at{budget}.csv",
            [r for r in lambda_rows if r["architecture"] == arch],
        )

    # ------------------------------------------------------------------
    # 2) Controlled ablations at the nominal campaign weight lambda_J=10.
    # ------------------------------------------------------------------

    # Standard FNO capacity: Ntrain=10k, J10.
    fno_cap = select(
        exact,
        lambda r: (
            r["architecture"] == "FNO"
            and r["train_samples"] == 10000
            and float(r["dino_weight"]) == 10.0
        ),
        lambda r: int(r["width"]),
    )
    write_csv(
        outdir / f"fno_capacity_at{budget}.csv",
        reduced_table(
            fno_cap, "capacity_setting",
            lambda r: f"w={r['width']},m={r['modes']}"
        ),
    )

    # Standard FNO training size: frozen w24,m10,L4, J10.
    fno_size = select(
        exact,
        lambda r: (
            r["architecture"] == "FNO"
            and int(r["width"]) == 24
            and int(r["modes"]) == 10
            and int(r["num_layers"]) == 4
            and float(r["dino_weight"]) == 10.0
        ),
        lambda r: int(r["train_samples"]),
    )
    write_csv(
        outdir / f"fno_training_size_at{budget}.csv",
        reduced_table(
            fno_size, "training_size",
            lambda r: int(r["train_samples"])
        ),
    )

    # Standard DeepONet capacity: Ntrain=10k, J10.
    do_cap = select(
        exact,
        lambda r: (
            r["architecture"] == "DeepONet"
            and r["train_samples"] == 10000
            and float(r["dino_weight"]) == 10.0
        ),
        lambda r: int(r["Dp"]),
    )
    write_csv(
        outdir / f"deeponet_capacity_at{budget}.csv",
        reduced_table(
            do_cap, "capacity_setting",
            lambda r: f"Dp={r['Dp']}"
        ),
    )

    # Standard DeepONet training size: frozen Dp400,L6,J10.
    do_size = select(
        exact,
        lambda r: (
            r["architecture"] == "DeepONet"
            and int(r["Dp"]) == 400
            and int(r["num_layers"]) == 6
            and float(r["dino_weight"]) == 10.0
        ),
        lambda r: int(r["train_samples"]),
    )
    write_csv(
        outdir / f"deeponet_training_size_at{budget}.csv",
        reduced_table(
            do_size, "training_size",
            lambda r: int(r["train_samples"])
        ),
    )

    # GenFNO output truncation/capacity:
    # Ntrain=10k, im16,h512,rb4,J10; om varies.
    gf_cap = select(
        exact,
        lambda r: (
            r["architecture"] == "GenFNO"
            and r["train_samples"] == 10000
            and int(r["input_modes"]) == 16
            and int(r["hidden_width"]) == 512
            and int(r["residual_blocks"]) == 4
            and float(r["dino_weight"]) == 10.0
        ),
        lambda r: int(r["output_modes"]),
    )
    write_csv(
        outdir / f"genfno_output_modes_at{budget}.csv",
        reduced_table(
            gf_cap, "output_modes",
            lambda r: int(r["output_modes"])
        ),
    )

    # GenFNO training size: im16,om24,h512,rb4,J10.
    gf_size = select(
        exact,
        lambda r: (
            r["architecture"] == "GenFNO"
            and int(r["input_modes"]) == 16
            and int(r["output_modes"]) == 24
            and int(r["hidden_width"]) == 512
            and int(r["residual_blocks"]) == 4
            and float(r["dino_weight"]) == 10.0
        ),
        lambda r: int(r["train_samples"]),
    )
    write_csv(
        outdir / f"genfno_training_size_at{budget}.csv",
        reduced_table(
            gf_size, "training_size",
            lambda r: int(r["train_samples"])
        ),
    )

    # GenDeepONet neural capacity:
    # Ntrain=10k,M400,N300,rb4,J10, hidden width varies.
    gd_cap = select(
        exact,
        lambda r: (
            r["architecture"] == "GenDeepONet"
            and r["train_samples"] == 10000
            and int(r["input_rank"]) == 400
            and int(r["output_rank"]) == 300
            and int(r["residual_blocks"]) == 4
            and float(r["dino_weight"]) == 10.0
        ),
        lambda r: int(r["hidden_width"]),
    )
    write_csv(
        outdir / f"gendeeponet_hidden_width_at{budget}.csv",
        reduced_table(
            gd_cap, "hidden_width",
            lambda r: int(r["hidden_width"])
        ),
    )

    # GenDeepONet training size: M400,N300,h1024,rb4,J10.
    gd_size = select(
        exact,
        lambda r: (
            r["architecture"] == "GenDeepONet"
            and int(r["input_rank"]) == 400
            and int(r["output_rank"]) == 300
            and int(r["hidden_width"]) == 1024
            and int(r["residual_blocks"]) == 4
            and float(r["dino_weight"]) == 10.0
        ),
        lambda r: int(r["train_samples"]),
    )
    write_csv(
        outdir / f"gendeeponet_training_size_at{budget}.csv",
        reduced_table(
            gd_size, "training_size",
            lambda r: int(r["train_samples"])
        ),
    )

    # Controlled low-data representatives: same representative architecture,
    # Ntrain=1000, lambda_J=10. These are candidates for the deliberately
    # lower-accuracy Landweber study; they are NOT automatically declared bad.
    low_data = []
    low_defs = {
        "FNO": "fno_N1000_w24_m10_L4",
        "DeepONet": "do_N1000_Dp400_L6",
        "GenFNO": "gfno_N1000_im16_om24_h512_rb4",
        "GenDeepONet": "gdo_N1000_M400_N300_h1024_rb4",
    }
    for arch in ARCH_ORDER:
        base = low_defs[arch]
        matches = [
            r for r in exact
            if r["base_architecture_id"] == base
            and float(r["dino_weight"]) == 10.0
        ]
        for r in matches:
            low_data.append(
                dict(
                    architecture=arch,
                    capacity=r["capacity"],
                    train_samples=r["train_samples"],
                    lambda_J=r["dino_weight"],
                    epoch=budget,
                    E_F_val=r["E_F_val_at_budget"],
                    E_J_val=r["E_J_val_at_budget"],
                    S_val=r["S_val_at_budget"],
                    run_id=r["run_id"],
                )
            )
    write_csv(
        outdir / f"low_data_representatives_at{budget}.csv",
        low_data,
    )

    # ------------------------------------------------------------------
    # Human-readable audit summary.
    # ------------------------------------------------------------------
    lines = []
    lines.append(f"Reaction validation summary with training budget B={budget}")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Validation runs found: {len(at_budget)}")
    lines.append(f"Runs with exact epoch {budget}: {len(exact)}")
    lines.append(f"Runs missing exact epoch {budget}: {len(missing)}")
    lines.append(
        f"Historical runs outside supported manifest ignored: {len(ignored_run_ids)}"
    )
    lines.append("No test data were read.")
    lines.append("")

    if missing:
        lines.append(f"RUNS MISSING EXACT EPOCH {budget}")
        lines.append("-" * 78)
        for r in missing:
            lines.append(
                f"{r['run_id']}: validated through "
                f"{r['validated_through_epoch']}"
            )
        lines.append("")

    lines.append(f"HEADLINE LAMBDA_J SELECTION AT EXACT EPOCH {budget}")
    lines.append("-" * 78)
    for r in selected_lambda_rows:
        lines.append(
            f"{r['architecture']:<16s} "
            f"lambda_J*={float(r['lambda_J_star']):g}  "
            f"E_F={float(r['E_F_val']):.6e}  "
            f"E_J={float(r['E_J_val']):.6e}  "
            f"S={float(r['S_val']):.6e}"
        )

    lines.append("")
    lines.append("CONTROLLED ABLATIONS PRODUCED")
    lines.append("-" * 78)
    lines.append("Standard FNO:")
    lines.append("  capacity @ N=10k, lambda_J=10")
    lines.append("  training size @ w=24,m=10,L=4, lambda_J=10")
    lines.append("Standard DeepONet:")
    lines.append("  capacity @ N=10k, lambda_J=10")
    lines.append("  training size @ Dp=400,L=6, lambda_J=10")
    lines.append("GenFNO:")
    lines.append("  output modes @ N=10k, im=16,h=512,rb=4, lambda_J=10")
    lines.append("  training size @ im=16,om=24,h=512,rb=4, lambda_J=10")
    lines.append("GenDeepONet:")
    lines.append("  hidden width @ N=10k,M=400,N=300,rb=4, lambda_J=10")
    lines.append("  training size @ M=400,N=300,h=1024,rb=4, lambda_J=10")
    lines.append("")
    lines.append(
        "NOTE: om64 is not included because that configuration is invalid "
        "on the 120x120 grid (2*modes_x <= nx)."
    )

    summary_path = outdir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print()
    print(f"Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
