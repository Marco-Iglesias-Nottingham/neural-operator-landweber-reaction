#!/usr/bin/env python3
"""
Build frozen downstream registries for the paper experiments.

Creates three validation-only registries:

1) headline_opt.tsv
   Frozen headline architecture; lambda_J selected by minimum S_val at EXACT
   epoch 2500; checkpoint selected by minimum S_val within epoch <= 2500.

2) headline_J0.tsv
   Same four frozen headline architectures, but lambda_J = 0 (no DINO/JVP
   supervision). Checkpoint selected by minimum S_val within epoch <= 2500.

3) ablations_J10.tsv
   Unique union of the supported one-axis capacity and training-set-size
   ablation tables. All tables share lambda_J = 10. Each checkpoint is the
   minimum-S_val validation checkpoint within epoch <= 2500.

No test data are read.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ARCH_ORDER = ["FNO", "DeepONet", "GenFNO", "GenDeepONet"]

HEADLINE_BASE_IDS = {
    "FNO": "fno_N10000_w24_m10_L4",
    "DeepONet": "do_N10000_Dp400_L6",
    "GenFNO": "gfno_N10000_im16_om24_h512_rb4",
    "GenDeepONet": "gdo_N10000_M400_N300_h1024_rb4",
}

EXPECTED_HEADLINE_LAMBDAS = {
    0.0, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0
}

ABLATION_FILES = [
    ("capacity", "fno_capacity_at2500.csv"),
    ("training_size", "fno_training_size_at2500.csv"),

    ("capacity", "deeponet_capacity_at2500.csv"),
    ("training_size", "deeponet_training_size_at2500.csv"),

    # Generalized architectures have controlled one-axis ablations.
    ("capacity", "genfno_output_modes_at2500.csv"),
    ("training_size", "genfno_training_size_at2500.csv"),
    ("capacity", "gendeeponet_hidden_width_at2500.csv"),
    ("training_size", "gendeeponet_training_size_at2500.csv"),
]

FIELDS = [
    "study",
    "ablation_axis",
    "architecture",
    "train_samples",
    "capacity",
    "dino_weight",
    "checkpoint_epoch",
    "E_F_val",
    "E_J_val",
    "S_val",
    "trainable_parameters",
    "run_id",
    "checkpoint",
]


def parse_args():
    p = argparse.ArgumentParser()
    root = Path("results/paper_T2_broad/budget2500_validation")
    p.add_argument("--input-dir", type=Path, default=root)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "landweber_registries",
    )
    p.add_argument(
        "--ablation-lambda",
        type=float,
        default=10.0,
        help="Common lambda_J required for the controlled ablation registries.",
    )
    p.add_argument(
        "--allow-missing-checkpoints",
        action="store_true",
        help=(
            "Exclude missing checkpoint files instead of failing. Missing runs "
            "are listed in missing_checkpoints.tsv."
        ),
    )
    p.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Build registries without checking GPFS checkpoint existence.",
    )
    p.add_argument(
        "--expected-models",
        type=Path,
        default=Path("configs/paper/downstream_models.tsv"),
        help="Frozen model/checkpoint choices that the regenerated selection must match.",
    )
    p.add_argument(
        "--skip-expected-check",
        action="store_true",
        help="Permit a newly selected model set to differ from the frozen paper set.",
    )
    return p.parse_args()


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def maybe_parameters(path: Path):
    if not path.is_file():
        return {}
    out = {}
    for r in read_csv(path):
        out[r["base_architecture_id"]] = int(float(r["trainable_parameters"]))
    return out


def f(x):
    return float(x)


def i(x):
    return int(float(x))


def is_available_at_budget(row):
    """Return whether an ablation-table row has reached the stated budget."""
    status = (row.get("completion_status") or "").strip()
    if status:
        if status == "available_at_budget":
            return True
        if status == "pending_completion":
            return False
        raise RuntimeError(f"Unknown completion_status={status!r}.")

    exact = (row.get("has_exact_budget_epoch") or "").strip().lower()
    if exact:
        if exact in {"true", "1", "yes"}:
            return True
        if exact in {"false", "0", "no"}:
            return False
        raise RuntimeError(f"Unknown has_exact_budget_epoch={exact!r}.")

    # Older exact-budget tables predate the explicit status columns.
    return True


def write_tsv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as fobj:
        w = csv.DictWriter(
            fobj,
            fieldnames=FIELDS,
            delimiter="\t",
            lineterminator="\n",  # avoids the CRLF problem in bash
        )
        w.writeheader()
        w.writerows(rows)


def row_from_best(best, *, study, axis, pmap):
    base = best["base_architecture_id"]
    return {
        "study": study,
        "ablation_axis": axis,
        "architecture": best["architecture"],
        "train_samples": i(best["train_samples"]),
        "capacity": best["capacity"],
        "dino_weight": f(best["dino_weight"]),
        "checkpoint_epoch": i(best["best_epoch_within_budget"]),
        "E_F_val": f(best["E_F_val_best"]),
        "E_J_val": f(best["E_J_val_best"]),
        "S_val": f(best["S_val_best"]),
        "trainable_parameters": pmap.get(base, ""),
        "run_id": best["run_id"],
        "checkpoint": best["checkpoint_best"],
    }


def validate_frozen_choices(path, headline_opt, headline_j0, ablations):
    with path.open(newline="", encoding="utf-8-sig") as fobj:
        expected_rows = list(csv.DictReader(fobj, delimiter="\t"))
    expected = {
        row["run_id"]: (
            row["downstream_role"],
            row["ablation_axis"],
            i(row["checkpoint_epoch"]),
            Path(row["checkpoint"]).name,
        )
        for row in expected_rows
    }
    actual = {}
    for role, rows in (
        ("headline_opt", headline_opt),
        ("headline_j0", headline_j0),
        ("ablation_j10", ablations),
    ):
        for row in rows:
            actual[row["run_id"]] = (
                role,
                row["ablation_axis"],
                i(row["checkpoint_epoch"]),
                Path(row["checkpoint"]).name,
            )
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        changed = sorted(
            run_id for run_id in set(actual) & set(expected)
            if actual[run_id] != expected[run_id]
        )
        raise RuntimeError(
            "Regenerated validation selection differs from the frozen downstream "
            f"manifest. Missing={missing}; unexpected={unexpected}; changed={changed}. "
            "Inspect the validation/retraining difference; use --skip-expected-check "
            "only to prepare a deliberate new model-selection decision."
        )


def main():
    args = parse_args()
    inp = args.input_dir
    out = args.output_dir

    best_rows = read_csv(inp / "all_runs_best_within2500.csv")
    best_by_run = {r["run_id"]: r for r in best_rows}
    pmap = maybe_parameters(inp / "trainable_parameters_at2500.csv")

    # ---------------------------------------------------------------
    # A. Headline optimal-lambda registry.
    # ---------------------------------------------------------------
    sweep = read_csv(inp / "headline_lambda_sweep_at2500.csv")
    headline_opt = []

    for arch in ARCH_ORDER:
        g = [r for r in sweep if r["architecture"] == arch]
        if not g:
            raise RuntimeError(f"No headline lambda sweep rows for {arch}.")
        epochs = {i(r["epoch"]) for r in g}
        if epochs != {2500}:
            raise RuntimeError(
                f"{arch}: headline lambda rows are not all exact epoch 2500."
            )
        got = {f(r["lambda_J"]) for r in g}
        if got != EXPECTED_HEADLINE_LAMBDAS:
            raise RuntimeError(
                f"{arch}: expected lambda set {sorted(EXPECTED_HEADLINE_LAMBDAS)}, "
                f"found {sorted(got)}."
            )
        at2500 = min(g, key=lambda r: f(r["S_val"]))
        run_id = at2500["run_id"]
        if run_id not in best_by_run:
            raise RuntimeError(f"Missing best-within-budget row for {run_id}.")
        headline_opt.append(
            row_from_best(
                best_by_run[run_id],
                study="headline_opt",
                axis="headline",
                pmap=pmap,
            )
        )

    # ---------------------------------------------------------------
    # B. Same headline architectures with lambda_J = 0.
    # ---------------------------------------------------------------
    headline_j0 = []
    for arch in ARCH_ORDER:
        base = HEADLINE_BASE_IDS[arch]
        candidates = [
            r for r in best_rows
            if r["base_architecture_id"] == base
            and abs(f(r["dino_weight"])) < 1e-14
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"{arch}: expected one lambda_J=0 row for {base}, "
                f"found {len(candidates)}."
            )
        headline_j0.append(
            row_from_best(
                candidates[0],
                study="headline_J0",
                axis="headline_no_dino",
                pmap=pmap,
            )
        )

    # ---------------------------------------------------------------
    # C. Controlled capacity + training-size ablation union.
    # ---------------------------------------------------------------
    axes_by_run = defaultdict(set)
    ablation_rows_by_run = {}
    common_lambdas = None
    pending_ablation_runs = set()

    for axis, filename in ABLATION_FILES:
        source_rows = read_csv(inp / filename)
        pending_ablation_runs.update(
            r["run_id"] for r in source_rows if not is_available_at_budget(r)
        )
        rows = [r for r in source_rows if is_available_at_budget(r)]
        if not rows:
            raise RuntimeError(
                f"No completed budget rows remain in ablation table {filename}."
            )
        lambdas = {f(r["dino_weight"]) for r in rows}
        common_lambdas = (
            lambdas if common_lambdas is None
            else common_lambdas.intersection(lambdas)
        )
        for r in rows:
            axes_by_run[r["run_id"]].add(axis)
            ablation_rows_by_run[r["run_id"]] = r

    if common_lambdas is None or not common_lambdas:
        raise RuntimeError("Ablation tables have no common lambda_J.")
    if args.ablation_lambda not in common_lambdas:
        raise RuntimeError(
            f"Requested ablation lambda_J={args.ablation_lambda:g}, but common "
            f"available values are {sorted(common_lambdas)}."
        )

    ablations = []
    for run_id in sorted(axes_by_run):
        source = ablation_rows_by_run[run_id]
        if abs(f(source["dino_weight"]) - args.ablation_lambda) > 1e-12:
            # Defensive: current controlled tables contain only J10.
            continue
        if run_id not in best_by_run:
            raise RuntimeError(f"Missing best-within-budget row for {run_id}.")
        axes = "+".join(sorted(axes_by_run[run_id]))
        ablations.append(
            row_from_best(
                best_by_run[run_id],
                study=f"ablations_J{args.ablation_lambda:g}",
                axis=axes,
                pmap=pmap,
            )
        )

    # Stable paper-friendly ordering.
    arch_rank = {a: j for j, a in enumerate(ARCH_ORDER)}
    def key(r):
        return (
            arch_rank[r["architecture"]],
            r["ablation_axis"],
            int(r["train_samples"]),
            r["capacity"],
            r["run_id"],
        )
    headline_opt.sort(key=key)
    headline_j0.sort(key=key)
    ablations.sort(key=key)

    expected_counts = {
        "headline_opt": (len(headline_opt), 4),
        "headline_j0": (len(headline_j0), 4),
        "ablation_j10": (len(ablations), 24),
    }
    wrong_counts = {
        name: (actual, expected)
        for name, (actual, expected) in expected_counts.items()
        if actual != expected
    }
    if wrong_counts:
        raise RuntimeError(f"Unexpected supported registry counts: {wrong_counts}")

    if not args.skip_expected_check:
        validate_frozen_choices(
            args.expected_models,
            headline_opt,
            headline_j0,
            ablations,
        )

    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Physical checkpoint audit.
    # ---------------------------------------------------------------
    all_sets = [
        ("headline_opt", headline_opt),
        ("headline_J0", headline_j0),
        ("ablations", ablations),
    ]
    missing = []
    if not args.skip_file_check:
        for set_name, rows in all_sets:
            for r in rows:
                if not Path(r["checkpoint"]).is_file():
                    missing.append({
                        "registry": set_name,
                        "run_id": r["run_id"],
                        "checkpoint": r["checkpoint"],
                    })

    missing_path = out / "missing_checkpoints.tsv"
    if missing:
        with missing_path.open("w", newline="", encoding="utf-8") as fobj:
            w = csv.DictWriter(
                fobj,
                fieldnames=["registry", "run_id", "checkpoint"],
                delimiter="\t",
                lineterminator="\n",
            )
            w.writeheader()
            w.writerows(missing)

        if not args.allow_missing_checkpoints:
            raise RuntimeError(
                f"{len(missing)} selected checkpoint(s) are missing. "
                f"See {missing_path}. No registry was written. "
                "Use --allow-missing-checkpoints only if you deliberately want "
                "to exclude unavailable models."
            )

        missing_ids = {m["run_id"] for m in missing}
        headline_opt = [r for r in headline_opt if r["run_id"] not in missing_ids]
        headline_j0 = [r for r in headline_j0 if r["run_id"] not in missing_ids]
        ablations = [r for r in ablations if r["run_id"] not in missing_ids]
    elif missing_path.exists():
        missing_path.unlink()

    write_tsv(out / "headline_opt.tsv", headline_opt)
    write_tsv(out / "headline_J0.tsv", headline_j0)
    write_tsv(out / "ablations_J10.tsv", ablations)

    if pending_ablation_runs:
        pending = ", ".join(sorted(pending_ablation_runs))
        print(f"Excluded pending budget run(s) from ablation registry: {pending}")

    summary = [
        "Reaction Landweber frozen registries",
        "=" * 80,
        "",
        "Selection uses validation only; no test data are read.",
        "",
        f"Headline optimal models : {len(headline_opt)}",
        f"Headline lambda_J=0     : {len(headline_j0)}",
        f"Ablation J10 models     : {len(ablations)}",
        f"Pending ablation runs   : {len(pending_ablation_runs)} (excluded)",
        f"Common ablation lambda  : {args.ablation_lambda:g}",
        "",
        "Headline optimal:",
    ]
    for r in headline_opt:
        summary.append(
            f"  {r['architecture']:<14s} J={r['dino_weight']:g} "
            f"epoch={r['checkpoint_epoch']:4d} S={r['S_val']:.6e} "
            f"{Path(r['checkpoint']).name}"
        )
    summary += ["", "Headline no-DINO (J=0):"]
    for r in headline_j0:
        summary.append(
            f"  {r['architecture']:<14s} epoch={r['checkpoint_epoch']:4d} "
            f"S={r['S_val']:.6e} {Path(r['checkpoint']).name}"
        )
    summary += [
        "",
        f"Ablation unique-run count: {len(ablations)} "
        "(union of all available capacity/training-size ablation tables).",
    ]
    for run_id in sorted(pending_ablation_runs):
        summary.append(f"  PENDING (excluded): {run_id}")

    text = "\n".join(summary) + "\n"
    (out / "registry_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"Registries: {out}")


if __name__ == "__main__":
    main()
