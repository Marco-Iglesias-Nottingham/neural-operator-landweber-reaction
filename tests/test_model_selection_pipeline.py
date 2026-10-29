#!/usr/bin/env python3
"""Integration checks for manifest-scoped validation summary and selection."""

from __future__ import annotations

import csv
import math
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "paper"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_trajectory(
    directory: Path,
    run_id: str,
    budget_score: float,
    selected_epoch: int | None,
    *,
    source_suffix: str = "",
    include_post_budget: bool = False,
) -> None:
    fields = [
        "epoch",
        "forward_pooled_relative_l2",
        "jvp_pooled_relative_l2",
        "joint_score",
        "checkpoint",
    ]

    def row(epoch: int, score: float) -> dict[str, object]:
        component = score / math.sqrt(2.0)
        return {
            "epoch": epoch,
            "forward_pooled_relative_l2": component,
            "jvp_pooled_relative_l2": component,
            "joint_score": score,
            "checkpoint": f"checkpoints/paper_T2_broad/{run_id}.epoch{epoch:04d}.pt",
        }

    rows = []
    if selected_epoch is not None and selected_epoch != 2500:
        rows.append(row(selected_epoch, 0.01))
    rows.append(row(2500, budget_score))
    if include_post_budget:
        rows.append(row(3000, budget_score / 10.0))
    with (directory / f"{run_id}{source_suffix}.validation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    training = read_tsv(CONFIG / "training_campaign.tsv")
    downstream = read_tsv(CONFIG / "downstream_models.tsv")
    expected = {row["run_id"]: row for row in downstream}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        evaluations = root / "evaluations"
        summary = root / "summary"
        registries = root / "registries"
        evaluations.mkdir()

        for row in training:
            run_id = row["run_id"]
            selected = expected.get(run_id)
            # The frozen optimized lambda must win each exact-budget sweep.
            budget_score = (
                0.1
                if selected and selected["downstream_role"] == "headline_opt"
                else 1.0
            )
            selected_epoch = (
                int(selected["checkpoint_epoch"]) if selected else None
            )
            write_trajectory(
                evaluations,
                run_id,
                budget_score,
                selected_epoch,
            )

        # A recognized historical cross-grid run must not enter any output.
        write_trajectory(
            evaluations,
            "fno_N5000_w8_m4_L4_J10",
            0.001,
            2500,
        )
        first_run = training[0]["run_id"]
        write_trajectory(
            evaluations,
            first_run,
            1.0e-9,
            2500,
            source_suffix="_historical_postbudget",
            include_post_budget=True,
        )

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluation" / "summarize_validation.py"),
                "--evaluation-dir",
                str(evaluations),
                "--campaign-manifest",
                str(CONFIG / "training_campaign.tsv"),
                "--output-dir",
                str(summary),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )

        with (summary / "all_runs_at2500.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            at_budget = list(csv.DictReader(handle))
        assert len(at_budget) == 52
        assert {row["run_id"] for row in at_budget} == {
            row["run_id"] for row in training
        }
        first_summary = next(row for row in at_budget if row["run_id"] == first_run)
        assert first_summary["validation_source"] == "canonical"
        assert not list(summary.glob("*capacity_x_training_size*"))
        with (summary / "ignored_runs_outside_campaign.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            ignored = list(csv.DictReader(handle))
        assert ignored == [{"run_id": "fno_N5000_w8_m4_L4_J10"}]

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluation" / "select_inversion_models.py"),
                "--input-dir",
                str(summary),
                "--output-dir",
                str(registries),
                "--expected-models",
                str(CONFIG / "downstream_models.tsv"),
                "--skip-file-check",
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )

        actual: dict[str, tuple[str, str, int, str]] = {}
        for filename, role in (
            ("headline_opt.tsv", "headline_opt"),
            ("headline_J0.tsv", "headline_j0"),
            ("ablations_J10.tsv", "ablation_j10"),
        ):
            for row in read_tsv(registries / filename):
                actual[row["run_id"]] = (
                    role,
                    row["ablation_axis"],
                    int(row["checkpoint_epoch"]),
                    Path(row["checkpoint"]).name,
                )
        frozen = {
            run_id: (
                row["downstream_role"],
                row["ablation_axis"],
                int(row["checkpoint_epoch"]),
                Path(row["checkpoint"]).name,
            )
            for run_id, row in expected.items()
        }
        assert actual == frozen

    print("Manifest-scoped model-selection pipeline checks passed.")


if __name__ == "__main__":
    main()
