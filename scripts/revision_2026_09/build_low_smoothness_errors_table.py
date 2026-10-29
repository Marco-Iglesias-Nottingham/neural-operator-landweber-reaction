#!/usr/bin/env python3
"""Build the low-smoothness minimum-error table from the frozen E3 CSVs."""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
E3 = REPO / "paper_products/generated/revision/E3"
CAMPAIGN = E3 / "trajectory_outcomes_pre_geometry.csv"
REFERENCES = E3 / "reference_arms.csv"
WORKING_OUTPUT = E3 / "preconditioner_low_smoothness_errors.tex"
REFERENCE_TABLE = REPO / "reference/paper/preconditioner_low_smoothness_errors.tex"
DISPLAY = REPO / "outputs/tables"
DISPLAY_OUTPUT = DISPLAY / "preconditioner_low_smoothness_errors.tex"
PROVENANCE = DISPLAY / "PROVENANCE.json"

TRUTHS = ("T1", "T2")
METHODS = ("ExactFEM", "FNO", "DeepONet", "GenFNO", "GenDeepONet")
DISPLAY_METHOD = {"ExactFEM": "Exact FEM", "FNO": "FNO", "DeepONet": "DeepONet",
                  "GenFNO": "GenFNO", "GenDeepONet": "GenDeepONet"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def unique(rows: list[dict[str, str]], *, source: Path, **criteria: str) -> dict[str, str] | None:
    matches = [row for row in rows if all(row[key] == value for key, value in criteria.items())]
    if len(matches) > 1:
        raise AssertionError(f"Duplicate rows in {source}: {criteria}")
    return matches[0] if matches else None


def percent(row: dict[str, str] | None) -> str:
    if row is None:
        return "--"
    return f"{100.0 * float(row['min_relative_coefficient_error']):.2f}"


def build_table(campaign: list[dict[str, str]], references: list[dict[str, str]]) -> tuple[str, int]:
    body: list[str] = []
    checked_cells = 0
    for truth_index, truth in enumerate(TRUTHS):
        for method in METHODS:
            baseline = unique(
                references,
                source=REFERENCES,
                truth=truth,
                architecture=method,
                configuration_id="baseline_nu1p5_ell0p1",
            )
            nu025 = unique(
                campaign,
                source=CAMPAIGN,
                truth=truth,
                architecture=method,
                matern_nu="0.25",
                matern_length_scale="0.1",
            )
            nu04 = unique(
                campaign,
                source=CAMPAIGN,
                truth=truth,
                architecture=method,
                matern_nu="0.4",
                matern_length_scale="0.1",
            )
            identity = unique(
                references,
                source=REFERENCES,
                truth=truth,
                architecture=method,
                configuration_id="identity",
            )
            if baseline is None or nu025 is None or nu04 is None:
                raise AssertionError(f"Missing required Matérn arm for {truth}/{method}")
            if identity is None and not (truth == "T1" and method == "ExactFEM"):
                raise AssertionError(f"Unexpected missing identity arm for {truth}/{method}")
            values = tuple(percent(row) for row in (baseline, nu025, nu04, identity))
            checked_cells += len(values)
            body.append(f"{truth} & {DISPLAY_METHOD[method]} & " + " & ".join(values) + r" \\")
        if truth_index == 0:
            body.append(r"\midrule")

    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Mat\'ern, $\ell_C=0.1$} & \\",
        r"\cmidrule(lr){3-5}",
        r"Truth & Method & $\nu_C=3/2$ & $\nu_C=0.25$ & $\nu_C=0.4$ & Identity \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n", checked_cells


def update_provenance() -> None:
    payload: dict[str, Any] = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    output_name = "display_assets/preconditioner_low_smoothness_errors.tex"
    payload["products"] = [row for row in payload["products"] if row.get("output") != output_name]
    payload["products"].append(
        {
            "source": "revision/E3/preconditioner_low_smoothness_errors.tex",
            "source_sha256": sha256(WORKING_OUTPUT),
            "output": output_name,
            "output_sha256": sha256(DISPLAY_OUTPUT),
            "change": (
                "Generate the 40-cell minimum relative coefficient-error comparison from the frozen "
                "E3 low-smoothness and reference-arm CSVs; two-decimal percentages match the reviewed table."
            ),
            "data_sources": [
                {
                    "path": "revision/E3/trajectory_outcomes_pre_geometry.csv",
                    "sha256": sha256(CAMPAIGN),
                },
                {"path": "revision/E3/reference_arms.csv", "sha256": sha256(REFERENCES)},
            ],
        }
    )
    PROVENANCE.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    campaign = read_rows(CAMPAIGN)
    references = read_rows(REFERENCES)
    if len(campaign) != 60 or len(references) != 19:
        raise AssertionError(f"Unexpected E3 row counts: campaign={len(campaign)}, references={len(references)}")
    generated, checked_cells = build_table(campaign, references)
    reviewed = REFERENCE_TABLE.read_text(encoding="utf-8")
    if generated != reviewed:
        difference = "".join(
            difflib.unified_diff(
                reviewed.splitlines(keepends=True),
                generated.splitlines(keepends=True),
                fromfile=str(REFERENCE_TABLE),
                tofile=str(WORKING_OUTPUT),
            )
        )
        raise AssertionError("Generated table does not match the reviewed table:\n" + difference)

    WORKING_OUTPUT.write_text(generated, encoding="utf-8")
    DISPLAY.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(WORKING_OUTPUT, DISPLAY_OUTPUT)
    if DISPLAY_OUTPUT.read_bytes() != REFERENCE_TABLE.read_bytes():
        raise AssertionError("Display table differs from the reviewed table after writing")
    update_provenance()
    print(f"Wrote {DISPLAY_OUTPUT.relative_to(REPO)}")
    print(f"Confirmed all {checked_cells} displayed cells against both frozen CSV sources and the reviewed table.")


if __name__ == "__main__":
    main()
