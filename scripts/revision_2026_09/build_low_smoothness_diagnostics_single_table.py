#!/usr/bin/env python3
"""Build the single-geometry E1/E2 diagnostic table from its frozen CSV."""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
E1_E2 = REPO / "paper_products/generated/revision/E1_E2"
SUMMARY = E1_E2 / "trajectory_diagnostics_summary.csv"
WORKING_OUTPUT = E1_E2 / "low_smoothness_diagnostics_single.tex"
REFERENCE_TABLE = REPO / "reference/paper/table_low_smoothness_diagnostics_single.tex"
DISPLAY = REPO / "outputs/tables"
DISPLAY_OUTPUT = DISPLAY / "low_smoothness_diagnostics_single.tex"
PROVENANCE = DISPLAY / "PROVENANCE.json"

CONFIGURATION = "nu0p25_ell0p1"
TRUTHS = ("T1", "T2")
METHODS = ("ExactFEM", "FNO", "DeepONet", "GenFNO", "GenDeepONet")
DISPLAY_METHOD = {
    "ExactFEM": "Exact FEM",
    "FNO": "FNO",
    "DeepONet": "DeepONet",
    "GenFNO": "GenFNO",
    "GenDeepONet": "GenDeepONet",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def unique(rows: list[dict[str, str]], *, truth: str, architecture: str) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["configuration_id"] == CONFIGURATION
        and row["truth"] == truth
        and row["architecture"] == architecture
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one {CONFIGURATION} row for {truth}/{architecture}; "
            f"found {len(matches)}"
        )
    return matches[0]


def integer(value: str) -> str:
    return f"{int(value):,}"


def crossing_cells(row: dict[str, str]) -> tuple[str, str]:
    iteration = row["first_diagnostic_gamma_at_least_half"]
    ratio = row["first_gamma_half_error_over_oracle"]
    if not iteration and not ratio:
        return "--", "--"
    if not iteration or not ratio:
        raise AssertionError(
            "Gamma-crossing iteration and error ratio must either both be present or both be absent: "
            f"{row['truth']}/{row['architecture']}"
        )
    return integer(iteration), f"{float(ratio):.2f}"


def build_table(rows: list[dict[str, str]]) -> tuple[str, int]:
    body: list[str] = []
    checked_cells = 0
    for truth_index, truth in enumerate(TRUTHS):
        for architecture in METHODS:
            row = unique(rows, truth=truth, architecture=architecture)
            crossing_iteration, crossing_ratio = crossing_cells(row)
            monotone_raw = row["geometry_monotone_to_oracle"]
            if monotone_raw not in {"0", "1"}:
                raise AssertionError(
                    f"Unexpected geometry monotonicity flag for {truth}/{architecture}: {monotone_raw!r}"
                )
            values = (
                f"{float(row['gamma_at_oracle']):.2f}",
                crossing_iteration,
                integer(row["oracle_iteration"]),
                crossing_ratio,
                "yes" if monotone_raw == "1" else "no",
                f"{float(row['maximum_s']):.2f}",
            )
            checked_cells += len(values)
            body.append(
                f"{truth} & {DISPLAY_METHOD[architecture]} & " + " & ".join(values) + r" \\"
            )
        if truth_index == 0:
            body.append(r"\midrule")

    lines = [
        r"\begin{tabular}{llrrrrcr}",
        r"\toprule",
        (
            r"Truth & Method & $\gamma_{k_{\mathrm{or}}}$ & $k_\gamma$ & $k_{\mathrm{or}}$ & "
            r"$e_m(k_\gamma)/e_m(k_{\mathrm{or}})$ & $\M_C$ monotone & $\max_k s_k$ \\"
        ),
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n", checked_cells


def update_provenance() -> None:
    payload: dict[str, Any] = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    output_name = "display_assets/low_smoothness_diagnostics_single.tex"
    payload["products"] = [
        row for row in payload["products"] if row.get("output") != output_name
    ]
    payload["products"].append(
        {
            "source": "revision/E1_E2/low_smoothness_diagnostics_single.tex",
            "source_sha256": sha256(WORKING_OUTPUT),
            "output": output_name,
            "output_sha256": sha256(DISPLAY_OUTPUT),
            "change": (
                "Generate the 60-cell nu_C=0.25, ell_C=0.1 E1/E2 trajectory-diagnostic "
                "table from the frozen detailed summary; displayed values match the reviewed table."
            ),
            "data_sources": [
                {
                    "path": "revision/E1_E2/trajectory_diagnostics_summary.csv",
                    "sha256": sha256(SUMMARY),
                    "filter": f"configuration_id={CONFIGURATION}",
                }
            ],
        }
    )
    PROVENANCE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    rows = read_rows(SUMMARY)
    if len(rows) != 60:
        raise AssertionError(f"Unexpected E1/E2 summary row count: {len(rows)}")
    selected = [row for row in rows if row["configuration_id"] == CONFIGURATION]
    if len(selected) != 10:
        raise AssertionError(
            f"Expected 10 {CONFIGURATION} rows in {SUMMARY}; found {len(selected)}"
        )

    generated, checked_cells = build_table(rows)
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
    print(
        f"Confirmed all {checked_cells} displayed cells against the frozen filtered CSV rows "
        "and the reviewed table."
    )


if __name__ == "__main__":
    main()
