#!/usr/bin/env python3
"""Build the frozen FNO capacity and training-size ablation tables.

This is post-processing only.  It reads existing validation/test summaries and
frozen noiseless Landweber archives; it never loads a model or runs an inverse
solve.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CAPACITY_SOURCE = (
    ROOT
    / "results/lw_noiseless_fno_capacity_precond/analysis"
    / "fno_capacity_preconditioner_summary.csv"
)
REGISTRY = (
    ROOT
    / "results/paper_T2_broad/budget2500_validation"
    / "landweber_registries/ablations_J10.tsv"
)
TEST_SOURCE = ROOT / "tables/paper_T2_broad/test/ablations_J10_test.csv"
T2_ROOT = ROOT / "results/lw_noiseless_ablation_J10/T2_broad"
T1_ROOT = (
    ROOT
    / "sulis_snapshot_2026-09-14/provenance/results/paper_T1"
    / "noiseless_landweber/ablations_J10"
)
HAND_TABLE = (
    ROOT / "reference/paper/fno_capacity_semiconvergence.tex"
)
DISPLAY_DIR = ROOT / "outputs/tables"
PROVENANCE = DISPLAY_DIR / "PROVENANCE.json"
DEFAULT_OUTPUT = ROOT / "paper_products/generated/revision/ablations"

CAPACITY_WIDTHS = (8, 12, 16, 24, 32, 40)
GRID_CELLS = (
    (8, 1000),
    (8, 10000),
    (12, 1000),
    (12, 5000),
    (12, 10000),
    (16, 1000),
    (16, 5000),
    (16, 10000),
    (24, 1000),
    (24, 5000),
    (24, 10000),
)
MODES_BY_WIDTH = {8: 4, 12: 6, 16: 8, 24: 10, 32: 12, 40: 16}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_csv(path: Path, *, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def parse_capacity(value: str) -> tuple[int, int]:
    pieces = dict(item.split("=", 1) for item in value.split(","))
    return int(pieces["w"]), int(pieces["m"])


def run_id(train_samples: int, width: int) -> str:
    return f"fno_N{train_samples}_w{width}_m{MODES_BY_WIDTH[width]}_L4_J10"


def display_capacity_row(row: dict[str, Any]) -> list[str]:
    return [
        f"{row['width']}/{row['modes']}",
        f"{100.0 * float(row['E_F_test']):.3f}",
        f"{100.0 * float(row['E_J_test']):.3f}",
        f"{100.0 * float(row['matern_min_error']):.2f}",
        str(int(row["matern_oracle_iteration"])),
        f"{float(row['matern_final_over_min']):.2f}",
        f"{100.0 * float(row['identity_min_error']):.2f}",
        str(int(row["identity_oracle_iteration"])),
        f"{float(row['identity_final_over_min']):.2f}",
    ]


def parse_hand_table_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    pattern = re.compile(r"^\s*(\d+/\d+)\s*&")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not pattern.match(line):
            continue
        cells = [cell.strip() for cell in line.removesuffix(" \\\\").split("&")]
        rows.append(cells)
    return rows


def build_capacity_rows() -> list[dict[str, Any]]:
    source = read_csv(CAPACITY_SOURCE)
    selected = [row for row in source if row["arm_id"] in {"ell0p1", "identity"}]
    by_key = {(row["run_id"], row["arm_id"]): row for row in selected}
    rows: list[dict[str, Any]] = []
    for width in CAPACITY_WIDTHS:
        rid = run_id(10000, width)
        matern = by_key.get((rid, "ell0p1"))
        identity = by_key.get((rid, "identity"))
        if matern is None or identity is None:
            raise AssertionError(f"Missing frozen capacity arm for {rid}")
        modes = MODES_BY_WIDTH[width]
        if int(matern["width"]) != width or int(matern["modes"]) != modes:
            raise AssertionError(f"Capacity metadata mismatch for {rid}")
        if "/T2_broad/" not in matern["output_dir"] or "/T2_broad/" not in identity["output_dir"]:
            raise AssertionError(f"Capacity study is not identified as T2: {rid}")
        if not math.isclose(float(matern["E_F_test"]), float(identity["E_F_test"])):
            raise AssertionError(f"Checkpoint mismatch between geometries for {rid}")
        rows.append(
            {
                "truth": "T2",
                "train_samples": 10000,
                "lambda_J": 10.0,
                "run_id": rid,
                "width": width,
                "modes": modes,
                "checkpoint_epoch": int(matern["checkpoint_epoch"]),
                "E_F_test": float(matern["E_F_test"]),
                "E_J_test": float(matern["E_J_test"]),
                "matern_nu": 1.5,
                "matern_length_scale": 0.1,
                "matern_min_error": float(matern["min_relative_coefficient_error"]),
                "matern_oracle_iteration": int(matern["argmin_k"]),
                "matern_final_over_min": float(matern["final_over_min"]),
                "identity_min_error": float(identity["min_relative_coefficient_error"]),
                "identity_oracle_iteration": int(identity["argmin_k"]),
                "identity_final_over_min": float(identity["final_over_min"]),
                "canonical_source": relative(CAPACITY_SOURCE),
            }
        )
    return rows


def write_capacity_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lrr@{\hspace{10pt}}rrr@{\hspace{10pt}}rrr}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{Mat\'ern, $\nu_C=3/2$, $\ell_C=0.1$} & \multicolumn{3}{c}{Identity} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r"Width/modes & $E_F$ (\%) & $E_J$ (\%) & $e_m^\star$ (\%) & $k_{\rm or}$ & Final/min & $e_m^\star$ (\%) & $k_{\rm or}$ & Final/min \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(display_capacity_row(row)) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def unique_result(root: Path, rid: str) -> tuple[Path, Path] | None:
    matches = sorted(root.glob(f"{rid}_e*_minu/result.npz"))
    if not matches:
        return None
    if len(matches) != 1:
        raise AssertionError(f"Expected one result for {rid} below {root}; found {matches}")
    settings = matches[0].with_name("settings.json")
    if not settings.is_file():
        raise FileNotFoundError(settings)
    return matches[0], settings


def load_outcome(result_path: Path, settings_path: Path, truth: str) -> dict[str, Any]:
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    arguments = settings["command_arguments"]
    if int(arguments["iterations"]) != 10000:
        raise AssertionError(f"Expected 10000 iterations: {settings_path}")
    if not math.isclose(float(arguments["matern_nu"]), 1.5) or not math.isclose(
        float(arguments["matern_length_scale"]), 0.1
    ):
        raise AssertionError(f"Not the baseline Matérn geometry: {settings_path}")
    if f"results/truths/{truth}/truth.npz" != arguments["synthetic_file"]:
        raise AssertionError(f"Truth mismatch: {settings_path}")
    with np.load(result_path, allow_pickle=False) as archive:
        iterations = np.asarray(archive["history_iteration"], dtype=np.int64)
        errors = np.asarray(archive["history_relative_u_error"], dtype=np.float64)
        oracle_error = float(np.asarray(archive["minimum_u_error"]).reshape(()))
        oracle_iteration = int(np.asarray(archive["minimum_u_error_iteration"]).reshape(()))
    if iterations[0] != 0 or iterations[-1] != 10000 or errors.size != iterations.size:
        raise AssertionError(f"Incomplete inversion history: {result_path}")
    oracle_index = int(np.nanargmin(errors))
    if int(iterations[oracle_index]) != oracle_iteration or not math.isclose(
        float(errors[oracle_index]), oracle_error, rel_tol=1.0e-10, abs_tol=1.0e-12
    ):
        raise AssertionError(f"Stored oracle mismatch: {result_path}")
    return {
        "min_error": oracle_error,
        "oracle_iteration": oracle_iteration,
        "final_error": float(errors[-1]),
        "final_over_min": float(errors[-1]) / oracle_error,
        "result_path": relative(result_path),
        "settings_path": relative(settings_path),
    }


def build_training_rows() -> tuple[list[dict[str, Any]], list[Path]]:
    registry_rows = read_csv(REGISTRY, delimiter="\t")
    registry = {row["run_id"]: row for row in registry_rows if row["architecture"] == "FNO"}
    test = {row["run_id"]: row for row in read_csv(TEST_SOURCE) if row["architecture"] == "FNO"}
    sources: list[Path] = [REGISTRY, TEST_SOURCE]
    rows: list[dict[str, Any]] = []
    for width, train_samples in GRID_CELLS:
        rid = run_id(train_samples, width)
        metadata = registry.get(rid)
        if metadata is None:
            raise AssertionError(f"Missing frozen registry row: {rid}")
        if not math.isclose(float(metadata["dino_weight"]), 10.0):
            raise AssertionError(f"Expected lambda_J=10: {rid}")
        actual_width, actual_modes = parse_capacity(metadata["capacity"])
        if (actual_width, actual_modes) != (width, MODES_BY_WIDTH[width]):
            raise AssertionError(f"Capacity mismatch: {rid}")

        t2_pair = unique_result(T2_ROOT, rid)
        if t2_pair is None:
            raise FileNotFoundError(f"Missing T2 trajectory for {rid}")
        t2 = load_outcome(*t2_pair, truth="T2_broad")
        sources.extend(t2_pair)
        t1_pair = unique_result(T1_ROOT, rid)
        t1 = load_outcome(*t1_pair, truth="T1") if t1_pair is not None else None
        if t1_pair is not None:
            sources.extend(t1_pair)

        test_row = test.get(rid)
        category = (
            "one_axis_training_size"
            if width == 24
            else "capacity_reference"
            if train_samples == 10000
            else "historical_cross_grid"
        )
        rows.append(
            {
                "category": category,
                "run_id": rid,
                "train_samples": train_samples,
                "width": width,
                "modes": actual_modes,
                "lambda_J": float(metadata["dino_weight"]),
                "checkpoint_epoch": int(metadata["checkpoint_epoch"]),
                "E_F_val": float(metadata["E_F_val"]),
                "E_J_val": float(metadata["E_J_val"]),
                "independent_test_available": int(test_row is not None),
                "E_F_test": float(test_row["E_F_test"]) if test_row else "",
                "E_J_test": float(test_row["E_J_test"]) if test_row else "",
                "preconditioner": "matern",
                "matern_nu": 1.5,
                "matern_length_scale": 0.1,
                "T2_min_error": t2["min_error"],
                "T2_oracle_iteration": t2["oracle_iteration"],
                "T2_final_over_min": t2["final_over_min"],
                "T1_available": int(t1 is not None),
                "T1_min_error": t1["min_error"] if t1 else "",
                "T1_oracle_iteration": t1["oracle_iteration"] if t1 else "",
                "T1_final_over_min": t1["final_over_min"] if t1 else "",
                "T2_source": t2["result_path"],
                "T1_source": t1["result_path"] if t1 else "",
            }
        )
    expected_historical = 5
    if sum(row["category"] == "historical_cross_grid" for row in rows) != expected_historical:
        raise AssertionError("Expected five usable historical FNO cross-grid cells.")
    if any(row["independent_test_available"] for row in rows if row["category"] == "historical_cross_grid"):
        raise AssertionError("Historical cross-grid rows unexpectedly entered final-test evaluation.")
    if unique_result(T2_ROOT, run_id(5000, 8)) is not None:
        raise AssertionError("The historically incomplete N=5000, width/modes=8/4 cell now exists; review scope.")
    return rows, sources


def value(value: Any, *, percent: bool = False, digits: int = 2) -> str:
    if value == "" or value is None:
        return "--"
    number = float(value)
    if not math.isfinite(number):
        return "--"
    if percent:
        number *= 100.0
    return f"{number:.{digits}f}"


def write_training_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{rrcrr@{\hspace{8pt}}rrr@{\hspace{8pt}}rrr}",
        r"\toprule",
        r" & & & & & \multicolumn{3}{c}{T2} & \multicolumn{3}{c}{T1} \\",
        r"\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"Training fields & Width/modes & $\lambda_J$ & $E_F^{\rm val}$ (\%) & $E_J^{\rm val}$ (\%) & $e_m^\star$ (\%) & $k_{\rm or}$ & Final/min & $e_m^\star$ (\%) & $k_{\rm or}$ & Final/min \\",
        r"\midrule",
    ]
    previous_width: int | None = None
    for row in rows:
        width = int(row["width"])
        if previous_width is not None and width != previous_width:
            lines.append(r"\addlinespace")
        cells = [
            f"{int(row['train_samples']):,}",
            f"{width}/{int(row['modes'])}",
            f"{float(row['lambda_J']):g}",
            value(row["E_F_val"], percent=True, digits=3),
            value(row["E_J_val"], percent=True, digits=3),
            value(row["T2_min_error"], percent=True, digits=2),
            str(int(row["T2_oracle_iteration"])),
            value(row["T2_final_over_min"], digits=2),
            value(row["T1_min_error"], percent=True, digits=2),
            str(int(row["T1_oracle_iteration"])) if row["T1_oracle_iteration"] != "" else "--",
            value(row["T1_final_over_min"], digits=2),
        ]
        lines.append(" & ".join(cells) + r" \\")
        previous_width = width
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def update_display_provenance(
    capacity_working: Path,
    capacity_display: Path,
    training_working: Path,
    training_display: Path,
    *,
    replace: bool = False,
) -> None:
    document = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    additions = [
        {
            "source": relative(capacity_working).removeprefix("paper_products/generated/"),
            "source_sha256": sha256(capacity_working),
            "output": f"display_assets/{capacity_display.name}",
            "output_sha256": sha256(capacity_display),
            "change": "Canonical T2 FNO capacity table from the frozen capacity-preconditioner summary; Matérn baseline and identity only.",
        },
        {
            "source": relative(training_working).removeprefix("paper_products/generated/"),
            "source_sha256": sha256(training_working),
            "output": f"display_assets/{training_display.name}",
            "output_sha256": sha256(training_display),
            "change": "Matérn-only FNO training-size/cross-grid table; validation errors are used uniformly because historical cross-grid cells lack independent-test evaluation.",
        },
    ]
    names = {entry["output"] for entry in additions}
    existing = [entry for entry in document["products"] if entry.get("output") in names]
    if existing and not replace:
        raise FileExistsError("Display provenance already contains an ablation-table entry.")
    if replace:
        document["products"] = [
            entry for entry in document["products"] if entry.get("output") not in names
        ]
    document["products"].extend(additions)
    PROVENANCE.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def manifest_records(paths: Iterable[Path]) -> list[dict[str, str]]:
    return [
        {"path": relative(path), "sha256": sha256(path)}
        for path in sorted(set(paths), key=lambda item: relative(item))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--display-dir", type=Path, default=DISPLAY_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only the products owned by this builder.",
    )
    args = parser.parse_args()

    output_names = (
        "fno_capacity_semiconvergence.csv",
        "fno_capacity_semiconvergence.tex",
        "fno_training_size_semiconvergence.csv",
        "fno_training_size_semiconvergence.tex",
        "NOTE.md",
        "manifest.json",
    )
    display_names = (
        "fno_capacity_semiconvergence.tex",
        "fno_training_size_semiconvergence.tex",
    )
    for path in [*(args.output_dir / name for name in output_names), *(args.display_dir / name for name in display_names)]:
        if path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite existing product: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.display_dir.mkdir(parents=True, exist_ok=True)

    capacity_rows = build_capacity_rows()
    generated_cells = [display_capacity_row(row) for row in capacity_rows]
    hand_cells = parse_hand_table_rows(HAND_TABLE)
    if generated_cells != hand_cells:
        raise AssertionError(
            "Canonical capacity cells differ from the hand-built draft:\n"
            f"canonical={generated_cells!r}\nhand={hand_cells!r}"
        )
    training_rows, training_sources = build_training_rows()

    capacity_csv = args.output_dir / output_names[0]
    capacity_tex = args.output_dir / output_names[1]
    training_csv = args.output_dir / output_names[2]
    training_tex = args.output_dir / output_names[3]
    note = args.output_dir / output_names[4]
    write_csv(capacity_csv, capacity_rows)
    write_capacity_table(capacity_tex, capacity_rows)
    write_csv(training_csv, training_rows)
    write_training_table(training_tex, training_rows)

    capacity_display = args.display_dir / display_names[0]
    training_display = args.display_dir / display_names[1]
    capacity_display.write_bytes(capacity_tex.read_bytes())
    training_display.write_bytes(training_tex.read_bytes())

    low_data = [row for row in training_rows if int(row["train_samples"]) < 10000]
    t1_low_data = [row for row in low_data if row["T1_available"]]
    note.write_text(
        """# Capacity and training-size ablations

## Capacity table

The canonical capacity-preconditioner summary contains six FNO checkpoints,
all trained on 10,000 fields with `lambda_J=10`, and applies each frozen
checkpoint to noiseless T2.  The display table uses the baseline Matérn
geometry (`nu_C=3/2`, `ell_C=0.1`) and identity.  Its 54 displayed numerical
cells match `reference/paper/fno_capacity_semiconvergence.tex`
exactly after the draft's stated rounding; no difference was found.

## Training size and historical cross-grid

Seven usable FNO checkpoints were trained on fewer than 10,000 fields.  Two
are the supported one-axis training-size arms at width/modes 24/10, and five
are historical off-central capacity-by-training-size cells.  All seven have
noiseless T2 inversions with the baseline Matérn geometry.  T1 inversions exist
only for the two supported 24/10 low-data arms.  The attempted N=5,000,
width/modes=8/4 model did not yield a usable frozen checkpoint or inversion.

The table adds the matching N=10,000 cells as within-width references.  It
uses validation `E_F` and `E_J` uniformly, because the five historical
cross-grid checkpoints were deliberately excluded from the independent
final-test campaign.  The full-precision CSV retains independent-test errors
where they exist and marks their availability, so validation and test values
cannot be confused.

All inversions in this table are noiseless, use Matérn `nu_C=3/2`,
`ell_C=0.1`, and run for 10,000 iterations.  Missing T1 cells mean "not run",
not a failed reconstruction.
""",
        encoding="utf-8",
    )
    if len(low_data) != 7 or len(t1_low_data) != 2:
        raise AssertionError("Unexpected low-data/T1 coverage count.")

    update_display_provenance(
        capacity_tex,
        capacity_display,
        training_tex,
        training_display,
        replace=args.force,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    inputs = [CAPACITY_SOURCE, REGISTRY, TEST_SOURCE, HAND_TABLE, *training_sources]
    outputs = [
        capacity_csv,
        capacity_tex,
        training_csv,
        training_tex,
        note,
        capacity_display,
        training_display,
        PROVENANCE,
    ]
    manifest = {
        "format": "reaction-revision-capacity-training-size-ablations-v1",
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "configuration": {
            "capacity_truth": "T2",
            "capacity_train_samples": 10000,
            "lambda_J": 10.0,
            "preconditioners": [
                {"kind": "matern", "nu_C": 1.5, "ell_C": 0.1},
                {"kind": "identity"},
            ],
            "training_size_metric_split": "validation",
            "training_size_preconditioner": {"kind": "matern", "nu_C": 1.5, "ell_C": 0.1},
            "training_size_truths": ["T2", "T1 where available"],
            "iterations": 10000,
            "capacity_hand_table_match": True,
            "usable_low_data_models": 7,
            "usable_historical_cross_grid_models": 5,
            "incomplete_cell": "fno_N5000_w8_m4_L4_J10",
        },
        "inputs": manifest_records(inputs),
        "outputs": manifest_records(outputs),
    }
    (args.output_dir / output_names[5]).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "Built 6 capacity rows and 11 training-size/grid rows; "
        "all hand-table cells matched."
    )


if __name__ == "__main__":
    main()
