#!/usr/bin/env python3
"""Summarise the frozen E1/E2 diagnostics for the 60 E3 trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARCHITECTURES = ("ExactFEM", "FNO", "DeepONet", "GenFNO", "GenDeepONet")
DISPLAY_ARCHITECTURE = {"ExactFEM": "Exact FEM"}
CONFIG_ORDER = (
    "nu0p25_ell0p1",
    "nu0p25_ell0p5",
    "nu0p25_ell1p0",
    "nu0p4_ell0p1",
    "nu0p4_ell0p5",
    "nu0p4_ell1p0",
)
CONFIG_LABELS = {
    "nu0p25_ell0p1": r"$\nu_C=0.25,\ \ell_C=0.1$",
    "nu0p25_ell0p5": r"$\nu_C=0.25,\ \ell_C=0.5$",
    "nu0p25_ell1p0": r"$\nu_C=0.25,\ \ell_C=1.0$",
    "nu0p4_ell0p1": r"$\nu_C=0.4,\ \ell_C=0.1$",
    "nu0p4_ell0p5": r"$\nu_C=0.4,\ \ell_C=0.5$",
    "nu0p4_ell1p0": r"$\nu_C=0.4,\ \ell_C=1.0$",
}
CONFIG_COLOURS = {
    "nu0p25_ell0p1": "#0072B2",
    "nu0p25_ell0p5": "#56B4E9",
    "nu0p25_ell1p0": "#003F5C",
    "nu0p4_ell0p1": "#D55E00",
    "nu0p4_ell0p5": "#E69F00",
    "nu0p4_ell1p0": "#8C510A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=ROOT
        / "sulis_snapshot_2026-09-25/provenance/results/revision_2026-09/E1_E2/E3",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=ROOT
        / "paper_products/generated/revision/E3/trajectory_outcomes_pre_geometry.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "paper_products/generated/revision/E1_E2",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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
        return str(path.resolve())


def load_diagnostics(
    diagnostic_root: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Path]],
    list[Path],
]:
    exact_summaries = sorted(diagnostic_root.rglob("exact_geometry_summary.json"))
    gamma_summaries = sorted(diagnostic_root.rglob("surrogate_gamma_summary.json"))
    if len(exact_summaries) != 60 or len(gamma_summaries) != 48:
        raise ValueError(
            f"Expected 60 exact/geometry and 48 surrogate-gamma summaries; "
            f"found {len(exact_summaries)} and {len(gamma_summaries)}."
        )

    gamma_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    gamma_csv_by_key: dict[tuple[str, str, str], Path] = {}
    source_paths: list[Path] = []
    for path in gamma_summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        identity = data["identity"]
        key = (
            identity["truth"],
            identity["run_id"],
            identity["configuration_id"],
        )
        if key in gamma_by_key:
            raise ValueError(f"Duplicate surrogate gamma summary: {key}")
        gamma_by_key[key] = data
        gamma_csv = path.with_name(data["csv"])
        if not gamma_csv.is_file() or sha256(gamma_csv) != data["csv_sha256"]:
            raise ValueError(f"Surrogate gamma CSV hash mismatch: {gamma_csv}")
        gamma_csv_by_key[key] = gamma_csv
        source_paths.extend((path, gamma_csv))

    rows: list[dict[str, Any]] = []
    series_csv_by_key: dict[tuple[str, str, str], dict[str, Path]] = {}
    for path in exact_summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        identity = data["identity"]
        key = (
            identity["truth"],
            identity["run_id"],
            identity["configuration_id"],
        )
        csv_path = path.with_name(data["csv"])
        if not csv_path.is_file() or sha256(csv_path) != data["csv_sha256"]:
            raise ValueError(f"Exact/geometry CSV hash mismatch: {csv_path}")
        source_paths.extend((path, csv_path))
        trajectory_rows = read_csv(csv_path)
        if int(data["states_evaluated"]) != len(trajectory_rows):
            raise ValueError(f"State-count mismatch: {csv_path}")
        error_by_iteration = {
            int(item["iteration"]): float(item["coefficient_relative_l2_error"])
            for item in trajectory_rows
        }
        oracle_iteration = int(data["oracle_iteration"])
        if oracle_iteration not in error_by_iteration:
            raise ValueError(
                f"Off-cadence oracle state is missing from {csv_path}: "
                f"k_or={oracle_iteration}."
            )
        oracle_relative_error = error_by_iteration[oracle_iteration]

        if identity["method"] == "exact":
            gamma_values = np.asarray(
                [float(item["gamma_model"]) for item in trajectory_rows], dtype=float
            )
            iterations = np.asarray(
                [int(item["iteration"]) for item in trajectory_rows], dtype=int
            )
            crossing = np.flatnonzero(gamma_values >= 0.5)
            first_crossing = int(iterations[crossing[0]]) if crossing.size else ""
            maximum_gamma = float(np.max(gamma_values))
            oracle_gamma = float(data["gamma_at_oracle"])
            fraction_gamma_below_half = float(np.mean(gamma_values < 0.5))
            gamma_source = csv_path
        else:
            gamma = gamma_by_key.pop(key)
            first_crossing = gamma["first_diagnostic_gamma_at_least_half"]
            first_crossing = "" if first_crossing is None else int(first_crossing)
            maximum_gamma = float(gamma["maximum_gamma"])
            oracle_gamma = float(gamma["gamma_at_oracle"])
            fraction_gamma_below_half = float(gamma["fraction_gamma_below_half"])
            gamma_source = gamma_csv_by_key[key]

        if first_crossing == "":
            crossing_relative_error: float | str = ""
            crossing_error_over_oracle: float | str = ""
        else:
            if int(first_crossing) not in error_by_iteration:
                raise ValueError(
                    f"Gamma crossing k={first_crossing} has no matching coefficient "
                    f"error in {csv_path}."
                )
            crossing_relative_error = error_by_iteration[int(first_crossing)]
            crossing_error_over_oracle = (
                crossing_relative_error / oracle_relative_error
            )

        series_csv_by_key[key] = {"eta": csv_path, "gamma": gamma_source}

        controls = data["controls_from_complete_scalar_history"]
        geometry = data["geometry"]
        rows.append(
            {
                "truth": identity["truth"],
                "method": identity["method"],
                "architecture": identity["architecture"],
                "run_id": identity["run_id"],
                "configuration_id": identity["configuration_id"],
                "matern_nu": data["configuration"]["matern_nu"],
                "matern_length_scale": data["configuration"]["matern_length_scale"],
                "states_evaluated": data["states_evaluated"],
                "oracle_iteration": data["oracle_iteration"],
                "maximum_eta": data["maximum_eta"],
                "eta_at_oracle": data["eta_at_oracle"],
                "fraction_eta_below_half": data["fraction_eta_below_half"],
                "maximum_gamma": maximum_gamma,
                "gamma_at_oracle": oracle_gamma,
                "first_diagnostic_gamma_at_least_half": first_crossing,
                "relative_coefficient_error_at_first_gamma_half": crossing_relative_error,
                "first_gamma_half_error_over_oracle": crossing_error_over_oracle,
                "oracle_relative_coefficient_error": oracle_relative_error,
                "fraction_gamma_below_half": fraction_gamma_below_half,
                "geometry_monotone_to_oracle": int(
                    geometry["monotone_nonincreasing_through_oracle"]
                ),
                "maximum_positive_geometry_change_to_oracle": geometry[
                    "maximum_positive_change_through_oracle"
                ],
                "maximum_s": controls["maximum_s"],
                "fraction_s_at_most_one": controls["fraction_s_at_most_one"],
                "physical_change_cap_count_inferred": controls[
                    "physical_change_cap_applied_count_inferred"
                ],
                "physical_change_cap_fraction_inferred": controls[
                    "physical_change_cap_applied_fraction_inferred"
                ],
                "projection_active_update_count": controls[
                    "projection_active_update_count"
                ],
                "projection_active_update_fraction": controls[
                    "projection_active_update_fraction"
                ],
                "maximum_active_bound_fraction": controls[
                    "maximum_active_bound_fraction"
                ],
                "maximum_cg_iterations": data["maximum_cg_iterations"],
                "maximum_cg_relative_residual": data["maximum_cg_relative_residual"],
                "total_fem_seconds": data["total_fem_seconds"],
                "total_geometry_cg_seconds": data["total_cg_seconds"],
                "exact_geometry_csv": relative(csv_path),
                "gamma_csv": relative(gamma_source),
            }
        )

    if gamma_by_key:
        raise ValueError(f"Unmatched surrogate gamma summaries: {sorted(gamma_by_key)}")
    rows.sort(
        key=lambda item: (
            ("T1", "T2").index(item["truth"]),
            ARCHITECTURES.index(item["architecture"]),
            CONFIG_ORDER.index(item["configuration_id"]),
        )
    )
    return rows, series_csv_by_key, source_paths


def join_outcomes(
    diagnostics: list[dict[str, Any]], outcomes_path: Path
) -> list[dict[str, Any]]:
    outcomes = read_csv(outcomes_path)
    if len(outcomes) != 60:
        raise ValueError(f"{outcomes_path}: expected 60 outcomes.")
    diagnostic_by_key = {
        (row["truth"], row["run_id"], row["configuration_id"]): row
        for row in diagnostics
    }
    merged: list[dict[str, Any]] = []
    for outcome in outcomes:
        key = (outcome["truth"], outcome["run_id"], outcome["configuration_id"])
        diagnostic = diagnostic_by_key[key]
        merged.append(
            {
                "truth": outcome["truth"],
                "architecture": outcome["architecture"],
                "run_id": outcome["run_id"],
                "configuration_id": outcome["configuration_id"],
                "matern_nu": outcome["matern_nu"],
                "matern_length_scale": outcome["matern_length_scale"],
                "min_relative_coefficient_error": outcome[
                    "min_relative_coefficient_error"
                ],
                "oracle_iteration": outcome["oracle_iteration"],
                "final_relative_coefficient_error": outcome[
                    "final_relative_coefficient_error"
                ],
                "final_over_min": outcome["final_over_min"],
                "geometry_monotone_to_oracle": diagnostic[
                    "geometry_monotone_to_oracle"
                ],
                "maximum_eta": diagnostic["maximum_eta"],
                "eta_at_oracle": diagnostic["eta_at_oracle"],
                "maximum_gamma": diagnostic["maximum_gamma"],
                "gamma_at_oracle": diagnostic["gamma_at_oracle"],
                "first_diagnostic_gamma_at_least_half": diagnostic[
                    "first_diagnostic_gamma_at_least_half"
                ],
                "maximum_s": diagnostic["maximum_s"],
                "physical_change_cap_fraction_inferred": diagnostic[
                    "physical_change_cap_fraction_inferred"
                ],
                "projection_active_update_fraction": diagnostic[
                    "projection_active_update_fraction"
                ],
            }
        )
    return merged


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for truth in ("T1", "T2"):
        for architecture in ARCHITECTURES:
            selected = [
                row
                for row in rows
                if row["truth"] == truth and row["architecture"] == architecture
            ]
            if len(selected) != 6:
                raise ValueError(f"Expected six rows for {truth}/{architecture}.")
            crossing_rows = [
                row
                for row in selected
                if row["first_diagnostic_gamma_at_least_half"] != ""
            ]

            def crossing_summary(column: str) -> tuple[float | str, float | str, float | str]:
                if not crossing_rows:
                    return "", "", ""
                values = np.asarray(
                    [float(row[column]) for row in crossing_rows], dtype=float
                )
                return float(np.median(values)), float(np.min(values)), float(np.max(values))

            iteration_median, iteration_minimum, iteration_maximum = crossing_summary(
                "first_diagnostic_gamma_at_least_half"
            )
            error_median, error_minimum, error_maximum = crossing_summary(
                "relative_coefficient_error_at_first_gamma_half"
            )
            ratio_median, ratio_minimum, ratio_maximum = crossing_summary(
                "first_gamma_half_error_over_oracle"
            )
            compact.append(
                {
                    "truth": truth,
                    "architecture": architecture,
                    "maximum_eta_over_geometries": max(
                        float(row["maximum_eta"]) for row in selected
                    ),
                    "minimum_eta_at_oracle": min(
                        float(row["eta_at_oracle"]) for row in selected
                    ),
                    "maximum_eta_at_oracle": max(
                        float(row["eta_at_oracle"]) for row in selected
                    ),
                    "maximum_gamma_over_geometries": max(
                        float(row["maximum_gamma"]) for row in selected
                    ),
                    "minimum_gamma_at_oracle": min(
                        float(row["gamma_at_oracle"]) for row in selected
                    ),
                    "maximum_gamma_at_oracle": max(
                        float(row["gamma_at_oracle"]) for row in selected
                    ),
                    "geometries_crossing_gamma_half": sum(
                        row["first_diagnostic_gamma_at_least_half"] != ""
                        for row in selected
                    ),
                    "gamma_half_iteration_median": iteration_median,
                    "gamma_half_iteration_minimum": iteration_minimum,
                    "gamma_half_iteration_maximum": iteration_maximum,
                    "gamma_half_relative_error_median": error_median,
                    "gamma_half_relative_error_minimum": error_minimum,
                    "gamma_half_relative_error_maximum": error_maximum,
                    "gamma_half_error_over_oracle_median": ratio_median,
                    "gamma_half_error_over_oracle_minimum": ratio_minimum,
                    "gamma_half_error_over_oracle_maximum": ratio_maximum,
                    "geometry_monotone_count": sum(
                        int(row["geometry_monotone_to_oracle"]) for row in selected
                    ),
                    "maximum_s_over_geometries": max(
                        float(row["maximum_s"]) for row in selected
                    ),
                }
            )
    return compact


def write_compact_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Truth & Method & $\max\eta_k$ & $\gamma_{k_{\rm or}}$ range & "
        r"$\gamma\geq 1/2$ & $e_m(k_{\gamma})/e_m(k_{\rm or})$ & "
        r"$M_C$ monotone & $\max s_k$ \\",
        r"\midrule",
    ]
    for row in rows:
        architecture = DISPLAY_ARCHITECTURE.get(row["architecture"], row["architecture"])
        gamma_range = (
            f"{float(row['minimum_gamma_at_oracle']):.3f}--"
            f"{float(row['maximum_gamma_at_oracle']):.3f}"
        )
        if row["gamma_half_error_over_oracle_median"] == "":
            crossing_ratio = "--"
        else:
            crossing_ratio = (
                f"{float(row['gamma_half_error_over_oracle_median']):.3f} "
                f"[{float(row['gamma_half_error_over_oracle_minimum']):.3f}, "
                f"{float(row['gamma_half_error_over_oracle_maximum']):.3f}]"
            )
        lines.append(
            f"{row['truth']} & {architecture} & "
            f"{float(row['maximum_eta_over_geometries']):.3f} & "
            f"{gamma_range} & {row['geometries_crossing_gamma_half']}/6 & "
            f"{crossing_ratio} & "
            f"{row['geometry_monotone_count']}/6 & "
            f"{float(row['maximum_s_over_geometries']):.3f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _series(path: Path, column: str) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(path)
    return (
        np.asarray([int(row["iteration"]) for row in rows], dtype=int),
        np.asarray([float(row[column]) for row in rows], dtype=float),
    )


def plot_diagnostic(
    path_stem: Path,
    rows: list[dict[str, Any]],
    csv_by_key: dict[tuple[str, str, str], dict[str, Path]],
    *,
    diagnostic: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from scripts.paper_products.common import configure_matplotlib, save_figure

    configure_matplotlib()
    figure, axes = plt.subplots(
        2, 5, figsize=(14.8, 6.8), sharex=True, sharey="row", layout="constrained"
    )
    layout = figure.get_layout_engine()
    if layout is not None:
        layout.set(w_pad=0.025, h_pad=0.025, wspace=0.035, hspace=0.06)

    for row_index, truth in enumerate(("T1", "T2")):
        for column_index, architecture in enumerate(ARCHITECTURES):
            axis = axes[row_index, column_index]
            selected = [
                row
                for row in rows
                if row["truth"] == truth and row["architecture"] == architecture
            ]
            for row in selected:
                key = (truth, row["run_id"], row["configuration_id"])
                column = "eta_exact_fem" if diagnostic == "eta" else "gamma_model"
                iteration, values = _series(csv_by_key[key][diagnostic], column)
                plotted_x = iteration + 1
                colour = CONFIG_COLOURS[row["configuration_id"]]
                axis.plot(plotted_x, values, color=colour, linewidth=1.05, alpha=0.9)
                if diagnostic == "gamma":
                    oracle = int(row["oracle_iteration"])
                    oracle_index = int(np.argmin(np.abs(iteration - oracle)))
                    axis.plot(
                        plotted_x[oracle_index],
                        values[oracle_index],
                        marker="x",
                        color=colour,
                        markersize=4.5,
                        markeredgewidth=1.0,
                        linestyle="none",
                    )
                    crossing = row["first_diagnostic_gamma_at_least_half"]
                    if crossing != "":
                        crossing_index = int(np.flatnonzero(iteration == int(crossing))[0])
                        axis.plot(
                            plotted_x[crossing_index],
                            values[crossing_index],
                            marker="o",
                            markerfacecolor="white",
                            markeredgecolor=colour,
                            markersize=3.8,
                            linestyle="none",
                        )
            axis.axhline(0.5, color="#555555", linestyle="--", linewidth=0.8)
            axis.set_xscale("log")
            axis.set_xticks(
                [1, 11, 101, 1001, 10001],
                ["0", "10", "100", "1000", "10000"],
            )
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
            if row_index == 0:
                axis.set_title(DISPLAY_ARCHITECTURE.get(architecture, architecture), pad=2)
            if column_index == 0:
                axis.text(
                    -0.16,
                    1.01,
                    truth,
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    weight="bold",
                )

    ylabel = (
        r"exact-FEM tangential-cone diagnostic $\eta_k$"
        if diagnostic == "eta"
        else r"model-driven descent diagnostic $\gamma_k$"
    )
    figure.supylabel(ylabel)
    figure.supxlabel(r"Landweber iteration $k$")
    handles = [
        Line2D([0], [0], color=CONFIG_COLOURS[config], label=CONFIG_LABELS[config])
        for config in CONFIG_ORDER
    ]
    handles.append(
        Line2D([0], [0], color="#555555", linestyle="--", label=r"threshold $1/2$")
    )
    if diagnostic == "gamma":
        handles.extend(
            (
                Line2D(
                    [0], [0], color="#333333", marker="x", linestyle="none", label="oracle"
                ),
                Line2D(
                    [0],
                    [0],
                    color="#333333",
                    marker="o",
                    markerfacecolor="white",
                    linestyle="none",
                    label=r"first $\gamma_k\geq1/2$",
                ),
            )
        )
    figure.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False)
    save_figure(figure, path_stem, dpi=dpi)


def write_note(path: Path, rows: list[dict[str, Any]]) -> None:
    nonmonotone = [row for row in rows if not row["geometry_monotone_to_oracle"]]
    surrogate = [row for row in rows if row["method"] == "surrogate"]
    lines = [
        "# E1/E2 diagnostics on the E3 low-smoothness trajectories",
        "",
        "## Scope",
        "",
        "These products cover the 60 completed E3 trajectories only: two truths,",
        "six low-smoothness Matérn geometries, Exact FEM and four frozen headline",
        "surrogates. They do not contain the historical nu_C=3/2 baseline or",
        "identity trajectories. Consequently, the original E2 question about the",
        "T2 identity-run crossings remains unresolved until those trajectories are",
        "rerun with saved states.",
        "",
        "## E1",
        "",
        f"All {len(rows)} arms have maximum sampled eta below 1/2. The largest",
        f"value is {max(float(row['maximum_eta']) for row in rows):.6g}; every arm",
        "therefore has fraction_eta_below_half equal to one on the saved-state grid.",
        "",
        "## E2 and E3 geometry",
        "",
        f"Gamma reaches or exceeds 1/2 in {sum(row['first_diagnostic_gamma_at_least_half'] != '' for row in surrogate)}",
        f"of {len(surrogate)} surrogate arms. Its largest sampled value is",
        f"{max(float(row['maximum_gamma']) for row in surrogate):.6g}. The step-size",
        f"quantity satisfies s_k <= 1 in every arm; the largest maximum_s is",
        f"{max(float(row['maximum_s']) for row in rows):.6g}.",
        "For every sampled gamma crossing, the detailed CSV now records the",
        "relative coefficient error at that same saved iterate and its ratio",
        "to the oracle error. The compact CSV/table report the median and range",
        "of those quantities over the crossing geometries; non-crossing arms are",
        "not silently assigned a terminal value.",
        "",
        f"The sampled M_C error is non-increasing through the oracle in {len(rows) - len(nonmonotone)}",
        f"of {len(rows)} arms. The six exceptions are:",
        "",
    ]
    for row in nonmonotone:
        lines.append(
            f"- {row['truth']}, {DISPLAY_ARCHITECTURE.get(row['architecture'], row['architecture'])}, "
            f"{row['configuration_id']} (maximum positive sampled change "
            f"{float(row['maximum_positive_geometry_change_to_oracle']):.6g})."
        )
    lines.extend(
        [
            "",
            "The changes are between saved diagnostic states, normally ten",
            "Landweber iterations apart; they are not adjacent-iteration changes.",
            "Regularity of T1 alone therefore does not establish monotonicity for",
            "every surrogate arm: the remaining theorem hypotheses and the model",
            "driving the iteration still matter.",
            "",
            "The figures use k+1 internally to display k=0 on a logarithmic axis;",
            "the tick labels report the actual Landweber iteration k.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    path: Path, *, sources: list[Path], outputs: list[Path], diagnostic_root: Path
) -> None:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = ""
    payload = {
        "format": "reaction-revision-e1-e2-e3-summary-v1",
        "scope": "60 completed low-smoothness E3 trajectories; excludes baseline and identity",
        "diagnostic_root": relative(diagnostic_root),
        "git_commit_before_generation": git_commit,
        "sources": [
            {"path": relative(source), "sha256": sha256(source)}
            for source in sorted(set(sources))
        ],
        "outputs": [
            {"path": relative(output), "sha256": sha256(output)}
            for output in outputs
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics, csv_by_key, sources = load_diagnostics(args.diagnostic_root)
    merged = join_outcomes(diagnostics, args.outcomes)
    compact = compact_rows(diagnostics)

    detailed_csv = args.output_dir / "trajectory_diagnostics_summary.csv"
    merged_csv = args.output_dir / "e3_outcomes_with_diagnostics.csv"
    compact_csv = args.output_dir / "trajectory_diagnostics_compact.csv"
    compact_tex = args.output_dir / "trajectory_diagnostics_compact.tex"
    note = args.output_dir / "NOTE.md"
    eta_stem = args.output_dir / "eta_by_geometry"
    gamma_stem = args.output_dir / "gamma_by_geometry"
    write_csv(detailed_csv, diagnostics)
    write_csv(merged_csv, merged)
    write_csv(compact_csv, compact)
    write_compact_tex(compact_tex, compact)
    write_note(note, diagnostics)
    plot_diagnostic(
        eta_stem, diagnostics, csv_by_key, diagnostic="eta", dpi=args.dpi
    )
    plot_diagnostic(
        gamma_stem, diagnostics, csv_by_key, diagnostic="gamma", dpi=args.dpi
    )
    outputs = [
        detailed_csv,
        merged_csv,
        compact_csv,
        compact_tex,
        note,
        eta_stem.with_suffix(".pdf"),
        eta_stem.with_suffix(".png"),
        gamma_stem.with_suffix(".pdf"),
        gamma_stem.with_suffix(".png"),
    ]
    write_manifest(
        args.output_dir / "manifest.json",
        sources=sources + [args.outcomes, Path(__file__)],
        outputs=outputs,
        diagnostic_root=args.diagnostic_root,
    )
    print(f"Saved E1/E2 E3-trajectory summary in {args.output_dir}")


if __name__ == "__main__":
    main()
