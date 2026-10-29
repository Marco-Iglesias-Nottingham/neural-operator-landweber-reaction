#!/usr/bin/env python3
"""Summarise the frozen E3 low-smoothness Matérn geometry sweep.

This first-layer analysis reads only the completed result archives.  It does
not evaluate a FEM operator and does not solve a covariance system.  The
trajectory-wide ``M_C`` monotonicity fields are deliberately marked pending
until the separate E1/E2 diagnostic stage has run.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
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
    "nu0p25_ell0p1": r"$(.25,.1)$",
    "nu0p25_ell0p5": r"$(.25,.5)$",
    "nu0p25_ell1p0": r"$(.25,1)$",
    "nu0p4_ell0p1": r"$(.4,.1)$",
    "nu0p4_ell0p5": r"$(.4,.5)$",
    "nu0p4_ell1p0": r"$(.4,1)$",
}
CONFIG_LABELS_V2 = {
    "nu0p25_ell0p1": "$\\nu=0.25$\n$\\ell=0.1$",
    "nu0p25_ell0p5": "$\\nu=0.25$\n$\\ell=0.5$",
    "nu0p25_ell1p0": "$\\nu=0.25$\n$\\ell=1.0$",
    "nu0p4_ell0p1": "$\\nu=0.4$\n$\\ell=0.1$",
    "nu0p4_ell0p5": "$\\nu=0.4$\n$\\ell=0.5$",
    "nu0p4_ell1p0": "$\\nu=0.4$\n$\\ell=1.0$",
}
COLORS = {
    "ExactFEM": "#222222",
    "FNO": "#0072B2",
    "DeepONet": "#E69F00",
    "GenFNO": "#009E73",
    "GenDeepONet": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-results-root",
        type=Path,
        default=ROOT / "sulis_snapshot_2026-09-25/provenance/results",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "sulis_snapshot_2026-09-25/provenance/results/revision_2026-09/"
        "E3/full/submissions.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "paper_products/generated/revision/E3",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--figures-v2-only",
        action="store_true",
        help=(
            "Read the existing trajectory/reference CSVs and write only the "
            "label/layout-corrected _v2 figures."
        ),
    )
    parser.add_argument(
        "--minimum-figure-v3-only",
        action="store_true",
        help=(
            "Read the existing trajectory/reference CSVs and write only the "
            "two-level-axis _v3 minimum-error figure."
        ),
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_history(archive: Any, key: str) -> np.ndarray:
    values = np.asarray(archive[key], dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{key}: expected a non-empty vector.")
    return values


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def summarize_archive(
    archive_path: Path,
    *,
    source: str,
    method: str,
    architecture: str,
    run_id: str,
    truth: str,
    configuration_id: str,
    nu: float | None,
    length_scale: float | None,
    job_id: str,
) -> dict[str, Any]:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        iteration = finite_history(archive, "history_iteration")
        error = finite_history(archive, "history_relative_u_error")
        residual = finite_history(archive, "history_relative_residual")
        max_change = finite_history(archive, "history_max_proposed_change")
        active_bounds = finite_history(archive, "history_active_bound_fraction")
    if not (
        iteration.shape
        == error.shape
        == residual.shape
        == max_change.shape
        == active_bounds.shape
    ):
        raise ValueError(f"{archive_path}: histories are not aligned.")
    finite_error = np.isfinite(error)
    indices = np.flatnonzero(finite_error)
    if indices.size == 0:
        raise ValueError(f"{archive_path}: no finite coefficient errors.")
    oracle_index = int(indices[np.argmin(error[finite_error])])
    final_index = int(indices[-1])
    oracle_error = float(error[oracle_index])
    final_error = float(error[final_index])

    proposed = max_change[np.isfinite(max_change)]
    bounds = active_bounds[np.isfinite(active_bounds)]
    cap_active = np.isclose(proposed, 5.0, rtol=0.0, atol=1.0e-8)
    clipping_active = bounds > 0.0
    # This is ordinary L2 monotonicity, not the requested M_C monotonicity.
    l2_prefix = error[: oracle_index + 1]
    l2_monotone = bool(np.all(np.diff(l2_prefix) <= 1.0e-12))
    return {
        "source": source,
        "method": method,
        "architecture": architecture,
        "run_id": run_id,
        "truth": truth,
        "configuration_id": configuration_id,
        "matern_nu": "" if nu is None else nu,
        "matern_length_scale": "" if length_scale is None else length_scale,
        "job_id": job_id,
        "iterations_completed": int(iteration[final_index]),
        "min_relative_coefficient_error": oracle_error,
        "oracle_iteration": int(iteration[oracle_index]),
        "final_relative_coefficient_error": final_error,
        "final_over_min": final_error / oracle_error,
        "min_relative_data_misfit": float(np.nanmin(residual)),
        "L2_error_monotone_to_oracle": int(l2_monotone),
        "MC_error_monotone_to_oracle": "pending_E2_trajectory_diagnostic",
        "physical_change_cap_active_count_inferred": int(np.count_nonzero(cap_active)),
        "physical_change_cap_active_fraction_inferred": (
            float(np.mean(cap_active)) if cap_active.size else float("nan")
        ),
        "box_projection_active_count": int(np.count_nonzero(clipping_active)),
        "box_projection_active_fraction": (
            float(np.mean(clipping_active)) if clipping_active.size else float("nan")
        ),
        "maximum_active_bound_fraction": (
            float(np.max(bounds)) if bounds.size else float("nan")
        ),
        "archive": display_path(archive_path),
    }


def campaign_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = read_tsv(args.manifest)
    if len(manifest) != 60:
        raise ValueError(f"{args.manifest}: expected 60 campaign rows.")
    rows: list[dict[str, Any]] = []
    for item in manifest:
        output = args.snapshot_results_root / item["output_dir"]
        archive = output / (
            "landweber_exact_multires_results.npz"
            if item["method"] == "exact"
            else "result.npz"
        )
        rows.append(
            summarize_archive(
                archive,
                source=item["source"],
                method=item["method"],
                architecture=item["architecture"],
                run_id=item["run_id"],
                truth=item["truth"],
                configuration_id=item["configuration_id"],
                nu=float(item["nu"]),
                length_scale=float(item["length_scale"]),
                job_id=item["job_id"],
            )
        )
    keys = {
        (row["truth"], row["configuration_id"], row["architecture"])
        for row in rows
    }
    if len(keys) != 60:
        raise ValueError("E3 manifest does not identify 60 unique arms.")
    return rows


def _unique_directory(root: Path, run_id: str) -> Path:
    matches = sorted(root.glob(f"{run_id}_e*_minu"))
    if len(matches) != 1:
        raise ValueError(f"{root}: expected one directory for {run_id}, found {matches}.")
    return matches[0]


def reference_rows() -> list[dict[str, Any]]:
    registry = read_tsv(
        ROOT
        / "results/paper_T2_broad/budget2500_validation/"
        "landweber_registries/headline_opt.tsv"
    )
    models = [row for row in registry if row["study"] == "headline_opt"]
    if len(models) != 4:
        raise ValueError("Frozen headline registry must contain four rows.")
    rows: list[dict[str, Any]] = []

    baseline_t1 = (
        ROOT
        / "sulis_snapshot_2026-09-14/provenance/results/paper_T1/"
        "noiseless_landweber/headline_opt"
    )
    baseline_t2 = ROOT / "results/lw_noiseless_headline_min_u/T2_broad"
    identity = ROOT / "results/lw_noiseless_precond_ablation"
    for truth, baseline in (("T1", baseline_t1), ("T2", baseline_t2)):
        for model in models:
            directory = _unique_directory(baseline, model["run_id"])
            rows.append(
                summarize_archive(
                    directory / "result.npz",
                    source="reused_reference",
                    method="surrogate",
                    architecture=model["architecture"],
                    run_id=model["run_id"],
                    truth=truth,
                    configuration_id="baseline_nu1p5_ell0p1",
                    nu=1.5,
                    length_scale=0.1,
                    job_id="",
                )
            )
            identity_directory = identity / ("T1" if truth == "T1" else "T2_broad") / (
                model["run_id"] + "__precond-identity"
            )
            rows.append(
                summarize_archive(
                    identity_directory / "result.npz",
                    source="reused_reference",
                    method="surrogate",
                    architecture=model["architecture"],
                    run_id=model["run_id"],
                    truth=truth,
                    configuration_id="identity",
                    nu=None,
                    length_scale=None,
                    job_id="",
                )
            )

    exact_t1 = (
        ROOT
        / "sulis_snapshot_2026-09-14/provenance/results/paper_T1/"
        "noiseless_landweber/exact_fem1x_10k/landweber_exact_multires_results.npz"
    )
    exact_t2 = (
        ROOT
        / "results/lw_noiseless_headline/T2_broad/exact_fem1x/"
        "landweber_exact_multires_results.npz"
    )
    for truth, archive in (("T1", exact_t1), ("T2", exact_t2)):
        rows.append(
            summarize_archive(
                archive,
                source="reused_reference",
                method="exact",
                architecture="ExactFEM",
                run_id="exact_fem1x",
                truth=truth,
                configuration_id="baseline_nu1p5_ell0p1",
                nu=1.5,
                length_scale=0.1,
                job_id="",
            )
        )
    exact_identity_t2 = (
        ROOT
        / "results/lw_noiseless_precond_ablation/T2_broad/"
        "exact_fem1x_10k__precond-identity/landweber_exact_multires_results.npz"
    )
    rows.append(
        summarize_archive(
            exact_identity_t2,
            source="reused_reference",
            method="exact",
            architecture="ExactFEM",
            run_id="exact_fem1x",
            truth="T2",
            configuration_id="identity",
            nu=None,
            length_scale=None,
            job_id="",
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    path_stem: Path,
    campaign: list[dict[str, Any]],
    references: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, 5, figsize=(14.2, 6.0), sharex=True, sharey="row", constrained_layout=True
    )
    x = np.arange(len(CONFIG_ORDER))
    for row_index, truth in enumerate(("T1", "T2")):
        for col_index, architecture in enumerate(ARCHITECTURES):
            axis = axes[row_index, col_index]
            selected = [
                row
                for row in campaign
                if row["truth"] == truth and row["architecture"] == architecture
            ]
            values = [
                next(
                    float(row[metric])
                    for row in selected
                    if row["configuration_id"] == config
                )
                for config in CONFIG_ORDER
            ]
            axis.plot(x, values, marker="o", color=COLORS[architecture], linewidth=1.5)
            for config, linestyle, label in (
                ("baseline_nu1p5_ell0p1", "--", r"baseline $(1.5,.1)$"),
                ("identity", ":", "identity"),
            ):
                matches = [
                    row
                    for row in references
                    if row["truth"] == truth
                    and row["architecture"] == architecture
                    and row["configuration_id"] == config
                ]
                if len(matches) == 1:
                    axis.axhline(
                        float(matches[0][metric]),
                        color="#666666" if config.startswith("baseline") else "#B2182B",
                        linestyle=linestyle,
                        linewidth=1.0,
                        label=label,
                    )
            if row_index == 0:
                axis.set_title(DISPLAY_ARCHITECTURE.get(architecture, architecture))
            if col_index == 0:
                axis.set_ylabel(f"{truth}: {ylabel}")
            axis.set_xticks(x, [CONFIG_LABELS[item] for item in CONFIG_ORDER], rotation=45, ha="right")
            axis.grid(axis="y", alpha=0.2)
            if metric in {"min_relative_coefficient_error", "final_over_min"}:
                axis.set_yscale("log")
            if row_index == 0 and col_index == 0:
                axis.legend(frameon=False, fontsize=7)
    fig.supxlabel(r"Matérn geometry $(\nu_C,\ell_C)$")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _metric_values(
    rows: list[dict[str, Any]],
    *,
    truth: str,
    architecture: str,
    metric: str,
) -> np.ndarray:
    selected = [
        row
        for row in rows
        if row["truth"] == truth and row["architecture"] == architecture
    ]
    values = []
    for configuration in CONFIG_ORDER:
        matches = [
            row for row in selected if row["configuration_id"] == configuration
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {truth}/{architecture}/{configuration} row; "
                f"found {len(matches)}."
            )
        values.append(float(matches[0][metric]))
    return np.asarray(values, dtype=np.float64)


def _reference_value(
    rows: list[dict[str, Any]],
    *,
    truth: str,
    architecture: str,
    configuration: str,
    metric: str,
) -> float | None:
    matches = [
        row
        for row in rows
        if row["truth"] == truth
        and row["architecture"] == architecture
        and row["configuration_id"] == configuration
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(
            f"Expected at most one {truth}/{architecture}/{configuration} "
            f"reference; found {len(matches)}."
        )
    return float(matches[0][metric])


def plot_metric_v2(
    path_stem: Path,
    campaign: list[dict[str, Any]],
    references: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
    dpi: int,
) -> None:
    """Write the requested label/layout-only E3 revision.

    Every plotted ordinate is read directly from one of the two frozen CSVs;
    this function performs no numerical transformation other than converting
    the CSV text to ``float``.
    """

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from scripts.paper_products.common import configure_matplotlib, save_figure

    configure_matplotlib()
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(14.8, 6.0),
        sharex=True,
        sharey=True if metric == "final_over_min" else "row",
    )
    x = np.arange(len(CONFIG_ORDER), dtype=np.float64)
    baseline_colour = "#666666"
    identity_colour = "#B2182B"
    for row_index, truth in enumerate(("T1", "T2")):
        for column_index, architecture in enumerate(ARCHITECTURES):
            axis = axes[row_index, column_index]
            values = _metric_values(
                campaign,
                truth=truth,
                architecture=architecture,
                metric=metric,
            )
            axis.plot(
                x,
                values,
                marker="o",
                markersize=3.8,
                color=COLORS[architecture],
                linewidth=1.45,
                zorder=3,
            )
            baseline = _reference_value(
                references,
                truth=truth,
                architecture=architecture,
                configuration="baseline_nu1p5_ell0p1",
                metric=metric,
            )
            identity = _reference_value(
                references,
                truth=truth,
                architecture=architecture,
                configuration="identity",
                metric=metric,
            )
            if baseline is not None:
                axis.axhline(
                    baseline,
                    color=baseline_colour,
                    linestyle="--",
                    linewidth=1.0,
                    zorder=1,
                )
            if identity is not None:
                axis.axhline(
                    identity,
                    color=identity_colour,
                    linestyle=":",
                    linewidth=1.2,
                    zorder=1,
                )
            elif truth == "T1" and architecture == "ExactFEM":
                axis.text(
                    0.97,
                    0.95,
                    "identity not run",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.5,
                    color=identity_colour,
                )
            axis.axvline(2.5, color="#D0D0D0", linewidth=0.8, zorder=0)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
            axis.set_xticks(x)
            axis.set_xticklabels(
                [CONFIG_LABELS_V2[item] for item in CONFIG_ORDER],
                rotation=28,
                ha="right",
                rotation_mode="anchor",
                linespacing=1.15,
                fontsize=6.4,
            )
            axis.tick_params(axis="x", pad=3.5)
            if row_index == 0:
                axis.set_title(DISPLAY_ARCHITECTURE.get(architecture, architecture))
            if metric == "min_relative_coefficient_error":
                axis.set_yscale("log")
            else:
                axis.set_ylim(0.9, 3.1)
                axis.axhline(1.0, color="#999999", linewidth=0.7, zorder=0)
                if row_index == 0 and column_index == 0:
                    axis.text(
                        0.03,
                        1.015,
                        "no late-iteration deterioration",
                        transform=axis.get_yaxis_transform(),
                        ha="left",
                        va="bottom",
                        fontsize=6.2,
                        color="#666666",
                    )

    # Row identifiers and one scientific y-axis title avoid repeating labels
    # in ten narrow panels.
    axes[0, 0].text(
        -0.17,
        1.01,
        "T1",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="bottom",
        weight="bold",
    )
    axes[1, 0].text(
        -0.17,
        1.01,
        "T2",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="bottom",
        weight="bold",
    )
    figure.supylabel(ylabel, x=0.006)
    figure.supxlabel(r"Matérn preconditioner parameters $(\nu_C,\ell_C)$", y=0.015)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            marker="o",
            markersize=4,
            linewidth=1.45,
            label=r"Matérn, $\nu_C<1/2$ (this sweep)",
        ),
        Line2D(
            [0],
            [0],
            color=baseline_colour,
            linestyle="--",
            linewidth=1.0,
            label=r"Matérn $\nu_C=3/2$, $\ell_C=0.1$ (baseline)",
        ),
        Line2D(
            [0],
            [0],
            color=identity_colour,
            linestyle=":",
            linewidth=1.2,
            label="identity (no preconditioning)",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        columnspacing=2.2,
        handlelength=2.8,
    )
    figure.subplots_adjust(
        left=0.078,
        right=0.995,
        bottom=0.195,
        top=0.875,
        wspace=0.16,
        hspace=0.22,
    )
    save_figure(figure, path_stem, dpi=dpi)


def build_v2_figures_from_existing_csvs(args: argparse.Namespace) -> list[Path]:
    campaign_path = args.output_dir / "trajectory_outcomes_pre_geometry.csv"
    reference_path = args.output_dir / "reference_arms.csv"
    campaign = read_csv(campaign_path)
    references = read_csv(reference_path)
    if len(campaign) != 60:
        raise ValueError(f"{campaign_path}: expected 60 rows, found {len(campaign)}.")
    if len(references) != 19:
        raise ValueError(f"{reference_path}: expected 19 rows, found {len(references)}.")
    for metric, filename, ylabel in (
        (
            "min_relative_coefficient_error",
            "minimum_coefficient_error_by_geometry_v2",
            r"min relative error in $m$",
        ),
        (
            "final_over_min",
            "final_over_min_by_geometry_v2",
            "final error / minimum error",
        ),
    ):
        plot_metric_v2(
            args.output_dir / filename,
            campaign,
            references,
            metric=metric,
            ylabel=ylabel,
            dpi=args.dpi,
        )
    return [
        args.output_dir / "minimum_coefficient_error_by_geometry_v2.pdf",
        args.output_dir / "minimum_coefficient_error_by_geometry_v2.png",
        args.output_dir / "final_over_min_by_geometry_v2.pdf",
        args.output_dir / "final_over_min_by_geometry_v2.png",
    ]


def plot_minimum_metric_v3(
    path_stem: Path,
    campaign: list[dict[str, Any]],
    references: list[dict[str, Any]],
    *,
    dpi: int,
) -> None:
    """Write the compact two-level-axis revision of the minimum-error figure."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from scripts.paper_products.common import configure_matplotlib, save_figure

    configure_matplotlib()
    figure, axes = plt.subplots(
        2,
        5,
        figsize=(14.8, 7.15),
        sharex=True,
        sharey="row",
        layout="constrained",
    )
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(w_pad=0.025, h_pad=0.025, wspace=0.035, hspace=0.06)

    x = np.arange(len(CONFIG_ORDER), dtype=np.float64)
    ell_labels = ("0.1", "0.5", "1.0", "0.1", "0.5", "1.0")
    baseline_colour = "#666666"
    identity_colour = "#B2182B"
    for row_index, truth in enumerate(("T1", "T2")):
        for column_index, architecture in enumerate(ARCHITECTURES):
            axis = axes[row_index, column_index]
            values = _metric_values(
                campaign,
                truth=truth,
                architecture=architecture,
                metric="min_relative_coefficient_error",
            )
            axis.plot(
                x,
                values,
                marker="o",
                markersize=3.8,
                color=COLORS[architecture],
                linewidth=1.45,
                zorder=3,
            )
            baseline = _reference_value(
                references,
                truth=truth,
                architecture=architecture,
                configuration="baseline_nu1p5_ell0p1",
                metric="min_relative_coefficient_error",
            )
            identity = _reference_value(
                references,
                truth=truth,
                architecture=architecture,
                configuration="identity",
                metric="min_relative_coefficient_error",
            )
            if baseline is not None:
                axis.axhline(
                    baseline,
                    color=baseline_colour,
                    linestyle="--",
                    linewidth=1.0,
                    zorder=1,
                )
            if identity is not None:
                axis.axhline(
                    identity,
                    color=identity_colour,
                    linestyle=":",
                    linewidth=1.2,
                    zorder=1,
                )
            elif truth == "T1" and architecture == "ExactFEM":
                axis.text(
                    0.97,
                    0.95,
                    "identity not run",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.5,
                    color=identity_colour,
                )
            axis.axvline(2.5, color="#D0D0D0", linewidth=0.8, zorder=0)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
            axis.set_yscale("log")
            axis.set_xticks(x, ell_labels)
            axis.tick_params(
                axis="x",
                pad=1.5,
                labelsize=9,
                labelbottom=row_index == 1,
            )
            if row_index == 0:
                axis.set_title(
                    DISPLAY_ARCHITECTURE.get(architecture, architecture),
                    pad=2.0,
                )
            if row_index == 1:
                for centre, label in (
                    (1.0, r"$\nu_C=0.25$"),
                    (4.0, r"$\nu_C=0.4$"),
                ):
                    axis.text(
                        centre,
                        -0.135,
                        label,
                        transform=axis.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=9,
                        clip_on=False,
                    )

    axes[0, 0].text(
        -0.17,
        1.01,
        "T1",
        transform=axes[0, 0].transAxes,
        ha="right",
        va="bottom",
        weight="bold",
    )
    axes[1, 0].text(
        -0.17,
        1.01,
        "T2",
        transform=axes[1, 0].transAxes,
        ha="right",
        va="bottom",
        weight="bold",
    )
    figure.supylabel(r"min relative error in $m$")
    figure.supxlabel(r"Matérn preconditioner length scale $\ell_C$")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            marker="o",
            markersize=4,
            linewidth=1.45,
            label=r"Matérn, $\nu_C<1/2$ (this sweep)",
        ),
        Line2D(
            [0],
            [0],
            color=baseline_colour,
            linestyle="--",
            linewidth=1.0,
            label=r"Matérn $\nu_C=3/2$, $\ell_C=0.1$ (baseline)",
        ),
        Line2D(
            [0],
            [0],
            color=identity_colour,
            linestyle=":",
            linewidth=1.2,
            label="identity (no preconditioning)",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="outside upper center",
        ncol=3,
        frameon=False,
        columnspacing=2.2,
        handlelength=2.8,
    )
    save_figure(figure, path_stem, dpi=dpi)


def build_v3_minimum_figure_from_existing_csvs(args: argparse.Namespace) -> list[Path]:
    campaign_path = args.output_dir / "trajectory_outcomes_pre_geometry.csv"
    reference_path = args.output_dir / "reference_arms.csv"
    campaign = read_csv(campaign_path)
    references = read_csv(reference_path)
    if len(campaign) != 60:
        raise ValueError(f"{campaign_path}: expected 60 rows, found {len(campaign)}.")
    if len(references) != 19:
        raise ValueError(f"{reference_path}: expected 19 rows, found {len(references)}.")
    stem = args.output_dir / "minimum_coefficient_error_by_geometry_v3"
    plot_minimum_metric_v3(stem, campaign, references, dpi=args.dpi)
    return [stem.with_suffix(".pdf"), stem.with_suffix(".png")]


def write_note(
    path: Path,
    campaign: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> None:
    lines = [
        "# E3 low-smoothness Matérn geometry sweep: first-layer summary",
        "",
        "All 60 preregistered trajectories completed. This summary uses the stored",
        "10,001-point scalar histories. It does not yet make the E2 covariance-geometry",
        "monotonicity claim; that column is explicitly pending the saved-state diagnostic.",
        "",
        "## Best low-smoothness configuration by minimum relative coefficient error",
        "",
        "| Truth | Method | Configuration | Minimum error | Oracle iteration | Final/min |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for truth in ("T1", "T2"):
        for architecture in ARCHITECTURES:
            selected = [
                row
                for row in campaign
                if row["truth"] == truth and row["architecture"] == architecture
            ]
            best = min(selected, key=lambda row: float(row["min_relative_coefficient_error"]))
            lines.append(
                f"| {truth} | {DISPLAY_ARCHITECTURE.get(architecture, architecture)} | "
                f"{best['configuration_id']} | {float(best['min_relative_coefficient_error']):.6g} | "
                f"{best['oracle_iteration']} | {float(best['final_over_min']):.4g} |"
            )
    lines.extend(
        [
            "",
            "## Reference arms",
            "",
            "The figures reuse the frozen Matérn `(nu_C, ell_C)=(1.5,0.1)` and",
            "identity trajectories. The exact-FEM T1 identity arm does not exist and",
            "is therefore not drawn; it was not silently imputed.",
            "",
            "The physical-change-cap count is inferred from stored accepted proposed",
            "changes equal to the configured maximum of 5. The box-projection count",
            "uses the stored active-bound fraction. These are screening summaries; the",
            "full E1/E2 analysis will use the saved fields and exact definitions.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.figures_v2_only and args.minimum_figure_v3_only:
        raise ValueError("Choose at most one figure-only mode.")
    if args.minimum_figure_v3_only:
        outputs = build_v3_minimum_figure_from_existing_csvs(args)
        print("Saved two-level-axis E3 v3 minimum-error figure:")
        for output in outputs:
            print(f"  {output}")
        return
    if args.figures_v2_only:
        outputs = build_v2_figures_from_existing_csvs(args)
        print("Saved label/layout-only E3 v2 figures:")
        for output in outputs:
            print(f"  {output}")
        return
    campaign = campaign_rows(args)
    references = reference_rows()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "trajectory_outcomes_pre_geometry.csv", campaign)
    write_csv(args.output_dir / "reference_arms.csv", references)
    write_note(args.output_dir / "NOTE_PRE_GEOMETRY.md", campaign, references)
    plot_metric(
        args.output_dir / "minimum_coefficient_error_by_geometry",
        campaign,
        references,
        metric="min_relative_coefficient_error",
        ylabel=r"minimum relative $m$ error",
        dpi=args.dpi,
    )
    plot_metric(
        args.output_dir / "final_over_min_by_geometry",
        campaign,
        references,
        metric="final_over_min",
        ylabel="final/minimum error ratio",
        dpi=args.dpi,
    )
    print(f"Saved first-layer E3 summary in {args.output_dir}")


if __name__ == "__main__":
    main()
