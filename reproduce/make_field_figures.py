#!/usr/bin/env python3
"""Regenerate the two frozen coefficient-field figures from extracted arrays."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(os.environ.get("REACTION_DATA_ROOT", ROOT / "data/reaction-paper-v1"))
METHODS = ("Exact FEM", "FNO", "DeepONet", "GenFNO", "GenDeepONet")
PANEL_LABELS = ("Truth",) + METHODS


def configure():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "reaction-release-fields-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.45,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
        }
    )
    return plt


def image(axis, field, *, vmin: float, vmax: float):
    rendered = axis.imshow(
        np.asarray(field, dtype=float).T,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        interpolation="nearest",
        aspect="equal",
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    return rendered


def save(plt, fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        bbox_inches="tight",
        metadata={
            "Creator": "scripts/revision_2026_09/build_reconstruction_field_figures.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def plot_baseline(plt, t1_fields, t1_errors, t2_fields, t2_errors, output: Path) -> None:
    fig = plt.figure(figsize=(6.38, 2.62), layout="constrained")
    grid = fig.add_gridspec(2, 7, width_ratios=(1, 1, 1, 1, 1, 1, 0.055), hspace=0.07, wspace=0.045)
    for row, (truth, fields, errors) in enumerate((("T1", t1_fields, t1_errors), ("T2", t2_fields, t2_errors))):
        row_minimum = min(float(np.min(field)) for field in fields)
        row_maximum = max(float(np.max(field)) for field in fields)
        rendered = None
        for column, (label, field, error) in enumerate(zip(PANEL_LABELS, fields, errors)):
            axis = fig.add_subplot(grid[row, column])
            rendered = image(axis, field, vmin=row_minimum, vmax=row_maximum)
            if row == 0:
                axis.set_title("Truth" if column == 0 else label, fontsize=7.7, pad=2.5)
            if column > 0:
                axis.text(
                    0.5, 0.02, f"$e_m={100.0 * float(error):.2f}\\%$",
                    transform=axis.transAxes, ha="center", va="bottom", fontsize=6.8, color="black",
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
                )
            if column == 0:
                axis.text(-0.08, 0.5, truth, transform=axis.transAxes, rotation=90,
                          ha="right", va="center", fontsize=8, fontweight="bold")
        colour_axis = fig.add_subplot(grid[row, 6])
        ticks = np.linspace(row_minimum, row_maximum, 5)
        colourbar = fig.colorbar(rendered, cax=colour_axis, ticks=ticks)
        colourbar.set_ticklabels([f"{tick:.1f}" for tick in ticks])
        colourbar.set_label(r"$m$", labelpad=2)
        colourbar.ax.tick_params(labelsize=6.5, pad=1.5)
    save(plt, fig, output / "noiseless_oracle_reconstructions_rows.pdf")


def plot_identity(plt, data, output: Path) -> None:
    truth = np.asarray(data["identity_truth"], dtype=float)
    oracle_fields = np.asarray(data["identity_oracle_fields"], dtype=float)
    final_fields = np.asarray(data["identity_final_fields"], dtype=float)
    oracle_iterations = np.asarray(data["identity_oracle_iterations"], dtype=int)
    oracle_errors = np.asarray(data["identity_oracle_errors"], dtype=float)
    final_errors = np.asarray(data["identity_final_errors"], dtype=float)
    # The frozen PDF's tight bounding box is 163.416 pt high.  The extra
    # 1/3000 inch compensates for the backend's final-point rounding and keeps
    # the two field rows on the same rendered pixels as the paper asset.
    fig = plt.figure(figsize=(6.38, 2.150333333333333), layout="constrained")
    grid = fig.add_gridspec(2, 7, width_ratios=(1, 1, 1, 1, 1, 1, 0.06), height_ratios=(1, 1), hspace=0.0, wspace=0.045)
    axes = [[fig.add_subplot(grid[row, column]) for column in range(6)] for row in range(2)]
    field_image = image(axes[0][0], truth, vmin=0.0, vmax=200.0)
    axes[0][0].set_title("Truth", fontsize=7.5, pad=2.5)
    axes[0][0].text(-0.10, 0.5, "Oracle", transform=axes[0][0].transAxes, rotation=90,
                    ha="right", va="center", fontsize=8, fontweight="bold")
    for index, method in enumerate(METHODS):
        column = index + 1
        field_image = image(axes[0][column], oracle_fields[index], vmin=0.0, vmax=200.0)
        axes[0][column].set_title(method, fontsize=7.4, pad=2.5)
        axes[0][column].text(
            0.5, 0.02,
            f"$k_{{\\rm or}}={int(oracle_iterations[index]):,}$\n$e_m={100.0 * oracle_errors[index]:.2f}\\%$",
            transform=axes[0][column].transAxes, ha="center", va="bottom", fontsize=6.5, linespacing=1.0,
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
        image(axes[1][column], final_fields[index], vmin=0.0, vmax=200.0)
        axes[1][column].text(
            0.5, 0.02, f"$e_m={100.0 * final_errors[index]:.2f}\\%$",
            transform=axes[1][column].transAxes, ha="center", va="bottom", fontsize=6.7,
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
    axes[1][0].axis("off")
    axes[1][0].text(0.50, 0.50, "Final\n$k=10{,}000$", transform=axes[1][0].transAxes,
                    ha="center", va="center", fontsize=8, fontweight="bold")
    colour_axis = fig.add_subplot(grid[:, 6])
    colourbar = fig.colorbar(field_image, cax=colour_axis, ticks=(0, 50, 100, 150, 200))
    colourbar.set_label(r"$m$", labelpad=2)
    colourbar.ax.tick_params(labelsize=7, pad=1.5)
    save(plt, fig, output / "preconditioner_identity_fields_T2.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/figures")
    args = parser.parse_args()
    plt = configure()
    source = args.data_root / "figure_data/reconstruction_fields.npz"
    with np.load(source, allow_pickle=False) as data:
        plot_baseline(plt, data["t1_fields"], data["t1_oracle_errors"], data["t2_fields"], data["t2_oracle_errors"], args.output_dir)
        plot_identity(plt, data, args.output_dir)
    print(f"Wrote field figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
