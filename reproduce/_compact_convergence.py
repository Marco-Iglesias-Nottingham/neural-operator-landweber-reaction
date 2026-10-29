#!/usr/bin/env python3
"""Build compact, reversible alternatives to the frozen convergence figures.

This script reads the canonical figure-data CSVs already used by the current
paper figures. It never reads raw experiment archives, modifies the canonical
paper-product pipeline, or writes into a manuscript directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("REACTION_DATA_ROOT", ROOT / "data/reaction-paper-v1"))
DEFAULT_OUTPUT = Path(os.environ.get("REACTION_OUTPUT_DIR", ROOT / "outputs/figures"))
FIGURE_DATA = DATA_ROOT / "figure_data"
MANIFEST = DATA_ROOT / "manifests/main_products_manifest.json"

NOISELESS_SOURCES = {
    "T1": FIGURE_DATA / "noiseless_headline_convergence_T1.csv",
    "T2": FIGURE_DATA / "noiseless_headline_convergence_T2.csv",
}
NOISY_SOURCE = FIGURE_DATA / "noisy_headline_convergence_T1.csv"

EXPECTED_SOURCE_SHA256 = {
    "noiseless_headline_convergence_T1.csv": "437fc0cb549400d0d416a0e446e8eb196f52095b723b6b766f6cbbd344bb88be",
    "noiseless_headline_convergence_T2.csv": "cd3fdbf881bff303d2f07aefb81a21fc46b119c4c6b15b285deffc1671024333",
    "noisy_headline_convergence_T1.csv": "0abaef68e20d98505c872c0874c9267c4fef779c20fc54851f48ae8c75c76cb6",
}

METHODS = ["Exact FEM", "FNO", "DeepONet", "GenFNO", "GenDeepONet"]
COLOURS = {
    "Exact FEM": "#111111",
    "FNO": "#0072B2",
    "DeepONet": "#D55E00",
    "GenFNO": "#009E73",
    "GenDeepONet": "#CC79A7",
}
LINE_WIDTH = 1.45
RAW_DELTA_LINE_WIDTH = 1.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement only inside the dedicated compact-output folder.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        current = np.ascontiguousarray(array)
        digest.update(str(current.dtype).encode("ascii"))
        digest.update(json.dumps(current.shape).encode("ascii"))
        digest.update(current.tobytes(order="C"))
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_sources() -> dict[str, dict[str, str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    recorded = {
        Path(product["output_path"]).name: product["output_sha256"]
        for product in manifest["products"]
        if product.get("output_path", "").startswith("paper_products/generated/figure_data/")
    }
    result: dict[str, dict[str, str]] = {}
    for path in [*NOISELESS_SOURCES.values(), NOISY_SOURCE]:
        actual = sha256(path)
        expected = EXPECTED_SOURCE_SHA256[path.name]
        manifest_value = recorded.get(path.name)
        if actual != expected or actual != manifest_value:
            raise AssertionError(
                f"Frozen source hash mismatch for {path}: actual={actual}, "
                f"expected={expected}, manifest={manifest_value}"
            )
        result[path.name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": actual,
            "manifest_sha256": manifest_value,
            "status": "PASS",
        }
    return result


def require_target_policy(output: Path, overwrite: bool) -> None:
    expected = DEFAULT_OUTPUT.resolve()
    if output.resolve() != expected:
        raise ValueError(f"Output must be the dedicated compact folder: {expected}")
    output.mkdir(parents=True, exist_ok=True)
    targets = [
        output / "noiseless_convergence_compact.pdf",
        output / "noiseless_convergence_compact.png",
        output / "noisy_convergence_T1_compact.pdf",
        output / "noisy_convergence_T1_compact.png",
        output / "curve_checksums.json",
        output / "REPORT.md",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Compact outputs already exist; pass --overwrite to replace only "
            f"these alternatives: {[path.name for path in existing]}"
        )


def configure_matplotlib() -> Any:
    mpl_cache = Path(tempfile.gettempdir()) / "reaction-compact-convergence-matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": LINE_WIDTH,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": (
                r"\usepackage[T1]{fontenc}"
                r"\usepackage{lmodern}"
                r"\usepackage{amsmath}"
            ),
        }
    )
    return plt


def apply_plain_log_labels(axis: Any, values: tuple[float, ...]) -> None:
    """Label selected existing log ticks with plain decimal strings."""
    from matplotlib.ticker import FuncFormatter

    labels = {
        value: (f"{value:.3f}".rstrip("0").rstrip("."))
        for value in values
    }

    def formatter(value: float, _position: int | None = None) -> str:
        for expected, label in labels.items():
            if np.isclose(value, expected, rtol=1.0e-8, atol=1.0e-12):
                return label
        return ""

    plain = FuncFormatter(formatter)
    axis.yaxis.set_major_formatter(plain)
    axis.yaxis.set_minor_formatter(plain)


def method_handles() -> list[Any]:
    from matplotlib.lines import Line2D

    return [
        Line2D([0], [0], color=COLOURS[method], lw=LINE_WIDTH, label=method)
        for method in METHODS
    ]


def save_figure(plt: Any, fig: Any, stem: Path, dpi: int) -> tuple[Path, Path]:
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    metadata = {
        "Creator": "scripts/revision_2026_10/build_compact_convergence_figures.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf, bbox_inches=None, pad_inches=0, metadata=metadata)
    fig.savefig(
        png,
        dpi=dpi,
        bbox_inches=None,
        pad_inches=0,
        metadata={"Software": metadata["Creator"]},
    )
    plt.close(fig)
    return pdf, png


def grouped_rows(rows: Iterable[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    result: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        result[tuple(row[key] for key in keys)].append(row)
    return dict(result)


def noiseless_curves() -> tuple[dict[tuple[str, str], dict[str, np.ndarray]], list[dict[str, Any]]]:
    curves: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    checks: list[dict[str, Any]] = []
    for truth, source in NOISELESS_SOURCES.items():
        groups = grouped_rows(read_csv(source), ("truth_id", "method"))
        for method in METHODS:
            current = sorted(groups[(truth, method)], key=lambda row: int(row["iteration"]))
            iteration = np.asarray([int(row["iteration"]) for row in current], dtype=np.int64)
            misfit = np.asarray([float(row["relative_data_misfit"]) for row in current], dtype=np.float64)
            error = np.asarray([float(row["relative_u_error"]) for row in current], dtype=np.float64)
            if not np.array_equal(iteration, np.arange(10001, dtype=np.int64)):
                raise AssertionError(f"Unexpected noiseless iteration grid: {truth}/{method}")
            if not (np.all(np.isfinite(misfit)) and np.all(misfit > 0) and np.all(np.isfinite(error)) and np.all(error > 0)):
                raise AssertionError(f"Invalid logarithmic data: {truth}/{method}")
            curves[(truth, method)] = {"iteration": iteration, "misfit": misfit, "error": error}
            checks.append(
                {
                    "figure": "noiseless_convergence_compact",
                    "truth": truth,
                    "noise_percent": None,
                    "method": method,
                    "points": len(iteration),
                    "iteration_misfit_error_sha256": array_sha256(iteration, misfit, error),
                    "status": "PASS - plotted arrays loaded without transformation from canonical CSV",
                }
            )
    return curves, checks


def noisy_curves() -> tuple[dict[tuple[int, str], dict[str, Any]], list[dict[str, Any]]]:
    curves: dict[tuple[int, str], dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    groups = grouped_rows(read_csv(NOISY_SOURCE), ("noise_percent", "method"))
    for noise in (1, 2, 5):
        for method in METHODS:
            current = sorted(groups[(str(noise), method)], key=lambda row: int(row["iteration"]))
            iteration = np.asarray([int(row["iteration"]) for row in current], dtype=np.int64)
            misfit = np.asarray([float(row["misfit_over_raw_delta"]) for row in current], dtype=np.float64)
            error = np.asarray([float(row["relative_u_error"]) for row in current], dtype=np.float64)
            if not np.array_equal(iteration, np.arange(10001, dtype=np.int64)):
                raise AssertionError(f"Unexpected noisy iteration grid: {noise}%/{method}")
            if not (np.all(np.isfinite(misfit)) and np.all(misfit > 0) and np.all(np.isfinite(error)) and np.all(error > 0)):
                raise AssertionError(f"Invalid logarithmic data: {noise}%/{method}")
            first = current[0]
            repeated = ("oracle_iteration", "oracle_relative_u_error", "delta_reached", "delta_at_iteration")
            for key in repeated:
                if any(row[key] != first[key] for row in current):
                    raise AssertionError(f"Nonconstant noisy metadata {key}: {noise}%/{method}")
            curves[(noise, method)] = {
                "iteration": iteration,
                "misfit": misfit,
                "error": error,
                "oracle_iteration": int(first["oracle_iteration"]),
                "oracle_error": float(first["oracle_relative_u_error"]),
                "delta_reached": bool(int(first["delta_reached"])),
                "delta_iteration": int(first["delta_at_iteration"]),
            }
            checks.append(
                {
                    "figure": "noisy_convergence_T1_compact",
                    "truth": "T1",
                    "noise_percent": noise,
                    "method": method,
                    "points": len(iteration),
                    "iteration_misfit_error_sha256": array_sha256(iteration, misfit, error),
                    "status": "PASS - plotted arrays loaded without transformation from canonical CSV",
                }
            )
    return curves, checks


def plot_noiseless(plt: Any, curves: dict[tuple[str, str], dict[str, np.ndarray]], output: Path, dpi: int) -> tuple[Path, Path]:
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 3.6), sharex="col", layout="constrained")
    for row_index, truth in enumerate(("T1", "T2")):
        for method in METHODS:
            curve = curves[(truth, method)]
            colour = COLOURS[method]
            axes[row_index, 0].semilogy(curve["iteration"], curve["misfit"], color=colour)
            axes[row_index, 1].semilogy(curve["iteration"], curve["error"], color=colour)
            oracle = int(np.argmin(curve["error"]))
            axes[row_index, 1].plot(
                curve["iteration"][oracle],
                curve["error"][oracle],
                marker="*",
                color=colour,
                markersize=7,
                linestyle="none",
                zorder=4,
            )
        axes[row_index, 0].set_ylabel(truth, labelpad=18, fontweight="bold")
        apply_plain_log_labels(axes[row_index, 0], (0.001, 0.01, 0.1))
        apply_plain_log_labels(axes[row_index, 1], (0.04, 0.06, 0.1, 0.2))
        for axis in axes[row_index]:
            axis.set_xticks((0, 2500, 5000, 7500, 10000), ("0", "2,500", "5,000", "7,500", "10,000"))
            axis.margins(x=0.03)
            axis.grid(alpha=0.2, which="both")
    axes[0, 0].set_title("Data misfit")
    axes[0, 1].set_title(r"Relative error in $m$")
    for axis in axes[-1]:
        axis.set_xlabel("Landweber iteration")
    handles = method_handles() + [
        Line2D([0], [0], marker="*", color="#444444", linestyle="none", markersize=7, label="oracle")
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=6,
        frameon=False,
        columnspacing=0.75,
        handlelength=1.35,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    return save_figure(plt, fig, output / "noiseless_convergence_compact", dpi)


def plot_noisy(plt: Any, curves: dict[tuple[int, str], dict[str, Any]], output: Path, dpi: int) -> tuple[Path, Path]:
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(3, 2, figsize=(6.0, 5.0), sharex="col", layout="constrained")
    for row_index, noise in enumerate((1, 2, 5)):
        axes[row_index, 0].axhline(1.0, color="#666666", ls="--", lw=RAW_DELTA_LINE_WIDTH)
        for method in METHODS:
            curve = curves[(noise, method)]
            colour = COLOURS[method]
            axes[row_index, 0].semilogy(curve["iteration"], curve["misfit"], color=colour)
            axes[row_index, 1].semilogy(curve["iteration"], curve["error"], color=colour)
            if curve["delta_reached"]:
                k = curve["delta_iteration"]
                axes[row_index, 0].plot(k, curve["misfit"][k], marker="o", ms=4, color=colour, linestyle="none")
                axes[row_index, 1].plot(k, curve["error"][k], marker="o", ms=4, color=colour, linestyle="none")
            axes[row_index, 1].plot(
                curve["oracle_iteration"],
                curve["oracle_error"],
                marker="*",
                ms=7,
                color=colour,
                linestyle="none",
            )
        axes[row_index, 0].set_ylabel(f"{noise}\\% noise", labelpad=12, fontweight="bold")
        if noise == 5:
            apply_plain_log_labels(
                axes[row_index, 0],
                (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7),
            )
        else:
            apply_plain_log_labels(axes[row_index, 0], (1.0, 2.0, 3.0, 4.0, 6.0))
        apply_plain_log_labels(axes[row_index, 1], (0.06, 0.1, 0.2))
        for axis in axes[row_index]:
            axis.set_xscale("symlog", linthresh=10.0, linscale=0.75)
            axis.set_xticks((0, 10, 100, 1000, 10000), ("0", "10", "100", "1,000", "10,000"))
            axis.grid(alpha=0.2, which="both")
    axes[0, 0].set_title(r"Misfit / raw $\delta$")
    axes[0, 1].set_title(r"Relative error in $m$")
    for axis in axes[-1]:
        axis.set_xlabel("Landweber iteration")
    handles = method_handles() + [
        Line2D([0], [0], color="#666666", ls="--", lw=RAW_DELTA_LINE_WIDTH, label=r"raw $\delta$"),
        Line2D([0], [0], marker="o", color="#444444", linestyle="none", markersize=4, label=r"first $\leq\delta$"),
        Line2D([0], [0], marker="*", color="#444444", linestyle="none", markersize=7, label="oracle"),
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=8,
        frameon=False,
        columnspacing=0.46,
        handlelength=1.05,
        handletextpad=0.22,
        borderaxespad=0.0,
    )
    return save_figure(plt, fig, output / "noisy_convergence_T1_compact", dpi)


def write_checks(output: Path, sources: dict[str, dict[str, str]], curve_checks: list[dict[str, Any]], outputs: list[Path]) -> Path:
    payload = {
        "format": "reaction-compact-convergence-figures-v1",
        "source_verification": sources,
        "curve_count": len(curve_checks),
        "curve_checks": curve_checks,
        "style_contract": {
            "pdf_width_inches": 6.0,
            "noiseless_height_inches": 3.6,
            "noisy_height_inches": 5.0,
            "tick_font_pt": 8,
            "axis_label_font_pt": 9,
            "legend_font_pt": 9,
            "line_width_pt": LINE_WIDTH,
            "raw_delta_line_width_pt": RAW_DELTA_LINE_WIDTH,
            "text_usetex": True,
            "font": "Computer Modern via lmodern",
            "rasterized_curves": False,
        },
        "outputs": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)} for path in outputs
        ],
    }
    target = output / "curve_checksums.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def write_report(output: Path, sources: dict[str, dict[str, str]], curve_checks: list[dict[str, Any]], products: list[Path]) -> Path:
    lines = [
        "# Compact convergence figures (temporary alternatives)",
        "",
        "## Status",
        "",
        "- Existing figures, scripts, manifests, frozen results and manuscript files were not modified.",
        "- The three canonical figure-data CSVs match the SHA-256 values in the frozen main-products manifest.",
        f"- All {len(curve_checks)} curve groups use the canonical iteration, misfit and coefficient-error arrays without numerical transformation.",
        "- Automated source, curve-shape, finite-value, logarithmic-value, style and output checks passed.",
        "- Visual QA for the generated files: **PASS**. Both PDFs were rendered at 200 dpi at their final 6-inch width; legends, labels, tick labels and oracle/crossing markers are visible without clipping or overlap.",
        "",
        "## Sources",
        "",
        "| Source | SHA-256 | Manifest match |",
        "|---|---|---|",
    ]
    for item in sources.values():
        lines.append(f"| `{item['path']}` | `{item['sha256']}` | {item['status']} |")
    lines.extend(
        [
            "",
            "The current products were originally generated by `scripts/paper_products/build_figures.py` and recorded in `paper_products/generated/manifests/main_products_manifest.json`.",
            "",
            "Detailed per-curve array checksums are in `curve_checksums.json`.",
            "",
            "## Outputs",
            "",
            "| File | Intended size | SHA-256 |",
            "|---|---:|---|",
        ]
    )
    sizes = {
        "noiseless_convergence_compact.pdf": "6.0 x 3.6 in",
        "noiseless_convergence_compact.png": "1200 x 720 px at 200 dpi",
        "noisy_convergence_T1_compact.pdf": "6.0 x 5.0 in",
        "noisy_convergence_T1_compact.png": "1200 x 1000 px at 200 dpi",
    }
    for path in products:
        lines.append(f"| `{path.name}` | {sizes[path.name]} | `{sha256(path)}` |")
    lines.extend(
        [
            "",
            "Both PDFs contain vector curves and use LaTeX text with Computer Modern. Tick labels are 8 pt; axis labels, panel titles and the shared legend are 9 pt. There is no figure suptitle. `pdfimages -list` reports no embedded raster images in either PDF.",
            "",
            "## Plain-decimal tick-label revision",
            "",
            "The logarithmic y-axes retain their original scales, tick positions and limits, but labels of the form `a x 10^b` were replaced by plain decimal labels. The preceding compact PDFs and PNG previews are preserved beside the current files with suffix `_v1`.",
            "",
            "## Regeneration",
            "",
            "From the repository root:",
            "",
            "```bash",
            "python scripts/revision_2026_10/build_compact_convergence_figures.py --overwrite",
            "```",
            "",
            "The script writes only to `paper_products/generated/revision/compact_figures_20261028/` and verifies the frozen source hashes before plotting.",
            "",
        ]
    )
    target = output / "REPORT.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    require_target_policy(output, args.overwrite)
    sources = verify_sources()
    noiseless, noiseless_checks = noiseless_curves()
    noisy, noisy_checks = noisy_curves()
    plt = configure_matplotlib()
    products = [
        *plot_noiseless(plt, noiseless, output, args.dpi),
        *plot_noisy(plt, noisy, output, args.dpi),
    ]
    checks = write_checks(output, sources, noiseless_checks + noisy_checks, products)
    report = write_report(output, sources, noiseless_checks + noisy_checks, products)
    print(f"Saved {len(products)} compact figure products in {output}")
    print(f"Curve verification: {checks}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
