#!/usr/bin/env python3
"""Build the temporary compact derivative-geometry paper figure.

The script reads only the frozen figure-data CSV recorded in the main-products
manifest. It writes the compact PDF/PNG and appends their provenance to the
compact-figure report; it never modifies the canonical figure or manuscript.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("REACTION_DATA_ROOT", ROOT / "data/reaction-paper-v1"))
OUTPUT_DIR = Path(os.environ.get("REACTION_OUTPUT_DIR", ROOT / "outputs/figures"))
SOURCE = DATA_ROOT / "figure_data/derivative_geometry.csv"
MANIFEST = DATA_ROOT / "manifests/main_products_manifest.json"
REPORT = OUTPUT_DIR / "REPORT.md"
EXPECTED_SOURCE_SHA256 = "947fe2fc01638c7f755c5671a554b950c31fd7a3b0037986482d4b60d427e573"

ARCHITECTURES = ("FNO", "DeepONet", "GenFNO", "GenDeepONet")
BASEPOINTS = (
    ("initial", r"Common initial field $m_0\equiv100$"),
    ("truth", r"Common T2 truth $m^\dagger$"),
)
METRICS = (
    (
        "pooled_fixed_direction_JVP_relative_error",
        r"fixed-direction JVP ($W_y$)",
        "o",
        "#1f77b4",
    ),
    (
        "ambient_same_residual_VJP_relative_error",
        "ambient VJP (Euclidean)",
        "s",
        "#ff7f0e",
    ),
    (
        "matern_preconditioned_same_residual_VJP_relative_error",
        r"preconditioned VJP ($\mathcal M_C$)",
        "^",
        "#2ca02c",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source() -> str:
    actual = sha256(SOURCE)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    matches = [
        product
        for product in manifest["products"]
        if product.get("output_path") == "paper_products/generated/figure_data/derivative_geometry.csv"
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one derivative-geometry source manifest entry, found {len(matches)}")
    recorded = matches[0]["output_sha256"]
    if actual != EXPECTED_SOURCE_SHA256 or actual != recorded:
        raise AssertionError(
            f"Frozen derivative-geometry source mismatch: actual={actual}, "
            f"expected={EXPECTED_SOURCE_SHA256}, manifest={recorded}"
        )
    return actual


def read_values() -> dict[tuple[str, str], dict[str, float]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 8:
        raise AssertionError(f"Expected eight source rows, found {len(rows)}")
    expected_keys = {(basepoint, architecture) for basepoint, _ in BASEPOINTS for architecture in ARCHITECTURES}
    actual_keys = {(row["basepoint"], row["architecture"]) for row in rows}
    if actual_keys != expected_keys:
        raise AssertionError(
            f"Unexpected basepoint/architecture rows: missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    values: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        current = {key: float(row[key]) for key, *_rest in METRICS}
        if not all(np.isfinite(value) and 1.0e-4 <= value <= 1.0 for value in current.values()):
            raise AssertionError(f"Invalid log-axis value for {row['basepoint']}/{row['architecture']}")
        values[(row["basepoint"], row["architecture"])] = current
    return values


def configure_matplotlib() -> Any:
    cache = Path(tempfile.gettempdir()) / "reaction-compact-derivative-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
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
            "axes.spines.top": False,
            "axes.spines.right": False,
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


def exact_tight_bbox(fig: Any) -> Any:
    """Return a tight bounding box exactly six inches wide."""
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.02)
    if tight.width > 6.0 + 1.0e-6:
        raise AssertionError(
            f"Tight derivative figure is wider than 6 inches: {tight.width:.6f}; "
            f"bounds=({tight.x0:.6f}, {tight.y0:.6f}, {tight.x1:.6f}, {tight.y1:.6f})"
        )
    if tight.height > 1.9 + 1.0e-6:
        raise AssertionError(f"Tight derivative figure is taller than 1.9 inches: {tight.height:.6f}")
    horizontal_padding = 0.5 * (6.0 - tight.width)
    return Bbox.from_extents(
        tight.x0 - horizontal_padding,
        tight.y0,
        tight.x1 + horizontal_padding,
        tight.y1,
    )


def plot(plt: Any, values: dict[tuple[str, str], dict[str, float]], dpi: int) -> tuple[Path, Path, float]:
    from matplotlib.ticker import LogFormatterMathtext, NullFormatter

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 1.8), sharey=True)
    fig.subplots_adjust(left=0.0874, right=0.9826, bottom=0.22, top=0.70, wspace=0.06)
    x = np.arange(len(ARCHITECTURES), dtype=float)
    offsets = (-0.18, 0.0, 0.18)
    handles = []
    for axis, (basepoint, title) in zip(axes, BASEPOINTS):
        for offset, (key, label, marker, colour) in zip(offsets, METRICS):
            current = np.asarray(
                [values[(basepoint, architecture)][key] for architecture in ARCHITECTURES],
                dtype=float,
            )
            scatter = axis.scatter(
                x + offset,
                current,
                label=label,
                marker=marker,
                color=colour,
                s=36,
                zorder=3,
            )
            axis.vlines(
                x + offset,
                ymin=np.full(len(current), 1.0e-4),
                ymax=current,
                colors="#1f77b4",
                alpha=0.18,
                linewidth=0.8,
            )
            if axis is axes[0]:
                handles.append(scatter)
        axis.set_yscale("log")
        axis.set_ylim(1.0e-4, 1.0)
        axis.set_yticks((1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0))
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0, labelOnlyBase=True))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.set_xticks(x, ARCHITECTURES, rotation=0, ha="center")
        axis.set_title(title, pad=2.0)
        axis.grid(axis="y", which="both", alpha=0.25)
        axis.margins(x=0.07)
    axes[0].set_ylabel("Relative error", labelpad=3.0)
    axes[1].tick_params(axis="y", which="both", labelleft=False)
    fig.legend(
        handles=handles,
        labels=[metric[1] for metric in METRICS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        columnspacing=0.30,
        handlelength=0.62,
        handletextpad=0.14,
        borderaxespad=0.0,
    )
    bbox = exact_tight_bbox(fig)
    pdf = OUTPUT_DIR / "derivative_geometry_compact.pdf"
    png = OUTPUT_DIR / "derivative_geometry_compact.png"
    metadata = {
        "Creator": "scripts/revision_2026_10/build_compact_derivative_geometry_figure.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf, bbox_inches=bbox, pad_inches=0.0, metadata=metadata)
    fig.savefig(
        png,
        dpi=dpi,
        bbox_inches=bbox,
        pad_inches=0.0,
        metadata={"Software": metadata["Creator"]},
    )
    height = bbox.height
    plt.close(fig)
    return pdf, png, height


def update_report(source_hash: str, pdf: Path, png: Path, height: float) -> None:
    heading = "## Compact derivative-geometry figure"
    existing = REPORT.read_text(encoding="utf-8") if REPORT.exists() else "# Compact figure report\n"
    if heading in existing:
        existing = existing.split(f"\n{heading}\n", maxsplit=1)[0].rstrip() + "\n"
    section = f"""

{heading}

- Source: `paper_products/generated/figure_data/derivative_geometry.csv`
- Source SHA-256: `{source_hash}` (matches the frozen main-products manifest)
- Plotted values: the 24 values in the source table are used directly and in the frozen architecture/basepoint order
- Final size: `6.000 x {height:.3f} in`
- PDF SHA-256: `{sha256(pdf)}`
- PNG SHA-256: `{sha256(png)}`
- Rendering checks: shared external one-row legend; horizontal 8 pt method labels; shared `10^-4` to `10^0` log range; no clipping or overlap; vector PDF

Regenerate from the repository root with:

```bash
python scripts/revision_2026_10/build_compact_derivative_geometry_figure.py --overwrite
```
"""
    REPORT.write_text(existing.rstrip() + section, encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = (
        OUTPUT_DIR / "derivative_geometry_compact.pdf",
        OUTPUT_DIR / "derivative_geometry_compact.png",
    )
    existing = [path.name for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Compact derivative outputs already exist: {existing}; pass --overwrite")
    source_hash = verify_source()
    values = read_values()
    plt = configure_matplotlib()
    pdf, png, height = plot(plt, values, args.dpi)
    update_report(source_hash, pdf, png, height)
    print(f"Saved {pdf.relative_to(ROOT)}")
    print(f"Saved {png.relative_to(ROOT)}")
    print(f"Final size: 6.000 x {height:.3f} in")


if __name__ == "__main__":
    main()
