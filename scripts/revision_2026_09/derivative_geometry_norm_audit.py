#!/usr/bin/env python3
"""Audit the norms used in the frozen headline derivative diagnostic.

This is post-processing only.  It reconstructs the pooled JVP, ambient VJP,
coefficient-L2 preconditioned VJP, and covariance-geometry preconditioned VJP
errors from the archived arrays.  No FEM, neural model, or covariance solve is
performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_ORDER = ("FNO", "DeepONet", "GenFNO", "GenDeepONet")
BASEPOINT_ORDER = ("initial", "truth")
COLORS = {
    "FNO": "#0072B2",
    "DeepONet": "#E69F00",
    "GenFNO": "#009E73",
    "GenDeepONet": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT
        / "sulis_snapshot_2026-09-06/provenance/results/paper_T2_broad/"
        "derivative_diagnostics/headline_fem_comparison",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT
        / "results/paper_T2_broad/budget2500_validation/"
        "landweber_registries/headline_opt.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "paper_products/generated/revision/E8",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def weighted_relative_error(
    predicted: np.ndarray,
    exact: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Return one pooled weighted error ratio over all array entries."""

    predicted = np.asarray(predicted, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if predicted.shape != exact.shape:
        raise ValueError("Predicted and exact arrays must have identical shapes.")
    if weights.shape != predicted.shape[-2:]:
        raise ValueError("Weights must match the trailing spatial dimensions.")
    effective = np.broadcast_to(weights, predicted.shape)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != weights.shape:
            raise ValueError("Mask and weights must have identical shapes.")
        effective = effective * np.broadcast_to(mask, predicted.shape)
    numerator = float(np.sum(effective * (predicted - exact) ** 2))
    denominator = float(np.sum(effective * exact**2))
    if not denominator > 0.0:
        raise ValueError("Exact weighted norm is zero.")
    return math.sqrt(max(numerator, 0.0) / denominator)


def covariance_geometry_relative_error(
    predicted_covector: np.ndarray,
    exact_covector: np.ndarray,
    predicted_direction: np.ndarray,
    exact_direction: np.ndarray,
) -> float:
    """Return ``||C(q_theta-q)||_C / ||C q||_C`` by energy pairing.

    For ``d=Cq``, the covariance geometry satisfies
    ``||d||_C^2 = q^T d``.  The archived coordinate covectors and directions
    therefore suffice; no covariance inverse is needed.
    """

    q_theta = np.asarray(predicted_covector, dtype=np.float64)
    q_exact = np.asarray(exact_covector, dtype=np.float64)
    d_theta = np.asarray(predicted_direction, dtype=np.float64)
    d_exact = np.asarray(exact_direction, dtype=np.float64)
    if not (q_theta.shape == q_exact.shape == d_theta.shape == d_exact.shape):
        raise ValueError("Covectors and covariance directions must align.")
    delta_q = q_theta - q_exact
    delta_d = d_theta - d_exact
    numerator = float(np.sum(delta_q * delta_d))
    denominator = float(np.sum(q_exact * d_exact))
    scale = max(abs(denominator), 1.0)
    if numerator < -1.0e-12 * scale:
        raise ValueError(f"Negative covariance error energy: {numerator}.")
    if not denominator > 0.0:
        raise ValueError(f"Non-positive exact covariance energy: {denominator}.")
    return math.sqrt(max(numerator, 0.0) / denominator)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registry_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = [row for row in rows if row.get("study") == "headline_opt"]
    if len(selected) != 4:
        raise ValueError(f"{path}: expected four headline models.")
    if {row["architecture"] for row in selected} != set(ARCHITECTURE_ORDER):
        raise ValueError(f"{path}: unexpected architecture set.")
    return selected


def audit_row(
    path: Path,
    *,
    architecture: str,
    run_id: str,
    basepoint: str,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        weights = np.asarray(data["observation_weights"], dtype=np.float64)
        mask = np.asarray(data["observation_mask"], dtype=bool)
        surrogate_jvp = np.asarray(data["surrogate_jvp"], dtype=np.float64)
        exact_jvp = np.asarray(data["fem_jvp"], dtype=np.float64)
        q_theta = np.asarray(data["surrogate_coordinate_covector"], dtype=np.float64)
        q_exact = np.asarray(
            data["fem_coordinate_covector_same_residual"], dtype=np.float64
        )
        d_theta = np.asarray(data["surrogate_landweber_direction"], dtype=np.float64)
        d_exact = np.asarray(
            data["fem_landweber_direction_same_residual"], dtype=np.float64
        )
        surrogate_forward = np.asarray(data["surrogate_forward"], dtype=np.float64)
        observed = np.asarray(data["data_observed"], dtype=np.float64)

    residual = surrogate_forward - observed
    residual_norm = math.sqrt(float(np.sum(weights * mask * residual**2)))
    return {
        "basepoint": basepoint,
        "architecture": architecture,
        "run_id": run_id,
        "residual_definition": "surrogate_forward_minus_observed_data",
        "residual_weighted_norm": residual_norm,
        "pooled_fixed_direction_JVP_relative_error": weighted_relative_error(
            surrogate_jvp, exact_jvp, weights, mask
        ),
        "ambient_same_residual_VJP_relative_error_euclidean": weighted_relative_error(
            q_theta, q_exact, np.ones_like(weights)
        ),
        "preconditioned_same_residual_VJP_relative_error_L2_Wu": weighted_relative_error(
            d_theta, d_exact, weights
        ),
        "preconditioned_same_residual_VJP_relative_error_MC": (
            covariance_geometry_relative_error(q_theta, q_exact, d_theta, d_exact)
        ),
        "source_npz": str(path.relative_to(ROOT)),
        "source_sha256": sha256(path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty audit.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_note(path: Path, rows: list[dict[str, Any]]) -> None:
    by_basepoint = {
        basepoint: [row for row in rows if row["basepoint"] == basepoint]
        for basepoint in BASEPOINT_ORDER
    }
    lines = [
        "# E8 derivative-geometry norm audit",
        "",
        "This audit is post-processing of the frozen headline derivative arrays; no ",
        "FEM solve, neural-model evaluation or covariance solve was rerun.",
        "",
        "## Residual used by the VJP comparison",
        "",
        "For each architecture and basepoint, the residual is",
        "`R_theta = G_theta(m) - y`. The same `R_theta` is supplied to the",
        "surrogate VJP and the exact-FEM VJP in that row, so the comparison",
        "isolates the derivative at fixed residual. It is not the same numerical",
        "residual across architectures because `G_theta(m)` is model-dependent.",
        "The initial and truth fields are common across architectures.",
        "",
        "## Norms",
        "",
        "- Pooled JVP: one observation-weighted ratio after summing squared errors",
        "  over all 16 directions and all observed grid entries. It is not the",
        "  mean or median of 16 relative errors.",
        "- Ambient VJP: unweighted Euclidean norm of the nodal coordinate covector.",
        "- Preconditioned L2(W_m): trapezoidal coefficient-space norm of `C_h q`.",
        "  This is the norm used by the original Figure 6 column.",
        "- Preconditioned M_C: covariance geometry computed from",
        "  `||C_h delta q||_C^2 = delta q^T C_h delta q`. This is the theorem-aligned",
        "  quantity requested in E8.",
        "",
        "## Result",
        "",
        "The theorem-aligned M_C values are:",
        "",
        "| Basepoint | Architecture | L2(W_m) | M_C |",
        "|---|---:|---:|---:|",
    ]
    for basepoint in BASEPOINT_ORDER:
        for architecture in ARCHITECTURE_ORDER:
            row = next(
                item
                for item in by_basepoint[basepoint]
                if item["architecture"] == architecture
            )
            lines.append(
                f"| {basepoint} | {architecture} | "
                f"{row['preconditioned_same_residual_VJP_relative_error_L2_Wu']:.6g} | "
                f"{row['preconditioned_same_residual_VJP_relative_error_MC']:.6g} |"
            )
    lines.extend(
        [
            "",
            "The revised CSV retains both preconditioned norms. Any manuscript",
            "discussion must name the norm explicitly rather than treating them as",
            "interchangeable.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot(path_stem: Path, rows: list[dict[str, Any]], *, dpi: int) -> None:
    import matplotlib.pyplot as plt

    metrics = (
        ("pooled_fixed_direction_JVP_relative_error", "fixed-direction JVP", "o"),
        (
            "ambient_same_residual_VJP_relative_error_euclidean",
            "ambient VJP (Euclidean)",
            "s",
        ),
        (
            "preconditioned_same_residual_VJP_relative_error_L2_Wu",
            r"preconditioned VJP ($L^2(W_m)$)",
            "^",
        ),
        (
            "preconditioned_same_residual_VJP_relative_error_MC",
            r"preconditioned VJP ($M_C$)",
            "D",
        ),
    )
    x = np.arange(len(ARCHITECTURE_ORDER), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(metrics))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.55), sharey=True, constrained_layout=True)
    for axis, basepoint, title in zip(
        axes,
        BASEPOINT_ORDER,
        (r"Common initial field $m_0\equiv100$", r"Common T2 truth $m^\dagger$"),
    ):
        current = [row for row in rows if row["basepoint"] == basepoint]
        for offset, (key, label, marker) in zip(offsets, metrics):
            values = [
                next(
                    float(row[key])
                    for row in current
                    if row["architecture"] == architecture
                )
                for architecture in ARCHITECTURE_ORDER
            ]
            axis.scatter(x + offset, values, label=label, marker=marker, s=36, zorder=3)
        axis.set_yscale("log")
        axis.set_xticks(x, ARCHITECTURE_ORDER, rotation=22, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", which="both", alpha=0.25)
    axes[0].set_ylabel("Relative derivative error")
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows_by_architecture = registry_rows(args.registry)
    rows: list[dict[str, Any]] = []
    for basepoint in BASEPOINT_ORDER:
        for architecture in ARCHITECTURE_ORDER:
            registry_row = next(
                row for row in rows_by_architecture if row["architecture"] == architecture
            )
            source = args.raw_root / registry_row["run_id"] / f"{basepoint}.npz"
            if not source.is_file():
                raise FileNotFoundError(source)
            rows.append(
                audit_row(
                    source,
                    architecture=architecture,
                    run_id=registry_row["run_id"],
                    basepoint=basepoint,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "derivative_geometry_norm_audit.csv", rows)
    write_note(args.output_dir / "NOTE.md", rows)
    plot(args.output_dir / "derivative_geometry_norm_audit", rows, dpi=args.dpi)
    print(f"Saved E8 audit in {args.output_dir}")


if __name__ == "__main__":
    main()
