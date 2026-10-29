#!/usr/bin/env python3
"""E5: apply fixed Morozov multipliers to frozen noisy trajectories.

This is retrospective post-processing.  It never changes a trajectory or
selects a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "results/revision_2026-09/E5"
DEFAULT_PRODUCTS = ROOT / "paper_products/generated/revision/E5"
TAUS = (1.0, 1.5, 2.0, 2.5, 3.0)
METHODS = ("Exact FEM", "FNO", "DeepONet", "GenFNO", "GenDeepONet")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_crossing(residual: np.ndarray, threshold: float) -> int | None:
    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Residual history must be a finite non-negative vector.")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Threshold must be finite and non-negative.")
    indices = np.flatnonzero(values <= threshold)
    return int(indices[0]) if indices.size else None


def read_config() -> dict[str, Any]:
    path = ROOT / "paper_products/config/main_products.json"
    return json.loads(path.read_text(encoding="utf-8"))


def archive_path(
    config: dict[str, Any], truth: str, noise: int, method: str
) -> tuple[Path, str, str]:
    root = ROOT / config["archives"][truth]["noisy_root"] / f"noise{noise}"
    if method == "Exact FEM":
        return root / "exact_fem1x_10k/landweber_exact_multires_results.npz", "ExactFEM", "exact_fem1x_10k"
    model = next(row for row in config["headline_models"] if row["architecture"] == method)
    stem = f"{model['run_id']}_e{int(model['epoch'])}"
    return root / "surrogate" / stem / "result.npz", method, str(model["run_id"])


def scalar(archive: Any, key: str) -> float:
    return float(np.asarray(archive[key]).reshape(()))


def analyze_archive(
    path: Path,
    *,
    truth: str,
    noise_percent: int,
    method: str,
    architecture: str,
    run_id: str,
) -> list[dict[str, Any]]:
    required = {
        "history_iteration",
        "history_residual_norm",
        "history_relative_u_error",
        "minimum_u_error_iteration",
        "minimum_u_error",
        "discrepancy_crossing_noise_norm_threshold",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        iteration = np.asarray(archive["history_iteration"], dtype=np.int64)
        residual = np.asarray(archive["history_residual_norm"], dtype=np.float64)
        u_error = np.asarray(archive["history_relative_u_error"], dtype=np.float64)
        oracle_iteration = int(scalar(archive, "minimum_u_error_iteration"))
        oracle_error = scalar(archive, "minimum_u_error")
        delta = scalar(archive, "discrepancy_crossing_noise_norm_threshold")

    if not (
        iteration.ndim == residual.ndim == u_error.ndim == 1
        and iteration.size == residual.size == u_error.size
        and np.array_equal(iteration, np.arange(iteration.size, dtype=np.int64))
    ):
        raise ValueError(f"{path}: inconsistent or non-contiguous histories")
    if not np.all(np.isfinite(u_error)) or np.any(u_error < 0.0):
        raise ValueError(f"{path}: invalid coefficient-error history")
    minimum_index = int(np.argmin(u_error))
    if int(iteration[minimum_index]) != oracle_iteration or not math.isclose(
        float(u_error[minimum_index]), oracle_error, rel_tol=1.0e-10, abs_tol=1.0e-12
    ):
        raise ValueError(f"{path}: saved oracle does not match the history")

    rows: list[dict[str, Any]] = []
    for tau in TAUS:
        threshold = tau * delta
        index = first_crossing(residual, threshold)
        stopped = index is not None
        rows.append(
            {
                "truth": truth,
                "noise_percent": noise_percent,
                "method": method,
                "architecture": architecture,
                "run_id": run_id,
                "tau": tau,
                "delta": delta,
                "threshold": threshold,
                "stopped": int(stopped),
                "k_tau": "" if index is None else int(iteration[index]),
                "residual_at_k_tau": "" if index is None else float(residual[index]),
                "relative_u_error_at_k_tau": "" if index is None else float(u_error[index]),
                "oracle_iteration": oracle_iteration,
                "oracle_relative_u_error": oracle_error,
                "k_tau_minus_oracle": "" if index is None else int(iteration[index]) - oracle_iteration,
                "reconstruction_penalty_ratio": ""
                if index is None
                else float(u_error[index]) / max(oracle_error, 1.0e-300),
                "archive": path.relative_to(ROOT).as_posix(),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def latex_value(value: Any, *, digits: int = 3) -> str:
    if value == "":
        return "--"
    return f"{float(value):.{digits}f}"


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lookup = {
        (str(row["truth"]), int(row["noise_percent"]), str(row["method"]), float(row["tau"])): row
        for row in rows
    }
    lines = [
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Truth & Noise & Method & $\tau=1$ & $1.5$ & $2$ & $2.5$ & $3$ \\",
        r"\midrule",
    ]
    for truth in ("T1", "T2"):
        for noise in (1, 2, 5):
            for method in METHODS:
                cells = []
                for tau in TAUS:
                    row = lookup[(truth, noise, method, tau)]
                    if not int(row["stopped"]):
                        cells.append("--")
                    else:
                        cells.append(
                            f"{int(row['k_tau']):,} / {latex_value(row['reconstruction_penalty_ratio'], digits=2)}"
                        )
                safe_method = method.replace("DeepONet", r"DeepONet")
                lines.append(
                    f"{truth} & {noise}\\% & {safe_method} & " + " & ".join(cells) + r" \\"
                )
        if truth == "T1":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_definitions(path: Path) -> None:
    text = """# Definitions inherited from the earlier empirical stopping diagnostic

The earlier `empirical_stopping_diagnostic` is a retrospective diagnostic, not
an implemented stopping rule and not a checkpoint-selection device.

- Its **broader frozen T2 campaign** consists of 32 noiseless T2 surrogate
  Landweber trajectories: four derivative-informed headline models, four
  forward-only (`lambda_J=0`) headline controls, and 24 one-axis `lambda_J=10`
  ablations. Exact FEM and noisy trajectories are not in that cohort.
- A fixed forward threshold is `Q_q(e_F)` for `q` equal to median, p95, p99 or
  maximum over the independent test errors.
- The combined effective scale is
  `delta + 3 Q_q(e_F) + D Q_q(max_j e_J)/(1+eta)`. The recorded threshold is
  `tau` times this scale. The published compact diagnostic uses the observed
  trajectory distance bound, `eta=0`, and `tau=1`; this is not the theorem's
  admissible lower-bound choice of `tau`.
- The **reconstruction penalty** is
  `e_u(k_cross)/e_u(k_or)`. The percentage penalty is 100 times this ratio
  minus 100. A value of one means the crossing happens at an iterate with
  oracle-level reconstruction error.

The new E5 calculation is different: it uses the 30 frozen noisy headline
trajectories (T1/T2, 1/2/5 percent, exact FEM plus four surrogates) and applies
the raw-noise Morozov threshold `tau*delta` for five fixed multipliers.
"""
    path.write_text(text, encoding="utf-8")


def ensure_empty(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing revision output: {path}")


def output_records(paths: Iterable[Path]) -> list[dict[str, str]]:
    def display_path(path: Path) -> str:
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return path.as_posix()

    return [
        {"path": display_path(path), "sha256": sha256(path)}
        for path in paths
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--product-output", type=Path, default=DEFAULT_PRODUCTS)
    args = parser.parse_args(argv)
    raw_root = args.raw_output.resolve()
    product_root = args.product_output.resolve()
    ensure_empty(raw_root)
    ensure_empty(product_root)

    config = read_config()
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    for truth in ("T1", "T2"):
        for noise in (1, 2, 5):
            for method in METHODS:
                path, architecture, run_id = archive_path(config, truth, noise, method)
                if not path.is_file():
                    raise FileNotFoundError(path)
                sources.append(path)
                rows.extend(
                    analyze_archive(
                        path,
                        truth=truth,
                        noise_percent=noise,
                        method=method,
                        architecture=architecture,
                        run_id=run_id,
                    )
                )
    if len(rows) != 150:
        raise AssertionError(f"Expected 150 model/noise/tau rows, found {len(rows)}")

    raw_csv = raw_root / "noisy_tau_stopping.csv"
    product_csv = product_root / "noisy_tau_stopping.csv"
    table = product_root / "noisy_tau_stopping.tex"
    definitions = product_root / "DEFINITIONS.md"
    write_csv(raw_csv, rows)
    write_csv(product_csv, rows)
    write_table(table, rows)
    write_definitions(definitions)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    manifest = {
        "format": "reaction-revision-E5-v1",
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": commit,
        "tau_values": list(TAUS),
        "stopping_rule": "first k with ||R_k||_Wy <= tau*delta",
        "inputs": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for path in sources
        ],
        "outputs": output_records((raw_csv, product_csv, table, definitions)),
    }
    manifest_path = product_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"E5 complete: {len(rows)} rows from {len(sources)} frozen trajectories")
    print(f"Raw output: {raw_csv}")
    print(f"Products: {product_root}")


if __name__ == "__main__":
    main()
