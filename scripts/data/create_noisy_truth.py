#!/usr/bin/env python3
"""Create a reproducible noisy archive from one clean truth archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

EXPECTED_FORMAT = "reaction-fine-truth-data-v1"

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Copy an existing Reaction truth/data archive and add exactly scaled Gaussian observation noise without re-solving the PDE.",
    )
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--noise-percent", type=float, required=True)
    p.add_argument("--noise-seed", type=int, default=20260821)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def scalar_text(value):
    item = np.asarray(value).reshape(()).item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)

def weighted_norm(field, weights, mask):
    field = np.asarray(field, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sqrt(np.sum(weights[mask] * field[mask] ** 2, dtype=np.float64)))

def main():
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Source archive does not exist: {source}")
    if args.noise_percent < 0:
        raise ValueError("--noise-percent must be non-negative")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite")

    with np.load(source, allow_pickle=False) as z:
        a = {k: np.array(z[k], copy=True) for k in z.files}

    if scalar_text(a["format"]) != EXPECTED_FORMAT:
        raise ValueError(f"Expected format {EXPECTED_FORMAT!r}")

    clean = np.asarray(a["data_noiseless"], dtype=np.float64)
    mask = np.asarray(a["observation_mask"], dtype=bool)
    weights = np.asarray(a["observation_weights_trapezoid"], dtype=np.float64)

    if clean.shape != mask.shape or clean.shape != weights.shape:
        raise ValueError("clean data, mask and weights must have the same shape")
    if not np.any(mask):
        raise ValueError("observation mask is empty")

    clean_norm = weighted_norm(clean, weights, mask)
    fraction = float(args.noise_percent) / 100.0
    target_norm = fraction * clean_norm

    raw = np.zeros_like(clean)
    if target_norm > 0:
        rng = np.random.default_rng(int(args.noise_seed))
        raw[mask] = rng.standard_normal(int(np.count_nonzero(mask)))
        raw_norm = weighted_norm(raw, weights, mask)
        scale = target_norm / raw_norm
    else:
        raw_norm = 0.0
        scale = 0.0

    noise = scale * raw
    noisy = clean + noise
    realized_norm = weighted_norm(noise, weights, mask)
    realized_percent = 100.0 * realized_norm / clean_norm

    # Generic fields used by the LW readers.
    a["data_noisy"] = noisy
    a["noise_fem_P1_vertices"] = noise

    # Shape-specific aliases written by the truth generator.
    for key in list(a):
        if key == "p_noisy_fem_P1_vertices" or key.startswith("p_noisy_fem_P1_vertices_"):
            a[key] = noisy.copy()
        elif key.startswith("noise_fem_P1_vertices_"):
            a[key] = noise.copy()

    a["noise_seed"] = np.asarray(args.noise_seed, dtype=np.int64)
    a["noise_level_percent_requested"] = np.asarray(args.noise_percent, dtype=np.float64)
    a["noiseless_data_norm"] = np.asarray(clean_norm, dtype=np.float64)
    a["noise_norm_target"] = np.asarray(target_norm, dtype=np.float64)
    a["noise_norm_realized"] = np.asarray(realized_norm, dtype=np.float64)
    a["relative_noise_percent_realized"] = np.asarray(realized_percent, dtype=np.float64)

    noise_metadata = {
        "seed": int(args.noise_seed),
        "requested_noise_percent": float(args.noise_percent),
        "requested_relative_noise_fraction": fraction,
        "noiseless_data_norm": clean_norm,
        "noise_norm_target": target_norm,
        "noise_norm_realized": realized_norm,
        "relative_noise_fraction_realized": realized_norm / clean_norm,
        "relative_noise_percent_realized": realized_percent,
        "gaussian_direction_weighted_norm": raw_norm,
        "gaussian_scale": scale,
        "number_of_noisy_observations": int(np.count_nonzero(mask)),
    }
    a["noise_metadata_json"] = np.asarray(json.dumps(noise_metadata, sort_keys=True))

    if "report_json" in a:
        try:
            report = json.loads(scalar_text(a["report_json"]))
            if isinstance(report, dict):
                report["noise"] = noise_metadata
                ca = report.get("command_arguments")
                if isinstance(ca, dict):
                    ca["noise_level_percent"] = float(args.noise_percent)
                    ca["noise_seed"] = int(args.noise_seed)
                    ca["output_dir"] = str(output.parent)
                a["report_json"] = np.asarray(json.dumps(report, sort_keys=True))
        except Exception:
            pass

    if not np.allclose(
        a["data_noisy"] - a["data_noiseless"],
        a["noise_fem_P1_vertices"],
        rtol=0.0, atol=5.0e-12,
    ):
        raise RuntimeError("Consistency check failed: noisy-clean != noise")

    tol = 5.0e-13 * max(1.0, target_norm)
    if abs(realized_norm - target_norm) > tol:
        raise RuntimeError("Realised noise norm does not match requested norm")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **a)

    print("Reaction noisy archive created")
    print(f"  source                  : {source}")
    print(f"  output                  : {output}")
    print(f"  noise seed              : {args.noise_seed}")
    print(f"  clean weighted norm     : {clean_norm:.16e}")
    print(f"  target noise norm       : {target_norm:.16e}")
    print(f"  realised noise norm     : {realized_norm:.16e}")
    print(f"  realised relative noise : {realized_percent:.12g}%")
    print(f"  observed points         : {np.count_nonzero(mask)}")

if __name__ == "__main__":
    main()
