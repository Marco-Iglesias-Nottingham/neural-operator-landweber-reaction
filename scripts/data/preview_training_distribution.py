#!/usr/bin/env python3
"""Quick audit of the broad-amplitude training distribution.

No FEM solves are performed.  This is intended to be run before committing to
KL/shard generation, chiefly to quantify how often the global bound-preserving
rescaling is active under the proposed [0,200] support.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reaction_core.config import DistributionConfig  # noqa: E402
from reaction_core.grid import RegularGrid2D  # noqa: E402
from reaction_core.sampling import MaternUSampler  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Audit the broad-amplitude direct-u Matérn distribution without FEM solves.",
    )
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--base-seed", type=int, default=987654321)
    p.add_argument("--nx", type=int, default=120)
    p.add_argument("--ny", type=int, default=120)
    p.add_argument("--nu-values", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5])
    p.add_argument("--length-scales", type=float, nargs="+", default=[0.075, 0.10, 0.15, 0.20, 0.25])
    p.add_argument("--mean-u-range", type=float, nargs=2, default=[30.0, 120.0])
    p.add_argument("--std-u-range", type=float, nargs=2, default=[10.0, 35.0])
    p.add_argument("--u-bounds", type=float, nargs=2, default=[0.0, 200.0])
    p.add_argument("--fft-workers", type=int, default=None)
    p.add_argument("--output", type=Path, default=Path("results/broad_amplitude_distribution_audit.json"))
    return p.parse_args()


def quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(np.asarray(values, dtype=np.float64), [0, .01, .05, .25, .5, .75, .95, .99, 1])
    names = ["min", "q01", "q05", "q25", "median", "q75", "q95", "q99", "max"]
    return {k: float(v) for k, v in zip(names, q)}


def main() -> None:
    args = parse_args()
    if args.n_samples < 1:
        raise ValueError("--n-samples must be positive")
    distribution = DistributionConfig(
        nu_values=tuple(args.nu_values),
        length_scales=tuple(args.length_scales),
        mean_u_range=(float(args.mean_u_range[0]), float(args.mean_u_range[1])),
        std_u_range=(float(args.std_u_range[0]), float(args.std_u_range[1])),
        u_min=float(args.u_bounds[0]),
        u_max=float(args.u_bounds[1]),
    )
    distribution.validate()
    grid = RegularGrid2D.from_domain(1.0, 1.0, args.nx, args.ny)
    sampler = MaternUSampler(
        grid,
        distribution,
        precompute_matern_bank=True,
        fft_workers=args.fft_workers,
    )

    fields_min = np.empty(args.n_samples)
    fields_max = np.empty(args.n_samples)
    target_mean = np.empty(args.n_samples)
    actual_mean = np.empty(args.n_samples)
    target_std = np.empty(args.n_samples)
    actual_std = np.empty(args.n_samples)
    scale = np.empty(args.n_samples)
    rescaled = np.empty(args.n_samples, dtype=bool)
    nus = np.empty(args.n_samples)
    ells = np.empty(args.n_samples)

    for i in range(args.n_samples):
        sample = sampler.sample(args.base_seed + i)
        m = sample.metadata
        fields_min[i] = m.minimum_u
        fields_max[i] = m.maximum_u
        target_mean[i] = m.target_mean_u
        actual_mean[i] = m.actual_mean_u
        target_std[i] = m.target_std_u
        actual_std[i] = m.actual_std_u
        scale[i] = m.fluctuation_scale
        rescaled[i] = m.rescaled
        nus[i] = m.nu
        ells[i] = m.length_scale
        if (i + 1) % 250 == 0 or i + 1 == args.n_samples:
            print(f"sample {i+1}/{args.n_samples}", flush=True)

    report = {
        "distribution": distribution.to_dict(),
        "n_samples": int(args.n_samples),
        "rescaled_fraction": float(np.mean(rescaled)),
        "strongly_rescaled_fraction_scale_lt_0p75": float(np.mean(scale < 0.75)),
        "strongly_rescaled_fraction_scale_lt_0p5": float(np.mean(scale < 0.5)),
        "fluctuation_scale": quantiles(scale),
        "target_mean_u": quantiles(target_mean),
        "actual_mean_u": quantiles(actual_mean),
        "target_std_u": quantiles(target_std),
        "actual_std_u": quantiles(actual_std),
        "minimum_u": quantiles(fields_min),
        "maximum_u": quantiles(fields_max),
        "actual_over_target_std": quantiles(actual_std / np.maximum(target_std, 1e-14)),
        "nu_counts": {str(v): int(np.sum(nus == v)) for v in sorted(set(nus.tolist()))},
        "length_scale_counts": {str(v): int(np.sum(ells == v)) for v in sorted(set(ells.tolist()))},
    }

    out = args.output.expanduser()
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nBroad-amplitude distribution audit")
    print("----------------------------------")
    print(f"samples:                     {args.n_samples}")
    print(f"rescaled fraction:           {report['rescaled_fraction']:.3%}")
    print(f"scale < 0.75:                {report['strongly_rescaled_fraction_scale_lt_0p75']:.3%}")
    print(f"scale < 0.50:                {report['strongly_rescaled_fraction_scale_lt_0p5']:.3%}")
    print(f"actual std median/q05/q95:   {report['actual_std_u']['median']:.3f} / {report['actual_std_u']['q05']:.3f} / {report['actual_std_u']['q95']:.3f}")
    print(f"field min q05/median/q95:    {report['minimum_u']['q05']:.3f} / {report['minimum_u']['median']:.3f} / {report['minimum_u']['q95']:.3f}")
    print(f"field max q05/median/q95:    {report['maximum_u']['q05']:.3f} / {report['maximum_u']['median']:.3f} / {report['maximum_u']['q95']:.3f}")
    print(f"saved:                       {out}")


if __name__ == "__main__":
    main()
