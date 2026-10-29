#!/usr/bin/env python3
"""Read-only validation of paired noisy truth archives."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_archive(value: str) -> tuple[float, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NOISE_PERCENT=PATH.")
    level, path = value.split("=", 1)
    try:
        percent = float(level)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid noise percent: {level!r}.") from exc
    if percent <= 0.0 or not path:
        raise argparse.ArgumentTypeError("Noise percent and path must be positive/nonempty.")
    return percent, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Validate that noisy archives are scaled copies of one truth/data archive.",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--archive",
        type=parse_archive,
        action="append",
        required=True,
        help="Repeat as NOISE_PERCENT=path/to/truth.npz.",
    )
    parser.add_argument("--noise-seed", type=int, default=20260821)
    return parser.parse_args()


def scalar(archive: np.lib.npyio.NpzFile, name: str):
    if name not in archive:
        raise KeyError(f"Archive is missing scalar {name!r}.")
    return np.asarray(archive[name]).reshape(()).item()


def weighted_norm(field: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights[mask] * field[mask] ** 2, dtype=np.float64)))


def main() -> None:
    args = parse_args()
    source_path = args.source.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    with np.load(source_path, allow_pickle=False) as source:
        required = {
            "format",
            "u_input_grid",
            "data_noiseless",
            "observation_mask",
            "observation_weights_trapezoid",
        }
        missing = sorted(required.difference(source.files))
        if missing:
            raise ValueError(f"{source_path}: missing {missing}.")
        source_format = str(scalar(source, "format"))
        source_u = np.asarray(source["u_input_grid"], dtype=np.float64)
        source_clean = np.asarray(source["data_noiseless"], dtype=np.float64)
        source_mask = np.asarray(source["observation_mask"], dtype=bool)
        source_weights = np.asarray(
            source["observation_weights_trapezoid"], dtype=np.float64
        )

    clean_norm = weighted_norm(source_clean, source_weights, source_mask)
    if clean_norm <= 0.0:
        raise ValueError("Source noiseless-data norm must be positive.")

    reference_direction: np.ndarray | None = None
    seen_levels: set[float] = set()
    for percent, supplied_path in args.archive:
        if percent in seen_levels:
            raise ValueError(f"Duplicate requested noise level: {percent:g}%.")
        seen_levels.add(percent)
        path = supplied_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "format",
                "u_input_grid",
                "data_noiseless",
                "data_noisy",
                "noise_fem_P1_vertices",
                "observation_mask",
                "observation_weights_trapezoid",
                "noise_seed",
                "noise_level_percent_requested",
                "noise_norm_realized",
                "relative_noise_percent_realized",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(f"{path}: missing {missing}.")
            if str(scalar(archive, "format")) != source_format:
                raise ValueError(f"{path}: format differs from the source archive.")
            if int(scalar(archive, "noise_seed")) != int(args.noise_seed):
                raise ValueError(f"{path}: unexpected noise seed.")
            if not np.isclose(
                float(scalar(archive, "noise_level_percent_requested")),
                percent,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise ValueError(f"{path}: requested noise level disagrees with its path.")

            u_field = np.asarray(archive["u_input_grid"], dtype=np.float64)
            clean = np.asarray(archive["data_noiseless"], dtype=np.float64)
            noisy = np.asarray(archive["data_noisy"], dtype=np.float64)
            noise = np.asarray(archive["noise_fem_P1_vertices"], dtype=np.float64)
            mask = np.asarray(archive["observation_mask"], dtype=bool)
            weights = np.asarray(
                archive["observation_weights_trapezoid"], dtype=np.float64
            )
            recorded_norm = float(scalar(archive, "noise_norm_realized"))
            recorded_percent = float(
                scalar(archive, "relative_noise_percent_realized")
            )

        for label, actual, expected in (
            ("truth field", u_field, source_u),
            ("noiseless data", clean, source_clean),
            ("observation mask", mask, source_mask),
            ("observation weights", weights, source_weights),
        ):
            if actual.shape != expected.shape or not np.array_equal(actual, expected):
                raise ValueError(f"{path}: {label} differs from the source archive.")
        if not np.allclose(noisy - clean, noise, rtol=0.0, atol=5.0e-12):
            raise ValueError(f"{path}: noisy-clean does not equal the saved noise.")

        realized = weighted_norm(noise, weights, mask)
        expected_norm = percent / 100.0 * clean_norm
        if not np.isclose(realized, expected_norm, rtol=5.0e-13, atol=5.0e-15):
            raise ValueError(f"{path}: realized noise norm is not exactly {percent:g}%.")
        if not np.isclose(
            recorded_norm,
            realized,
            rtol=5.0e-13,
            atol=5.0e-15,
        ):
            raise ValueError(f"{path}: recorded noise norm is inconsistent.")
        if not np.isclose(recorded_percent, percent, rtol=0.0, atol=1.0e-10):
            raise ValueError(f"{path}: recorded relative noise percent is inconsistent.")

        direction = noise / expected_norm
        if reference_direction is None:
            reference_direction = direction
        elif not np.allclose(direction, reference_direction, rtol=2.0e-13, atol=2.0e-13):
            raise ValueError(f"{path}: noise direction differs from the paired archives.")

        print(
            f"validated {percent:g}%: {path} "
            f"(noise norm {realized:.16e}, seed {args.noise_seed})"
        )
if __name__ == "__main__":
    main()
