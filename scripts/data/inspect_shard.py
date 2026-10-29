#!/usr/bin/env python3
"""Validate and inspect one physical-u FNO/DINO HDF5 shard.

The script checks the schema, tensor shapes, finite values, positivity bounds,
seed bookkeeping, and homogeneous Dirichlet values of both p and its JVPs.
It can also save a compact plot for one sample without requiring FEniCSx.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


EXPECTED_FORMAT = "reaction_fno_dino_shard_v1_physical_u"
METADATA_NAMES = (
    "seed",
    "field_seed",
    "family_id",
    "nu",
    "length_scale",
    "target_mean_u",
    "actual_mean_u",
    "minimum_u",
    "maximum_u",
    "target_std_u",
    "actual_std_u",
    "fluctuation_scale",
    "rescaled",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a reaction shard containing physical u, pressure p, "
            "and KL-direction JVPs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("shard", type=Path, help="HDF5 shard to inspect.")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Completed sample index to describe and optionally plot.",
    )
    parser.add_argument(
        "--modes",
        default="0,1,2,3",
        help="Comma-separated mode indices, or 'all'.",
    )
    parser.add_argument(
        "--max-stat-samples",
        type=int,
        default=64,
        help="Samples used for detailed checks; use 0 for every completed sample.",
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=float,
        default=1.0e-7,
        help="Maximum permitted absolute p/JVP value on the boundary.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional PNG/PDF path for selected u, p, directions, and JVPs.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the stored project_config_json in full.",
    )
    return parser.parse_args()


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _project_config(handle: h5py.File) -> dict[str, Any]:
    raw = _decode(handle.attrs.get("project_config_json", "{}"))
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("project_config_json is not valid JSON.") from error
    if not isinstance(parsed, dict):
        raise ValueError("project_config_json must encode an object.")
    return parsed


def _parse_modes(specification: str, n_modes: int) -> list[int]:
    if specification.strip().lower() == "all":
        return list(range(n_modes))
    modes: list[int] = []
    for token in specification.split(","):
        token = token.strip()
        if not token:
            continue
        mode = int(token)
        if not 0 <= mode < n_modes:
            raise ValueError(
                f"Mode {mode} is outside the available range [0,{n_modes - 1}]."
            )
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("No modes selected.")
    return modes


def _stats(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float64)
    return (
        f"min={np.min(array):.6e}, max={np.max(array):.6e}, "
        f"mean={np.mean(array):.6e}, std={np.std(array):.6e}, "
        f"rms={np.sqrt(np.mean(array * array)):.6e}"
    )


def _boundary_max(values: np.ndarray) -> float:
    array = np.asarray(values)
    if array.shape[-2] < 2 or array.shape[-1] < 2:
        raise ValueError("A spatial field must have at least two points per axis.")
    return float(
        max(
            np.max(np.abs(array[..., 0, :])),
            np.max(np.abs(array[..., -1, :])),
            np.max(np.abs(array[..., :, 0])),
            np.max(np.abs(array[..., :, -1])),
        )
    )


def _validate_schema(
    handle: h5py.File,
) -> tuple[int, int, int, int, dict[str, Any]]:
    required = ("x_grid", "y_grid", "directions_u", "u", "p", "jvp", "metadata")
    missing = [name for name in required if name not in handle]
    if missing:
        raise KeyError(f"Shard is missing datasets/groups: {missing}")

    file_format = str(_decode(handle.attrs.get("format", "")))
    if file_format != EXPECTED_FORMAT:
        raise ValueError(
            f"Unexpected shard format {file_format!r}; expected {EXPECTED_FORMAT!r}."
        )

    x_grid = handle["x_grid"]
    y_grid = handle["y_grid"]
    if x_grid.ndim != 1 or y_grid.ndim != 1:
        raise ValueError("x_grid and y_grid must be one-dimensional.")
    nx, ny = int(x_grid.shape[0]), int(y_grid.shape[0])
    if nx < 2 or ny < 2:
        raise ValueError("The spatial grid must have at least two points per axis.")
    if np.any(np.diff(x_grid[:]) <= 0.0) or np.any(np.diff(y_grid[:]) <= 0.0):
        raise ValueError("x_grid and y_grid must be strictly increasing.")

    directions_shape = tuple(handle["directions_u"].shape)
    if len(directions_shape) != 3 or directions_shape[1:] != (nx, ny):
        raise ValueError(
            "directions_u must have shape (n_modes,nx,ny); received "
            f"{directions_shape}."
        )
    n_modes = int(directions_shape[0])
    if n_modes < 1:
        raise ValueError("The shard contains no DINO directions.")

    u_shape = tuple(handle["u"].shape)
    if len(u_shape) != 4 or u_shape[1:] != (1, nx, ny):
        raise ValueError(
            f"u must have shape (n_samples,1,{nx},{ny}); received {u_shape}."
        )
    capacity = int(u_shape[0])
    if tuple(handle["p"].shape) != u_shape:
        raise ValueError("p must have the same shape as u.")
    expected_jvp = (capacity, n_modes, 1, nx, ny)
    if tuple(handle["jvp"].shape) != expected_jvp:
        raise ValueError(
            f"jvp must have shape {expected_jvp}; received {handle['jvp'].shape}."
        )

    metadata = handle["metadata"]
    missing_metadata = [name for name in METADATA_NAMES if name not in metadata]
    if missing_metadata:
        raise KeyError(f"metadata is missing datasets: {missing_metadata}")
    for name in METADATA_NAMES:
        if tuple(metadata[name].shape) != (capacity,):
            raise ValueError(
                f"metadata/{name} must have shape {(capacity,)}, "
                f"received {metadata[name].shape}."
            )

    completed = int(handle.attrs.get("completed_samples", 0))
    if not 0 <= completed <= capacity:
        raise ValueError(
            f"completed_samples={completed} is incompatible with capacity={capacity}."
        )
    if int(handle.attrs.get("n_samples", capacity)) != capacity:
        raise ValueError("n_samples attribute does not match the u dataset.")
    if int(handle.attrs.get("n_modes", n_modes)) != n_modes:
        raise ValueError("n_modes attribute does not match directions_u.")

    config = _project_config(handle)
    return capacity, completed, n_modes, nx * ny, config


def _sample_ids(completed: int, maximum: int) -> np.ndarray:
    if completed < 1:
        return np.empty(0, dtype=np.int64)
    count = completed if maximum == 0 else min(completed, maximum)
    return np.unique(
        np.rint(np.linspace(0, completed - 1, count)).astype(np.int64)
    )


def _plot_sample(
    handle: h5py.File,
    sample: int,
    modes: list[int],
    path: Path,
    dpi: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Matplotlib is required only when --plot is used.") from error

    x = np.asarray(handle["x_grid"][:], dtype=np.float64)
    y = np.asarray(handle["y_grid"][:], dtype=np.float64)
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    u = np.asarray(handle["u"][sample, 0], dtype=np.float64)
    p = np.asarray(handle["p"][sample, 0], dtype=np.float64)

    rows = 1 + len(modes)
    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(9.0, 3.7 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    def draw(axis, values: np.ndarray, title: str, cmap: str) -> None:
        image = axis.imshow(
            values.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis, shrink=0.82)

    draw(axes[0, 0], u, f"sample {sample}: physical u", "viridis")
    draw(axes[0, 1], p, f"sample {sample}: pressure p", "viridis")
    for row, mode in enumerate(modes, start=1):
        direction = np.asarray(handle["directions_u"][mode], dtype=np.float64)
        jvp = np.asarray(handle["jvp"][sample, mode, 0], dtype=np.float64)
        draw(axes[row, 0], direction, f"physical-u direction {mode}", "coolwarm")
        draw(axes[row, 1], jvp, f"Dp(u)h, mode {mode}", "coolwarm")

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot: {path}")


def main() -> None:
    args = parse_args()
    shard = args.shard.expanduser().resolve()
    if not shard.is_file():
        raise FileNotFoundError(shard)
    if args.max_stat_samples < 0:
        raise ValueError("--max-stat-samples cannot be negative.")
    if args.boundary_tolerance < 0.0:
        raise ValueError("--boundary-tolerance cannot be negative.")
    if args.dpi < 1:
        raise ValueError("--dpi must be positive.")

    with h5py.File(shard, "r") as handle:
        capacity, completed, n_modes, grid_size, config = _validate_schema(handle)
        nx = int(handle["x_grid"].shape[0])
        ny = int(handle["y_grid"].shape[0])
        modes = _parse_modes(args.modes, n_modes)

        print("Reaction shard inspection")
        print(f"  file:               {shard}")
        print(f"  format:             {_decode(handle.attrs['format'])}")
        print(f"  completed/capacity: {completed}/{capacity}")
        print(f"  grid:               {nx} x {ny} ({grid_size} points)")
        print(f"  modes:              {n_modes}")
        print(f"  u shape/dtype:      {handle['u'].shape} / {handle['u'].dtype}")
        print(f"  p shape/dtype:      {handle['p'].shape} / {handle['p'].dtype}")
        print(f"  jvp shape/dtype:    {handle['jvp'].shape} / {handle['jvp'].dtype}")
        if completed < capacity:
            print("  status:             INCOMPLETE SHARD")
        else:
            print("  status:             complete")

        if args.print_config:
            print("Stored project configuration:")
            print(json.dumps(config, indent=2, sort_keys=True))
        if completed == 0:
            raise RuntimeError("The shard has no completed samples to inspect.")
        if not 0 <= args.sample < completed:
            raise ValueError(
                f"--sample must lie in [0,{completed - 1}], got {args.sample}."
            )

        directions = np.asarray(handle["directions_u"][:], dtype=np.float64)
        if not np.all(np.isfinite(directions)):
            raise ValueError("directions_u contains non-finite values.")
        direction_rms = np.sqrt(np.mean(directions * directions, axis=(1, 2)))
        print(
            "Direction RMS:        "
            + ", ".join(f"{value:.5g}" for value in direction_rms)
        )

        ids = _sample_ids(completed, args.max_stat_samples)
        u = np.asarray(handle["u"][ids], dtype=np.float64)
        p = np.asarray(handle["p"][ids], dtype=np.float64)
        jvp = np.asarray(handle["jvp"][ids], dtype=np.float64)
        if not all(np.all(np.isfinite(values)) for values in (u, p, jvp)):
            raise ValueError("A checked u, p, or jvp value is non-finite.")

        distribution = config.get("distribution", {})
        configured_minimum = float(distribution.get("u_min", 0.0))
        configured_maximum = float(distribution.get("u_max", np.inf))
        u_minimum = float(np.min(u))
        u_maximum = float(np.max(u))
        tolerance = 1.0e-5 * max(1.0, abs(configured_minimum), abs(configured_maximum))
        if u_minimum < configured_minimum - tolerance:
            raise ValueError(
                f"Physical u reaches {u_minimum:g}, below stored u_min={configured_minimum:g}."
            )
        if u_maximum > configured_maximum + tolerance:
            raise ValueError(
                f"Physical u reaches {u_maximum:g}, above stored u_max={configured_maximum:g}."
            )
        if u_minimum <= 0.0:
            raise ValueError(f"Physical u is not strictly positive: min={u_minimum:g}.")

        p_boundary = _boundary_max(p)
        jvp_boundary = _boundary_max(jvp)
        if p_boundary > args.boundary_tolerance:
            raise ValueError(
                f"p boundary maximum {p_boundary:.3e} exceeds tolerance "
                f"{args.boundary_tolerance:.3e}."
            )
        if jvp_boundary > args.boundary_tolerance:
            raise ValueError(
                f"JVP boundary maximum {jvp_boundary:.3e} exceeds tolerance "
                f"{args.boundary_tolerance:.3e}."
            )

        metadata = handle["metadata"]
        seeds = np.asarray(metadata["seed"][:completed], dtype=np.uint64)
        if np.unique(seeds).size != completed:
            raise ValueError("Completed samples contain duplicate deterministic seeds.")
        consecutive = bool(completed == 1 or np.all(np.diff(seeds) == 1))

        print(f"Checked samples:      {ids.size} (indices {ids[0]}..{ids[-1]})")
        print(f"u:                    {_stats(u)}")
        print(f"p:                    {_stats(p)}")
        print(f"jvp:                  {_stats(jvp)}")
        print(f"max |p| boundary:     {p_boundary:.6e}")
        print(f"max |JVP| boundary:   {jvp_boundary:.6e}")
        print(
            f"seed range:            {int(seeds[0])}..{int(seeds[-1])} "
            f"(unique=yes, consecutive={'yes' if consecutive else 'no'})"
        )

        nu, nu_counts = np.unique(metadata["nu"][:completed], return_counts=True)
        ell, ell_counts = np.unique(
            metadata["length_scale"][:completed], return_counts=True
        )
        print(
            "nu counts:            "
            + ", ".join(f"{value:g}:{count}" for value, count in zip(nu, nu_counts))
        )
        print(
            "length-scale counts:  "
            + ", ".join(
                f"{value:g}:{count}" for value, count in zip(ell, ell_counts)
            )
        )
        print(
            "actual sample means:  "
            f"[{np.min(metadata['actual_mean_u'][:completed]):.5g}, "
            f"{np.max(metadata['actual_mean_u'][:completed]):.5g}]"
        )
        print(
            "actual sample stds:   "
            f"[{np.min(metadata['actual_std_u'][:completed]):.5g}, "
            f"{np.max(metadata['actual_std_u'][:completed]):.5g}]"
        )
        rescaled = int(np.count_nonzero(metadata["rescaled"][:completed]))
        print(f"uniformly rescaled:  {rescaled}/{completed}")

        sample_u = np.asarray(handle["u"][args.sample, 0], dtype=np.float64)
        sample_p = np.asarray(handle["p"][args.sample, 0], dtype=np.float64)
        print(f"Selected sample {args.sample}")
        print(f"  u: {_stats(sample_u)}")
        print(f"  p: {_stats(sample_p)}")
        for mode in modes:
            values = np.asarray(handle["jvp"][args.sample, mode, 0], dtype=np.float64)
            print(f"  JVP mode {mode}: {_stats(values)}")

        if args.plot is not None:
            _plot_sample(handle, args.sample, modes, args.plot, args.dpi)

    print("All requested shard checks passed.")


if __name__ == "__main__":
    main()
