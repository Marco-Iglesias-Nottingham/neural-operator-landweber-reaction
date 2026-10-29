#!/usr/bin/env python3
"""Build the direct-``u`` KL/DINO data basis without scikit-learn.

This is the Reaction analogue of the active MRE ``*_hpc.py`` builder.  The
sample pool is written to a float32 memmap and the leading residual modes are
computed by a deterministic, blockwise NumPy randomized SVD.  Consequently a
production ``10000 x 120 x 120`` pool never has to reside in memory and no
``sklearn`` installation is needed.

The basis definition remains unchanged:

* mode 0 is an all-constant physical-``u`` direction whose pointwise
  amplitude is the empirical standard deviation of sample spatial means;
* modes 1..K are unit PCA modes of the per-sample mean-removed physical-``u``
  fields, scaled by their empirical one-standard-deviation coefficients.

The legacy Reaction HDF5 datasets and provenance contract are preserved for
the shard generator.  Additional datasets report variance capture relative
to the *full* empirical covariance trace, including discarded spatial modes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np


BUILDER_VERSION = "reaction-kl-numpy-randomized-svd-v1"


def find_project_root() -> Path:
    """Find the project directory containing ``reaction_core``."""

    environment_root = os.environ.get("REACTION_PROJECT_ROOT")
    if environment_root:
        candidate = Path(environment_root).expanduser().resolve()
        if (candidate / "reaction_core").is_dir():
            return candidate
        raise RuntimeError(
            f"REACTION_PROJECT_ROOT={candidate} does not contain reaction_core/."
        )

    script_path = Path(__file__).resolve()
    candidates = [
        script_path.parent,
        *script_path.parents,
        Path.cwd().resolve(),
        *Path.cwd().resolve().parents,
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "reaction_core").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate the Reaction project root containing reaction_core/. "
        "Run from the project or set REACTION_PROJECT_ROOT."
    )


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reaction_core.config import DistributionConfig  # noqa: E402
from reaction_core.grid import RegularGrid2D  # noqa: E402
from reaction_core.kl import (  # noqa: E402
    KLBasis2D,
    build_kl_basis_from_residual_matrix,
)
from reaction_core.sampling import MaternUSampler  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = DistributionConfig()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Build the shared direct-u Reaction KL basis with a float32 "
            "sample memmap and NumPy randomized SVD (no scikit-learn)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/reaction_matern_u_kl_modes16.h5",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=PROJECT_ROOT / "data/reaction_matern_u_kl_modes16.png",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not import matplotlib or create the optional mode plot.",
    )
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument(
        "--n-spatial-modes",
        type=int,
        default=15,
        help="Non-constant PCA modes; total DINO directions is this plus one.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help=(
            "Row-block size for memmap statistics/products. The option name is "
            "retained for compatibility with the former IncrementalPCA job."
        ),
    )
    parser.add_argument("--base-seed", type=int, default=314159)
    parser.add_argument("--nx", type=int, default=120)
    parser.add_argument("--ny", type=int, default=120)

    parser.add_argument(
        "--svd-oversampling",
        type=int,
        default=10,
        help="Randomized subspace vectors beyond n_spatial_modes.",
    )
    parser.add_argument(
        "--svd-power-iterations",
        type=int,
        default=2,
        help="Power iterations used to sharpen the leading singular subspace.",
    )
    parser.add_argument(
        "--svd-seed",
        type=int,
        default=20260810,
        help="Deterministic seed used only by the NumPy randomized SVD.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Temporary float32-memmap directory. If omitted, SLURM_TMPDIR or "
            "TMPDIR is preferred, then the output directory."
        ),
    )
    parser.add_argument(
        "--keep-work-file",
        action="store_true",
        help="Keep the temporary residual memmap for diagnostics.",
    )

    parser.add_argument(
        "--nu-values",
        type=float,
        nargs="+",
        default=list(defaults.nu_values),
    )
    parser.add_argument(
        "--length-scales",
        type=float,
        nargs="+",
        default=list(defaults.length_scales),
    )
    parser.add_argument(
        "--mean-u-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=list(defaults.mean_u_range),
    )
    parser.add_argument(
        "--std-u-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=list(defaults.std_u_range),
    )
    parser.add_argument(
        "--u-bounds",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[defaults.u_min, defaults.u_max],
    )
    parser.add_argument("--precompute-bank", action="store_true")
    parser.add_argument("--fft-workers", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output/plot files.",
    )
    return parser.parse_args(argv)


def resolve_project_path(path: Path) -> Path:
    expanded = path.expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (PROJECT_ROOT / expanded).resolve()
    )


def choose_work_dir(requested: Path | None, output: Path) -> Path:
    if requested is not None:
        result = resolve_project_path(requested)
    else:
        scratch = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR")
        result = (
            Path(scratch).expanduser().resolve()
            if scratch
            else output.parent.resolve()
        )
    result.mkdir(parents=True, exist_ok=True)
    if not result.is_dir():
        raise RuntimeError(f"KL work directory is not a directory: {result}.")
    return result


def sample_seed(base_seed: int, index: int) -> int:
    """Preserve the deterministic seed scheme of the original Reaction builder."""

    sequence = np.random.SeedSequence([int(base_seed), int(index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def centered_total_sample_variance(
    residual_memmap: np.memmap,
    residual_mean: np.ndarray,
    *,
    block_rows: int,
) -> float:
    """Return the full trace of the centered residual sample covariance."""

    n_samples, n_features = residual_memmap.shape
    mean = np.asarray(residual_mean, dtype=np.float64)
    if mean.shape != (n_features,):
        raise ValueError(
            f"residual_mean must have shape ({n_features},), got {mean.shape}."
        )
    if n_samples < 2:
        raise ValueError("At least two samples are required for sample variance.")
    sum_squares = 0.0
    for start in range(0, n_samples, block_rows):
        stop = min(start + block_rows, n_samples)
        block = np.asarray(residual_memmap[start:stop], dtype=np.float64)
        block -= mean[None, :]
        if not np.all(np.isfinite(block)):
            raise RuntimeError(
                f"Residual memmap is non-finite in rows {start}:{stop}."
            )
        sum_squares += float(np.sum(block * block, dtype=np.float64))
    return sum_squares / float(n_samples - 1)


def validate_args(args: argparse.Namespace) -> None:
    if args.n_samples < 2:
        raise ValueError("--n-samples must be at least 2.")
    if args.n_spatial_modes < 1:
        raise ValueError("--n-spatial-modes must be positive.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.nx < 2 or args.ny < 2:
        raise ValueError("--nx and --ny must be at least 2.")
    maximum_rank = min(args.n_samples - 1, args.nx * args.ny)
    if args.n_spatial_modes > maximum_rank:
        raise ValueError(
            f"--n-spatial-modes exceeds the maximum centered rank {maximum_rank}."
        )
    if args.svd_oversampling < 0:
        raise ValueError("--svd-oversampling cannot be negative.")
    if args.svd_power_iterations < 0:
        raise ValueError("--svd-power-iterations cannot be negative.")
    if args.fft_workers == 0:
        raise ValueError("--fft-workers cannot be zero.")


def build_basis(
    args: argparse.Namespace,
) -> tuple[KLBasis2D, dict[str, np.ndarray | float | str]]:
    validate_args(args)
    output = resolve_project_path(args.output)
    plot = resolve_project_path(args.plot)
    args.output = output
    args.plot = plot
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"KL output already exists: {output}. Use --overwrite explicitly."
        )
    if not args.no_plot and plot.exists() and not args.overwrite:
        raise FileExistsError(
            f"KL plot already exists: {plot}. Use --overwrite explicitly."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_plot:
        plot.parent.mkdir(parents=True, exist_ok=True)

    distribution = DistributionConfig(
        nu_values=tuple(float(value) for value in args.nu_values),
        length_scales=tuple(float(value) for value in args.length_scales),
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
        precompute_matern_bank=args.precompute_bank,
        max_cached_samplers=None if args.precompute_bank else 8,
        fft_workers=args.fft_workers,
    )

    work_dir = choose_work_dir(args.work_dir, output)
    work_file = work_dir / f"reaction_kl_residuals_{os.getpid()}.float32.dat"
    residual_memmap = np.memmap(
        work_file,
        mode="w+",
        dtype=np.float32,
        shape=(args.n_samples, grid.size),
        order="C",
    )

    print("Reaction direct-u Matérn KL distribution")
    print("----------------------------------------")
    print(f"project root:          {PROJECT_ROOT}")
    print(f"nu values:             {distribution.nu_values}")
    print(f"length scales:         {distribution.length_scales}")
    print(f"mean u range:          {distribution.mean_u_range}")
    print(f"requested std range:   {distribution.std_u_range}")
    print(f"physical safeguards:   [{distribution.u_min}, {distribution.u_max}]")
    print(f"samples:               {args.n_samples}")
    print(f"grid:                  {grid.nx} x {grid.ny}")
    print(f"spatial PCA modes:     {args.n_spatial_modes}")
    print(f"temporary memmap:      {work_file}")
    print(
        "temporary size [GiB]:  "
        f"{args.n_samples * grid.size * np.dtype(np.float32).itemsize / 1024**3:.3f}"
    )
    print("PCA backend:           NumPy blockwise randomized SVD")
    print(
        "SVD controls:          "
        f"oversampling={args.svd_oversampling}, "
        f"power_iterations={args.svd_power_iterations}, seed={args.svd_seed}"
    )
    print(flush=True)

    raw_sum = np.zeros(grid.size, dtype=np.float64)
    residual_sum = np.zeros(grid.size, dtype=np.float64)
    sample_means = np.empty(args.n_samples, dtype=np.float64)

    try:
        for index in range(args.n_samples):
            seed = sample_seed(args.base_seed, index)
            sample = sampler.sample(seed)
            flat = np.asarray(sample.u, dtype=np.float64).reshape(-1)
            if flat.shape != (grid.size,) or not np.all(np.isfinite(flat)):
                raise RuntimeError(f"Invalid sampled field at index {index}.")
            spatial_mean = float(np.mean(flat, dtype=np.float64))
            residual = flat - spatial_mean
            residual_memmap[index] = residual.astype(np.float32)
            raw_sum += flat
            residual_sum += residual
            sample_means[index] = spatial_mean

            if index == 0 or (index + 1) % 250 == 0 or index + 1 == args.n_samples:
                metadata = sample.metadata
                print(
                    f"sample {index + 1:6d}/{args.n_samples}: "
                    f"nu={metadata.nu:g}, ell={metadata.length_scale:g}, "
                    f"u=[{metadata.minimum_u:.2f},{metadata.maximum_u:.2f}], "
                    f"std={metadata.actual_std_u:.2f}",
                    flush=True,
                )

        residual_memmap.flush()
        parameter_mean_grid = (raw_sum / args.n_samples).reshape(grid.shape)
        residual_mean = residual_sum / args.n_samples
        constant_std = float(np.std(sample_means, ddof=1))
        spatial_total_variance = centered_total_sample_variance(
            residual_memmap,
            residual_mean,
            block_rows=args.batch_size,
        )
        constant_coordinate_variance = float(grid.size) * constant_std**2
        full_empirical_variance = (
            constant_coordinate_variance + spatial_total_variance
        )
        if not np.isfinite(full_empirical_variance) or full_empirical_variance <= 0.0:
            raise RuntimeError(
                f"Invalid full empirical variance: {full_empirical_variance}."
            )

        provenance: dict[str, Any] = {
            "parameter_variable": "physical_reaction_coefficient_u",
            "distribution": distribution.to_dict(),
            "domain": [1.0, 1.0],
            "grid_shape": [grid.nx, grid.ny],
            "n_source_samples": int(args.n_samples),
            "base_seed": int(args.base_seed),
            "sample_seed_scheme": (
                "SeedSequence([base_seed,sample_index]).generate_state(1,uint32)"
            ),
            "construction": (
                "constant spatial-mean mode plus centered NumPy blockwise "
                "randomized SVD"
            ),
            "builder_version": BUILDER_VERSION,
            "sample_matrix_storage": "float32_memmap",
            "constant_mode_scaling": (
                "direction=ones*std(spatial means); coordinate scale="
                "sqrt(n_grid)*std(spatial means)"
            ),
        }
        basis = build_kl_basis_from_residual_matrix(
            residual_memmap,
            grid,
            n_spatial_modes=args.n_spatial_modes,
            parameter_mean_grid=parameter_mean_grid,
            constant_mean_std=constant_std,
            residual_mean=residual_mean,
            block_size=args.batch_size,
            random_seed=args.svd_seed,
            oversampling=args.svd_oversampling,
            power_iterations=args.svd_power_iterations,
            provenance=provenance,
        )

        full_variance_contributions = np.asarray(
            basis.explained_variance,
            dtype=np.float64,
        )
        full_variance_ratio = (
            full_variance_contributions / full_empirical_variance
        )
        cumulative_full_variance_capture = np.cumsum(full_variance_ratio)
        retained_full_variance_capture = float(
            cumulative_full_variance_capture[-1]
        )
        if retained_full_variance_capture > 1.0 + 5.0e-10:
            raise RuntimeError(
                "Retained randomized-SVD variance exceeds the full empirical "
                f"trace: {retained_full_variance_capture:.16e}."
            )

        print()
        print(f"Number of DINO directions: {basis.n_modes}")
        print(f"Constant pointwise amplitude: {basis.constant_mean_std:.8e}")
        print("Coordinate-space direction scales:")
        print(basis.direction_scales)
        print()
        print("Full empirical variance diagnostics (coordinate-space trace)")
        print("----------------------------------------------------------")
        print(f"constant-mode variance:          {constant_coordinate_variance:.8e}")
        print(f"all spatial-residual variance:   {spatial_total_variance:.8e}")
        print(f"full empirical variance:         {full_empirical_variance:.8e}")
        print("Cumulative variance captured by stored DINO directions:")
        print(cumulative_full_variance_capture)
        print(
            f"TOTAL VARIANCE CAPTURED BY {basis.n_modes} DIRECTIONS: "
            f"{retained_full_variance_capture:.6f} "
            f"({100.0 * retained_full_variance_capture:.2f}%)"
        )

        diagnostics: dict[str, np.ndarray | float | str] = {
            "constant_coordinate_variance": constant_coordinate_variance,
            "spatial_total_variance": spatial_total_variance,
            "full_empirical_variance": full_empirical_variance,
            "full_variance_contributions": full_variance_contributions,
            "full_variance_ratio": full_variance_ratio,
            "cumulative_full_variance_capture": cumulative_full_variance_capture,
            "retained_full_variance_capture": retained_full_variance_capture,
            "work_file": str(work_file),
        }
        return basis, diagnostics
    finally:
        try:
            residual_memmap.flush()
        except Exception:
            pass
        mmap_object = getattr(residual_memmap, "_mmap", None)
        if mmap_object is not None:
            mmap_object.close()
        del residual_memmap
        if args.keep_work_file:
            print(f"Kept residual memmap: {work_file}", flush=True)
        else:
            try:
                work_file.unlink()
            except FileNotFoundError:
                pass


def save_basis_with_diagnostics(
    path: Path,
    basis: KLBasis2D,
    diagnostics: dict[str, np.ndarray | float | str],
) -> None:
    basis.save(path)
    with h5py.File(path, "a") as handle:
        for name in (
            "full_variance_contributions",
            "full_variance_ratio",
            "cumulative_full_variance_capture",
        ):
            handle.create_dataset(
                name,
                data=np.asarray(diagnostics[name], dtype=np.float64),
            )
        for name in (
            "constant_coordinate_variance",
            "spatial_total_variance",
            "full_empirical_variance",
            "retained_full_variance_capture",
        ):
            handle.attrs[name] = float(diagnostics[name])
        handle.attrs["builder_version"] = BUILDER_VERSION
        handle.attrs["pca_backend"] = "numpy_blockwise_randomized_svd"
        handle.attrs["variance_capture_definition"] = (
            "coordinate-space trace fraction: [n_grid*Var(sample spatial mean) "
            "+ retained spatial PCA eigenvalues] / [n_grid*Var(sample spatial "
            "mean) + total centered residual variance]"
        )


def save_mode_plot(path: Path, basis: KLBasis2D) -> None:
    """Import plotting only when explicitly requested/enabled."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_show = min(16, basis.n_modes)
    n_columns = 4
    n_rows = int(np.ceil(n_show / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(12, 3 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for mode in range(n_show):
        image = axes[mode].imshow(
            basis.directions[mode].T,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="coolwarm",
            aspect="equal",
        )
        axes[mode].set_title(f"physical-u direction {mode}")
        figure.colorbar(image, ax=axes[mode], shrink=0.8)
    for axis in axes[n_show:]:
        axis.axis("off")
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    basis, diagnostics = build_basis(args)
    save_basis_with_diagnostics(args.output, basis, diagnostics)
    print(f"Saved basis: {args.output}", flush=True)
    print(
        "Saved full-distribution variance-capture diagnostics in the HDF5 file.",
        flush=True,
    )
    if not args.no_plot:
        save_mode_plot(args.plot, basis)
        print(f"Saved plot: {args.plot}", flush=True)


if __name__ == "__main__":
    main()
