#!/usr/bin/env python3
"""Generate one FNO/DeepONet + DINO HDF5 shard.

The coefficient stored in the shard is the physical reaction coefficient
``u`` itself (never ``log(u)``).  For every sample this script solves

    -Delta p + u p = 1,  p|boundary = 0,

and the exact tangent equation for every retained KL direction ``h``:

    (-Delta + u) Dp(u)h = -h p.

The state matrix is assembled and factorised once per sampled coefficient and
then reused for the forward solve and all tangent solves.  Parallel data
generation is intentionally performed with independent one-rank shard jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from mpi4py import MPI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reaction_core.config import (  # noqa: E402
    DatasetConfig,
    DistributionConfig,
    GridConfig,
    ProjectConfig,
    ReactionConfig,
)
from reaction_core.datasets import ReactionShardWriter  # noqa: E402
from reaction_core.grid import (  # noqa: E402
    RegularGrid2D,
    StructuredCoefficientGridMap,
    StructuredLagrangeStateGridMap,
)
from reaction_core.kl import KLBasis2D  # noqa: E402
from reaction_core.observations import FullFieldObservationOperator  # noqa: E402
from reaction_core.sampling import MaternUSampler  # noqa: E402
from reaction_core.solver import ReactionSolver2D  # noqa: E402


UINT32_MAX = int(np.iinfo(np.uint32).max)


def parse_args() -> argparse.Namespace:
    defaults = DistributionConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Generate one HDF5 shard containing bounded physical u, the "
            "exact reaction solution p, and exact KL-direction JVPs. Run "
            "one MPI rank per shard."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination HDF5 shard.",
    )
    parser.add_argument(
        "--kl-basis",
        type=Path,
        required=True,
        help="KL-basis HDF5 file containing directions_u.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="Number of coefficient samples in this shard.",
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help=(
            "Global zero-based sample offset. Use non-overlapping offset "
            "ranges for independent shard jobs."
        ),
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=271828,
        help=(
            "First seed before applying sample-offset. Sample i uses "
            "base_seed + sample_offset + i."
        ),
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=16,
        help="Number of leading directions_u to retain and differentiate.",
    )

    parser.add_argument("--fem-nx", type=int, default=119)
    parser.add_argument("--fem-ny", type=int, default=119)
    parser.add_argument(
        "--fem-degree",
        type=int,
        choices=(1, 2),
        default=1,
        help="Lagrange degree for p; u always uses nodal P1.",
    )
    parser.add_argument(
        "--rhs",
        type=float,
        default=1.0,
        help="Constant right-hand side in -Delta p + u p = rhs.",
    )

    parser.add_argument(
        "--nu-values",
        type=float,
        nargs="+",
        default=list(defaults.nu_values),
        help="Finite bank of Matérn smoothness values.",
    )
    parser.add_argument(
        "--length-scales",
        type=float,
        nargs="+",
        default=list(defaults.length_scales),
        help="Finite bank of Matérn correlation lengths on the unit square.",
    )
    parser.add_argument(
        "--mean-u-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=list(defaults.mean_u_range),
        help="Uniform range for the physical-u spatial mean.",
    )
    parser.add_argument(
        "--std-u-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=list(defaults.std_u_range),
        help="Uniform range for the requested physical-u standard deviation.",
    )
    parser.add_argument(
        "--u-bounds",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[defaults.u_min, defaults.u_max],
        help=(
            "Hard physical-u safeguards. The lower bound may be zero. "
            "Fluctuations are uniformly reduced about their sampled mean "
            "when necessary."
        ),
    )
    parser.add_argument(
        "--minimum-length-scale-grid-cells",
        type=float,
        default=8.0,
        help=(
            "Reject correlation lengths shorter than this many intervals "
            "on the coefficient grid."
        ),
    )

    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Storage dtype for u, p, and JVP tensors.",
    )
    parser.add_argument(
        "--compression",
        choices=("lzf", "gzip", "none"),
        default="lzf",
        help="HDF5 compression for sample tensors.",
    )
    parser.add_argument(
        "--precompute-bank",
        action="store_true",
        help="Precompute every Matérn circulant-embedding sampler.",
    )
    parser.add_argument(
        "--max-cached-samplers",
        type=int,
        default=8,
        help=(
            "Maximum lazily cached Matérn samplers; use 0 for no limit. "
            "Ignored with --precompute-bank."
        ),
    )
    parser.add_argument(
        "--fft-workers",
        type=int,
        default=None,
        help="Optional worker count passed to FFT-based Matérn sampling.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=8,
        help="Flush HDF5 buffers and update completed_samples every N samples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output shard.",
    )
    return parser.parse_args()


def sample_seed(base_seed: int, global_index: int) -> int:
    """Return a collision-free uint32 seed for a disjoint global index range."""
    seed = int(base_seed) + int(global_index)
    if not 0 <= seed <= UINT32_MAX:
        raise ValueError(
            f"Derived seed {seed} is outside the uint32 range [0,{UINT32_MAX}]."
        )
    return seed


def _validate_args(args: argparse.Namespace) -> tuple[int, int]:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError(
            "Run exactly one MPI rank per shard. Parallelise over independent "
            "Slurm array tasks, each with a disjoint --sample-offset."
        )
    if args.n_samples < 1:
        raise ValueError("--n-samples must be positive.")
    if args.n_modes < 1:
        raise ValueError("--n-modes must be positive.")
    if args.sample_offset < 0:
        raise ValueError("--sample-offset must be non-negative.")
    if args.base_seed < 0:
        raise ValueError("--base-seed must be non-negative.")
    if args.flush_every < 1:
        raise ValueError("--flush-every must be positive.")
    if args.minimum_length_scale_grid_cells <= 0.0:
        raise ValueError("--minimum-length-scale-grid-cells must be positive.")
    if args.max_cached_samplers < 0:
        raise ValueError("--max-cached-samplers cannot be negative.")
    if float(args.u_bounds[0]) < ReactionSolver2D.HARD_MINIMUM_U:
        raise ValueError(
            "The lower --u-bounds value must be at least the solver hard "
            f"minimum {ReactionSolver2D.HARD_MINIMUM_U:g}."
        )

    first_seed = sample_seed(args.base_seed, args.sample_offset)
    last_global_index = args.sample_offset + args.n_samples - 1
    last_seed = sample_seed(args.base_seed, last_global_index)
    return first_seed, last_seed


def _build_config(args: argparse.Namespace) -> ProjectConfig:
    reaction = ReactionConfig(
        fem_nx=int(args.fem_nx),
        fem_ny=int(args.fem_ny),
        state_degree=int(args.fem_degree),
        rhs=float(args.rhs),
        u_min=float(args.u_bounds[0]),
    )
    grid = GridConfig(nx=reaction.fem_nx + 1, ny=reaction.fem_ny + 1)
    distribution = DistributionConfig(
        nu_values=tuple(float(value) for value in args.nu_values),
        length_scales=tuple(float(value) for value in args.length_scales),
        mean_u_range=(float(args.mean_u_range[0]), float(args.mean_u_range[1])),
        std_u_range=(float(args.std_u_range[0]), float(args.std_u_range[1])),
        u_min=float(args.u_bounds[0]),
        u_max=float(args.u_bounds[1]),
    )
    dataset = DatasetConfig(
        n_kl_modes=int(args.n_modes),
        compression=None if args.compression == "none" else args.compression,
        output_dtype=str(args.dtype),
    )
    config = ProjectConfig(
        reaction=reaction,
        grid=grid,
        distribution=distribution,
        dataset=dataset,
    )
    config.validate()

    largest_spacing = max(
        reaction.Lx / reaction.fem_nx,
        reaction.Ly / reaction.fem_ny,
    )
    shortest_length_scale = min(distribution.length_scales)
    required = args.minimum_length_scale_grid_cells * largest_spacing
    if shortest_length_scale < required:
        cells = shortest_length_scale / largest_spacing
        raise ValueError(
            "The shortest Matérn length scale is under-resolved: "
            f"{shortest_length_scale:g} spans {cells:.2f} grid intervals, "
            f"below the requested minimum of "
            f"{args.minimum_length_scale_grid_cells:g}."
        )
    return config


def _validate_paths(args: argparse.Namespace) -> None:
    args.output = args.output.expanduser().resolve()
    args.kl_basis = args.kl_basis.expanduser().resolve()
    if not args.kl_basis.is_file():
        raise FileNotFoundError(f"KL basis not found: {args.kl_basis}")
    if args.output == args.kl_basis:
        raise ValueError("--output must not be the KL-basis file.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Use --overwrite to replace it."
        )


def main() -> None:
    args = parse_args()
    first_seed, last_seed = _validate_args(args)
    _validate_paths(args)
    config = _build_config(args)

    reaction = config.reaction
    grid = RegularGrid2D.from_domain(
        reaction.Lx,
        reaction.Ly,
        config.grid.nx,
        config.grid.ny,
    )
    basis = KLBasis2D.load(args.kl_basis)
    if basis.grid.shape != grid.shape or not (
        np.allclose(basis.grid.x, grid.x, rtol=1.0e-12, atol=1.0e-12)
        and np.allclose(basis.grid.y, grid.y, rtol=1.0e-12, atol=1.0e-12)
    ):
        raise ValueError(
            "KL-basis grid does not match the requested FEM/neural grid: "
            f"basis={basis.grid.shape}, requested={grid.shape}."
        )
    basis_distribution = basis.provenance.get("distribution")
    requested_distribution = config.distribution.to_dict()
    if basis_distribution is None:
        raise ValueError(
            "KL basis has no stored distribution provenance. Rebuild it with "
            "scripts/data/build_kl_basis.py before generating shards."
        )
    if json.dumps(basis_distribution, sort_keys=True) != json.dumps(
        requested_distribution,
        sort_keys=True,
    ):
        raise ValueError(
            "KL-basis distribution does not match the shard distribution. "
            f"basis={basis_distribution}, requested={requested_distribution}."
        )
    # Compute tangents along exactly the float32 directions that are stored in
    # every shard and later supplied to the neural-network JVP calculation.
    directions = np.asarray(
        basis.select_directions(args.n_modes),
        dtype=np.float32,
    )
    if not np.all(np.isfinite(directions)):
        raise ValueError("KL directions_u contains non-finite values.")

    distribution = config.distribution
    max_cached_samplers = (
        None
        if args.precompute_bank or args.max_cached_samplers == 0
        else int(args.max_cached_samplers)
    )
    sampler = MaternUSampler(
        grid,
        distribution,
        precompute_matern_bank=args.precompute_bank,
        max_cached_samplers=max_cached_samplers,
        fft_workers=args.fft_workers,
    )

    spacing = max(grid.dx, grid.dy)
    print("Reaction physical-u shard generation")
    print(f"  output:            {args.output}")
    print(f"  KL basis:          {args.kl_basis}")
    print(f"  samples/modes:     {args.n_samples}/{args.n_modes}")
    print(
        f"  global indices:    {args.sample_offset}.."
        f"{args.sample_offset + args.n_samples - 1}"
    )
    print(f"  deterministic seeds: {first_seed}..{last_seed}")
    print(
        f"  FEM/grid:          P{reaction.state_degree} on "
        f"{reaction.fem_nx}x{reaction.fem_ny} cells / {grid.nx}x{grid.ny}"
    )
    print(f"  Matérn nu:          {distribution.nu_values}")
    print(f"  length scales:     {distribution.length_scales}")
    print(
        "  length scales/cells: "
        + ", ".join(f"{ell / spacing:.1f}" for ell in distribution.length_scales)
    )
    print(f"  mean/std ranges:   {distribution.mean_u_range} / {distribution.std_u_range}")
    print(f"  physical bounds:   [{distribution.u_min}, {distribution.u_max}]")
    print(flush=True)

    output_dtype = np.dtype(config.dataset.output_dtype)
    compression = config.dataset.compression
    solver: ReactionSolver2D | None = None
    writer: ReactionShardWriter | None = None
    final_diagnostics: dict[str, int | bool] | None = None
    total_start = perf_counter()

    try:
        initial_u = 0.5 * sum(distribution.mean_u_range)
        solver = ReactionSolver2D(
            reaction,
            minimum_u=distribution.u_min,
            initial_u=initial_u,
        )
        coefficient_map = StructuredCoefficientGridMap(solver, grid)
        state_map = StructuredLagrangeStateGridMap(solver, grid)
        observations = FullFieldObservationOperator(state_map)

        # Directions are physical increments h, not logarithmic increments.
        direction_functions = [
            coefficient_map.grid_to_function(
                direction,
                name=f"physical_u_direction_{mode_index}",
            )
            for mode_index, direction in enumerate(directions)
        ]

        writer = ReactionShardWriter(
            args.output,
            n_samples=args.n_samples,
            nx=grid.nx,
            ny=grid.ny,
            directions=directions,
            x_grid=grid.x,
            y_grid=grid.y,
            dtype=output_dtype,
            compression=compression,
            config=config.to_dict(),
        )
        writer.handle.attrs["base_seed"] = int(args.base_seed)
        writer.handle.attrs["sample_offset"] = int(args.sample_offset)
        writer.handle.attrs["first_seed"] = int(first_seed)
        writer.handle.attrs["last_seed"] = int(last_seed)
        writer.handle.attrs["seed_formula"] = (
            "seed = base_seed + sample_offset + local_index"
        )

        expected_output_shape = (1, grid.nx, grid.ny)
        for local_index in range(args.n_samples):
            global_index = args.sample_offset + local_index
            seed = sample_seed(args.base_seed, global_index)
            sample_start = perf_counter()

            before = solver.cache_diagnostics()
            factorisations_before = int(
                before["operator_assemblies_factorizations"]
            )

            sampled = sampler.sample(seed)
            coefficient_values = coefficient_map.grid_to_array(sampled.u)
            solver.set_coefficient(coefficient_values)
            state = solver.forward()
            pressure = observations.apply(state, dtype=output_dtype)
            if pressure.shape != expected_output_shape:
                raise RuntimeError(
                    f"Expected p shape {expected_output_shape}, got {pressure.shape}."
                )

            jvp = np.empty(
                (args.n_modes, 1, grid.nx, grid.ny),
                dtype=output_dtype,
            )
            for mode_index, direction_function in enumerate(direction_functions):
                tangent = solver.tangent(state, direction_function)
                tangent_grid = observations.apply(tangent, dtype=output_dtype)
                if tangent_grid.shape != expected_output_shape:
                    raise RuntimeError(
                        "Expected tangent shape "
                        f"{expected_output_shape}, got {tangent_grid.shape}."
                    )
                jvp[mode_index] = tangent_grid

            after = solver.cache_diagnostics()
            factorisations_this_sample = int(
                after["operator_assemblies_factorizations"]
            ) - factorisations_before
            if factorisations_this_sample != 1:
                raise RuntimeError(
                    "Expected one matrix assembly/factorisation per u sample, "
                    f"reused for one forward and {args.n_modes} tangent solves; "
                    f"observed {factorisations_this_sample}."
                )

            writer.write_sample(
                local_index,
                u=sampled.u,
                p=pressure,
                jvp=jvp,
                metadata=sampled.metadata,
            )

            completed = local_index + 1
            if completed % args.flush_every == 0 or completed == args.n_samples:
                writer.handle.attrs["completed_samples"] = completed
                writer.handle.flush()

            elapsed = perf_counter() - sample_start
            metadata = sampled.metadata
            print(
                f"sample {completed:5d}/{args.n_samples} "
                f"global={global_index:8d} seed={seed:10d} "
                f"nu={metadata.nu:3.1f} ell={metadata.length_scale:6.3f} "
                f"mean={metadata.actual_mean_u:7.3f} "
                f"std={metadata.actual_std_u:6.3f} "
                f"u=[{metadata.minimum_u:7.3f},{metadata.maximum_u:7.3f}] "
                f"rescaled={int(metadata.rescaled)} LU={factorisations_this_sample} "
                f"time={elapsed:7.2f}s",
                flush=True,
            )

        final_diagnostics = solver.cache_diagnostics()
    finally:
        # On an exception the writer records the number of fully written
        # samples, leaving a clearly incomplete shard for diagnosis/resume.
        try:
            if writer is not None:
                writer.close()
        finally:
            # PETSc resources are still released if an HDF5 flush/close fails.
            if solver is not None:
                if final_diagnostics is None:
                    try:
                        final_diagnostics = solver.cache_diagnostics()
                    except Exception:
                        final_diagnostics = None
                solver.close()

    elapsed = perf_counter() - total_start
    print(f"Saved complete shard: {args.output}")
    print(f"Total generation time: {elapsed / 3600.0:.3f} h")
    if final_diagnostics is not None:
        print(f"Cached-solver counters: {final_diagnostics}")


if __name__ == "__main__":
    main()
