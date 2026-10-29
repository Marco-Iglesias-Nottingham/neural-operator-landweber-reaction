#!/usr/bin/env python3
"""Build a non-overwriting covariance-weighted input DIS pilot basis.

The covariance factor is estimated from the same independent physical-u pool
and seed convention as the frozen KL basis.  The reduced sensitivity matrix is
assembled with physical FEM tangent solves and the Landweber interior
trapezoidal observation metric.  No neural surrogate, inverse trajectory, or
final-test datum enters the construction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np


BUILDER_VERSION = "reaction-covariance-input-dis-projected-v1"


def find_project_root() -> Path:
    environment_root = os.environ.get("REACTION_PROJECT_ROOT")
    if environment_root:
        candidate = Path(environment_root).expanduser().resolve()
        if (candidate / "reaction_core").is_dir():
            return candidate
        raise RuntimeError(
            f"REACTION_PROJECT_ROOT={candidate} does not contain reaction_core/."
        )
    script_path = Path(__file__).resolve()
    for candidate in [script_path.parent, *script_path.parents, Path.cwd().resolve()]:
        candidate = candidate.resolve()
        if (candidate / "reaction_core").is_dir():
            return candidate
    raise RuntimeError("Could not locate the project root containing reaction_core/.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reaction_core.config import DistributionConfig, ReactionConfig  # noqa: E402
from reaction_core.dis import (  # noqa: E402
    DISBasis2D,
    array_sha256,
    covariance_factor_from_svd,
    select_nested_sample_indices,
    solve_projected_dis,
)
from reaction_core.grid import RegularGrid2D  # noqa: E402
from reaction_core.kl import blockwise_column_mean, randomized_svd_centered  # noqa: E402
from reaction_core.sampling import MaternUSampler  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build the observation-aware covariance-weighted input DIS pilot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "results/dis_basis_pilot/covrank64_nb32/reaction_input_dis_obsw_v1.h5",
    )
    parser.add_argument("--n-covariance-samples", type=int, default=10000)
    parser.add_argument("--covariance-rank", type=int, default=64)
    parser.add_argument("--n-basis-samples", type=int, default=32)
    parser.add_argument("--n-directions", type=int, default=16)
    parser.add_argument("--pool-base-seed", type=int, default=314159)
    parser.add_argument("--basis-sample-seed", type=int, default=20260920)
    parser.add_argument("--covariance-svd-seed", type=int, default=20260920)
    parser.add_argument("--svd-oversampling", type=int, default=16)
    parser.add_argument("--svd-power-iterations", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--nx", type=int, default=120)
    parser.add_argument("--ny", type=int, default=120)
    parser.add_argument("--rhs", type=float, default=1.0)
    parser.add_argument(
        "--linear-solver", choices=("lu", "cg-hypre"), default="lu"
    )
    parser.add_argument("--ksp-rtol", type=float, default=1e-11)
    parser.add_argument("--ksp-max-it", type=int, default=10000)
    parser.add_argument("--adjoint-check-seed", type=int, default=20260920)
    parser.add_argument("--adjoint-check-tolerance", type=float, default=5e-8)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--keep-work-file", action="store_true")
    parser.add_argument("--fft-workers", type=int, default=None)
    parser.add_argument("--precompute-bank", action="store_true")
    parser.add_argument(
        "--nu-values", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5]
    )
    parser.add_argument(
        "--length-scales",
        type=float,
        nargs="+",
        default=[0.075, 0.1, 0.15, 0.2, 0.25],
    )
    parser.add_argument("--mean-u-range", type=float, nargs=2, default=[95.0, 105.0])
    parser.add_argument("--std-u-range", type=float, nargs=2, default=[10.0, 30.0])
    parser.add_argument("--u-bounds", type=float, nargs=2, default=[0.0, 200.0])
    return parser.parse_args(argv)


def resolve_project_path(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (PROJECT_ROOT / expanded).resolve()


def choose_work_dir(requested: Path | None, output: Path) -> Path:
    if requested is not None:
        result = resolve_project_path(requested)
    else:
        scratch = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR")
        result = Path(scratch).resolve() if scratch else output.parent
    result.mkdir(parents=True, exist_ok=True)
    return result


def sample_seed(base_seed: int, index: int) -> int:
    sequence = np.random.SeedSequence([int(base_seed), int(index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def git_metadata() -> dict[str, Any]:
    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    revision = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
        "worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def solver_options(args: argparse.Namespace) -> dict[str, Any]:
    if args.linear_solver == "lu":
        return {"ksp_type": "preonly", "pc_type": "lu"}
    return {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": float(args.ksp_rtol),
        "ksp_atol": 0.0,
        "ksp_max_it": int(args.ksp_max_it),
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.n_covariance_samples < 2:
        raise ValueError("--n-covariance-samples must be at least two.")
    if not 1 <= args.covariance_rank <= min(
        args.n_covariance_samples - 1, args.nx * args.ny
    ):
        raise ValueError("--covariance-rank exceeds the possible centered rank.")
    if not 1 <= args.n_directions <= args.covariance_rank:
        raise ValueError("--n-directions must not exceed --covariance-rank.")
    if not 1 <= args.n_basis_samples <= args.n_covariance_samples:
        raise ValueError("--n-basis-samples must lie inside the covariance pool.")
    if args.nx < 3 or args.ny < 3:
        raise ValueError("--nx and --ny must be at least three.")
    if args.svd_oversampling < 0 or args.svd_power_iterations < 0:
        raise ValueError("SVD controls must be non-negative.")
    if args.block_size < 1:
        raise ValueError("--block-size must be positive.")
    if args.fft_workers == 0:
        raise ValueError("--fft-workers cannot be zero.")
    if args.adjoint_check_tolerance <= 0.0:
        raise ValueError("--adjoint-check-tolerance must be positive.")


def centered_trace(
    sample_matrix: np.ndarray,
    mean: np.ndarray,
    *,
    block_size: int,
) -> float:
    total = 0.0
    for start in range(0, sample_matrix.shape[0], block_size):
        stop = min(start + block_size, sample_matrix.shape[0])
        block = np.asarray(sample_matrix[start:stop], dtype=np.float64)
        block -= mean[None, :]
        total += float(np.sum(block * block, dtype=np.float64))
    return total / float(sample_matrix.shape[0] - 1)


def make_covariance_pool(
    args: argparse.Namespace,
    *,
    grid: RegularGrid2D,
    distribution: DistributionConfig,
    work_file: Path,
) -> tuple[np.memmap, np.ndarray, dict[str, Any]]:
    pool = np.memmap(
        work_file,
        mode="w+",
        dtype=np.float32,
        shape=(args.n_covariance_samples, grid.size),
        order="C",
    )
    sampler = MaternUSampler(
        grid,
        distribution,
        precompute_matern_bank=args.precompute_bank,
        max_cached_samplers=None if args.precompute_bank else 8,
        fft_workers=args.fft_workers,
    )
    rescaled = 0
    actual_minimum = np.inf
    actual_maximum = -np.inf
    actual_mean_minimum = np.inf
    actual_mean_maximum = -np.inf
    actual_std_minimum = np.inf
    actual_std_maximum = -np.inf
    pool_digest = hashlib.sha256()
    pool_digest.update(b"float32-C-order")
    pool_digest.update(
        json.dumps(
            [args.n_covariance_samples, grid.nx, grid.ny],
            separators=(",", ":"),
        ).encode("ascii")
    )
    started = perf_counter()
    for index in range(args.n_covariance_samples):
        sampled = sampler.sample(sample_seed(args.pool_base_seed, index))
        flat = np.asarray(sampled.u, dtype=np.float32).reshape(-1)
        if flat.shape != (grid.size,) or not np.all(np.isfinite(flat)):
            raise RuntimeError(f"Invalid covariance sample {index}.")
        pool[index] = flat
        pool_digest.update(np.ascontiguousarray(flat).tobytes(order="C"))
        metadata = sampled.metadata
        rescaled += int(metadata.rescaled)
        actual_minimum = min(actual_minimum, float(metadata.minimum_u))
        actual_maximum = max(actual_maximum, float(metadata.maximum_u))
        actual_mean_minimum = min(actual_mean_minimum, float(metadata.actual_mean_u))
        actual_mean_maximum = max(actual_mean_maximum, float(metadata.actual_mean_u))
        actual_std_minimum = min(actual_std_minimum, float(metadata.actual_std_u))
        actual_std_maximum = max(actual_std_maximum, float(metadata.actual_std_u))
        if index == 0 or (index + 1) % 250 == 0 or index + 1 == args.n_covariance_samples:
            print(
                f"covariance sample {index + 1:6d}/{args.n_covariance_samples}: "
                f"u=[{metadata.minimum_u:.2f},{metadata.maximum_u:.2f}] "
                f"mean={metadata.actual_mean_u:.2f} std={metadata.actual_std_u:.2f}",
                flush=True,
            )
    pool.flush()
    mean = blockwise_column_mean(pool, block_size=args.block_size)
    diagnostics = {
        "sampling_seconds": float(perf_counter() - started),
        "rescaled_samples": int(rescaled),
        "actual_u_range": [actual_minimum, actual_maximum],
        "actual_mean_range": [actual_mean_minimum, actual_mean_maximum],
        "actual_std_range": [actual_std_minimum, actual_std_maximum],
        "pool_sha256": pool_digest.hexdigest(),
        "pool_hash_contract": (
            "sha256(header='float32-C-order'+JSON([N,nx,ny])) then each "
            "sample's C-order float32 bytes in sample-index order"
        ),
    }
    return pool, mean, diagnostics


def assemble_reduced_sensitivity(
    args: argparse.Namespace,
    *,
    grid: RegularGrid2D,
    pool: np.ndarray,
    covariance_factor: np.ndarray,
    basis_indices: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    from mpi4py import MPI

    from reaction_core.grid import (
        StructuredCoefficientGridMap,
        StructuredLagrangeStateGridMap,
    )
    from reaction_core.inversion import (
        ScalarVertexObservationAdjoint,
        interior_vertex_mask,
        trapezoid_weights,
        weighted_inner,
    )
    from reaction_core.solver import ReactionSolver2D

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Run the DIS builder with exactly one MPI rank.")
    config = ReactionConfig(
        fem_nx=grid.nx - 1,
        fem_ny=grid.ny - 1,
        state_degree=1,
        rhs=float(args.rhs),
        u_min=float(args.u_bounds[0]),
    )
    options = solver_options(args)
    weights = trapezoid_weights(grid)
    mask = interior_vertex_mask(grid)
    selected_weights = weights[mask]
    sqrt_weights = np.sqrt(selected_weights)
    rank = covariance_factor.shape[1]
    reduced_sum = np.zeros((rank, rank), dtype=np.float64)
    per_sample_reduced = np.empty(
        (basis_indices.size, rank, rank), dtype=np.float64
    )
    verification: dict[str, Any] | None = None
    started = perf_counter()

    with ReactionSolver2D(
        config,
        minimum_u=float(args.u_bounds[0]),
        initial_u=float(np.mean(args.mean_u_range)),
        solver_options=options,
    ) as solver:
        coefficient_map = StructuredCoefficientGridMap(solver, grid)
        state_map = StructuredLagrangeStateGridMap(solver, grid)
        direction_functions = [
            coefficient_map.grid_to_function(
                covariance_factor[:, column].reshape(grid.shape),
                name=f"covariance_factor_direction_{column:03d}",
            )
            for column in range(rank)
        ]
        with ScalarVertexObservationAdjoint(
            solver,
            state_map,
            solver_options=options,
        ) as observation_adjoint:
            for position, sample_index in enumerate(basis_indices):
                before = solver.cache_diagnostics()
                assemblies_before = int(before["operator_assemblies_factorizations"])
                field = np.asarray(pool[int(sample_index)], dtype=np.float64).reshape(
                    grid.shape
                )
                solver.set_coefficient(coefficient_map.grid_to_array(field))
                state = solver.forward()
                weighted_responses = np.empty((int(np.count_nonzero(mask)), rank))
                for column, direction in enumerate(direction_functions):
                    tangent = solver.tangent(state, direction)
                    tangent_grid = state_map.state_to_grid(tangent)
                    weighted_responses[:, column] = tangent_grid[mask] * sqrt_weights
                sample_reduced = weighted_responses.T @ weighted_responses
                sample_reduced = 0.5 * (sample_reduced + sample_reduced.T)
                per_sample_reduced[position] = sample_reduced
                reduced_sum += sample_reduced

                after = solver.cache_diagnostics()
                assemblies = int(after["operator_assemblies_factorizations"]) - assemblies_before
                if assemblies != 1:
                    raise RuntimeError(
                        "Expected one state-operator assembly per basis input, "
                        f"observed {assemblies} at pool index {sample_index}."
                    )

                if position == 0:
                    rng = np.random.default_rng(int(args.adjoint_check_seed))
                    a = rng.standard_normal(rank)
                    b = rng.standard_normal(rank)
                    a /= np.linalg.norm(a)
                    b /= np.linalg.norm(b)
                    h_a = covariance_factor @ a
                    h_b = covariance_factor @ b
                    tangent_a = solver.tangent(
                        state,
                        coefficient_map.grid_to_function(
                            h_a.reshape(grid.shape), name="dis_adjoint_check_h_a"
                        ),
                    )
                    tangent_b = solver.tangent(
                        state,
                        coefficient_map.grid_to_function(
                            h_b.reshape(grid.shape), name="dis_adjoint_check_h_b"
                        ),
                    )
                    tangent_a_grid = state_map.state_to_grid(tangent_a)
                    tangent_b_grid = state_map.state_to_grid(tangent_b)
                    source = observation_adjoint.lift_weighted_residual(
                        tangent_b_grid,
                        weights,
                        mask=mask,
                        name="dis_adjoint_check_source",
                    )
                    adjoint = solver.adjoint(source)
                    native_covector = solver.parameter_coordinate_adjoint(state, adjoint)
                    coordinate_covector = coefficient_map.array_to_grid(
                        native_covector
                    ).reshape(-1)
                    left = weighted_inner(
                        tangent_a_grid, tangent_b_grid, weights, mask=mask
                    )
                    right = float(h_a @ coordinate_covector)
                    relative = abs(left - right) / max(abs(left), abs(right), 1e-30)
                    if relative > args.adjoint_check_tolerance:
                        raise RuntimeError(
                            "DIS coordinate-adjoint check failed: "
                            f"left={left:.16e}, right={right:.16e}, "
                            f"relative={relative:.3e}."
                        )
                    verification = {
                        "left_weighted_jvp_pairing": left,
                        "right_coordinate_covector_pairing": right,
                        "relative_error": relative,
                        "tolerance": float(args.adjoint_check_tolerance),
                    }

                print(
                    f"basis input {position + 1:3d}/{basis_indices.size}: "
                    f"pool_index={int(sample_index):5d} "
                    f"assembly={assemblies} tangents={rank}",
                    flush=True,
                )
            cache = solver.cache_diagnostics()
            lift = observation_adjoint.diagnostics()

    if verification is None:
        raise RuntimeError("The coordinate-adjoint verification did not run.")
    reduced = reduced_sum / float(basis_indices.size)
    diagnostics = {
        "assembly_seconds": float(perf_counter() - started),
        "state_solver_options": options,
        "state_solver_cache": cache,
        "observation_lift": lift,
        "coordinate_adjoint_check": verification,
        "number_of_observations": int(np.count_nonzero(mask)),
        "weighted_observation_area": float(np.sum(weights[mask])),
    }
    return reduced, weights, mask, per_sample_reduced, diagnostics


def build(args: argparse.Namespace) -> DISBasis2D:
    validate_args(args)
    output = resolve_project_path(args.output)
    args.output = output
    if output.exists():
        raise FileExistsError(f"DIS output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    distribution = DistributionConfig(
        nu_values=tuple(args.nu_values),
        length_scales=tuple(args.length_scales),
        mean_u_range=tuple(args.mean_u_range),
        std_u_range=tuple(args.std_u_range),
        u_min=float(args.u_bounds[0]),
        u_max=float(args.u_bounds[1]),
    )
    distribution.validate()
    grid = RegularGrid2D.from_domain(1.0, 1.0, args.nx, args.ny)
    work_dir = choose_work_dir(args.work_dir, output)
    work_file = work_dir / f"reaction_dis_pool_{os.getpid()}.float32.dat"
    pool: np.memmap | None = None
    started = perf_counter()

    print("Reaction covariance-weighted input DIS pilot", flush=True)
    print(f"  output:                 {output}")
    print(f"  covariance pool/rank:   {args.n_covariance_samples}/{args.covariance_rank}")
    print(f"  basis inputs:           {args.n_basis_samples}")
    print(f"  supervision directions: {args.n_directions}")
    print(f"  grid/FEM cells:         {grid.shape}/{grid.nx - 1}x{grid.ny - 1}")
    print("  output metric:          interior trapezoidal pressure norm")
    print(f"  temporary pool:         {work_file}", flush=True)

    try:
        pool, parameter_mean, sampling_diagnostics = make_covariance_pool(
            args,
            grid=grid,
            distribution=distribution,
            work_file=work_file,
        )
        svd_started = perf_counter()
        svd = randomized_svd_centered(
            pool,
            n_components=args.covariance_rank,
            column_mean=parameter_mean,
            oversampling=args.svd_oversampling,
            power_iterations=args.svd_power_iterations,
            block_size=args.block_size,
            random_seed=args.covariance_svd_seed,
        )
        factor_flat, covariance_values = covariance_factor_from_svd(
            svd.right_singular_vectors,
            svd.singular_values,
            n_samples=args.n_covariance_samples,
        )
        svd_seconds = float(perf_counter() - svd_started)
        full_trace = centered_trace(
            pool, parameter_mean, block_size=args.block_size
        )
        retained_trace = float(np.sum(covariance_values))
        basis_indices = select_nested_sample_indices(
            args.n_covariance_samples,
            args.n_basis_samples,
            seed=args.basis_sample_seed,
        )
        basis_seeds = np.asarray(
            [sample_seed(args.pool_base_seed, int(index)) for index in basis_indices],
            dtype=np.uint32,
        )
        basis_hashes = tuple(
            array_sha256(
                np.asarray(pool[int(index)], dtype=np.float32).reshape(grid.shape)
            )
            for index in basis_indices
        )

        reduced, weights, mask, per_sample_reduced, fem_diagnostics = (
            assemble_reduced_sensitivity(
            args,
            grid=grid,
            pool=pool,
            covariance_factor=factor_flat,
            basis_indices=basis_indices,
            )
        )
        spectrum, leading_values, latent, directions_flat = solve_projected_dis(
            reduced,
            factor_flat,
            n_directions=args.n_directions,
        )
        provenance: dict[str, Any] = {
            "builder_version": BUILDER_VERSION,
            "parameter_variable": "physical_reaction_coefficient_u",
            "distribution": distribution.to_dict(),
            "domain": [1.0, 1.0],
            "grid_shape": [grid.nx, grid.ny],
            "covariance_pool": {
                "n_samples": int(args.n_covariance_samples),
                "base_seed": int(args.pool_base_seed),
                "sample_seed_scheme": (
                    "SeedSequence([base_seed,sample_index]).generate_state(1,uint32)"
                ),
                "centering": "full physical nodal ensemble column mean",
                "sample_covariance_denominator": "N-1",
                "storage": "float32_memmap; float64 statistics/products",
                "rank": int(args.covariance_rank),
                "svd_seed": int(args.covariance_svd_seed),
                "svd_oversampling": int(args.svd_oversampling),
                "svd_power_iterations": int(args.svd_power_iterations),
                "full_coordinate_trace": full_trace,
                "retained_coordinate_trace": retained_trace,
                "retained_trace_fraction": retained_trace / full_trace,
                "sampling_diagnostics": sampling_diagnostics,
            },
            "sensitivity": {
                "kind": "uncentred_mean_JtWyJ",
                "basis_sample_seed": int(args.basis_sample_seed),
                "basis_sample_count": int(args.n_basis_samples),
                "selection": "prefix of one deterministic random permutation",
                "output_map": "interior P1 pressure vertices",
                "output_metric": "tensor-product trapezoidal quadrature",
                "state_degree": 1,
                "fem_cells": [grid.nx - 1, grid.ny - 1],
                "rhs": float(args.rhs),
                "linear_solver": str(args.linear_solver),
                "fem_diagnostics": fem_diagnostics,
            },
            "direction_scaling": (
                "h_j=R a_j; covariance-scaled latent-unit probes; "
                "no multiplication by sqrt(DIS eigenvalue)"
            ),
            "execution": {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "host": platform.node(),
                "python": platform.python_version(),
                "software_versions": {
                    "numpy": np.__version__,
                    "h5py": package_version("h5py"),
                    "scipy": package_version("scipy"),
                    "dolfinx": package_version("fenics-dolfinx"),
                    "petsc4py": package_version("petsc4py"),
                },
                "git": git_metadata(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "builder_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            },
            "timing": {
                "svd_seconds": svd_seconds,
                "total_seconds_before_save": float(perf_counter() - started),
            },
        }
        basis = DISBasis2D(
            grid=grid,
            parameter_mean_grid=parameter_mean.reshape(grid.shape),
            covariance_factor=factor_flat.T.reshape(
                args.covariance_rank, grid.nx, grid.ny
            ),
            covariance_eigenvalues=covariance_values,
            reduced_sensitivity=reduced,
            sensitivity_spectrum=spectrum,
            sensitivity_eigenvalues=leading_values,
            latent_vectors=latent,
            raw_directions=directions_flat.reshape(
                args.n_directions, grid.nx, grid.ny
            ),
            directions=directions_flat.reshape(
                args.n_directions, grid.nx, grid.ny
            ).astype(np.float32),
            observation_weights=weights,
            observation_mask=mask,
            basis_sample_indices=basis_indices,
            basis_sample_seeds=basis_seeds,
            basis_sample_hashes=basis_hashes,
            per_sample_reduced_sensitivity=per_sample_reduced,
            provenance=provenance,
        )
        basis.save(output)
        loaded = DISBasis2D.load(output)
        if loaded.basis_hash != basis.basis_hash:
            raise RuntimeError("Saved DIS basis failed its immediate hash round trip.")
        print(f"Saved DIS pilot basis: {output}")
        print(f"Basis SHA-256 contract: {basis.basis_hash}")
        print(
            "Covariance trace retained: "
            f"{retained_trace / full_trace:.6f} ({100 * retained_trace / full_trace:.2f}%)"
        )
        print("Leading DIS sensitivity eigenvalues:")
        print(leading_values)
        print("DIS basis construction completed.", flush=True)
        return basis
    finally:
        if pool is not None:
            try:
                pool.flush()
            except Exception:
                pass
            mmap_object = getattr(pool, "_mmap", None)
            if mmap_object is not None:
                mmap_object.close()
            del pool
        if args.keep_work_file:
            print(f"Kept covariance-pool memmap: {work_file}", flush=True)
        else:
            try:
                work_file.unlink()
            except FileNotFoundError:
                pass


def main(argv: list[str] | None = None) -> None:
    build(parse_args(argv))


if __name__ == "__main__":
    main()
