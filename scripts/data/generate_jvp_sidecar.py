#!/usr/bin/env python3
"""Recompute exact JVP labels for one immutable Reaction coefficient shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import h5py
import numpy as np
from mpi4py import MPI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reaction_core.config import ReactionConfig  # noqa: E402
from reaction_core.dis import DISBasis2D  # noqa: E402
from reaction_core.grid import (  # noqa: E402
    RegularGrid2D,
    StructuredCoefficientGridMap,
    StructuredLagrangeStateGridMap,
)
from reaction_core.jvp_sidecars import ReactionJVPSidecarWriter  # noqa: E402
from reaction_core.observations import FullFieldObservationOperator  # noqa: E402
from reaction_core.solver import ReactionSolver2D  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a DIS-JVP sidecar for one existing Reaction shard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-shard", type=Path, required=True)
    parser.add_argument("--dis-basis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-basis-hash", required=True)
    parser.add_argument("--flush-every", type=int, default=8)
    parser.add_argument("--forward-rtol", type=float, default=5e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Run exactly one MPI rank per derivative sidecar.")
    source_path = args.source_shard.expanduser().resolve()
    basis_path = args.dis_basis.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not source_path.is_file() or not basis_path.is_file():
        raise FileNotFoundError("The source shard and DIS basis must both exist.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace {output_path}.")
    basis = DISBasis2D.load(basis_path)
    if basis.basis_hash != args.expected_basis_hash:
        raise ValueError(
            f"DIS basis hash {basis.basis_hash} differs from expected {args.expected_basis_hash}."
        )
    basis_file_hash = file_sha256(basis_path)

    with h5py.File(source_path, "r") as source:
        raw = source.attrs.get("project_config_json")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        config = json.loads(str(raw))
        reaction = ReactionConfig(**config["reaction"])
        reaction.validate()
        grid = RegularGrid2D(
            np.asarray(source["x_grid"][:], dtype=np.float64),
            np.asarray(source["y_grid"][:], dtype=np.float64),
        )
        if basis.grid.shape != grid.shape or not (
            np.array_equal(basis.grid.x, grid.x) and np.array_equal(basis.grid.y, grid.y)
        ):
            raise ValueError("DIS basis and source shard grids differ.")
        n_samples = int(source["u"].shape[0])
        source_pressure = source["p"]

        solver = ReactionSolver2D(
            reaction,
            minimum_u=float(config["distribution"]["u_min"]),
            initial_u=float(np.mean(source["u"][0])),
        )
        coefficient_map = StructuredCoefficientGridMap(solver, grid)
        state_map = StructuredLagrangeStateGridMap(solver, grid)
        observations = FullFieldObservationOperator(state_map)
        direction_functions = [
            coefficient_map.grid_to_function(values, name=f"dis_direction_{i}")
            for i, values in enumerate(basis.directions)
        ]
        maximum_forward_relative_error = 0.0
        started = perf_counter()
        with ReactionJVPSidecarWriter(
            output_path,
            source_shard=source_path,
            directions=basis.directions,
            x_grid=grid.x,
            y_grid=grid.y,
            basis_logical_sha256=basis.basis_hash,
            basis_file_sha256=basis_file_hash,
        ) as writer:
            for index in range(n_samples):
                before = solver.cache_diagnostics()
                u = np.asarray(source["u"][index, 0], dtype=np.float64)
                solver.set_coefficient(coefficient_map.grid_to_array(u))
                state = solver.forward()
                pressure = observations.apply(state, dtype=np.float32)
                stored = np.asarray(source_pressure[index], dtype=np.float32)
                relative = float(
                    np.linalg.norm(pressure.astype(np.float64) - stored.astype(np.float64))
                    / max(np.linalg.norm(stored.astype(np.float64)), np.finfo(float).tiny)
                )
                maximum_forward_relative_error = max(maximum_forward_relative_error, relative)
                if relative > args.forward_rtol:
                    raise RuntimeError(
                        f"Recomputed pressure differs from source at sample {index}: {relative:.3e}."
                    )
                jvp = np.empty((basis.n_directions, 1, grid.nx, grid.ny), dtype=np.float32)
                for mode, direction in enumerate(direction_functions):
                    jvp[mode] = observations.apply(solver.tangent(state, direction), dtype=np.float32)
                after = solver.cache_diagnostics()
                assemblies = int(after["operator_assemblies_factorizations"]) - int(
                    before["operator_assemblies_factorizations"]
                )
                if assemblies != 1:
                    raise RuntimeError(f"Expected one operator assembly for sample {index}; got {assemblies}.")
                writer.write(index, jvp)
                if (index + 1) % args.flush_every == 0 or index + 1 == n_samples:
                    writer.handle.attrs["maximum_forward_relative_error"] = maximum_forward_relative_error
                    writer.handle.flush()
                    print(f"sample {index + 1:4d}/{n_samples} forward-check={relative:.3e}", flush=True)
            writer.handle.attrs["elapsed_seconds"] = perf_counter() - started
            writer.handle.attrs["source_project_config_json"] = json.dumps(config, sort_keys=True)

    print(f"Saved derivative sidecar: {output_path}")
    print(f"DIS logical basis hash: {basis.basis_hash}")
    print(f"Maximum forward replay discrepancy: {maximum_forward_relative_error:.3e}")


if __name__ == "__main__":
    main()
