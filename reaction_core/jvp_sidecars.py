"""Integrity-checked derivative-label sidecars for immutable Reaction shards."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from .dis import array_sha256


SIDECAR_FORMAT = "reaction_jvp_sidecar_v1"


def source_tensor_hashes(source: h5py.File, key: str) -> np.ndarray:
    """Hash every stored float32 sample tensor using the DIS array contract."""

    return np.asarray(
        [
            array_sha256(np.asarray(source[key][i], dtype=np.float32))
            for i in range(source[key].shape[0])
        ],
        dtype="S64",
    )


def source_u_hashes(source: h5py.File) -> np.ndarray:
    return source_tensor_hashes(source, "u")


def source_p_hashes(source: h5py.File) -> np.ndarray:
    return source_tensor_hashes(source, "p")


class ReactionJVPSidecarWriter:
    """Write a sidecar atomically; an exception never replaces the destination."""

    def __init__(
        self,
        path: str | Path,
        *,
        source_shard: str | Path,
        directions: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        basis_logical_sha256: str,
        basis_file_sha256: str,
        dtype: np.dtype = np.float32,
        compression: str | None = "lzf",
    ) -> None:
        self.path = Path(path).resolve()
        if self.path.exists():
            raise FileExistsError(f"Refusing to replace derivative sidecar: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.path.with_name(self.path.name + f".partial-{os.getpid()}")
        if self.partial.exists():
            raise FileExistsError(f"Temporary sidecar already exists: {self.partial}")
        self.source_path = Path(source_shard).resolve()
        directions = np.asarray(directions, dtype=np.float32)
        x_grid = np.asarray(x_grid, dtype=np.float64)
        y_grid = np.asarray(y_grid, dtype=np.float64)
        if directions.ndim != 3 or directions.shape[1:] != (x_grid.size, y_grid.size):
            raise ValueError("directions/grid shape mismatch.")
        with h5py.File(self.source_path, "r") as source:
            self.n_samples = int(source["u"].shape[0])
            seeds = np.asarray(source["metadata/seed"][:], dtype=np.uint32)
            u_hashes = source_u_hashes(source)
            p_hashes = source_p_hashes(source)
            source_name = self.source_path.name
        self.n_modes = int(directions.shape[0])
        self.handle = h5py.File(self.partial, "x")
        h = self.handle
        h.attrs["format"] = SIDECAR_FORMAT
        h.attrs["n_samples"] = self.n_samples
        h.attrs["n_modes"] = self.n_modes
        h.attrs["completed_samples"] = 0
        h.attrs["source_shard_name"] = source_name
        h.attrs["basis_logical_sha256"] = str(basis_logical_sha256)
        h.attrs["basis_file_sha256"] = str(basis_file_sha256)
        h.attrs["directions_sha256"] = array_sha256(directions)
        h.create_dataset("x_grid", data=x_grid)
        h.create_dataset("y_grid", data=y_grid)
        h.create_dataset("directions_u", data=directions)
        h.create_dataset("source_seed", data=seeds)
        h.create_dataset("source_u_sha256", data=u_hashes)
        h.create_dataset("source_p_sha256", data=p_hashes)
        h.create_dataset(
            "jvp",
            shape=(self.n_samples, self.n_modes, 1, x_grid.size, y_grid.size),
            dtype=np.dtype(dtype),
            chunks=(1, 1, 1, x_grid.size, y_grid.size),
            compression=compression,
        )
        self.next_index = 0

    def write(self, index: int, jvp: np.ndarray) -> None:
        if index != self.next_index:
            raise ValueError(f"Expected sidecar sample {self.next_index}, received {index}.")
        values = np.asarray(jvp)
        expected = tuple(self.handle["jvp"].shape[1:])
        if values.shape == (expected[0], expected[2], expected[3]):
            values = values[:, None]
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid JVP tensor {values.shape}; expected {expected}.")
        self.handle["jvp"][index] = values
        self.next_index += 1
        self.handle.attrs["completed_samples"] = self.next_index

    def close(self, *, success: bool) -> None:
        if self.handle is None:
            return
        complete = self.next_index == self.n_samples
        self.handle.flush()
        self.handle.close()
        self.handle = None
        if success and complete:
            # Atomic publication that fails rather than replacing a path
            # created concurrently after the constructor's existence check.
            os.link(self.partial, self.path)
            self.partial.unlink()
        elif success:
            raise RuntimeError(f"Sidecar is incomplete: {self.next_index}/{self.n_samples}.")

    def __enter__(self) -> "ReactionJVPSidecarWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(success=exc_type is None)


def audit_jvp_sidecars(
    source_paths: Sequence[str | Path],
    sidecar_paths: Sequence[str | Path],
    *,
    expected_basis_logical_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify one-to-one sidecar/source linkage and common DIS directions."""

    sources = [Path(p).resolve() for p in source_paths]
    sidecars = [Path(p).resolve() for p in sidecar_paths]
    if len(sources) != len(sidecars):
        raise ValueError(f"Need one JVP sidecar per source shard; got {len(sidecars)} for {len(sources)}.")
    reference_directions = None
    reference_basis = None
    total = 0
    for source_path, sidecar_path in zip(sources, sidecars, strict=True):
        with h5py.File(source_path, "r") as source, h5py.File(sidecar_path, "r") as sidecar:
            if str(sidecar.attrs.get("format", "")) != SIDECAR_FORMAT:
                raise ValueError(f"Not a Reaction JVP sidecar: {sidecar_path}")
            n = int(source["u"].shape[0])
            if int(sidecar.attrs.get("n_samples", -1)) != n or int(sidecar.attrs.get("completed_samples", -1)) != n:
                raise ValueError(f"Incomplete or wrongly sized sidecar: {sidecar_path}")
            if str(sidecar.attrs.get("source_shard_name", "")) != source_path.name:
                raise ValueError(f"Sidecar/source filename mismatch for {sidecar_path}.")
            if not np.array_equal(sidecar["source_seed"][:], source["metadata/seed"][:]):
                raise ValueError(f"Sidecar/source seeds differ for {sidecar_path}.")
            if not np.array_equal(sidecar["source_u_sha256"][:], source_u_hashes(source)):
                raise ValueError(f"Sidecar/source coefficient hashes differ for {sidecar_path}.")
            if not np.array_equal(sidecar["source_p_sha256"][:], source_p_hashes(source)):
                raise ValueError(f"Sidecar/source pressure hashes differ for {sidecar_path}.")
            if not np.array_equal(sidecar["x_grid"][:], source["x_grid"][:]) or not np.array_equal(sidecar["y_grid"][:], source["y_grid"][:]):
                raise ValueError(f"Sidecar/source grids differ for {sidecar_path}.")
            directions = np.asarray(sidecar["directions_u"][:], dtype=np.float32)
            expected_shape = (n, directions.shape[0], 1, source["u"].shape[-2], source["u"].shape[-1])
            if tuple(sidecar["jvp"].shape) != expected_shape:
                raise ValueError(f"JVP tensor shape mismatch in {sidecar_path}.")
            if str(sidecar.attrs.get("directions_sha256", "")) != array_sha256(directions):
                raise ValueError(f"Direction hash mismatch in {sidecar_path}.")
            basis_hash = str(sidecar.attrs.get("basis_logical_sha256", ""))
            if expected_basis_logical_sha256 and basis_hash != expected_basis_logical_sha256:
                raise ValueError(f"Unexpected DIS basis hash in {sidecar_path}: {basis_hash}.")
            if reference_basis is None:
                reference_basis = basis_hash
                reference_directions = directions
            elif basis_hash != reference_basis or not np.array_equal(directions, reference_directions):
                raise ValueError("JVP sidecars do not share one exact basis/direction set.")
            total += n
    assert reference_directions is not None
    return {
        "paths": sidecars,
        "n_sidecars": len(sidecars),
        "n_samples": total,
        "basis_logical_sha256": reference_basis,
        "directions": reference_directions,
        "directions_sha256": array_sha256(reference_directions),
    }
