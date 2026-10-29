from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


class ReactionShardWriter:
    """Write one fixed-size physical-``u`` reaction FNO+DINO shard."""

    def __init__(
        self,
        path: str | Path,
        *,
        n_samples: int,
        nx: int,
        ny: int,
        directions: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        dtype=np.float32,
        compression: str | None = "lzf",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n_samples = int(n_samples)
        self.nx = int(nx)
        self.ny = int(ny)
        self.dtype = np.dtype(dtype)
        self.directions = np.asarray(directions, dtype=np.float32)
        if self.n_samples < 1:
            raise ValueError("n_samples must be positive.")
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx and ny must be at least two.")
        if self.directions.ndim != 3 or self.directions.shape[1:] != (
            self.nx,
            self.ny,
        ):
            raise ValueError("directions must have shape (n_modes,nx,ny).")
        if self.directions.shape[0] < 1:
            raise ValueError("At least one DINO direction is required.")
        self.n_modes = int(self.directions.shape[0])

        x_grid = np.asarray(x_grid, dtype=np.float64)
        y_grid = np.asarray(y_grid, dtype=np.float64)
        if x_grid.shape != (self.nx,) or y_grid.shape != (self.ny,):
            raise ValueError("x_grid/y_grid lengths must match nx/ny.")
        if not np.all(np.isfinite(self.directions)):
            raise ValueError("directions contains non-finite values.")

        self.handle = h5py.File(self.path, "w")
        self.handle.attrs["format"] = "reaction_fno_dino_shard_v1_physical_u"
        self.handle.attrs["n_samples"] = self.n_samples
        self.handle.attrs["n_modes"] = self.n_modes
        self.handle.attrs["input_variable"] = "physical_reaction_coefficient_u"
        self.handle.attrs["output_channels"] = json.dumps(["p"])
        self.handle.attrs["direction_variable"] = "physical_u"
        if config is not None:
            self.handle.attrs["project_config_json"] = json.dumps(config)

        self.handle.create_dataset("x_grid", data=x_grid)
        self.handle.create_dataset("y_grid", data=y_grid)
        self.handle.create_dataset("directions_u", data=self.directions)

        scalar_chunks = (1, 1, self.nx, self.ny)
        jvp_chunks = (1, 1, 1, self.nx, self.ny)
        self.u = self.handle.create_dataset(
            "u",
            shape=(self.n_samples, 1, self.nx, self.ny),
            dtype=self.dtype,
            chunks=scalar_chunks,
            compression=compression,
        )
        self.p = self.handle.create_dataset(
            "p",
            shape=(self.n_samples, 1, self.nx, self.ny),
            dtype=self.dtype,
            chunks=scalar_chunks,
            compression=compression,
        )
        self.jvp = self.handle.create_dataset(
            "jvp",
            shape=(self.n_samples, self.n_modes, 1, self.nx, self.ny),
            dtype=self.dtype,
            chunks=jvp_chunks,
            compression=compression,
        )

        metadata = self.handle.create_group("metadata")
        self.metadata = {
            "seed": metadata.create_dataset(
                "seed", shape=(self.n_samples,), dtype=np.uint32
            ),
            "field_seed": metadata.create_dataset(
                "field_seed", shape=(self.n_samples,), dtype=np.uint32
            ),
            "family_id": metadata.create_dataset(
                "family_id", shape=(self.n_samples,), dtype=np.int16
            ),
            "nu": metadata.create_dataset(
                "nu", shape=(self.n_samples,), dtype=np.float32
            ),
            "length_scale": metadata.create_dataset(
                "length_scale", shape=(self.n_samples,), dtype=np.float32
            ),
            "target_mean_u": metadata.create_dataset(
                "target_mean_u", shape=(self.n_samples,), dtype=np.float32
            ),
            "actual_mean_u": metadata.create_dataset(
                "actual_mean_u", shape=(self.n_samples,), dtype=np.float32
            ),
            "minimum_u": metadata.create_dataset(
                "minimum_u", shape=(self.n_samples,), dtype=np.float32
            ),
            "maximum_u": metadata.create_dataset(
                "maximum_u", shape=(self.n_samples,), dtype=np.float32
            ),
            "target_std_u": metadata.create_dataset(
                "target_std_u", shape=(self.n_samples,), dtype=np.float32
            ),
            "actual_std_u": metadata.create_dataset(
                "actual_std_u", shape=(self.n_samples,), dtype=np.float32
            ),
            "fluctuation_scale": metadata.create_dataset(
                "fluctuation_scale", shape=(self.n_samples,), dtype=np.float32
            ),
            "rescaled": metadata.create_dataset(
                "rescaled", shape=(self.n_samples,), dtype=np.uint8
            ),
        }
        self._next_index = 0

    def write_sample(
        self,
        index: int,
        *,
        u: np.ndarray,
        p: np.ndarray,
        jvp: np.ndarray,
        metadata,
    ) -> None:
        if self.handle is None:
            raise RuntimeError("Cannot write to a closed shard.")
        if index != self._next_index:
            raise ValueError(
                f"Samples must be written sequentially; expected "
                f"{self._next_index}, received {index}."
            )
        if not 0 <= index < self.n_samples:
            raise IndexError(index)

        u = np.asarray(u)
        p = np.asarray(p)
        jvp = np.asarray(jvp)
        if u.shape != (self.nx, self.ny):
            raise ValueError(f"u must have shape {(self.nx, self.ny)}.")
        if p.shape == (self.nx, self.ny):
            p = p[None]
        if p.shape != (1, self.nx, self.ny):
            raise ValueError(f"p must have shape {(1, self.nx, self.ny)}.")
        if jvp.shape == (self.n_modes, self.nx, self.ny):
            jvp = jvp[:, None]
        if jvp.shape != (self.n_modes, 1, self.nx, self.ny):
            raise ValueError(
                "jvp must have shape (n_modes,1,nx,ny); received "
                f"{jvp.shape}."
            )
        if not all(np.all(np.isfinite(value)) for value in (u, p, jvp)):
            raise ValueError("A sample contains non-finite values.")

        self.u[index, 0] = u
        self.p[index] = p
        self.jvp[index] = jvp
        values = {
            "seed": metadata.seed,
            "field_seed": metadata.field_seed,
            "family_id": metadata.family_id,
            "nu": metadata.nu,
            "length_scale": metadata.length_scale,
            "target_mean_u": metadata.target_mean_u,
            "actual_mean_u": metadata.actual_mean_u,
            "minimum_u": metadata.minimum_u,
            "maximum_u": metadata.maximum_u,
            "target_std_u": metadata.target_std_u,
            "actual_std_u": metadata.actual_std_u,
            "fluctuation_scale": metadata.fluctuation_scale,
            "rescaled": int(bool(metadata.rescaled)),
        }
        for name, value in values.items():
            self.metadata[name][index] = value
        self._next_index += 1

    def close(self) -> None:
        if self.handle is None:
            return
        self.handle.attrs["completed_samples"] = self._next_index
        self.handle.flush()
        self.handle.close()
        self.handle = None

    def __enter__(self) -> "ReactionShardWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class ReactionH5Dataset:
    """PyTorch-compatible lazy dataset spanning several reaction shards."""

    def __init__(
        self,
        shard_paths: Sequence[str | Path],
        *,
        include_jvp: bool = True,
    ) -> None:
        self.paths = [Path(path).resolve() for path in shard_paths]
        if not self.paths:
            raise ValueError("At least one shard is required.")
        self.include_jvp = bool(include_jvp)
        self._handles: dict[int, h5py.File] = {}
        self.lengths: list[int] = []
        self.cumulative: list[int] = []
        total = 0
        reference_directions: np.ndarray | None = None
        reference_grid: tuple[np.ndarray, np.ndarray] | None = None

        for path in self.paths:
            with h5py.File(path, "r") as handle:
                for key in ("u", "p", "directions_u", "x_grid", "y_grid"):
                    if key not in handle:
                        raise KeyError(f"Shard {path} is missing {key!r}.")
                u_shape = tuple(handle["u"].shape)
                p_shape = tuple(handle["p"].shape)
                if len(u_shape) != 4 or u_shape[1] != 1:
                    raise ValueError(f"Invalid u shape {u_shape} in {path}.")
                if p_shape != u_shape:
                    raise ValueError(
                        f"p shape {p_shape} does not match u shape {u_shape} in {path}."
                    )
                length = int(u_shape[0])
                completed = int(handle.attrs.get("completed_samples", length))
                if completed != length:
                    raise ValueError(
                        f"Shard {path} is incomplete: {completed}/{length} samples."
                    )
                if self.include_jvp:
                    if "jvp" not in handle:
                        raise KeyError(f"Shard {path} is missing 'jvp'.")
                    expected_jvp_tail = (1, u_shape[2], u_shape[3])
                    if tuple(handle["jvp"].shape)[0] != length or tuple(
                        handle["jvp"].shape
                    )[2:] != expected_jvp_tail:
                        raise ValueError(f"Invalid jvp shape in {path}.")

                self.lengths.append(length)
                total += length
                self.cumulative.append(total)
                directions = np.asarray(handle["directions_u"][:], dtype=np.float32)
                if (
                    directions.ndim != 3
                    or tuple(directions.shape[1:]) != tuple(u_shape[2:])
                ):
                    raise ValueError(
                        f"Invalid directions_u shape {directions.shape} in {path}."
                    )
                if self.include_jvp and int(handle["jvp"].shape[1]) != int(
                    directions.shape[0]
                ):
                    raise ValueError(
                        f"JVP/direction mode mismatch in {path}: "
                        f"{handle['jvp'].shape[1]} versus {directions.shape[0]}."
                    )
                if reference_directions is None:
                    reference_directions = directions
                elif (
                    directions.shape != reference_directions.shape
                    or not np.allclose(
                        directions,
                        reference_directions,
                        rtol=1.0e-6,
                        atol=1.0e-7,
                    )
                ):
                    raise ValueError("All shards must use identical DINO directions.")

                shard_grid = (
                    np.asarray(handle["x_grid"][:], dtype=np.float64),
                    np.asarray(handle["y_grid"][:], dtype=np.float64),
                )
                if reference_grid is None:
                    reference_grid = shard_grid
                elif not (
                    np.array_equal(shard_grid[0], reference_grid[0])
                    and np.array_equal(shard_grid[1], reference_grid[1])
                ):
                    raise ValueError("All shards must use identical regular grids.")

        assert reference_directions is not None and reference_grid is not None
        self.directions = reference_directions
        self.x_grid, self.y_grid = reference_grid

    def __len__(self) -> int:
        return self.cumulative[-1]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative, index)
        start = 0 if shard_index == 0 else self.cumulative[shard_index - 1]
        return shard_index, index - start

    def _handle(self, shard_index: int) -> h5py.File:
        handle = self._handles.get(shard_index)
        if handle is None or not bool(handle.id.valid):
            handle = h5py.File(self.paths[shard_index], "r")
            self._handles[shard_index] = handle
        return handle

    def __getitem__(self, index: int):
        import torch

        shard_index, local_index = self._locate(index)
        handle = self._handle(shard_index)
        u = torch.from_numpy(
            np.asarray(handle["u"][local_index], dtype=np.float32)
        )
        p = torch.from_numpy(
            np.asarray(handle["p"][local_index], dtype=np.float32)
        )
        if not self.include_jvp:
            return u, p
        derivative = torch.from_numpy(
            np.asarray(handle["jvp"][local_index], dtype=np.float32)
        )
        return u, p, derivative

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()

    def __del__(self):
        self.close()


class SelectiveReactionH5Dataset(ReactionH5Dataset):
    """Lazy Reaction dataset that reads exactly one requested JVP mode.

    When ``include_jvp`` is true, indices must be ``(sample_index,
    mode_index)`` pairs.  This is intentionally paired with a sampler that
    chooses one mode per physical-``u`` sample.  The HDF5 writer chunks JVPs
    as ``(1,1,1,nx,ny)``, so this selection avoids reading or decompressing
    the other stored modes.

    ``include_output=False`` is useful for derivative-only validation: it
    avoids reading ``p`` while retaining the branch input ``u``.
    """

    def __init__(
        self,
        shard_paths: Sequence[str | Path],
        *,
        include_jvp: bool,
        include_output: bool = True,
        jvp_sidecar_paths: Sequence[str | Path] | None = None,
    ) -> None:
        self.jvp_sidecar_paths = (
            [Path(path).resolve() for path in jvp_sidecar_paths]
            if jvp_sidecar_paths is not None
            else None
        )
        self._jvp_handles: dict[int, h5py.File] = {}
        # Source JVPs need not be inspected when an overlay supplies them.
        super().__init__(
            shard_paths,
            include_jvp=include_jvp and self.jvp_sidecar_paths is None,
        )
        self.include_jvp = bool(include_jvp)
        if self.jvp_sidecar_paths is not None:
            if not include_jvp:
                raise ValueError("JVP sidecars were supplied to a forward-only dataset.")
            from .jvp_sidecars import audit_jvp_sidecars

            overlay = audit_jvp_sidecars(self.paths, self.jvp_sidecar_paths)
            self.directions = np.asarray(overlay["directions"], dtype=np.float32)
        self.include_output = bool(include_output)

    def _jvp_handle(self, shard_index: int) -> h5py.File:
        if self.jvp_sidecar_paths is None:
            return self._handle(shard_index)
        handle = self._jvp_handles.get(shard_index)
        if handle is None or not bool(handle.id.valid):
            handle = h5py.File(self.jvp_sidecar_paths[shard_index], "r")
            self._jvp_handles[shard_index] = handle
        return handle

    def __getstate__(self):
        state = super().__getstate__()
        state["_jvp_handles"] = {}
        return state

    def __getitem__(self, item):
        import torch

        mode_index: int | None = None
        if isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError(
                    "Derivative dataset indices must be "
                    "(sample_index, mode_index)."
                )
            sample_index = int(item[0])
            mode_index = int(item[1])
        else:
            sample_index = int(item)

        if self.include_jvp:
            if mode_index is None:
                raise ValueError(
                    "A DINO mode index is required when include_jvp=True."
                )
            if not 0 <= mode_index < int(self.directions.shape[0]):
                raise IndexError(
                    f"mode_index={mode_index} is outside "
                    f"[0,{self.directions.shape[0]})."
                )

        shard_index, local_index = self._locate(sample_index)
        handle = self._handle(shard_index)
        result: list[torch.Tensor] = [
            torch.from_numpy(
                np.asarray(handle["u"][local_index], dtype=np.float32)
            )
        ]

        if self.include_output:
            result.append(
                torch.from_numpy(
                    np.asarray(handle["p"][local_index], dtype=np.float32)
                )
            )

        if self.include_jvp:
            assert mode_index is not None
            jvp_handle = self._jvp_handle(shard_index)
            derivative = torch.from_numpy(
                np.asarray(
                    jvp_handle["jvp"][local_index, mode_index],
                    dtype=np.float32,
                )
            )
            result.append(torch.tensor(mode_index, dtype=torch.long))
            result.append(derivative)

        return tuple(result)

    def close(self) -> None:
        for handle in getattr(self, "_jvp_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass
        getattr(self, "_jvp_handles", {}).clear()
        super().close()


def compute_scalar_channel_stats(
    dataset: ReactionH5Dataset,
    *,
    max_samples: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute one scalar mean and standard deviation for ``u`` and ``p``."""
    count = min(len(dataset), max_samples) if max_samples else len(dataset)
    if count < 1:
        raise ValueError("Dataset is empty.")
    u_sum = 0.0
    u_square_sum = 0.0
    p_sum = 0.0
    p_square_sum = 0.0
    total_pixels = 0
    for index in range(count):
        u_tensor, p_tensor = dataset[index][:2]
        u = u_tensor.numpy().astype(np.float64)
        p = p_tensor.numpy().astype(np.float64)
        u_sum += float(np.sum(u))
        u_square_sum += float(np.sum(u * u))
        p_sum += float(np.sum(p))
        p_square_sum += float(np.sum(p * p))
        total_pixels += int(u.size)

    u_mean = u_sum / total_pixels
    p_mean = p_sum / total_pixels
    u_variance = max(u_square_sum / total_pixels - u_mean**2, 1.0e-20)
    p_variance = max(p_square_sum / total_pixels - p_mean**2, 1.0e-20)
    return {
        "x_mean": np.asarray([u_mean], dtype=np.float32),
        "x_std": np.asarray([np.sqrt(u_variance)], dtype=np.float32),
        "y_mean": np.asarray([p_mean], dtype=np.float32),
        "y_std": np.asarray([np.sqrt(p_variance)], dtype=np.float32),
    }


# Concise aliases used by some generator/trainer templates.
ReactionDataset = ReactionH5Dataset
ReactionH5ShardWriter = ReactionShardWriter


__all__ = [
    "ReactionDataset",
    "ReactionH5Dataset",
    "ReactionH5ShardWriter",
    "ReactionShardWriter",
    "SelectiveReactionH5Dataset",
    "compute_scalar_channel_stats",
]
