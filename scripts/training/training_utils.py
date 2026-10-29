"""Shared shard and training helpers for the Reaction emulators."""

from __future__ import annotations

import glob
import json
from pathlib import Path
import random
import re
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Sampler

from reaction_core.jvp_sidecars import audit_jvp_sidecars


def expand_shard_patterns(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        if matches:
            paths.extend(Path(match).resolve() for match in matches)
        elif Path(pattern).is_file():
            paths.append(Path(pattern).resolve())
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise FileNotFoundError(f"No shards matched {list(patterns)!r}.")
    return unique


def prepare_jvp_overlays(
    args,
    train_paths: list[Path],
    val_paths: list[Path],
    *,
    use_dino: bool,
) -> tuple[list[Path] | None, list[Path] | None, dict[str, object] | None]:
    """Resolve and verify optional paired derivative-label sidecars."""

    train_patterns = getattr(args, "train_jvp_sidecars", None)
    val_patterns = getattr(args, "val_jvp_sidecars", None)
    expected_hash = getattr(args, "expected_jvp_basis_hash", None)
    supplied = train_patterns is not None or val_patterns is not None or expected_hash is not None
    if not supplied:
        return None, None, None
    if not use_dino:
        raise ValueError("Derivative sidecars cannot be used when --dino-weight is zero.")
    if train_patterns is None or val_patterns is None or not expected_hash:
        raise ValueError(
            "DIS supervision requires --train-jvp-sidecars, --val-jvp-sidecars, "
            "and --expected-jvp-basis-hash together."
        )
    train_sidecars = expand_shard_patterns(train_patterns)
    val_sidecars = expand_shard_patterns(val_patterns)
    train = audit_jvp_sidecars(
        train_paths,
        train_sidecars,
        expected_basis_logical_sha256=expected_hash,
    )
    validation = audit_jvp_sidecars(
        val_paths,
        val_sidecars,
        expected_basis_logical_sha256=expected_hash,
    )
    if not np.array_equal(train["directions"], validation["directions"]):
        raise ValueError("Training and validation sidecar directions differ.")
    return train_sidecars, val_sidecars, {
        "kind": "DIS_sidecar",
        "basis_logical_sha256": str(expected_hash),
        "directions_sha256": str(train["directions_sha256"]),
        "directions": train["directions"],
        "train_sidecars": [str(path) for path in train_sidecars],
        "val_sidecars": [str(path) for path in val_sidecars],
        "train_samples": int(train["n_samples"]),
        "val_samples": int(validation["n_samples"]),
    }


def audit_shards(paths: list[Path], *, label: str) -> dict[str, object]:
    """Reject incomplete or mutually incompatible shard collections."""

    resolved_paths = [Path(path).resolve() for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError(f"Duplicate paths were supplied for {label} shards.")

    required = (
        "u",
        "p",
        "jvp",
        "directions_u",
        "x_grid",
        "y_grid",
        "metadata/seed",
        "metadata/minimum_u",
        "metadata/maximum_u",
    )
    reference_directions: np.ndarray | None = None
    reference_grid: tuple[np.ndarray, np.ndarray] | None = None
    reference_config: dict[str, object] | None = None
    seeds: list[int] = []
    sample_counts: list[int] = []
    seed_records: list[dict[str, int]] = []
    total = 0

    for path in resolved_paths:
        with h5py.File(path, "r") as handle:
            if str(handle.attrs.get("format", "")) != (
                "reaction_fno_dino_shard_v1_physical_u"
            ):
                raise ValueError(f"{label} shard {path} is not a physical-u Reaction shard.")
            if str(handle.attrs.get("input_variable", "")) != (
                "physical_reaction_coefficient_u"
            ):
                raise ValueError(f"{label} shard {path} does not store direct physical u.")
            if str(handle.attrs.get("direction_variable", "")) != "physical_u":
                raise ValueError(f"{label} shard {path} has non-physical directions.")

            raw_config = handle.attrs.get("project_config_json")
            if raw_config is None:
                raise ValueError(f"{label} shard {path} has no project configuration.")
            if isinstance(raw_config, bytes):
                raw_config = raw_config.decode("utf-8")
            try:
                project_config = json.loads(str(raw_config))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{label} shard {path} has invalid project_config_json."
                ) from error
            if not isinstance(project_config, dict):
                raise ValueError(f"{label} shard {path} configuration is not an object.")
            try:
                reaction_config = project_config["reaction"]
                grid_config = project_config["grid"]
                distribution_config = project_config["distribution"]
            except KeyError as error:
                raise ValueError(
                    f"{label} shard {path} has an incomplete project configuration."
                ) from error
            if reference_config is None:
                reference_config = project_config
            elif project_config != reference_config:
                raise ValueError(f"Project configuration differs in {label} shard {path}.")

            missing = [name for name in required if name not in handle]
            if missing:
                raise ValueError(f"{label} shard {path} is missing {missing}.")

            allocated = int(handle["u"].shape[0])
            declared = int(handle.attrs.get("n_samples", -1))
            completed = int(handle.attrs.get("completed_samples", -1))
            if allocated < 1 or declared != allocated or completed != allocated:
                raise ValueError(
                    f"Incomplete {label} shard {path}: allocated={allocated}, "
                    f"n_samples={declared}, completed_samples={completed}."
                )

            u_shape = tuple(handle["u"].shape)
            p_shape = tuple(handle["p"].shape)
            jvp_shape = tuple(handle["jvp"].shape)
            if len(u_shape) != 4 or u_shape[1] != 1:
                raise ValueError(f"Unexpected u shape in {path}: {u_shape}.")
            expected_p = u_shape
            if p_shape != expected_p:
                raise ValueError(f"p shape {p_shape} does not match u {u_shape} in {path}.")
            expected_jvp_tail = (1, u_shape[-2], u_shape[-1])
            if (
                len(jvp_shape) != 5
                or jvp_shape[0] != allocated
                or tuple(jvp_shape[-3:]) != expected_jvp_tail
            ):
                raise ValueError(f"Unexpected jvp shape in {path}: {jvp_shape}.")
            declared_modes = int(handle.attrs.get("n_modes", -1))
            if declared_modes != int(jvp_shape[1]):
                raise ValueError(
                    f"n_modes={declared_modes} but jvp stores "
                    f"{jvp_shape[1]} modes in {path}."
                )

            directions = np.asarray(handle["directions_u"][:], dtype=np.float32)
            if directions.shape != (
                jvp_shape[1],
                u_shape[-2],
                u_shape[-1],
            ):
                raise ValueError(
                    f"Direction/JVP mismatch in {path}: "
                    f"{directions.shape} versus {jvp_shape}."
                )
            if reference_directions is None:
                reference_directions = directions
            elif not np.array_equal(reference_directions, directions):
                raise ValueError(f"DINO directions differ in {path}.")

            x_grid = np.asarray(handle["x_grid"][:], dtype=np.float64)
            y_grid = np.asarray(handle["y_grid"][:], dtype=np.float64)
            if x_grid.shape != (u_shape[-2],) or y_grid.shape != (u_shape[-1],):
                raise ValueError(f"Grid shape mismatch in {path}.")
            expected_grid_shape = (
                int(grid_config["nx"]),
                int(grid_config["ny"]),
            )
            expected_vertex_shape = (
                int(reaction_config["fem_nx"]) + 1,
                int(reaction_config["fem_ny"]) + 1,
            )
            if u_shape[-2:] != expected_grid_shape or expected_grid_shape != expected_vertex_shape:
                raise ValueError(
                    f"Stored grid/configuration mismatch in {path}: "
                    f"data={u_shape[-2:]}, grid={expected_grid_shape}, "
                    f"FEM vertices={expected_vertex_shape}."
                )
            if not (
                np.isclose(x_grid[0], 0.0)
                and np.isclose(y_grid[0], 0.0)
                and np.isclose(x_grid[-1], float(reaction_config["Lx"]))
                and np.isclose(y_grid[-1], float(reaction_config["Ly"]))
            ):
                raise ValueError(f"Grid endpoints do not match the domain in {path}.")
            if reference_grid is None:
                reference_grid = (x_grid, y_grid)
            elif not (
                np.array_equal(reference_grid[0], x_grid)
                and np.array_equal(reference_grid[1], y_grid)
            ):
                raise ValueError(f"Neural grids differ in {path}.")

            shard_seeds = np.asarray(handle["metadata/seed"][:], dtype=np.uint64)
            if shard_seeds.shape != (allocated,):
                raise ValueError(f"Seed metadata has wrong shape in {path}.")
            required_seed_attrs = (
                "base_seed",
                "sample_offset",
                "first_seed",
                "last_seed",
            )
            missing_seed_attrs = [
                name for name in required_seed_attrs if name not in handle.attrs
            ]
            if missing_seed_attrs:
                raise ValueError(
                    f"Seed provenance attributes are missing in {path}: "
                    f"{missing_seed_attrs}."
                )
            base_seed = int(handle.attrs["base_seed"])
            sample_offset = int(handle.attrs["sample_offset"])
            first_seed = int(handle.attrs["first_seed"])
            last_seed = int(handle.attrs["last_seed"])
            if first_seed != base_seed + sample_offset:
                raise ValueError(
                    f"Inconsistent first_seed/base_seed/sample_offset in {path}."
                )
            if last_seed != first_seed + allocated - 1:
                raise ValueError(f"Inconsistent last_seed in {path}.")
            expected_seeds = np.arange(
                first_seed,
                last_seed + 1,
                dtype=np.uint64,
            )
            if not np.array_equal(shard_seeds, expected_seeds):
                raise ValueError(
                    f"metadata/seed is not the declared consecutive range in {path}."
                )
            seeds.extend(int(value) for value in shard_seeds)
            seed_records.append(
                {
                    "base_seed": base_seed,
                    "sample_offset": sample_offset,
                    "first_seed": first_seed,
                    "last_seed": last_seed,
                }
            )
            stored_minimum = np.asarray(
                handle["metadata/minimum_u"][:], dtype=np.float64
            )
            stored_maximum = np.asarray(
                handle["metadata/maximum_u"][:], dtype=np.float64
            )
            lower = float(distribution_config["u_min"])
            upper = float(distribution_config["u_max"])
            tolerance = 2.0e-5 * max(1.0, abs(lower), abs(upper))
            if np.any(stored_minimum < lower - tolerance) or np.any(
                stored_maximum > upper + tolerance
            ):
                raise ValueError(f"Physical-u metadata violates configured bounds in {path}.")
            sample_counts.append(allocated)
            total += allocated

    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate sample seeds found within {label} shards.")
    assert reference_directions is not None
    assert reference_grid is not None
    assert reference_config is not None
    return {
        "paths": resolved_paths,
        "n_shards": len(resolved_paths),
        "n_samples": total,
        "sample_counts": sample_counts,
        "seed_records": seed_records,
        "seeds": set(seeds),
        "directions": reference_directions,
        "x_grid": reference_grid[0],
        "y_grid": reference_grid[1],
        "shape": reference_directions.shape[-2:],
        "project_config": reference_config,
    }


def assert_disjoint_splits(train_audit: dict[str, object], val_audit: dict[str, object]) -> None:
    path_overlap = set(train_audit["paths"]) & set(val_audit["paths"])
    if path_overlap:
        raise ValueError(
            "Training and validation shard paths overlap: "
            + ", ".join(str(path) for path in sorted(path_overlap))
        )
    overlap = set(train_audit["seeds"]) & set(val_audit["seeds"])
    if overlap:
        examples = sorted(overlap)[:8]
        raise ValueError(f"Training and validation seeds overlap: {examples}.")
    if not np.array_equal(train_audit["directions"], val_audit["directions"]):
        raise ValueError("Training and validation DINO directions differ.")
    if tuple(train_audit["shape"]) != tuple(val_audit["shape"]):
        raise ValueError("Training and validation grids differ.")
    if train_audit["project_config"] != val_audit["project_config"]:
        raise ValueError("Training and validation project configurations differ.")


def assert_expected_shard_layout(
    train_audit: dict[str, object],
    val_audit: dict[str, object],
    *,
    expected_total_shards: int,
    expected_train_shards: int,
    expected_samples_per_shard: int,
    expected_base_seed: int,
) -> dict[str, int]:
    """Require the exact production shard IDs, split, and sample counts.

    Production uses ``reaction_shard_000.h5`` through ``063.h5``.  Checking
    the names as well as the counts prevents a missing shard, stale extra
    shard, or shifted train/validation boundary from silently changing the
    experiment.
    """

    expected_total_shards = int(expected_total_shards)
    expected_train_shards = int(expected_train_shards)
    expected_samples_per_shard = int(expected_samples_per_shard)
    expected_base_seed = int(expected_base_seed)
    if expected_total_shards < 2:
        raise ValueError("expected_total_shards must be at least two.")
    if not 1 <= expected_train_shards < expected_total_shards:
        raise ValueError(
            "expected_train_shards must lie strictly between zero and "
            "expected_total_shards."
        )
    if expected_samples_per_shard < 1:
        raise ValueError("expected_samples_per_shard must be positive.")

    train_paths = [Path(path) for path in train_audit["paths"]]
    val_paths = [Path(path) for path in val_audit["paths"]]
    all_paths = train_paths + val_paths
    if len(all_paths) != expected_total_shards:
        raise ValueError(
            f"Expected exactly {expected_total_shards} Reaction shards, "
            f"received {len(all_paths)}."
        )
    if len(train_paths) != expected_train_shards:
        raise ValueError(
            f"Expected exactly {expected_train_shards} training shards, "
            f"received {len(train_paths)}."
        )

    pattern = re.compile(r"^reaction_shard_(\d{3})\.h5$")
    ids: list[int] = []
    for path in all_paths:
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(
                f"Noncanonical Reaction shard filename: {path.name!r}."
            )
        ids.append(int(match.group(1)))

    expected_ids = list(range(expected_total_shards))
    if ids != expected_ids:
        raise ValueError(
            "Reaction shards must be supplied in the exact contiguous order "
            f"000--{expected_total_shards - 1:03d}; received IDs {ids}."
        )
    if ids[:expected_train_shards] != list(range(expected_train_shards)):
        raise ValueError("The training split does not contain the leading shard IDs.")

    counts = list(train_audit["sample_counts"]) + list(
        val_audit["sample_counts"]
    )
    bad_counts = [
        (index, int(count))
        for index, count in enumerate(counts)
        if int(count) != expected_samples_per_shard
    ]
    if bad_counts:
        raise ValueError(
            "Every Reaction shard must contain exactly "
            f"{expected_samples_per_shard} completed samples; mismatches: "
            f"{bad_counts}."
        )

    seed_records = list(train_audit["seed_records"]) + list(
        val_audit["seed_records"]
    )
    for shard_id, record in enumerate(seed_records):
        expected_offset = shard_id * expected_samples_per_shard
        expected_first = expected_base_seed + expected_offset
        expected_last = expected_first + expected_samples_per_shard - 1
        observed = {
            "base_seed": int(record["base_seed"]),
            "sample_offset": int(record["sample_offset"]),
            "first_seed": int(record["first_seed"]),
            "last_seed": int(record["last_seed"]),
        }
        expected = {
            "base_seed": expected_base_seed,
            "sample_offset": expected_offset,
            "first_seed": expected_first,
            "last_seed": expected_last,
        }
        if observed != expected:
            raise ValueError(
                f"Shard {shard_id:03d} seed provenance is {observed}, "
                f"expected {expected}."
            )

    expected_total_samples = expected_total_shards * expected_samples_per_shard
    actual_total_samples = int(train_audit["n_samples"]) + int(
        val_audit["n_samples"]
    )
    if actual_total_samples != expected_total_samples:
        raise ValueError(
            f"Expected {expected_total_samples} total samples, received "
            f"{actual_total_samples}."
        )

    return {
        "total_shards": expected_total_shards,
        "train_shards": expected_train_shards,
        "validation_shards": expected_total_shards - expected_train_shards,
        "samples_per_shard": expected_samples_per_shard,
        "total_samples": expected_total_samples,
        "training_samples": expected_train_shards * expected_samples_per_shard,
        "validation_samples": (
            expected_total_shards - expected_train_shards
        )
        * expected_samples_per_shard,
        "base_seed": expected_base_seed,
    }


class EpochRandomModeSampler(Sampler):
    """Visit each sample once per epoch with one uniform random mode."""

    def __init__(self, dataset_size: int, n_modes: int, *, seed: int) -> None:
        if dataset_size < 1 or n_modes < 1:
            raise ValueError("dataset_size and n_modes must be positive.")
        self.dataset_size = int(dataset_size)
        self.n_modes = int(n_modes)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.dataset_size

    def __iter__(self):
        permutation_generator = torch.Generator()
        permutation_generator.manual_seed(
            self.seed + 1_000_003 * self.epoch
        )
        indices = torch.randperm(
            self.dataset_size,
            generator=permutation_generator,
        ).tolist()

        mode_generator = torch.Generator()
        mode_generator.manual_seed(
            self.seed + 20_000_033 * self.epoch + 17
        )
        modes = torch.randint(
            0,
            self.n_modes,
            (self.dataset_size,),
            generator=mode_generator,
        ).tolist()
        return iter(
            (int(sample_index), int(mode_index))
            for sample_index, mode_index in zip(indices, modes)
        )


def assert_expected_independent_layout(
    train_audit: dict[str, object],
    val_audit: dict[str, object],
    *,
    expected_train_shards: int,
    expected_val_shards: int,
    expected_samples_per_shard: int,
    expected_train_base_seed: int,
    expected_val_base_seed: int,
) -> dict[str, int]:
    """Validate independent train/validation shard pools.

    Unlike :func:`assert_expected_shard_layout`, this helper allows the
    training and validation pools to live in different directories and to use
    different base seeds.  Each pool must independently contain canonical
    ``reaction_shard_000.h5, ...`` files with consecutive sample offsets.
    This is the layout used by the 5k/10k/20k/40k training ladder.
    """

    assert_disjoint_splits(train_audit, val_audit)

    expected_train_shards = int(expected_train_shards)
    expected_val_shards = int(expected_val_shards)
    expected_samples_per_shard = int(expected_samples_per_shard)
    expected_train_base_seed = int(expected_train_base_seed)
    expected_val_base_seed = int(expected_val_base_seed)
    if expected_train_shards < 1 or expected_val_shards < 1:
        raise ValueError("Training and validation shard counts must be positive.")
    if expected_samples_per_shard < 1:
        raise ValueError("expected_samples_per_shard must be positive.")

    pattern = re.compile(r"^reaction_shard_(\d{3})\.h5$")

    def check_pool(
        audit: dict[str, object],
        *,
        label: str,
        expected_shards: int,
        expected_base_seed: int,
    ) -> None:
        paths = [Path(path) for path in audit["paths"]]
        if len(paths) != expected_shards:
            raise ValueError(
                f"Expected {expected_shards} {label} shards, received {len(paths)}."
            )
        ids: list[int] = []
        for path in paths:
            match = pattern.fullmatch(path.name)
            if match is None:
                raise ValueError(
                    f"Noncanonical {label} shard filename: {path.name!r}."
                )
            ids.append(int(match.group(1)))
        expected_ids = list(range(expected_shards))
        if ids != expected_ids:
            raise ValueError(
                f"{label.capitalize()} shards must be supplied in contiguous "
                f"order 000--{expected_shards - 1:03d}; received {ids}."
            )
        counts = [int(value) for value in audit["sample_counts"]]
        if any(value != expected_samples_per_shard for value in counts):
            raise ValueError(
                f"Every {label} shard must contain "
                f"{expected_samples_per_shard} samples; received {counts}."
            )
        records = list(audit["seed_records"])
        for shard_id, record in enumerate(records):
            expected_offset = shard_id * expected_samples_per_shard
            expected = {
                "base_seed": expected_base_seed,
                "sample_offset": expected_offset,
                "first_seed": expected_base_seed + expected_offset,
                "last_seed": (
                    expected_base_seed
                    + expected_offset
                    + expected_samples_per_shard
                    - 1
                ),
            }
            observed = {name: int(record[name]) for name in expected}
            if observed != expected:
                raise ValueError(
                    f"{label.capitalize()} shard {shard_id:03d} seed provenance "
                    f"is {observed}, expected {expected}."
                )

    check_pool(
        train_audit,
        label="training",
        expected_shards=expected_train_shards,
        expected_base_seed=expected_train_base_seed,
    )
    check_pool(
        val_audit,
        label="validation",
        expected_shards=expected_val_shards,
        expected_base_seed=expected_val_base_seed,
    )
    return {
        "train_shards": expected_train_shards,
        "val_shards": expected_val_shards,
        "samples_per_shard": expected_samples_per_shard,
        "train_samples": expected_train_shards * expected_samples_per_shard,
        "val_samples": expected_val_shards * expected_samples_per_shard,
        "train_base_seed": expected_train_base_seed,
        "val_base_seed": expected_val_base_seed,
    }


class EpochShuffleSampler(Sampler):
    """Deterministic epoch-keyed sample permutation without DINO modes."""

    def __init__(self, dataset_size: int, *, seed: int) -> None:
        if dataset_size < 1:
            raise ValueError("dataset_size must be positive.")
        self.dataset_size = int(dataset_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.dataset_size

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + 1_000_003 * self.epoch)
        return iter(
            int(index)
            for index in torch.randperm(
                self.dataset_size,
                generator=generator,
            ).tolist()
        )


class BalancedValidationModeSampler(Sampler):
    """Assign one mode per sample with counts differing by at most one."""

    def __init__(self, dataset_size: int, n_modes: int) -> None:
        if dataset_size < 1 or n_modes < 1:
            raise ValueError("dataset_size and n_modes must be positive.")
        self.dataset_size = int(dataset_size)
        self.n_modes = int(n_modes)
        self.round_index = 0

    def set_round(self, round_index: int) -> None:
        self.round_index = int(round_index)

    def __len__(self) -> int:
        return self.dataset_size

    def __iter__(self):
        offset = self.round_index % self.n_modes
        return iter(
            (
                sample_index,
                (sample_index + offset) % self.n_modes,
            )
            for sample_index in range(self.dataset_size)
        )


def dino_validation_is_due(
    epoch: int,
    final_epoch: int,
    interval: int,
) -> bool:
    if interval < 1:
        raise ValueError("DINO validation interval must be positive.")
    return epoch == 1 or epoch == final_epoch or epoch % interval == 0


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = int(torch.initial_seed() % 2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def first_difference_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dx_prediction = prediction[..., 1:, :] - prediction[..., :-1, :]
    dx_target = target[..., 1:, :] - target[..., :-1, :]
    dy_prediction = prediction[..., :, 1:] - prediction[..., :, :-1]
    dy_target = target[..., :, 1:] - target[..., :, :-1]
    return 0.5 * (
        torch.mean((dx_prediction - dx_target) ** 2)
        + torch.mean((dy_prediction - dy_target) ** 2)
    )


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.vector_norm((prediction - target).reshape(prediction.shape[0], -1), dim=1)
    denominator = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(1.0e-12)
    return torch.mean(numerator / denominator)


__all__ = [
    "BalancedValidationModeSampler",
    "EpochRandomModeSampler",
    "EpochShuffleSampler",
    "assert_disjoint_splits",
    "assert_expected_shard_layout",
    "audit_shards",
    "dino_validation_is_due",
    "expand_shard_patterns",
    "first_difference_loss",
    "relative_l2",
    "seed_everything",
    "seed_worker",
]
