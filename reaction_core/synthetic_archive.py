"""Validation and loading of saved fine-grid synthetic truth archives.

This module contains no FEniCS or Torch imports. Exact and surrogate inverse
drivers can therefore share one archive contract while remaining usable in
their separate Sulis environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .grid import RegularGrid2D
from .inversion import interior_vertex_mask, trapezoid_weights


SYNTHETIC_FILE_FORMAT = "reaction-fine-truth-data-v1"
HARD_MINIMUM_U = 0.0


@dataclass(frozen=True)
class FineSyntheticArchive:
    """Validated fine FEniCS truth and observation data."""

    path: Path
    fine_grid: RegularGrid2D
    fine_fem_nx: int
    fine_fem_ny: int
    rhs: float
    u_fine: np.ndarray
    data_noiseless_fine: np.ndarray
    data_noisy_fine: np.ndarray
    noise_fine: np.ndarray
    observation_mask_fine: np.ndarray
    observation_weights_fine: np.ndarray
    truth_kind: str
    truth_seed: int | None
    noise_seed: int | None
    metadata: dict[str, Any]


def _resolve_npz(path: Path, project_root: Path | None) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = (project_root if project_root is not None else Path.cwd()) / expanded
    resolved = expanded.resolve()
    return resolved if resolved.suffix == ".npz" else resolved.with_suffix(".npz")


def _archive_text(archive: Any, key: str) -> str:
    if key not in archive:
        raise ValueError(f"Synthetic archive is missing required key {key!r}.")
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"Synthetic archive field {key!r} must be scalar.")
    return str(value.reshape(()).item())


def _archive_scalar(archive: Any, key: str, cast) -> Any:
    if key not in archive:
        raise ValueError(f"Synthetic archive is missing required key {key!r}.")
    value = np.asarray(archive[key])
    if value.size != 1:
        raise ValueError(f"Synthetic archive field {key!r} must be scalar.")
    return cast(value.reshape(()).item())


def _archive_field(
    archive: Any,
    key: str,
    shape: tuple[int, int],
    *,
    dtype=np.float64,
) -> np.ndarray:
    if key not in archive:
        raise ValueError(f"Synthetic archive is missing required key {key!r}.")
    values = np.asarray(archive[key], dtype=dtype)
    if values.shape != shape:
        raise ValueError(
            f"Synthetic archive {key!r} has shape {values.shape}; expected {shape}."
        )
    if dtype is not bool and not np.all(np.isfinite(values)):
        raise ValueError(f"Synthetic archive {key!r} contains non-finite values.")
    return values.copy()


def _parse_archive_json(archive: Any, key: str) -> dict[str, Any]:
    if key not in archive:
        return {}
    try:
        decoded = json.loads(_archive_text(archive, key))
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not decode synthetic archive JSON field {key!r}."
        ) from exc
    return decoded if isinstance(decoded, dict) else {"value": decoded}


def load_fine_synthetic_archive(
    path: Path,
    *,
    project_root: Path | None = None,
    supported_u_bounds: tuple[float, float] | None = None,
) -> FineSyntheticArchive:
    """Load and validate a fine P1 truth/data archive without solving.

    Relative paths are resolved against ``project_root`` when it is supplied,
    otherwise against the current working directory. ``supported_u_bounds``
    lets an emulator driver additionally enforce its supervised input support.
    """

    source_path = _resolve_npz(path, project_root)
    if not source_path.is_file():
        raise FileNotFoundError(f"Synthetic file does not exist: {source_path}")

    with np.load(source_path, allow_pickle=False) as archive:
        if _archive_text(archive, "format") != SYNTHETIC_FILE_FORMAT:
            raise ValueError(
                "Unsupported synthetic archive format; expected "
                f"{SYNTHETIC_FILE_FORMAT!r}."
            )
        if _archive_text(archive, "parameterisation") != "direct physical u":
            raise ValueError("Synthetic archive is not direct physical-u data.")
        if _archive_scalar(archive, "state_degree", int) != 1:
            raise ValueError("Only P1 synthetic archives are supported.")

        fine_grid = RegularGrid2D(
            x=np.asarray(archive["fine_fem_x_grid"], dtype=np.float64),
            y=np.asarray(archive["fine_fem_y_grid"], dtype=np.float64),
        )
        fine_nx = _archive_scalar(archive, "fine_fem_nx_cells", int)
        fine_ny = _archive_scalar(archive, "fine_fem_ny_cells", int)
        if fine_grid.shape != (fine_nx + 1, fine_ny + 1):
            raise ValueError(
                "Fine FEM coordinate dimensions do not match stored cell counts."
            )
        if not (
            np.isclose(fine_grid.x[0], 0.0)
            and np.isclose(fine_grid.y[0], 0.0)
            and np.isclose(fine_grid.x[-1], 1.0)
            and np.isclose(fine_grid.y[-1], 1.0)
        ):
            raise ValueError("Only unit-square synthetic archives are supported.")

        rhs = _archive_scalar(archive, "rhs", float)
        if not np.isfinite(rhs) or rhs <= 0.0:
            raise ValueError("Synthetic archive has an invalid positive RHS.")
        shape = fine_grid.shape
        u_fine = _archive_field(archive, "u_fem_P1_vertices", shape)
        clean = _archive_field(archive, "data_noiseless", shape)
        noisy = _archive_field(archive, "data_noisy", shape)
        noise = _archive_field(archive, "noise_fem_P1_vertices", shape)
        mask = _archive_field(
            archive, "observation_mask", shape, dtype=bool
        ).astype(bool)
        weights = _archive_field(
            archive, "observation_weights_trapezoid", shape
        )

        if not np.allclose(
            weights, trapezoid_weights(fine_grid), rtol=0.0, atol=5.0e-14
        ):
            raise ValueError("Synthetic archive has incompatible observation weights.")
        if not np.any(mask):
            raise ValueError("Synthetic archive has an empty observation mask.")
        if np.any(mask & ~interior_vertex_mask(fine_grid)):
            raise ValueError("Synthetic archive observes a Dirichlet boundary vertex.")
        if not np.allclose(noisy - clean, noise, rtol=0.0, atol=5.0e-12):
            raise ValueError(
                "Synthetic archive noisy data do not equal noiseless data + noise."
            )
        if np.any(np.abs(noise[~mask]) > 5.0e-12):
            raise ValueError("Synthetic archive has noise outside its observation mask.")
        if np.min(u_fine) < HARD_MINIMUM_U - 1.0e-10:
            raise ValueError("Synthetic archive u falls below the physical lower bound.")

        if supported_u_bounds is not None:
            lower, upper = map(float, supported_u_bounds)
            if not HARD_MINIMUM_U <= lower < upper:
                raise ValueError("Invalid supported physical-u bounds.")
            if np.min(u_fine) < lower - 1.0e-9:
                raise ValueError(
                    "Synthetic archive u falls below the emulator input support."
                )
            if np.max(u_fine) > upper + 1.0e-9:
                raise ValueError(
                    "Synthetic archive u exceeds the emulator input support."
                )

        truth_kind = (
            _archive_text(archive, "truth_kind")
            if "truth_kind" in archive
            else "unknown"
        )
        truth_seed = (
            _archive_scalar(archive, "truth_seed", int)
            if "truth_seed" in archive
            else None
        )
        noise_seed = (
            _archive_scalar(archive, "noise_seed", int)
            if "noise_seed" in archive
            else None
        )
        metadata = {
            "truth": _parse_archive_json(archive, "truth_metadata_json"),
            "noise": _parse_archive_json(archive, "noise_metadata_json"),
            "report": _parse_archive_json(archive, "report_json"),
        }

    return FineSyntheticArchive(
        path=source_path,
        fine_grid=fine_grid,
        fine_fem_nx=int(fine_nx),
        fine_fem_ny=int(fine_ny),
        rhs=float(rhs),
        u_fine=u_fine,
        data_noiseless_fine=clean,
        data_noisy_fine=noisy,
        noise_fine=noise,
        observation_mask_fine=mask,
        observation_weights_fine=weights,
        truth_kind=truth_kind,
        truth_seed=truth_seed,
        noise_seed=noise_seed,
        metadata=metadata,
    )
