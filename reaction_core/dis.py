"""Pure-NumPy data structures for covariance-weighted input DIS bases.

This module contains no FEniCS or neural-network dependency.  The physical
PDE actions used to assemble the reduced sensitivity matrix live in the
standalone data builder; keeping the basis contract here makes its validation
and future dataset linkage independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .grid import RegularGrid2D


DIS_FORMAT = "reaction_covariance_weighted_input_dis_v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def array_sha256(values: np.ndarray) -> str:
    """Hash an array with an explicit dtype, shape, and C-order byte contract."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def select_nested_sample_indices(
    pool_size: int,
    selected_size: int,
    *,
    seed: int,
) -> np.ndarray:
    """Return a deterministic prefix of one permutation of the sample pool."""

    pool_size = int(pool_size)
    selected_size = int(selected_size)
    if pool_size < 1:
        raise ValueError("pool_size must be positive.")
    if not 1 <= selected_size <= pool_size:
        raise ValueError("selected_size must lie between one and pool_size.")
    permutation = np.random.default_rng(int(seed)).permutation(pool_size)
    return np.asarray(permutation[:selected_size], dtype=np.int64)


def covariance_factor_from_svd(
    right_singular_vectors: np.ndarray,
    singular_values: np.ndarray,
    *,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``R`` with ``RR.T`` equal to the truncated sample covariance."""

    vectors = np.asarray(right_singular_vectors, dtype=np.float64)
    values = np.asarray(singular_values, dtype=np.float64)
    n_samples = int(n_samples)
    if vectors.ndim != 2 or values.shape != (vectors.shape[0],):
        raise ValueError("SVD vectors/scales have incompatible shapes.")
    if n_samples < 2:
        raise ValueError("At least two samples are required.")
    if not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(values)):
        raise ValueError("SVD values must be finite.")
    if np.any(values <= 0.0):
        raise ValueError("Retained singular values must be positive.")
    gram = vectors @ vectors.T
    if not np.allclose(gram, np.eye(vectors.shape[0]), rtol=2e-10, atol=2e-10):
        raise ValueError("Right singular vectors are not row-orthonormal.")

    eigenvalues = values**2 / float(n_samples - 1)
    factor = vectors.T * np.sqrt(eigenvalues)[None, :]
    return np.asarray(factor, dtype=np.float64), eigenvalues


def solve_projected_dis(
    reduced_sensitivity: np.ndarray,
    covariance_factor: np.ndarray,
    *,
    n_directions: int,
    psd_relative_tolerance: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve ``R.T G R`` and reconstruct the leading physical directions.

    Returns the full descending sensitivity spectrum, the leading eigenvalues,
    leading latent vectors stored by row, and physical directions stored by row.
    """

    reduced = np.asarray(reduced_sensitivity, dtype=np.float64)
    factor = np.asarray(covariance_factor, dtype=np.float64)
    n_directions = int(n_directions)
    if reduced.ndim != 2 or reduced.shape[0] != reduced.shape[1]:
        raise ValueError("reduced_sensitivity must be square.")
    if factor.ndim != 2 or factor.shape[1] != reduced.shape[0]:
        raise ValueError("covariance_factor must have shape (n_parameters, rank).")
    if not 1 <= n_directions <= reduced.shape[0]:
        raise ValueError("n_directions must lie between one and the factor rank.")
    if not np.all(np.isfinite(reduced)) or not np.all(np.isfinite(factor)):
        raise ValueError("DIS inputs must be finite.")

    asymmetry = np.linalg.norm(reduced - reduced.T)
    scale = max(np.linalg.norm(reduced), 1.0)
    if asymmetry > 1e-11 * scale:
        raise ValueError(
            "reduced_sensitivity is not symmetric: "
            f"relative asymmetry={asymmetry / scale:.3e}."
        )
    symmetric = 0.5 * (reduced + reduced.T)
    values, vectors = np.linalg.eigh(symmetric)
    order = np.argsort(values)[::-1]
    spectrum = np.asarray(values[order], dtype=np.float64)
    spectral_scale = max(abs(spectrum[0]), np.finfo(np.float64).tiny)
    tolerance = float(psd_relative_tolerance) * spectral_scale
    if spectrum[-1] < -tolerance:
        raise ValueError(
            "reduced_sensitivity is not positive semidefinite: "
            f"minimum eigenvalue={spectrum[-1]:.6e}."
        )
    spectrum[np.abs(spectrum) <= tolerance] = 0.0
    leading_values = spectrum[:n_directions].copy()
    latent = np.asarray(vectors[:, order[:n_directions]].T, dtype=np.float64)
    directions = latent @ factor.T
    return spectrum, leading_values, latent, directions


@dataclass(frozen=True)
class DISBasis2D:
    """Validated covariance-factor and observation-aware DIS artifact."""

    grid: RegularGrid2D
    parameter_mean_grid: np.ndarray
    covariance_factor: np.ndarray
    covariance_eigenvalues: np.ndarray
    reduced_sensitivity: np.ndarray
    sensitivity_spectrum: np.ndarray
    sensitivity_eigenvalues: np.ndarray
    latent_vectors: np.ndarray
    raw_directions: np.ndarray
    directions: np.ndarray
    observation_weights: np.ndarray
    observation_mask: np.ndarray
    basis_sample_indices: np.ndarray
    basis_sample_seeds: np.ndarray
    basis_sample_hashes: tuple[str, ...]
    per_sample_reduced_sensitivity: np.ndarray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = self.grid.shape
        mean = np.asarray(self.parameter_mean_grid, dtype=np.float64)
        factor = np.asarray(self.covariance_factor, dtype=np.float64)
        covariance_values = np.asarray(self.covariance_eigenvalues, dtype=np.float64)
        reduced = np.asarray(self.reduced_sensitivity, dtype=np.float64)
        spectrum = np.asarray(self.sensitivity_spectrum, dtype=np.float64)
        sensitivity_values = np.asarray(self.sensitivity_eigenvalues, dtype=np.float64)
        latent = np.asarray(self.latent_vectors, dtype=np.float64)
        raw_directions = np.asarray(self.raw_directions, dtype=np.float64)
        directions = np.asarray(self.directions)
        weights = np.asarray(self.observation_weights, dtype=np.float64)
        mask = np.asarray(self.observation_mask, dtype=bool)
        indices = np.asarray(self.basis_sample_indices, dtype=np.int64)
        seeds = np.asarray(self.basis_sample_seeds, dtype=np.uint32)
        per_sample = (
            None
            if self.per_sample_reduced_sensitivity is None
            else np.asarray(self.per_sample_reduced_sensitivity, dtype=np.float64)
        )

        if mean.shape != shape:
            raise ValueError(f"parameter_mean_grid must have shape {shape}.")
        if factor.ndim != 3 or factor.shape[1:] != shape:
            raise ValueError("covariance_factor must have shape (rank,nx,ny).")
        covariance_rank = factor.shape[0]
        if covariance_values.shape != (covariance_rank,):
            raise ValueError("covariance_eigenvalues has the wrong shape.")
        if reduced.shape != (covariance_rank, covariance_rank):
            raise ValueError("reduced_sensitivity has the wrong shape.")
        if spectrum.shape != (covariance_rank,):
            raise ValueError("sensitivity_spectrum has the wrong shape.")
        n_directions = sensitivity_values.size
        if not 1 <= n_directions <= covariance_rank:
            raise ValueError("The number of directions is invalid.")
        if latent.shape != (n_directions, covariance_rank):
            raise ValueError("latent_vectors has the wrong shape.")
        if raw_directions.shape != (n_directions, *shape):
            raise ValueError("raw_directions has the wrong shape.")
        if directions.shape != raw_directions.shape:
            raise ValueError("directions has the wrong shape.")
        if directions.dtype != np.dtype(np.float32):
            raise ValueError("Applied directions must use the training float32 dtype.")
        if weights.shape != shape or mask.shape != shape:
            raise ValueError("Observation arrays must use the basis grid.")
        if indices.ndim != 1 or seeds.shape != indices.shape:
            raise ValueError("Basis sample indices and seeds must be one-dimensional.")
        if len(self.basis_sample_hashes) != indices.size:
            raise ValueError("One hash is required for each basis sample.")
        if len(set(int(value) for value in indices)) != indices.size:
            raise ValueError("Basis sample indices must be unique.")
        if any(len(value) != 64 for value in self.basis_sample_hashes):
            raise ValueError("Basis sample hashes must be SHA-256 hex strings.")
        if per_sample is not None:
            expected_shape = (indices.size, covariance_rank, covariance_rank)
            if per_sample.shape != expected_shape:
                raise ValueError(
                    "per_sample_reduced_sensitivity must have shape "
                    f"{expected_shape}."
                )
            if not np.all(np.isfinite(per_sample)):
                raise ValueError("Per-sample reduced sensitivity is non-finite.")
            if not np.allclose(
                per_sample,
                np.swapaxes(per_sample, 1, 2),
                rtol=1e-11,
                atol=1e-13,
            ):
                raise ValueError("Per-sample reduced sensitivity is not symmetric.")
            if not np.allclose(
                np.mean(per_sample, axis=0),
                reduced,
                rtol=2e-12,
                atol=2e-14,
            ):
                raise ValueError(
                    "The mean per-sample sensitivity does not equal the stored matrix."
                )
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be a dictionary.")

        numeric = (
            mean,
            factor,
            covariance_values,
            reduced,
            spectrum,
            sensitivity_values,
            latent,
            raw_directions,
            directions,
            weights,
        )
        if any(not np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("DIS arrays must be finite.")
        if np.any(covariance_values <= 0.0):
            raise ValueError("Covariance eigenvalues must be positive.")
        if np.any(sensitivity_values < 0.0) or np.any(spectrum < 0.0):
            raise ValueError("Sensitivity eigenvalues must be non-negative.")
        if np.any(weights < 0.0) or not np.any(mask):
            raise ValueError("Observation weights/mask are invalid.")
        if not np.allclose(reduced, reduced.T, rtol=1e-11, atol=1e-12):
            raise ValueError("reduced_sensitivity must be symmetric.")
        if not np.allclose(
            latent @ latent.T,
            np.eye(n_directions),
            rtol=2e-10,
            atol=2e-10,
        ):
            raise ValueError("latent_vectors are not row-orthonormal.")

        expected = latent @ factor.reshape(covariance_rank, -1)
        if not np.allclose(
            raw_directions.reshape(n_directions, -1),
            expected,
            rtol=2e-10,
            atol=2e-11,
        ):
            raise ValueError("raw_directions do not equal R @ latent vectors.")
        if not np.array_equal(directions, raw_directions.astype(np.float32)):
            raise ValueError(
                "Applied directions must be the exact float32 rounding of raw_directions."
            )
        for mode in range(n_directions):
            residual = (
                reduced @ latent[mode]
                - sensitivity_values[mode] * latent[mode]
            )
            denominator = max(
                np.linalg.norm(reduced) * np.linalg.norm(latent[mode]),
                abs(sensitivity_values[mode]),
                1.0,
            )
            if np.linalg.norm(residual) / denominator > 2e-10:
                raise ValueError(f"DIS eigenpair {mode} has an excessive residual.")

        object.__setattr__(self, "parameter_mean_grid", mean)
        object.__setattr__(self, "covariance_factor", factor)
        object.__setattr__(self, "covariance_eigenvalues", covariance_values)
        object.__setattr__(self, "reduced_sensitivity", reduced)
        object.__setattr__(self, "sensitivity_spectrum", spectrum)
        object.__setattr__(self, "sensitivity_eigenvalues", sensitivity_values)
        object.__setattr__(self, "latent_vectors", latent)
        object.__setattr__(self, "raw_directions", raw_directions)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "observation_weights", weights)
        object.__setattr__(self, "observation_mask", mask)
        object.__setattr__(self, "basis_sample_indices", indices)
        object.__setattr__(self, "basis_sample_seeds", seeds)
        object.__setattr__(self, "basis_sample_hashes", tuple(self.basis_sample_hashes))
        object.__setattr__(self, "per_sample_reduced_sensitivity", per_sample)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def covariance_rank(self) -> int:
        return int(self.covariance_factor.shape[0])

    @property
    def n_directions(self) -> int:
        return int(self.directions.shape[0])

    @property
    def basis_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(DIS_FORMAT.encode("ascii"))
        for values in (
            self.grid.x,
            self.grid.y,
            self.parameter_mean_grid,
            self.covariance_factor,
            self.covariance_eigenvalues,
            self.reduced_sensitivity,
            self.sensitivity_spectrum,
            self.sensitivity_eigenvalues,
            self.latent_vectors,
            self.raw_directions,
            self.directions,
            self.observation_weights,
            self.observation_mask.astype(np.uint8),
            self.basis_sample_indices,
            self.basis_sample_seeds,
        ):
            digest.update(array_sha256(values).encode("ascii"))
        if self.per_sample_reduced_sensitivity is not None:
            digest.update(
                array_sha256(self.per_sample_reduced_sensitivity).encode("ascii")
            )
        digest.update(_canonical_json(list(self.basis_sample_hashes)).encode("utf-8"))
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        """Save atomically without ever replacing an existing artifact."""

        destination = Path(path).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"DIS basis already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.partial.{os.getpid()}"
        )
        if temporary.exists():
            raise FileExistsError(f"Temporary DIS path already exists: {temporary}")
        string_dtype = h5py.string_dtype(encoding="ascii", length=64)
        try:
            with h5py.File(temporary, "x") as handle:
                handle.attrs["format"] = DIS_FORMAT
                handle.attrs["basis_hash"] = self.basis_hash
                handle.attrs["provenance_json"] = _canonical_json(self.provenance)
                handle.attrs["parameter_variable"] = "physical_reaction_coefficient_u"
                handle.attrs["coordinate_order"] = "C-order (x-index,y-index)"
                handle.attrs["covariance_factor_definition"] = "Sigma ~= R R^T"
                handle.attrs["direction_definition"] = "h_j = R a_j; no sqrt(lambda_DIS) scaling"
                handle.attrs["applied_direction_dtype"] = "float32"
                handle.attrs["sensitivity_definition"] = "mean_i J_i^T W_y J_i (uncentred)"
                handle.create_dataset("x_grid", data=self.grid.x)
                handle.create_dataset("y_grid", data=self.grid.y)
                handle.create_dataset("parameter_mean_grid", data=self.parameter_mean_grid)
                handle.create_dataset(
                    "covariance_factor",
                    data=self.covariance_factor,
                    compression="lzf",
                )
                handle.create_dataset("covariance_eigenvalues", data=self.covariance_eigenvalues)
                handle.create_dataset("reduced_sensitivity", data=self.reduced_sensitivity)
                handle.create_dataset("sensitivity_spectrum", data=self.sensitivity_spectrum)
                handle.create_dataset("sensitivity_eigenvalues", data=self.sensitivity_eigenvalues)
                handle.create_dataset("latent_vectors", data=self.latent_vectors)
                handle.create_dataset(
                    "directions_u_raw_float64",
                    data=self.raw_directions,
                    compression="lzf",
                )
                handle.create_dataset("directions_u", data=self.directions, compression="lzf")
                handle.create_dataset(
                    "raw_direction_coordinate_norms",
                    data=np.linalg.norm(
                        self.raw_directions.reshape(self.n_directions, -1), axis=1
                    ),
                )
                handle.create_dataset(
                    "applied_direction_coordinate_norms",
                    data=np.linalg.norm(
                        self.directions.astype(np.float64).reshape(
                            self.n_directions, -1
                        ),
                        axis=1,
                    ),
                )
                handle.create_dataset("observation_weights", data=self.observation_weights)
                handle.create_dataset(
                    "observation_mask",
                    data=self.observation_mask.astype(np.uint8),
                )
                handle.create_dataset("basis_sample_indices", data=self.basis_sample_indices)
                handle.create_dataset("basis_sample_seeds", data=self.basis_sample_seeds)
                handle.create_dataset(
                    "basis_sample_sha256",
                    data=np.asarray(self.basis_sample_hashes, dtype=string_dtype),
                )
                if self.per_sample_reduced_sensitivity is not None:
                    handle.create_dataset(
                        "per_sample_reduced_sensitivity",
                        data=self.per_sample_reduced_sensitivity,
                        compression="lzf",
                    )
                handle.create_dataset(
                    "latent_gram",
                    data=self.latent_vectors @ self.latent_vectors.T,
                )
                flat = self.directions.astype(np.float64).reshape(
                    self.n_directions, -1
                )
                handle.create_dataset("direction_coordinate_gram", data=flat @ flat.T)
                handle.flush()
            os.link(temporary, destination)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    @classmethod
    def load(cls, path: str | Path) -> "DISBasis2D":
        source = Path(path).expanduser().resolve()
        with h5py.File(source, "r") as handle:
            if str(handle.attrs.get("format", "")) != DIS_FORMAT:
                raise ValueError(f"Unsupported DIS basis format in {source}.")
            provenance = json.loads(str(handle.attrs["provenance_json"]))
            result = cls(
                grid=RegularGrid2D(
                    np.asarray(handle["x_grid"], dtype=np.float64),
                    np.asarray(handle["y_grid"], dtype=np.float64),
                ),
                parameter_mean_grid=np.asarray(handle["parameter_mean_grid"]),
                covariance_factor=np.asarray(handle["covariance_factor"]),
                covariance_eigenvalues=np.asarray(handle["covariance_eigenvalues"]),
                reduced_sensitivity=np.asarray(handle["reduced_sensitivity"]),
                sensitivity_spectrum=np.asarray(handle["sensitivity_spectrum"]),
                sensitivity_eigenvalues=np.asarray(handle["sensitivity_eigenvalues"]),
                latent_vectors=np.asarray(handle["latent_vectors"]),
                raw_directions=np.asarray(handle["directions_u_raw_float64"]),
                directions=np.asarray(handle["directions_u"]),
                observation_weights=np.asarray(handle["observation_weights"]),
                observation_mask=np.asarray(handle["observation_mask"], dtype=bool),
                basis_sample_indices=np.asarray(handle["basis_sample_indices"]),
                basis_sample_seeds=np.asarray(handle["basis_sample_seeds"]),
                basis_sample_hashes=tuple(
                    value.decode("ascii") if isinstance(value, bytes) else str(value)
                    for value in np.asarray(handle["basis_sample_sha256"])
                ),
                per_sample_reduced_sensitivity=(
                    np.asarray(handle["per_sample_reduced_sensitivity"])
                    if "per_sample_reduced_sensitivity" in handle
                    else None
                ),
                provenance=provenance,
            )
            stored_hash = str(handle.attrs["basis_hash"])
        if result.basis_hash != stored_hash:
            raise ValueError(
                f"DIS basis hash mismatch in {source}: "
                f"stored={stored_hash}, computed={result.basis_hash}."
            )
        return result


__all__ = [
    "DISBasis2D",
    "DIS_FORMAT",
    "array_sha256",
    "covariance_factor_from_svd",
    "select_nested_sample_indices",
    "solve_projected_dis",
]
