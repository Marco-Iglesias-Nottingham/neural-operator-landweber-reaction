"""Empirical KL directions for the direct physical reaction coefficient.

This module deliberately depends only on NumPy and h5py.  Large sample
matrices may be NumPy memmaps: all matrix products and column statistics are
performed in row blocks, and the randomized SVD never materialises a second
``(n_samples, n_grid)`` array.

Each training field is decomposed into its scalar spatial mean and a
zero-spatial-mean residual.  KL mode zero represents one empirical standard
deviation of the scalar mean.  The remaining modes are right singular vectors
of the column-centred residual matrix, scaled by the corresponding empirical
coefficient standard deviations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import h5py
import numpy as np

from .grid import RegularGrid2D


Array = np.ndarray
KL_FORMAT = "reaction_physical_u_kl_basis_v1"


def _positive_integer(value: int, *, name: str) -> int:
    converted = int(value)
    if converted != value or converted < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return converted


def _matrix_shape(matrix: np.ndarray, *, name: str = "matrix") -> tuple[int, int]:
    if not hasattr(matrix, "shape") or len(matrix.shape) != 2:
        raise ValueError(f"{name} must be a two-dimensional array or memmap.")
    n_rows, n_columns = int(matrix.shape[0]), int(matrix.shape[1])
    if n_rows < 1 or n_columns < 1:
        raise ValueError(f"{name} must have nonzero dimensions; got {matrix.shape}.")
    if not np.issubdtype(np.dtype(matrix.dtype), np.number):
        raise TypeError(f"{name} must have a numeric dtype; got {matrix.dtype}.")
    return n_rows, n_columns


def _column_mean_array(
    column_mean: np.ndarray,
    n_columns: int,
    *,
    name: str = "column_mean",
) -> Array:
    mean = np.asarray(column_mean, dtype=np.float64)
    if mean.shape != (n_columns,):
        raise ValueError(f"{name} must have shape ({n_columns},); got {mean.shape}.")
    if not np.all(np.isfinite(mean)):
        raise ValueError(f"{name} contains non-finite values.")
    return mean


def _finite_float64_block(
    matrix: np.ndarray,
    start: int,
    stop: int,
    *,
    name: str,
) -> Array:
    block = np.asarray(matrix[start:stop], dtype=np.float64)
    if not np.all(np.isfinite(block)):
        raise ValueError(
            f"{name} contains non-finite values in rows {start}:{stop}."
        )
    return block


def blockwise_column_mean(
    matrix: np.ndarray,
    *,
    block_size: int = 128,
) -> Array:
    """Compute a float64 column mean from an array or row-major memmap.

    A batch-combination update is used instead of one very large sum, which is
    both memory bounded and numerically more stable for many samples.
    """

    n_rows, n_columns = _matrix_shape(matrix)
    block_size = _positive_integer(block_size, name="block_size")
    mean = np.zeros(n_columns, dtype=np.float64)
    count = 0
    for start in range(0, n_rows, block_size):
        stop = min(start + block_size, n_rows)
        block = _finite_float64_block(matrix, start, stop, name="matrix")
        block_count = stop - start
        block_mean = np.mean(block, axis=0, dtype=np.float64)
        new_count = count + block_count
        mean += (block_mean - mean) * (block_count / new_count)
        count = new_count
    return mean


def centered_matrix_right_product(
    matrix: np.ndarray,
    right: np.ndarray,
    *,
    column_mean: np.ndarray,
    block_size: int = 128,
    out: np.ndarray | None = None,
) -> Array:
    r"""Return ``(matrix-column_mean) @ right`` in bounded row blocks.

    ``right`` may be a vector or a matrix.  ``out`` may itself be a memmap and
    must have the resulting shape and a floating dtype.
    """

    n_rows, n_columns = _matrix_shape(matrix)
    block_size = _positive_integer(block_size, name="block_size")
    mean = _column_mean_array(column_mean, n_columns)
    multiplier = np.asarray(right, dtype=np.float64)
    was_vector = multiplier.ndim == 1
    if was_vector:
        multiplier = multiplier[:, None]
    if multiplier.ndim != 2 or multiplier.shape[0] != n_columns:
        raise ValueError(
            "right must have shape "
            f"({n_columns},) or ({n_columns},r); got {np.shape(right)}."
        )
    if not np.all(np.isfinite(multiplier)):
        raise ValueError("right contains non-finite values.")

    result_shape = (n_rows, multiplier.shape[1])
    if out is None:
        result = np.empty(result_shape, dtype=np.float64)
    else:
        result = out
        if result.shape != result_shape:
            raise ValueError(f"out must have shape {result_shape}; got {result.shape}.")
        if not np.issubdtype(np.dtype(result.dtype), np.floating):
            raise TypeError("out must have a floating dtype.")

    mean_product = mean @ multiplier
    for start in range(0, n_rows, block_size):
        stop = min(start + block_size, n_rows)
        block = _finite_float64_block(matrix, start, stop, name="matrix")
        result[start:stop] = block @ multiplier - mean_product[None, :]
    return result[:, 0] if was_vector else result


def centered_matrix_transpose_left_product(
    matrix: np.ndarray,
    left: np.ndarray,
    *,
    column_mean: np.ndarray,
    block_size: int = 128,
) -> Array:
    r"""Return ``(matrix-column_mean).T @ left`` in bounded row blocks."""

    n_rows, n_columns = _matrix_shape(matrix)
    block_size = _positive_integer(block_size, name="block_size")
    mean = _column_mean_array(column_mean, n_columns)
    multiplier = np.asarray(left, dtype=np.float64)
    was_vector = multiplier.ndim == 1
    if was_vector:
        multiplier = multiplier[:, None]
    if multiplier.ndim != 2 or multiplier.shape[0] != n_rows:
        raise ValueError(
            "left must have shape "
            f"({n_rows},) or ({n_rows},r); got {np.shape(left)}."
        )
    if not np.all(np.isfinite(multiplier)):
        raise ValueError("left contains non-finite values.")

    result = np.zeros((n_columns, multiplier.shape[1]), dtype=np.float64)
    for start in range(0, n_rows, block_size):
        stop = min(start + block_size, n_rows)
        block = _finite_float64_block(matrix, start, stop, name="matrix")
        left_block = multiplier[start:stop]
        result += block.T @ left_block
        result -= mean[:, None] * np.sum(left_block, axis=0, dtype=np.float64)
    return result[:, 0] if was_vector else result


def _orthonormal_range(
    matrix: np.ndarray,
    *,
    required_rank: int,
    name: str,
) -> Array:
    """Return a deterministic rank-revealing orthonormal column basis."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite two-dimensional matrix.")
    left, singular_values, _ = np.linalg.svd(values, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise ValueError(f"{name} has zero numerical rank.")
    tolerance = (
        np.finfo(np.float64).eps
        * max(values.shape)
        * float(singular_values[0])
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank < required_rank:
        raise ValueError(
            f"{name} has numerical rank {rank}, below the requested "
            f"{required_rank} components."
        )
    return np.asarray(left[:, :rank], dtype=np.float64)


def _canonicalize_row_signs(rows: np.ndarray) -> Array:
    """Resolve singular-vector sign ambiguity by a largest-entry convention."""

    result = np.asarray(rows, dtype=np.float64).copy()
    for row in result:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    return result


@dataclass(frozen=True)
class RandomizedSVDResult:
    """Leading right singular system of a column-centred sample matrix."""

    right_singular_vectors: np.ndarray
    singular_values: np.ndarray
    column_mean: np.ndarray
    n_samples: int
    n_features: int
    requested_components: int
    sketch_size: int
    numerical_subspace_rank: int
    oversampling: int
    power_iterations: int
    random_seed: int

    @property
    def explained_variance(self) -> Array:
        if self.n_samples < 2:
            raise ValueError("At least two samples are needed for variance.")
        return self.singular_values**2 / (self.n_samples - 1)


def randomized_svd_centered(
    matrix: np.ndarray,
    *,
    n_components: int,
    column_mean: np.ndarray | None = None,
    oversampling: int = 8,
    power_iterations: int = 2,
    block_size: int = 128,
    random_seed: int = 0,
) -> RandomizedSVDResult:
    """Compute deterministic leading right singular vectors with NumPy.

    The algorithm is a rank-revealing randomized range finder followed by a
    small dense SVD.  Power iterations alternate blockwise applications of
    the centred matrix and its transpose.  It is deterministic for fixed
    inputs and ``random_seed``; output signs are canonicalized.
    """

    n_samples, n_features = _matrix_shape(matrix)
    n_components = _positive_integer(n_components, name="n_components")
    block_size = _positive_integer(block_size, name="block_size")
    oversampling = int(oversampling)
    power_iterations = int(power_iterations)
    if oversampling < 0:
        raise ValueError("oversampling cannot be negative.")
    if power_iterations < 0:
        raise ValueError("power_iterations cannot be negative.")

    maximum_centered_rank = min(n_samples - 1, n_features)
    if n_components > maximum_centered_rank:
        raise ValueError(
            f"n_components={n_components} exceeds the maximum centred rank "
            f"{maximum_centered_rank} for shape {matrix.shape}."
        )
    mean = (
        blockwise_column_mean(matrix, block_size=block_size)
        if column_mean is None
        else _column_mean_array(column_mean, n_features)
    )
    sketch_size = min(
        n_components + oversampling,
        maximum_centered_rank,
    )
    if sketch_size < n_components:
        raise RuntimeError("Internal randomized-SVD sketch is too small.")

    rng = np.random.default_rng(int(random_seed))
    omega = rng.standard_normal((n_features, sketch_size))
    sample_sketch = centered_matrix_right_product(
        matrix,
        omega,
        column_mean=mean,
        block_size=block_size,
    )
    sample_basis = _orthonormal_range(
        sample_sketch,
        required_rank=n_components,
        name="initial randomized sample sketch",
    )

    for iteration in range(power_iterations):
        feature_sketch = centered_matrix_transpose_left_product(
            matrix,
            sample_basis,
            column_mean=mean,
            block_size=block_size,
        )
        feature_basis = _orthonormal_range(
            feature_sketch,
            required_rank=n_components,
            name=f"randomized feature sketch at power iteration {iteration + 1}",
        )
        sample_sketch = centered_matrix_right_product(
            matrix,
            feature_basis,
            column_mean=mean,
            block_size=block_size,
        )
        sample_basis = _orthonormal_range(
            sample_sketch,
            required_rank=n_components,
            name=f"randomized sample sketch at power iteration {iteration + 1}",
        )

    # B = Q.T @ A_centered.  Form A_centered.T @ Q blockwise and transpose;
    # only the small (subspace_rank,n_features) B matrix is dense in memory.
    feature_projection = centered_matrix_transpose_left_product(
        matrix,
        sample_basis,
        column_mean=mean,
        block_size=block_size,
    )
    small_matrix = feature_projection.T
    _, singular_values, right_vectors = np.linalg.svd(
        small_matrix,
        full_matrices=False,
    )
    if singular_values.size < n_components:
        raise ValueError(
            "The randomized subspace produced fewer singular values than "
            "the requested component count."
        )
    leading_values = np.asarray(singular_values[:n_components], dtype=np.float64)
    if not np.all(np.isfinite(leading_values)) or np.any(leading_values <= 0.0):
        raise ValueError(
            "The requested residual KL subspace is numerically rank deficient."
        )
    leading_vectors = _canonicalize_row_signs(right_vectors[:n_components])

    return RandomizedSVDResult(
        right_singular_vectors=leading_vectors,
        singular_values=leading_values,
        column_mean=mean.copy(),
        n_samples=n_samples,
        n_features=n_features,
        requested_components=n_components,
        sketch_size=sketch_size,
        numerical_subspace_rank=int(sample_basis.shape[1]),
        oversampling=oversampling,
        power_iterations=power_iterations,
        random_seed=int(random_seed),
    )


@dataclass(frozen=True)
class KLBasis2D:
    """Constant-plus-empirical KL directions in direct physical ``u``."""

    grid: RegularGrid2D
    unit_modes: np.ndarray
    direction_scales: np.ndarray
    directions: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    parameter_mean_grid: np.ndarray
    constant_mean_std: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unit_modes = np.asarray(self.unit_modes, dtype=np.float64)
        scales = np.asarray(self.direction_scales, dtype=np.float64)
        supplied_directions = np.asarray(self.directions, dtype=np.float64)
        variances = np.asarray(self.explained_variance, dtype=np.float64)
        ratios = np.asarray(self.explained_variance_ratio, dtype=np.float64)
        parameter_mean = np.asarray(self.parameter_mean_grid, dtype=np.float64)
        constant_std = float(self.constant_mean_std)

        if unit_modes.ndim != 3 or unit_modes.shape[1:] != self.grid.shape:
            raise ValueError(
                "unit_modes must have shape (n_modes,"
                f"{self.grid.nx},{self.grid.ny}); got {unit_modes.shape}."
            )
        n_modes = unit_modes.shape[0]
        if n_modes < 2:
            raise ValueError("A KL basis needs one constant and one spatial mode.")
        if scales.shape != (n_modes,):
            raise ValueError(f"direction_scales must have shape ({n_modes},).")
        if supplied_directions.shape != unit_modes.shape:
            raise ValueError("directions must have the same shape as unit_modes.")
        if variances.shape != (n_modes,) or ratios.shape != (n_modes,):
            raise ValueError(
                "explained_variance and explained_variance_ratio must have "
                f"shape ({n_modes},)."
            )
        if parameter_mean.shape != self.grid.shape:
            raise ValueError(
                f"parameter_mean_grid must have shape {self.grid.shape}."
            )
        arrays = (unit_modes, scales, supplied_directions, variances, ratios, parameter_mean)
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("KL basis arrays must be finite.")
        if constant_std < 0.0 or not np.isfinite(constant_std):
            raise ValueError("constant_mean_std must be finite and non-negative.")
        if np.any(scales < 0.0) or np.any(variances < 0.0) or np.any(ratios < 0.0):
            raise ValueError("KL scales, variances, and ratios must be non-negative.")

        flat_modes = unit_modes.reshape(n_modes, -1)
        gram = flat_modes @ flat_modes.T
        if not np.allclose(gram, np.eye(n_modes), rtol=2.0e-5, atol=2.0e-5):
            raise ValueError("unit_modes are not orthonormal on the flattened grid.")
        expected_constant = np.full(
            self.grid.size,
            1.0 / np.sqrt(self.grid.size),
            dtype=np.float64,
        )
        if not np.allclose(
            flat_modes[0],
            expected_constant,
            rtol=2.0e-5,
            atol=2.0e-7,
        ):
            raise ValueError("KL mode zero is not the normalized constant field.")
        if not np.allclose(variances, scales**2, rtol=2.0e-10, atol=2.0e-12):
            raise ValueError("explained_variance must equal direction_scales squared.")
        variance_total = float(np.sum(variances, dtype=np.float64))
        expected_ratios = (
            variances / variance_total
            if variance_total > 0.0
            else np.zeros_like(variances)
        )
        if not np.allclose(ratios, expected_ratios, rtol=2.0e-10, atol=2.0e-12):
            raise ValueError("explained_variance_ratio is inconsistent with variances.")
        expected_constant_scale = constant_std * np.sqrt(self.grid.size)
        if not np.isclose(scales[0], expected_constant_scale, rtol=2.0e-12, atol=2.0e-12):
            raise ValueError(
                "The constant direction scale must be sqrt(n_grid)*constant_mean_std."
            )

        reconstructed = unit_modes * scales[:, None, None]
        storage_tolerance = 8.0e-6 * max(
            1.0,
            float(np.max(np.abs(reconstructed), initial=0.0)),
        )
        if not np.allclose(
            supplied_directions,
            reconstructed,
            rtol=2.0e-5,
            atol=storage_tolerance,
        ):
            raise ValueError("directions must equal unit_modes * direction_scales.")
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be a dictionary.")

        # Preserve directions_u exactly as supplied.  In particular, an HDF5
        # round trip independently rounds unit_modes and directions_u to
        # float32; reconstructing here would change the saved directions by a
        # few ulps and would make newly generated shards differ from the basis
        # file.  Freshly built bases supply the exact product, while loaded
        # bases retain the authoritative stored directions after the tolerant
        # consistency check above.
        object.__setattr__(self, "unit_modes", unit_modes)
        object.__setattr__(self, "direction_scales", scales)
        object.__setattr__(self, "directions", supplied_directions)
        object.__setattr__(self, "explained_variance", variances)
        object.__setattr__(self, "explained_variance_ratio", ratios)
        object.__setattr__(self, "parameter_mean_grid", parameter_mean)
        object.__setattr__(self, "constant_mean_std", constant_std)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def n_modes(self) -> int:
        return int(self.directions.shape[0])

    @property
    def directions_u(self) -> np.ndarray:
        return self.directions

    @property
    def parameter_mean_u(self) -> np.ndarray:
        return self.parameter_mean_grid

    def select_directions(self, n_modes: int) -> np.ndarray:
        if not 1 <= int(n_modes) <= self.n_modes:
            raise ValueError(
                f"n_modes must lie in [1,{self.n_modes}], got {n_modes}."
            )
        return self.directions[: int(n_modes)]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.attrs["format"] = KL_FORMAT
            handle.attrs["basis_variable"] = "physical_reaction_coefficient_u"
            handle.attrs["mode_0"] = "constant one-sigma spatial-mean direction"
            handle.attrs["constant_mean_std"] = float(self.constant_mean_std)
            handle.attrs["provenance_json"] = json.dumps(
                self.provenance,
                sort_keys=True,
            )
            handle.create_dataset("x_grid", data=np.asarray(self.grid.x, dtype=np.float64))
            handle.create_dataset("y_grid", data=np.asarray(self.grid.y, dtype=np.float64))
            handle.create_dataset("unit_modes", data=self.unit_modes.astype(np.float32))
            handle.create_dataset(
                "direction_scales",
                data=self.direction_scales.astype(np.float64),
            )
            handle.create_dataset(
                "directions_u",
                data=self.directions.astype(np.float32),
            )
            handle.create_dataset(
                "explained_variance",
                data=self.explained_variance.astype(np.float64),
            )
            handle.create_dataset(
                "explained_variance_ratio",
                data=self.explained_variance_ratio.astype(np.float64),
            )
            handle.create_dataset(
                "parameter_mean_u",
                data=self.parameter_mean_grid.astype(np.float32),
            )

    @classmethod
    def load(cls, path: str | Path) -> "KLBasis2D":
        with h5py.File(path, "r") as handle:
            stored_format = str(handle.attrs.get("format", KL_FORMAT))
            if stored_format != KL_FORMAT:
                raise ValueError(
                    f"Unsupported KL format {stored_format!r}; expected {KL_FORMAT!r}."
                )
            basis_variable = str(handle.attrs.get("basis_variable", ""))
            if basis_variable not in ("", "physical_reaction_coefficient_u"):
                raise ValueError(
                    f"KL basis is for {basis_variable!r}, not physical u."
                )
            required = (
                "x_grid",
                "y_grid",
                "unit_modes",
                "direction_scales",
                "directions_u",
                "explained_variance",
                "explained_variance_ratio",
                "parameter_mean_u",
            )
            missing = [name for name in required if name not in handle]
            if missing:
                raise ValueError(f"KL basis is missing datasets: {missing}.")
            if "constant_mean_std" not in handle.attrs:
                raise ValueError("KL basis is missing constant_mean_std metadata.")
            grid = RegularGrid2D(
                x=np.asarray(handle["x_grid"][:], dtype=np.float64),
                y=np.asarray(handle["y_grid"][:], dtype=np.float64),
            )
            raw_provenance = handle.attrs.get("provenance_json", "{}")
            if isinstance(raw_provenance, bytes):
                raw_provenance = raw_provenance.decode("utf-8")
            try:
                provenance = json.loads(str(raw_provenance))
            except json.JSONDecodeError as error:
                raise ValueError("KL provenance_json is not valid JSON.") from error
            if not isinstance(provenance, dict):
                raise ValueError("KL provenance_json must encode an object.")
            return cls(
                grid=grid,
                unit_modes=np.asarray(handle["unit_modes"][:], dtype=np.float64),
                direction_scales=np.asarray(
                    handle["direction_scales"][:], dtype=np.float64
                ),
                directions=np.asarray(handle["directions_u"][:], dtype=np.float64),
                explained_variance=np.asarray(
                    handle["explained_variance"][:], dtype=np.float64
                ),
                explained_variance_ratio=np.asarray(
                    handle["explained_variance_ratio"][:], dtype=np.float64
                ),
                parameter_mean_grid=np.asarray(
                    handle["parameter_mean_u"][:], dtype=np.float64
                ),
                constant_mean_std=float(handle.attrs["constant_mean_std"]),
                provenance=provenance,
            )


def build_kl_basis_from_residual_matrix(
    residual_matrix: np.ndarray,
    grid: RegularGrid2D,
    *,
    n_spatial_modes: int,
    parameter_mean_grid: np.ndarray,
    constant_mean_std: float,
    residual_mean: np.ndarray | None = None,
    block_size: int = 128,
    random_seed: int = 0,
    oversampling: int = 8,
    power_iterations: int = 2,
    provenance: dict[str, Any] | None = None,
) -> KLBasis2D:
    """Build a KL basis from an existing residual array or memmap.

    ``residual_matrix`` has shape ``(n_samples, grid.size)`` and each row must
    already have had that sample's scalar spatial mean removed.  The matrix is
    then column-centred for PCA.  ``constant_mean_std`` is the sample standard
    deviation (``ddof=1``) of those removed scalar means.
    """

    n_samples, n_features = _matrix_shape(
        residual_matrix,
        name="residual_matrix",
    )
    if n_features != grid.size:
        raise ValueError(
            f"residual_matrix has {n_features} columns; grid requires {grid.size}."
        )
    n_spatial_modes = _positive_integer(
        n_spatial_modes,
        name="n_spatial_modes",
    )
    block_size = _positive_integer(block_size, name="block_size")
    if n_samples <= n_spatial_modes:
        raise ValueError(
            "Need at least n_spatial_modes+1 samples for empirical variance."
        )
    parameter_mean = np.asarray(parameter_mean_grid, dtype=np.float64)
    if parameter_mean.shape != grid.shape or not np.all(np.isfinite(parameter_mean)):
        raise ValueError(
            f"parameter_mean_grid must be finite with shape {grid.shape}."
        )
    constant_std = float(constant_mean_std)
    if not np.isfinite(constant_std) or constant_std < 0.0:
        raise ValueError("constant_mean_std must be finite and non-negative.")
    centred_mean = (
        blockwise_column_mean(residual_matrix, block_size=block_size)
        if residual_mean is None
        else _column_mean_array(residual_mean, grid.size, name="residual_mean")
    )

    decomposition = randomized_svd_centered(
        residual_matrix,
        n_components=n_spatial_modes,
        column_mean=centred_mean,
        oversampling=oversampling,
        power_iterations=power_iterations,
        block_size=block_size,
        random_seed=random_seed,
    )
    spatial_units = decomposition.right_singular_vectors.reshape(
        (n_spatial_modes, grid.nx, grid.ny)
    )
    spatial_variance = decomposition.explained_variance
    spatial_scales = np.sqrt(np.maximum(spatial_variance, 0.0))

    constant_unit = np.full(
        grid.shape,
        1.0 / np.sqrt(grid.size),
        dtype=np.float64,
    )
    constant_scale = constant_std * np.sqrt(grid.size)
    unit_modes = np.concatenate([constant_unit[None], spatial_units], axis=0)
    direction_scales = np.concatenate([[constant_scale], spatial_scales])
    directions = unit_modes * direction_scales[:, None, None]
    # Preserve the existing Reaction schema semantics: the spatial entries
    # are the empirical PCA variances S**2/(n_samples-1), while entry zero is
    # the variance of the constant-mode coefficient.
    explained_variance = np.concatenate(
        [[constant_scale**2], spatial_variance]
    )
    total = float(np.sum(explained_variance, dtype=np.float64))
    explained_variance_ratio = (
        explained_variance / total
        if total > 0.0
        else np.zeros_like(explained_variance)
    )

    recorded_provenance = {} if provenance is None else dict(provenance)
    recorded_provenance["kl_algorithm"] = {
        "implementation": "numpy_blockwise_randomized_svd",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_spatial_modes": n_spatial_modes,
        "block_size": block_size,
        "random_seed": int(random_seed),
        "oversampling": int(oversampling),
        "power_iterations": int(power_iterations),
        "sketch_size": decomposition.sketch_size,
        "numerical_subspace_rank": decomposition.numerical_subspace_rank,
        "residual_matrix_dtype": str(np.dtype(residual_matrix.dtype)),
    }
    return KLBasis2D(
        grid=grid,
        unit_modes=unit_modes,
        direction_scales=direction_scales,
        directions=directions,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        parameter_mean_grid=parameter_mean,
        constant_mean_std=constant_std,
        provenance=recorded_provenance,
    )


def build_incremental_kl_basis(
    samples: Iterable[np.ndarray],
    grid: RegularGrid2D,
    *,
    n_spatial_modes: int,
    batch_size: int = 128,
    provenance: dict[str, Any] | None = None,
    random_seed: int = 0,
    oversampling: int = 8,
    power_iterations: int = 2,
    work_directory: str | Path | None = None,
    residual_dtype: str | np.dtype = np.float64,
) -> KLBasis2D:
    """Build mean plus residual-PCA directions from a one-pass iterable.

    The iterable is consumed once.  Zero-spatial-mean rows are spooled to a
    temporary memmap, so memory use is bounded by the randomized-SVD sketch
    and ``batch_size`` row blocks rather than the full sample matrix.
    ``batch_size`` retains its existing public role as the processing block
    size; a short final block needs no duplication.
    """

    n_spatial_modes = _positive_integer(
        n_spatial_modes,
        name="n_spatial_modes",
    )
    batch_size = _positive_integer(batch_size, name="batch_size")
    dtype = np.dtype(residual_dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("residual_dtype must be float32 or float64.")
    temporary_parent = None if work_directory is None else Path(work_directory)
    if temporary_parent is not None:
        temporary_parent.mkdir(parents=True, exist_ok=True)

    running_sum = np.zeros(grid.size, dtype=np.float64)
    mean_of_sample_means = 0.0
    sample_mean_m2 = 0.0
    count = 0

    with tempfile.TemporaryDirectory(
        prefix="reaction-kl-",
        dir=temporary_parent,
    ) as temporary_directory:
        residual_path = Path(temporary_directory) / "residuals.bin"
        with residual_path.open("wb") as stream:
            for sample_index, field in enumerate(samples):
                values = np.asarray(field, dtype=np.float64)
                if values.shape != grid.shape:
                    raise ValueError(
                        f"KL sample {sample_index} has shape {values.shape}; "
                        f"expected {grid.shape}."
                    )
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"KL sample {sample_index} is non-finite.")
                flat = values.reshape(-1)
                spatial_mean = float(np.mean(flat, dtype=np.float64))
                residual = flat - spatial_mean
                np.asarray(residual, dtype=dtype).tofile(stream)

                running_sum += flat
                count += 1
                delta = spatial_mean - mean_of_sample_means
                mean_of_sample_means += delta / count
                sample_mean_m2 += delta * (spatial_mean - mean_of_sample_means)

        if count <= n_spatial_modes:
            raise ValueError(
                "Need at least n_spatial_modes+1 samples to construct the basis; "
                f"received {count}."
            )
        residuals = np.memmap(
            residual_path,
            mode="r",
            dtype=dtype,
            shape=(count, grid.size),
            order="C",
        )
        try:
            parameter_mean = (running_sum / count).reshape(grid.shape)
            constant_std = float(np.sqrt(sample_mean_m2 / (count - 1)))
            # mean(residual_i) by feature equals mean(full fields) minus the
            # scalar mean of sample spatial means.  This avoids another pass.
            residual_mean = (
                parameter_mean.reshape(-1) - mean_of_sample_means
                if dtype == np.dtype(np.float64)
                else blockwise_column_mean(residuals, block_size=batch_size)
            )
            basis = build_kl_basis_from_residual_matrix(
                residuals,
                grid,
                n_spatial_modes=n_spatial_modes,
                parameter_mean_grid=parameter_mean,
                constant_mean_std=constant_std,
                residual_mean=residual_mean,
                block_size=batch_size,
                random_seed=random_seed,
                oversampling=oversampling,
                power_iterations=power_iterations,
                provenance=provenance,
            )
        finally:
            # Close the mmap before TemporaryDirectory removes its file.  The
            # private mmap attribute is the documented NumPy cleanup pattern.
            mmap_object = getattr(residuals, "_mmap", None)
            if mmap_object is not None:
                mmap_object.close()
            del residuals
    return basis


__all__ = [
    "KLBasis2D",
    "RandomizedSVDResult",
    "blockwise_column_mean",
    "build_incremental_kl_basis",
    "build_kl_basis_from_residual_matrix",
    "centered_matrix_right_product",
    "centered_matrix_transpose_left_product",
    "randomized_svd_centered",
]
