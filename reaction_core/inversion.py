"""Exact-inversion utilities for the scalar reaction problem.

The inverse variable is the physical coefficient ``u`` itself.  In
particular, none of the functions in this module applies a logarithm,
exponential, or a log-parameter chain rule.

For vertex observations ``P p`` and a trapezoidal objective

    Phi(u) = 0.5 * (P p(u) - d)^T W (P p(u) - d),

the finite-element state adjoint needs the L2 source

    q = M_V^{-1} P^T W (P p(u) - d).

``ScalarVertexObservationAdjoint`` performs that exact discrete lift for
both P1 and P2 scalar state spaces.  After the state adjoint solve, the
solver's ``parameter_coordinate_adjoint`` returns the P1 nodal coordinate
covector.  ``StationaryMaternCovariance2D`` applies the non-periodic
stationary Matérn matrix to that coordinate covector using an exact BTTB
matrix-vector product implemented as zero-padded FFT convolution.

The FEM helpers deliberately require one MPI rank.  Returning a regular-grid
coordinate vector from a distributed finite-element vector requires an
explicit ownership and gather policy and is never done implicitly here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.fft import irfftn, next_fast_len, rfftn

from .grid import RegularGrid2D
from .random_fields import matern_covariance

# The grid weighting/resampling and FFT covariance utilities are intentionally
# usable in a PyTorch-only environment.  Importing DOLFINx at module scope can
# initialise PETSc just for ``--help`` of an emulator driver, so FEM imports
# are deferred until ScalarVertexObservationAdjoint is actually constructed.
ufl: Any | None = None
fem: Any | None = None
petsc: Any | None = None
PETSc: Any | None = None

if TYPE_CHECKING:
    from .grid import (
            StructuredLagrangeStateGridMap,
    )
    from .solver import ReactionSolver2D


FloatArray = np.ndarray
BoolArray = np.ndarray


def _field_array(
    values: np.ndarray,
    shape: tuple[int, int],
    *,
    name: str,
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (1, *shape):
        array = array[0]
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _boolean_mask(
    values: np.ndarray,
    shape: tuple[int, int],
    *,
    name: str,
) -> BoolArray:
    original = np.asarray(values)
    if original.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {original.shape}.")
    if original.dtype != np.bool_:
        if not np.all(np.isfinite(original)):
            raise ValueError(f"{name} contains non-finite values.")
        if not np.all((original == 0) | (original == 1)):
            raise ValueError(f"{name} must contain only Boolean/0/1 values.")
    return np.asarray(original, dtype=bool)


def trapezoid_weights(grid: RegularGrid2D) -> FloatArray:
    """Return tensor-product trapezoidal weights on ``grid``.

    The sum of the returned weights is the area of the rectangular domain.
    Observation masks are kept separate so it remains clear which vertices
    are measured and which factors arise from quadrature.
    """

    wx = np.full(grid.nx, grid.dx, dtype=np.float64)
    wy = np.full(grid.ny, grid.dy, dtype=np.float64)
    wx[[0, -1]] *= 0.5
    wy[[0, -1]] *= 0.5
    return wx[:, None] * wy[None, :]


def interior_vertex_mask(
    grid: RegularGrid2D,
    *,
    boundary_layers: int = 1,
) -> BoolArray:
    """Select vertices away from ``boundary_layers`` grid boundaries."""

    layers = int(boundary_layers)
    if layers < 0 or layers != boundary_layers:
        raise ValueError("boundary_layers must be a non-negative integer.")
    mask = np.ones(grid.shape, dtype=bool)
    if layers == 0:
        return mask
    if 2 * layers >= grid.nx or 2 * layers >= grid.ny:
        raise ValueError(
            f"boundary_layers={layers} leaves no interior of grid {grid.shape}."
        )
    mask[:layers, :] = False
    mask[-layers:, :] = False
    mask[:, :layers] = False
    mask[:, -layers:] = False
    return mask


def _integer_pair(
    value: int | tuple[int, int],
    *,
    name: str,
    minimum: int,
) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        pair = int(value), int(value)
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must be an integer or a pair of integers.")
        pair = int(value[0]), int(value[1])
        if pair != tuple(value):
            raise ValueError(f"{name} must contain integers.")
    if pair[0] < minimum or pair[1] < minimum:
        raise ValueError(f"Both entries of {name} must be at least {minimum}.")
    return pair


def regular_sensor_mask(
    grid: RegularGrid2D,
    *,
    stride: int | tuple[int, int] = 1,
    offset: int | tuple[int, int] = 0,
    interior_only: bool = True,
    boundary_layers: int = 1,
) -> BoolArray:
    """Return a deterministic rectangular sub-sampling mask.

    ``offset`` is an index on the full grid and ``stride`` is the separation
    between sensors.  With the defaults every interior vertex is observed and
    the Dirichlet boundary is excluded.  This makes sparse-sensor experiments
    reproducible without changing any observation/adjoint code.
    """

    stride_x, stride_y = _integer_pair(stride, name="stride", minimum=1)
    offset_x, offset_y = _integer_pair(offset, name="offset", minimum=0)
    if offset_x >= grid.nx or offset_y >= grid.ny:
        raise ValueError(
            f"offset {(offset_x, offset_y)} lies outside grid {grid.shape}."
        )

    mask = np.zeros(grid.shape, dtype=bool)
    mask[offset_x::stride_x, offset_y::stride_y] = True
    if interior_only:
        mask &= interior_vertex_mask(grid, boundary_layers=boundary_layers)
    if not np.any(mask):
        raise ValueError("The requested sensor layout contains no observations.")
    return mask


def weighted_inner(
    left: np.ndarray,
    right: np.ndarray,
    weights: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float:
    """Return the scalar weighted grid inner product.

    A Boolean ``mask`` zeros unobserved entries before summation.  Singleton
    channel arrays ``(1,nx,ny)`` are accepted for convenience and treated as
    scalar fields.
    """

    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.ndim != 2:
        raise ValueError("weights must be a two-dimensional scalar grid.")
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("weights must be finite and non-negative.")
    a = _field_array(left, weight_array.shape, name="left")
    b = _field_array(right, weight_array.shape, name="right")
    effective = weight_array
    if mask is not None:
        effective = effective * _boolean_mask(
            mask,
            weight_array.shape,
            name="mask",
        )
    return float(np.sum(effective * a * b, dtype=np.float64))


def weighted_norm(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float:
    """Return ``sqrt(weighted_inner(values, values))``."""

    squared = weighted_inner(values, values, weights, mask=mask)
    return float(np.sqrt(max(squared, 0.0)))


def weighted_residual_covector(
    residual: np.ndarray,
    weights: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> FloatArray:
    """Return ``W residual`` (and zero unobserved entries) on the grid."""

    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.ndim != 2:
        raise ValueError("weights must be a two-dimensional scalar grid.")
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("weights must be finite and non-negative.")
    field = _field_array(residual, weight_array.shape, name="residual")
    output = weight_array * field
    if mask is not None:
        output = output * _boolean_mask(mask, weight_array.shape, name="mask")
    return np.asarray(output, dtype=np.float64)


@dataclass(frozen=True)
class ScalarNoiseRealization:
    """One masked scalar heteroscedastic measurement-noise realization."""

    observed: FloatArray
    noise: FloatArray
    standard_deviation: FloatArray
    observation_mask: BoolArray
    weighted_rms: float
    diagnostics: dict[str, float | int]

    @property
    def noise_std(self) -> FloatArray:
        return self.standard_deviation

    @property
    def expected_noise_norm(self) -> float:
        """Weighted RMS expectation ``sqrt(sum_i w_i sigma_i^2)``."""

        return float(self.diagnostics["weighted_expected_noise_norm"])

    @property
    def realised_noise_norm(self) -> float:
        """Weighted norm of the particular generated noise realization."""

        return float(self.diagnostics["weighted_realised_noise_norm"])

    # US spelling for callers which use ``realization`` elsewhere.
    realized_noise_norm = realised_noise_norm


def add_scalar_heteroscedastic_noise(
    noiseless_data: np.ndarray,
    *,
    weights: np.ndarray,
    observation_mask: np.ndarray,
    noise_percent: float,
    floor_fraction: float = 0.05,
    seed: int = 0,
    exclude_boundary: bool = True,
) -> ScalarNoiseRealization:
    r"""Add independent entrywise Gaussian noise at observed interior nodes.

    At an observed entry ``i``, the standard deviation is

    ``noise_percent/100 * max(abs(data_i), floor_fraction * weighted_RMS)``.

    Noise, its standard deviation, and returned data are exactly zero at all
    unobserved entries.  With ``exclude_boundary=True`` (the default), all
    four edges are removed even if the supplied mask selected them.  This is
    important for the exactly-zero Dirichlet data: a noise floor must not turn
    the known boundary condition into synthetic measurements.
    """

    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.ndim != 2:
        raise ValueError("weights must be a two-dimensional scalar grid.")
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("weights must be finite and non-negative.")
    data = _field_array(noiseless_data, weight_array.shape, name="noiseless_data")
    selected = _boolean_mask(
        observation_mask,
        weight_array.shape,
        name="observation_mask",
    ).copy()
    if exclude_boundary:
        selected[[0, -1], :] = False
        selected[:, [0, -1]] = False
    if not np.any(selected):
        raise ValueError("The effective observation mask is empty.")

    percent = float(noise_percent)
    floor = float(floor_fraction)
    if not np.isfinite(percent) or percent < 0.0:
        raise ValueError("noise_percent must be finite and non-negative.")
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("floor_fraction must be finite and non-negative.")

    effective_weights = weight_array * selected
    observed_area = float(np.sum(effective_weights, dtype=np.float64))
    if observed_area <= 0.0:
        raise ValueError("Observed quadrature weights must have positive sum.")
    rms = float(
        np.sqrt(
            np.sum(effective_weights * data * data, dtype=np.float64)
            / observed_area
        )
    )

    standard_deviation = np.zeros_like(data)
    noise_fraction = percent / 100.0
    standard_deviation[selected] = noise_fraction * np.maximum(
        np.abs(data[selected]),
        floor * rms,
    )
    noise = np.zeros_like(data)
    if noise_fraction > 0.0:
        rng = np.random.default_rng(int(seed))
        noise[selected] = (
            rng.standard_normal(int(np.count_nonzero(selected)))
            * standard_deviation[selected]
        )

    observed = np.zeros_like(data)
    observed[selected] = data[selected] + noise[selected]
    noiseless_norm = weighted_norm(data, weight_array, mask=selected)
    noise_norm = weighted_norm(noise, weight_array, mask=selected)
    expected_noise_norm = float(
        np.sqrt(
            np.sum(
                effective_weights * standard_deviation * standard_deviation,
                dtype=np.float64,
            )
        )
    )
    diagnostics: dict[str, float | int] = {
        "noise_percent": percent,
        "noise_fraction": noise_fraction,
        "noise_floor_fraction": floor,
        "noise_seed": int(seed),
        "number_of_observations": int(np.count_nonzero(selected)),
        "weighted_observed_area": observed_area,
        "weighted_rms": rms,
        "weighted_noiseless_norm": noiseless_norm,
        "weighted_expected_noise_norm": expected_noise_norm,
        "weighted_realised_noise_norm": noise_norm,
        "weighted_noise_norm": noise_norm,
        "weighted_relative_noise_norm": float(
            noise_norm / max(noiseless_norm, 1.0e-30)
        ),
    }
    return ScalarNoiseRealization(
        observed=observed,
        noise=noise,
        standard_deviation=standard_deviation,
        observation_mask=selected,
        weighted_rms=rms,
        diagnostics=diagnostics,
    )


def _check_resampling_domains(
    source_grid: RegularGrid2D,
    target_grid: RegularGrid2D,
) -> None:
    coordinate_scale = max(
        1.0,
        abs(float(source_grid.x[0])),
        abs(float(source_grid.x[-1])),
        abs(float(source_grid.y[0])),
        abs(float(source_grid.y[-1])),
    )
    tolerance = 64.0 * np.finfo(np.float64).eps * coordinate_scale
    if (
        target_grid.x[0] < source_grid.x[0] - tolerance
        or target_grid.x[-1] > source_grid.x[-1] + tolerance
        or target_grid.y[0] < source_grid.y[0] - tolerance
        or target_grid.y[-1] > source_grid.y[-1] + tolerance
    ):
        raise ValueError(
            "The target grid extends outside the source grid; extrapolation "
            "is intentionally unsupported."
        )


def resample_regular_grid_field(
    values: np.ndarray,
    source_grid: RegularGrid2D,
    target_grid: RegularGrid2D,
) -> FloatArray:
    """Bilinearly resample one scalar tensor-product-grid field."""

    field = _field_array(values, source_grid.shape, name="values")
    _check_resampling_domains(source_grid, target_grid)

    intermediate = np.empty(
        (source_grid.nx, target_grid.ny),
        dtype=np.float64,
    )
    for i in range(source_grid.nx):
        intermediate[i] = np.interp(
            target_grid.y,
            source_grid.y,
            field[i],
        )

    sampled = np.empty(target_grid.shape, dtype=np.float64)
    for j in range(target_grid.ny):
        sampled[:, j] = np.interp(
            target_grid.x,
            source_grid.x,
            intermediate[:, j],
        )
    return sampled


def resample_regular_grid_fields(
    values: np.ndarray,
    source_grid: RegularGrid2D,
    target_grid: RegularGrid2D,
) -> FloatArray:
    """Bilinearly resample an array with trailing ``(nx,ny)`` dimensions."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[-2:] != source_grid.shape:
        raise ValueError(
            "values must have trailing source-grid shape "
            f"{source_grid.shape}; got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("values contains non-finite entries.")
    leading_shape = array.shape[:-2]
    flattened = array.reshape((-1, *source_grid.shape))
    result = np.stack(
        [
            resample_regular_grid_field(field, source_grid, target_grid)
            for field in flattened
        ],
        axis=0,
    )
    return result.reshape((*leading_shape, *target_grid.shape))


def explicit_dense_matern_covariance(
    grid: RegularGrid2D,
    *,
    nu: float,
    length_scale: float,
    variance: float = 1.0,
    max_points: int = 4096,
) -> FloatArray:
    """Build the exact dense nodal Matérn matrix for a small-grid check."""

    if grid.size > int(max_points):
        raise ValueError(
            f"Dense validation is limited to {max_points} points; "
            f"the grid has {grid.size}."
        )
    coordinates = grid.points()
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    return matern_covariance(
        distances,
        nu=float(nu),
        length_scale=float(length_scale),
        variance=float(variance),
    )


class IdentityCovariance2D:
    """Coordinate-space identity preconditioner for controlled ablations.

    The class deliberately exposes the small interface used by
    :class:`StationaryMaternCovariance2D`, allowing the Landweber iteration to
    remain unchanged.  ``apply_coordinate_covector`` always returns a copy so
    callers may safely mutate the resulting search direction.
    """

    def __init__(self, grid: RegularGrid2D, **_ignored: Any) -> None:
        self.grid = grid
        self._closed = False

    @property
    def approximate_workspace_megabytes(self) -> float:
        return 0.0

    def apply_coordinate_covector(
        self,
        coordinate_covector: np.ndarray,
    ) -> FloatArray:
        """Return a finite ``float64`` copy of the coordinate covector."""

        if self._closed:
            raise RuntimeError("The identity covariance operator is closed.")
        vector = _field_array(
            coordinate_covector,
            self.grid.shape,
            name="coordinate_covector",
        )
        return np.array(vector, dtype=np.float64, copy=True)

    apply = apply_coordinate_covector

    def close(self) -> None:
        self._closed = True

    destroy = close

    def __enter__(self) -> "IdentityCovariance2D":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class StationaryMaternCovariance2D:
    r"""Exact non-periodic nodal Matérn action via zero-padded FFTs.

    The implicit BTTB matrix is

    ``K[(i,j),(k,l)] = matern_covariance(distance((i,j),(k,l)))``.

    ``apply_coordinate_covector(q)`` returns ``K q``.  No quadrature weights
    are built into ``K``: the input must be the Euclidean nodal coordinate
    covector, e.g. the array returned by the solver after the observation
    quadrature has already been lifted through the state adjoint.

    The Toeplitz action is a *linear* convolution.  Zero padding to at least
    the full convolution size prevents periodic wrap-around, so opposite
    boundaries of the unit square are not treated as neighbours.
    """

    def __init__(
        self,
        grid: RegularGrid2D,
        *,
        nu: float,
        length_scale: float,
        variance: float = 1.0,
        fft_workers: int | None = None,
    ) -> None:
        self.grid = grid
        self.nu = float(nu)
        self.length_scale = float(length_scale)
        self.variance = float(variance)
        self.fft_workers = fft_workers
        if self.nu <= 0.0 or not np.isfinite(self.nu):
            raise ValueError("nu must be finite and positive.")
        if self.length_scale <= 0.0 or not np.isfinite(self.length_scale):
            raise ValueError("length_scale must be finite and positive.")
        if self.variance < 0.0 or not np.isfinite(self.variance):
            raise ValueError("variance must be finite and non-negative.")
        if fft_workers is not None and int(fft_workers) == 0:
            raise ValueError("fft_workers must be None or a non-zero integer.")

        lag_x = np.arange(-(grid.nx - 1), grid.nx, dtype=np.float64) * grid.dx
        lag_y = np.arange(-(grid.ny - 1), grid.ny, dtype=np.float64) * grid.dy
        delta_x, delta_y = np.meshgrid(lag_x, lag_y, indexing="ij")
        kernel = matern_covariance(
            np.hypot(delta_x, delta_y),
            nu=self.nu,
            length_scale=self.length_scale,
            variance=self.variance,
        )

        # q has shape (nx,ny), kernel has (2nx-1,2ny-1), hence the full
        # linear convolution has (3nx-2,3ny-2) entries.
        linear_shape = (3 * grid.nx - 2, 3 * grid.ny - 2)
        self.fft_shape = tuple(next_fast_len(size) for size in linear_shape)
        self._output_slices = (
            slice(grid.nx - 1, 2 * grid.nx - 1),
            slice(grid.ny - 1, 2 * grid.ny - 1),
        )
        self._kernel_spectrum = rfftn(
            kernel,
            s=self.fft_shape,
            workers=self.fft_workers,
        )
        self._closed = False

    @property
    def approximate_workspace_megabytes(self) -> float:
        if self._closed or self._kernel_spectrum is None:
            return 0.0
        return float(self._kernel_spectrum.nbytes / 1024.0**2)

    def apply_coordinate_covector(self, coordinate_covector: np.ndarray) -> FloatArray:
        """Return ``K q`` in the same ``(x-index,y-index)`` layout as ``q``."""

        if self._closed or self._kernel_spectrum is None:
            raise RuntimeError("The Matérn covariance operator is closed.")
        vector = _field_array(
            coordinate_covector,
            self.grid.shape,
            name="coordinate_covector",
        )
        vector_spectrum = rfftn(
            vector,
            s=self.fft_shape,
            workers=self.fft_workers,
        )
        full_convolution = irfftn(
            vector_spectrum * self._kernel_spectrum,
            s=self.fft_shape,
            workers=self.fft_workers,
        )
        result = full_convolution[self._output_slices]
        if result.shape != self.grid.shape:
            raise RuntimeError(
                f"Internal covariance slice has shape {result.shape}, "
                f"expected {self.grid.shape}."
            )
        if not np.all(np.isfinite(result)):
            raise RuntimeError("The Matérn covariance action is non-finite.")
        return np.asarray(result, dtype=np.float64).copy()

    apply = apply_coordinate_covector

    def validate_against_dense(
        self,
        *,
        seed: int = 20260818,
        trials: int = 3,
        rtol: float = 2.0e-12,
        atol: float = 2.0e-12,
        max_points: int = 4096,
        raise_on_failure: bool = True,
    ) -> dict[str, float | int | bool]:
        """Compare the FFT action with an explicit dense small-grid matrix."""

        if trials < 1:
            raise ValueError("trials must be at least one.")
        dense = explicit_dense_matern_covariance(
            self.grid,
            nu=self.nu,
            length_scale=self.length_scale,
            variance=self.variance,
            max_points=max_points,
        )
        rng = np.random.default_rng(int(seed))
        maximum_absolute_error = 0.0
        maximum_relative_error = 0.0
        for _ in range(int(trials)):
            coordinate_covector = rng.standard_normal(self.grid.shape)
            expected = (
                dense @ coordinate_covector.reshape(-1, order="C")
            ).reshape(self.grid.shape, order="C")
            actual = self.apply_coordinate_covector(coordinate_covector)
            error = np.linalg.norm((actual - expected).reshape(-1))
            reference = np.linalg.norm(expected.reshape(-1))
            maximum_absolute_error = max(
                maximum_absolute_error,
                float(np.max(np.abs(actual - expected))),
            )
            maximum_relative_error = max(
                maximum_relative_error,
                float(error / max(reference, 1.0e-30)),
            )

        threshold = float(atol) + float(rtol) * float(np.max(np.abs(dense)))
        passed = bool(
            maximum_absolute_error <= threshold
            or maximum_relative_error <= float(rtol)
        )
        diagnostics: dict[str, float | int | bool] = {
            "number_of_points": int(self.grid.size),
            "trials": int(trials),
            "maximum_absolute_error": maximum_absolute_error,
            "maximum_relative_l2_error": maximum_relative_error,
            "absolute_threshold": threshold,
            "passed": passed,
        }
        if raise_on_failure and not passed:
            raise AssertionError(
                "FFT/BTTB Matérn action did not match its explicit dense "
                f"matrix: {diagnostics}."
            )
        return diagnostics

    def close(self) -> None:
        self._kernel_spectrum = None
        self._closed = True

    destroy = close

    def __enter__(self) -> "StationaryMaternCovariance2D":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def validate_matern_covariance_action(
    *,
    nx: int = 7,
    ny: int = 6,
    nu: float = 1.5,
    length_scale: float = 0.2,
    variance: float = 9.0,
    seed: int = 20260818,
) -> dict[str, float | int | bool]:
    """Run the explicit dense self-check on a deliberately non-square grid."""

    grid = RegularGrid2D.from_domain(1.0, 0.8, int(nx), int(ny))
    with StationaryMaternCovariance2D(
        grid,
        nu=nu,
        length_scale=length_scale,
        variance=variance,
    ) as covariance:
        return covariance.validate_against_dense(seed=seed)


def _require_fem() -> None:
    """Load exact-FEM dependencies only for the vertex-adjoint lift."""

    global PETSc, fem, petsc, ufl
    if fem is not None and petsc is not None and PETSc is not None and ufl is not None:
        return
    try:
        import ufl as imported_ufl
        from dolfinx import fem as imported_fem
        from dolfinx.fem import petsc as imported_petsc
        from petsc4py import PETSc as imported_petsc_class
    except ImportError as exc:
        raise ImportError(
            "DOLFINx, UFL, petsc4py and mpi4py are required for the exact "
            "vertex-observation adjoint lift."
        ) from exc
    ufl = imported_ufl
    fem = imported_fem
    petsc = imported_petsc
    PETSc = imported_petsc_class


def _function_petsc_vector(function: Any):
    vector = getattr(function.x, "petsc_vec", None)
    if vector is not None:
        return vector
    legacy = getattr(function, "vector", None)
    if legacy is None:
        raise RuntimeError("Could not access the PETSc vector of a Function.")
    return legacy


class ScalarVertexObservationAdjoint:
    r"""Lift scalar vertex coordinate covectors into the state L2 space.

    For a scalar P1 or P2 state and its vertex extraction operator ``P``, the
    lift solves

    ``M_V q = P^T c``.

    Thus, if ``c = W * mask * residual``, calling ``solver.adjoint(q)`` uses
    exactly the transpose of the discrete trapezoidal observation objective.
    In P2, the right-hand side entries at all edge-midpoint DOFs are zero;
    their values in ``q`` are determined by the full P2 mass solve.
    """

    def __init__(
        self,
        solver: "ReactionSolver2D",
        state_map: "StructuredLagrangeStateGridMap",
        *,
        solver_options: dict[str, Any] | None = None,
    ) -> None:
        """Create the exact vertex-coordinate to L2 mass lift.

        Parameters
        ----------
        solver_options:
            PETSc KSP/PC options for the mass solve.  The default is the
            historical direct solve, ``{"ksp_type": "preonly", "pc_type":
            "lu"}``.  Passing the same option dictionary as
            :class:`reaction_core.solver.ReactionSolver2D` therefore permits,
            for example, ``cg`` with ``hypre`` on large serial meshes.  The
            options are installed with the solver's prefixed PETSc-option
            mechanism, so they do not leak into other KSP instances.
        """
        _require_fem()
        if solver.comm.size != 1:
            raise RuntimeError(
                "ScalarVertexObservationAdjoint requires one MPI rank; "
                "distributed vertex coordinates need an explicit gather policy."
            )
        if state_map.solver is not solver:
            raise ValueError("state_map was constructed for a different solver.")
        self.solver = solver
        self.state_map = state_map
        self.grid = state_map.grid
        self._closed = False
        self._solve_count = 0
        self.solver_options = dict(
            solver_options
            or {
                "ksp_type": "preonly",
                "pc_type": "lu",
            }
        )
        self._last_mass_ksp_reason: int | None = None
        self._last_mass_ksp_iterations: int | None = None

        trial = ufl.TrialFunction(solver.V)
        test = ufl.TestFunction(solver.V)
        self._mass_form = fem.form(ufl.inner(trial, test) * ufl.dx)
        self._mass_matrix = petsc.assemble_matrix(self._mass_form)
        self._mass_matrix.assemble()
        self._mass_ksp = PETSc.KSP().create(solver.comm)
        self._mass_ksp.setOptionsPrefix(
            f"reaction_vertex_observation_mass_{id(self):x}_"
        )
        self._mass_ksp.setOperators(self._mass_matrix)
        # Use the exact same scoped PETSc-option setup as the cached reaction
        # state solver.  In particular this lets one Landweber run choose a
        # compatible CG+HYPRE backend for both the state and this mass lift.
        solver._apply_petsc_options(self._mass_ksp, self.solver_options)
        self._mass_ksp.setUp()

    def lift_coordinate_covector(
        self,
        coordinate_covector: np.ndarray,
        *,
        name: str = "vertex_observation_L2_source",
        out=None,
    ):
        """Return ``q`` satisfying ``M_V q = P^T coordinate_covector``."""

        if self._closed or self._mass_matrix is None or self._mass_ksp is None:
            raise RuntimeError("The vertex-observation adjoint lift is closed.")
        values = _field_array(
            coordinate_covector,
            self.grid.shape,
            name="coordinate_covector",
        )
        state_dofs = np.asarray(
            self.state_map.flat_to_state_dof,
            dtype=np.int64,
        )
        if state_dofs.size != self.grid.size:
            raise RuntimeError(
                "The state-map vertex permutation has the wrong size: "
                f"{state_dofs.size} rather than {self.grid.size}."
            )

        rhs = self._mass_matrix.createVecRight()
        rhs.set(0.0)
        try:
            rhs_values = rhs.getArray()
            try:
                if np.max(state_dofs, initial=-1) >= rhs_values.size:
                    raise RuntimeError(
                        "A vertex state-DOF index lies outside the owned PETSc "
                        "vector. This lift is intentionally serial."
                    )
                rhs_values[state_dofs] = values.reshape(-1, order="C")
            finally:
                del rhs_values

            output = out if out is not None else self.solver.new_state_function()
            if output.x.array.shape != self.solver.new_state_function().x.array.shape:
                raise ValueError("out does not belong to the solver state space.")
            output.name = name
            solution = _function_petsc_vector(output)
            solution.set(0.0)
            self._mass_ksp.solve(rhs, solution)
            reason = int(self._mass_ksp.getConvergedReason())
            self._last_mass_ksp_reason = reason
            self._last_mass_ksp_iterations = int(
                self._mass_ksp.getIterationNumber()
            )
            if reason <= 0:
                raise RuntimeError(
                    "The state mass solve failed while lifting the vertex "
                    f"coordinate covector: KSP reason={reason}."
                )
            output.x.scatter_forward()
        finally:
            rhs.destroy()
        self._solve_count += 1
        return output

    def lift_weighted_residual(
        self,
        residual: np.ndarray,
        weights: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        name: str = "weighted_vertex_residual_L2_source",
        out=None,
    ):
        """Convenience lift of ``W * mask * residual``."""

        coordinate_covector = weighted_residual_covector(
            residual,
            weights,
            mask=mask,
        )
        return self.lift_coordinate_covector(
            coordinate_covector,
            name=name,
            out=out,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "closed": bool(self._closed),
            "mass_solves": int(self._solve_count),
            "state_degree": int(self.state_map.state_degree),
            "number_of_vertex_observations": int(self.grid.size),
            "mass_solver_options": dict(self.solver_options),
            "last_mass_ksp_reason": self._last_mass_ksp_reason,
            "last_mass_ksp_iterations": self._last_mass_ksp_iterations,
        }

    def close(self) -> None:
        if self._mass_ksp is not None:
            self._mass_ksp.destroy()
            self._mass_ksp = None
        if self._mass_matrix is not None:
            self._mass_matrix.destroy()
            self._mass_matrix = None
        self._closed = True

    destroy = close

    def __enter__(self) -> "ScalarVertexObservationAdjoint":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # PETSc may already be finalised during interpreter shutdown.
            pass


# Compatibility with the name used by the current exact MRE implementation.
TrapezoidalObservationAdjoint = ScalarVertexObservationAdjoint


__all__ = [
    "IdentityCovariance2D",
    "ScalarNoiseRealization",
    "ScalarVertexObservationAdjoint",
    "StationaryMaternCovariance2D",
    "TrapezoidalObservationAdjoint",
    "add_scalar_heteroscedastic_noise",
    "explicit_dense_matern_covariance",
    "interior_vertex_mask",
    "regular_sensor_mask",
    "resample_regular_grid_field",
    "resample_regular_grid_fields",
    "trapezoid_weights",
    "validate_matern_covariance_action",
    "weighted_inner",
    "weighted_norm",
    "weighted_residual_covector",
]


if __name__ == "__main__":
    import json

    print(json.dumps(validate_matern_covariance_action(), indent=2))
