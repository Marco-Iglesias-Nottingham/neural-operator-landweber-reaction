#!/usr/bin/env python3
"""Stationary Matérn fields on a regular 2D grid via circulant embedding.

This is the reaction-project copy of the validated MRE sampler.  A covariance
spectrum is built once for each ``(nu, length_scale)`` pair and subsequent
samples require one forward and one inverse FFT.  Cropping the larger periodic
embedding gives the requested non-periodic covariance on the retained domain;
opposite sides of the unit square are not treated as neighbours.

The covariance convention is

    C(r) = variance * 2**(1-nu) / Gamma(nu) * z**nu K_nu(z),
    z = sqrt(2*nu) * r / length_scale.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import fft2, ifft2
from scipy.special import gammaln, kve


FloatArray = NDArray[np.float64]


def _as_uniform_grid(values: ArrayLike, name: str) -> tuple[FloatArray, float]:
    """Validate a uniform increasing grid and return a float64 canonical copy."""
    original = np.asarray(values)
    if original.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if original.size < 2:
        raise ValueError(f"{name} must contain at least two points.")

    grid = np.asarray(original, dtype=np.float64)
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} contains non-finite values.")
    increments = np.diff(grid)
    if np.any(increments <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")

    spacing = float((grid[-1] - grid[0]) / (grid.size - 1))
    canonical = grid[0] + spacing * np.arange(grid.size, dtype=np.float64)
    if np.issubdtype(original.dtype, np.floating):
        source_epsilon = float(np.finfo(original.dtype).eps)
    else:
        source_epsilon = float(np.finfo(np.float64).eps)
    coordinate_scale = max(
        1.0,
        float(np.max(np.abs(grid))),
        abs(float(grid[-1] - grid[0])),
    )
    if not np.allclose(
        grid,
        canonical,
        rtol=max(32.0 * source_epsilon, 1.0e-11),
        atol=max(32.0 * source_epsilon * coordinate_scale, 1.0e-13),
    ):
        maximum_deviation = float(np.max(np.abs(grid - canonical)))
        raise ValueError(
            f"{name} must be uniformly spaced; maximum deviation is "
            f"{maximum_deviation:.6e}."
        )
    return canonical, spacing


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive.")
    return 1 << (int(value) - 1).bit_length()


def matern_covariance(
    distance: ArrayLike,
    nu: float,
    length_scale: float,
    variance: float = 1.0,
) -> FloatArray:
    """Evaluate the isotropic Matérn covariance in the project convention."""
    radius = np.asarray(distance, dtype=np.float64)
    nu = float(nu)
    length_scale = float(length_scale)
    variance = float(variance)
    if nu <= 0.0:
        raise ValueError("nu must be positive.")
    if length_scale <= 0.0:
        raise ValueError("length_scale must be positive.")
    if variance < 0.0:
        raise ValueError("variance must be non-negative.")
    if np.any(radius < 0.0):
        raise ValueError("distance must be non-negative.")
    if variance == 0.0:
        return np.zeros_like(radius, dtype=np.float64)

    covariance = np.empty_like(radius, dtype=np.float64)
    zero = radius == 0.0
    covariance[zero] = variance
    if np.any(~zero):
        z = np.sqrt(2.0 * nu) * radius[~zero] / length_scale
        log_prefactor = (
            np.log(variance)
            + (1.0 - nu) * np.log(2.0)
            - gammaln(nu)
            + nu * np.log(z)
            - z
        )
        with np.errstate(over="ignore", under="ignore", invalid="raise"):
            covariance[~zero] = np.exp(log_prefactor) * kve(nu, z)
    return covariance


@dataclass(frozen=True)
class EmbeddingDiagnostics:
    requested_shape: tuple[int, int]
    embedding_shape: tuple[int, int]
    attempts: int
    minimum_raw_eigenvalue: float
    maximum_raw_eigenvalue: float
    relative_minimum_eigenvalue: float
    clipped_eigenvalue_count: int
    approximate_spectrum_megabytes: float


class MaternCirculantEmbedding2D:
    """Reusable sampler for one fixed Matérn parameter pair."""

    def __init__(
        self,
        x_grid: ArrayLike,
        y_grid: ArrayLike,
        nu: float,
        length_scale: float,
        variance: float = 1.0,
        mean: float = 0.0,
        *,
        initial_embedding_shape: tuple[int, int] | None = None,
        max_embedding_attempts: int = 7,
        eigenvalue_tolerance: float = 1.0e-12,
        fft_workers: int | None = None,
    ) -> None:
        self.x_grid, self.dx = _as_uniform_grid(x_grid, "x_grid")
        self.y_grid, self.dy = _as_uniform_grid(y_grid, "y_grid")
        self.nx = int(self.x_grid.size)
        self.ny = int(self.y_grid.size)
        self.nu = float(nu)
        self.length_scale = float(length_scale)
        self.variance = float(variance)
        self.mean = float(mean)
        self.fft_workers = fft_workers

        if self.nu <= 0.0:
            raise ValueError("nu must be positive.")
        if self.length_scale <= 0.0:
            raise ValueError("length_scale must be positive.")
        if self.variance < 0.0:
            raise ValueError("variance must be non-negative.")
        if max_embedding_attempts < 1:
            raise ValueError("max_embedding_attempts must be at least one.")
        if eigenvalue_tolerance < 0.0:
            raise ValueError("eigenvalue_tolerance must be non-negative.")

        minimum_mx = 2 * (self.nx - 1)
        minimum_my = 2 * (self.ny - 1)
        if initial_embedding_shape is None:
            mx = _next_power_of_two(minimum_mx)
            my = _next_power_of_two(minimum_my)
        else:
            mx, my = map(int, initial_embedding_shape)
            if mx < minimum_mx or my < minimum_my:
                raise ValueError(
                    "initial_embedding_shape must be at least "
                    f"({minimum_mx}, {minimum_my})."
                )

        self._sqrt_spectrum: FloatArray | None = None
        self.diagnostics: EmbeddingDiagnostics | None = None
        for attempt in range(1, max_embedding_attempts + 1):
            raw_spectrum = self._embedding_spectrum(mx, my)
            maximum = float(np.max(raw_spectrum))
            minimum = float(np.min(raw_spectrum))
            scale = max(abs(maximum), self.variance, 1.0)
            tolerance = eigenvalue_tolerance * scale
            if minimum >= -tolerance:
                clipped = int(np.count_nonzero(raw_spectrum < 0.0))
                self.mx = int(mx)
                self.my = int(my)
                self._sqrt_spectrum = np.sqrt(np.maximum(raw_spectrum, 0.0))
                self.diagnostics = EmbeddingDiagnostics(
                    requested_shape=(self.nx, self.ny),
                    embedding_shape=(self.mx, self.my),
                    attempts=attempt,
                    minimum_raw_eigenvalue=minimum,
                    maximum_raw_eigenvalue=maximum,
                    relative_minimum_eigenvalue=minimum / scale,
                    clipped_eigenvalue_count=clipped,
                    approximate_spectrum_megabytes=(
                        self._sqrt_spectrum.nbytes / 1024.0**2
                    ),
                )
                break
            mx *= 2
            my *= 2

        if self._sqrt_spectrum is None or self.diagnostics is None:
            raise RuntimeError(
                "No non-negative circulant embedding was found after "
                f"{max_embedding_attempts} attempts."
            )

    @property
    def embedding_shape(self) -> tuple[int, int]:
        return self.mx, self.my

    def _embedding_spectrum(self, mx: int, my: int) -> FloatArray:
        index_x = np.arange(mx, dtype=np.float64)
        index_y = np.arange(my, dtype=np.float64)
        distance_x = np.minimum(index_x, mx - index_x) * self.dx
        distance_y = np.minimum(index_y, my - index_y) * self.dy
        lag_x, lag_y = np.meshgrid(distance_x, distance_y, indexing="ij")
        covariance_column = matern_covariance(
            np.hypot(lag_x, lag_y),
            nu=self.nu,
            length_scale=self.length_scale,
            variance=self.variance,
        )
        spectrum = fft2(covariance_column, workers=self.fft_workers)
        imaginary_scale = float(np.max(np.abs(spectrum.imag)))
        real_scale = max(float(np.max(np.abs(spectrum.real))), 1.0)
        if imaginary_scale > 1.0e-10 * real_scale:
            raise RuntimeError(
                "The circulant covariance FFT has an unexpectedly large "
                "imaginary component."
            )
        return np.asarray(spectrum.real, dtype=np.float64)

    def sample(
        self,
        seed: int | None = None,
        *,
        rng: np.random.Generator | None = None,
        dtype: np.dtype | type = np.float64,
    ) -> np.ndarray:
        if seed is not None and rng is not None:
            raise ValueError("Supply either seed or rng, not both.")
        generator = np.random.default_rng(seed) if rng is None else rng
        white_noise = generator.standard_normal((self.mx, self.my))
        white_spectrum = fft2(white_noise, workers=self.fft_workers)
        embedded = ifft2(
            self._sqrt_spectrum * white_spectrum,
            workers=self.fft_workers,
        ).real
        field = embedded[: self.nx, : self.ny] + self.mean
        return np.asarray(field, dtype=dtype)

    def sample_batch(
        self,
        n_samples: int,
        seed: int | None = None,
        *,
        dtype: np.dtype | type = np.float32,
    ) -> np.ndarray:
        n_samples = int(n_samples)
        if n_samples < 1:
            raise ValueError("n_samples must be positive.")
        rng = np.random.default_rng(seed)
        result = np.empty((n_samples, self.nx, self.ny), dtype=dtype)
        for index in range(n_samples):
            result[index] = self.sample(rng=rng, dtype=dtype)
        return result


def make_length_scale_bank(
    minimum: float,
    maximum: float,
    count: int,
    *,
    spacing: str = "log",
) -> FloatArray:
    minimum = float(minimum)
    maximum = float(maximum)
    count = int(count)
    if minimum <= 0.0 or maximum <= 0.0:
        raise ValueError("Length scales must be positive.")
    if maximum < minimum:
        raise ValueError("maximum must be at least minimum.")
    if count < 1:
        raise ValueError("count must be positive.")
    if count == 1:
        return np.asarray([minimum], dtype=np.float64)
    if spacing == "log":
        return np.geomspace(minimum, maximum, count, dtype=np.float64)
    if spacing == "linear":
        return np.linspace(minimum, maximum, count, dtype=np.float64)
    raise ValueError("spacing must be 'log' or 'linear'.")


class MaternSamplerBank2D:
    """Lazy LRU cache of samplers for a finite Matérn parameter bank."""

    def __init__(
        self,
        x_grid: ArrayLike,
        y_grid: ArrayLike,
        nu_values: Iterable[float],
        length_scale_values: Iterable[float],
        variance: float = 1.0,
        mean: float = 0.0,
        *,
        precompute: bool = False,
        max_cached_samplers: int | None = None,
        initial_embedding_shape: tuple[int, int] | None = None,
        max_embedding_attempts: int = 7,
        eigenvalue_tolerance: float = 1.0e-12,
        fft_workers: int | None = None,
    ) -> None:
        self.x_grid, _ = _as_uniform_grid(x_grid, "x_grid")
        self.y_grid, _ = _as_uniform_grid(y_grid, "y_grid")
        self.nu_values = np.unique(np.asarray(list(nu_values), dtype=np.float64))
        self.length_scale_values = np.unique(
            np.asarray(list(length_scale_values), dtype=np.float64)
        )
        if self.nu_values.size == 0 or np.any(self.nu_values <= 0.0):
            raise ValueError("nu_values must contain positive values.")
        if (
            self.length_scale_values.size == 0
            or np.any(self.length_scale_values <= 0.0)
        ):
            raise ValueError("length_scale_values must contain positive values.")
        if variance < 0.0:
            raise ValueError("variance must be non-negative.")
        if max_cached_samplers is not None and max_cached_samplers < 1:
            raise ValueError("max_cached_samplers must be positive or None.")

        self.variance = float(variance)
        self.mean = float(mean)
        self.initial_embedding_shape = initial_embedding_shape
        self.max_embedding_attempts = int(max_embedding_attempts)
        self.eigenvalue_tolerance = float(eigenvalue_tolerance)
        self.fft_workers = fft_workers
        self.max_cached_samplers = max_cached_samplers
        self._cache: OrderedDict[
            tuple[float, float], MaternCirculantEmbedding2D
        ] = OrderedDict()
        if precompute:
            self.precompute_all()

    def _canonical_key(
        self,
        nu: float,
        length_scale: float,
        *,
        nearest: bool,
    ) -> tuple[float, float]:
        nu = float(nu)
        length_scale = float(length_scale)
        if nearest:
            return (
                float(self.nu_values[np.argmin(np.abs(self.nu_values - nu))]),
                float(
                    self.length_scale_values[
                        np.argmin(np.abs(self.length_scale_values - length_scale))
                    ]
                ),
            )
        nu_matches = np.isclose(self.nu_values, nu, rtol=1.0e-12, atol=1.0e-14)
        ell_matches = np.isclose(
            self.length_scale_values,
            length_scale,
            rtol=1.0e-12,
            atol=1.0e-14,
        )
        if not np.any(nu_matches):
            raise KeyError(f"nu={nu} is not in the sampler bank.")
        if not np.any(ell_matches):
            raise KeyError(f"length_scale={length_scale} is not in the bank.")
        return (
            float(self.nu_values[np.flatnonzero(nu_matches)[0]]),
            float(self.length_scale_values[np.flatnonzero(ell_matches)[0]]),
        )

    def get_sampler(
        self,
        nu: float,
        length_scale: float,
        *,
        nearest: bool = False,
    ) -> MaternCirculantEmbedding2D:
        key = self._canonical_key(nu, length_scale, nearest=nearest)
        if key in self._cache:
            sampler = self._cache.pop(key)
            self._cache[key] = sampler
            return sampler
        sampler = MaternCirculantEmbedding2D(
            self.x_grid,
            self.y_grid,
            key[0],
            key[1],
            variance=self.variance,
            mean=self.mean,
            initial_embedding_shape=self.initial_embedding_shape,
            max_embedding_attempts=self.max_embedding_attempts,
            eigenvalue_tolerance=self.eigenvalue_tolerance,
            fft_workers=self.fft_workers,
        )
        self._cache[key] = sampler
        if (
            self.max_cached_samplers is not None
            and len(self._cache) > self.max_cached_samplers
        ):
            self._cache.popitem(last=False)
        return sampler

    def precompute_all(self) -> None:
        for nu in self.nu_values:
            for length_scale in self.length_scale_values:
                self.get_sampler(float(nu), float(length_scale))

    def sample(
        self,
        nu: float,
        length_scale: float,
        seed: int | None = None,
        *,
        rng: np.random.Generator | None = None,
        nearest: bool = False,
        dtype: np.dtype | type = np.float64,
    ) -> np.ndarray:
        return self.get_sampler(nu, length_scale, nearest=nearest).sample(
            seed=seed,
            rng=rng,
            dtype=dtype,
        )

    def sample_random(
        self,
        parameter_rng: np.random.Generator,
        *,
        field_seed: int | None = None,
        dtype: np.dtype | type = np.float64,
    ) -> tuple[np.ndarray, float, float, int]:
        if not isinstance(parameter_rng, np.random.Generator):
            raise TypeError("parameter_rng must be numpy.random.Generator.")
        nu = float(parameter_rng.choice(self.nu_values))
        length_scale = float(parameter_rng.choice(self.length_scale_values))
        if field_seed is None:
            field_seed = int(parameter_rng.integers(0, 2**31 - 1))
        field = self.sample(
            nu,
            length_scale,
            seed=field_seed,
            dtype=dtype,
        )
        return field, nu, length_scale, int(field_seed)

    def cache_diagnostics(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (nu, length_scale), sampler in self._cache.items():
            diagnostic = sampler.diagnostics
            assert diagnostic is not None
            rows.append(
                {
                    "nu": nu,
                    "length_scale": length_scale,
                    "embedding_shape": diagnostic.embedding_shape,
                    "attempts": diagnostic.attempts,
                    "relative_minimum_eigenvalue": (
                        diagnostic.relative_minimum_eigenvalue
                    ),
                    "spectrum_megabytes": diagnostic.approximate_spectrum_megabytes,
                }
            )
        return rows


def sample_matern_field_on_grid(
    x_grid: ArrayLike,
    y_grid: ArrayLike,
    nu: float,
    length_scale: float,
    variance: float,
    seed: int,
    mean: float = 0.0,
    *,
    initial_embedding_shape: tuple[int, int] | None = None,
    max_embedding_attempts: int = 7,
    eigenvalue_tolerance: float = 1.0e-12,
    fft_workers: int | None = None,
) -> np.ndarray:
    sampler = MaternCirculantEmbedding2D(
        x_grid,
        y_grid,
        nu,
        length_scale,
        variance=variance,
        mean=mean,
        initial_embedding_shape=initial_embedding_shape,
        max_embedding_attempts=max_embedding_attempts,
        eigenvalue_tolerance=eigenvalue_tolerance,
        fft_workers=fft_workers,
    )
    return sampler.sample(seed=seed)


def sample_nu(rng: np.random.Generator, choices: Iterable[float]) -> float:
    values = np.asarray(list(choices), dtype=np.float64)
    if values.size == 0 or np.any(values <= 0.0):
        raise ValueError("choices must contain positive values.")
    return float(rng.choice(values))


def sample_length_scale(
    rng: np.random.Generator,
    minimum: float,
    maximum: float,
    mode: str = "log_uniform",
) -> float:
    minimum = float(minimum)
    maximum = float(maximum)
    if minimum <= 0.0 or maximum <= 0.0 or maximum < minimum:
        raise ValueError("Require 0 < minimum <= maximum.")
    if mode == "fixed":
        return minimum
    if mode == "uniform":
        return float(rng.uniform(minimum, maximum))
    if mode == "log_uniform":
        return float(np.exp(rng.uniform(np.log(minimum), np.log(maximum))))
    raise ValueError("mode must be 'fixed', 'uniform', or 'log_uniform'.")


__all__ = [
    "EmbeddingDiagnostics",
    "MaternCirculantEmbedding2D",
    "MaternSamplerBank2D",
    "make_length_scale_bank",
    "matern_covariance",
    "sample_length_scale",
    "sample_matern_field_on_grid",
    "sample_nu",
]
