from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DistributionConfig
from .grid import RegularGrid2D
from .random_fields import MaternSamplerBank2D


FAMILY_TO_ID = {"matern": 0}
ID_TO_FAMILY = {0: "matern"}


@dataclass(frozen=True)
class SampleMetadata:
    family: str
    family_id: int
    seed: int
    field_seed: int
    nu: float
    length_scale: float
    target_mean_u: float
    actual_mean_u: float
    minimum_u: float
    maximum_u: float
    target_std_u: float
    actual_std_u: float
    fluctuation_scale: float
    rescaled: bool

    @property
    def length_scale_m(self) -> float:
        """Compatibility alias retained for ported diagnostic scripts."""
        return self.length_scale

    @property
    def physical_mean(self) -> float:
        return self.actual_mean_u

    @property
    def physical_min(self) -> float:
        return self.minimum_u

    @property
    def physical_max(self) -> float:
        return self.maximum_u

    @property
    def actual_std(self) -> float:
        return self.actual_std_u

    @property
    def was_rescaled(self) -> bool:
        return self.rescaled


@dataclass(frozen=True)
class SampledU:
    u: np.ndarray
    metadata: SampleMetadata

    @property
    def u_grid(self) -> np.ndarray:
        return self.u


def _standardise(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    if not np.all(np.isfinite(field)):
        raise ValueError("The sampled Matérn field contains non-finite values.")
    mean = float(np.mean(field))
    standard_deviation = float(np.std(field))
    if standard_deviation <= 1.0e-12:
        raise RuntimeError("Random field has essentially zero variance.")
    return (field - mean) / standard_deviation


def _bounded_physical_field(
    standardised_field: np.ndarray,
    *,
    target_mean: float,
    target_std: float,
    minimum: float,
    maximum: float,
) -> tuple[np.ndarray, float, bool]:
    """Construct ``mean + scale*std*z`` without pointwise distortion.

    The largest globally admissible fluctuation scale is found analytically.
    A final ``np.clip`` is applied only to remove floating-point excursions at
    an active bound; it is not used to shape the sample.
    """
    z = np.asarray(standardised_field, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("standardised_field must be two-dimensional.")
    if not minimum <= target_mean <= maximum:
        raise ValueError("target_mean must lie inside [minimum, maximum].")
    if target_std < 0.0:
        raise ValueError("target_std must be non-negative.")

    fluctuation = float(target_std) * z
    scale = 1.0
    negative_minimum = float(np.min(fluctuation))
    positive_maximum = float(np.max(fluctuation))
    if negative_minimum < 0.0:
        scale = min(scale, (target_mean - minimum) / (-negative_minimum))
    if positive_maximum > 0.0:
        scale = min(scale, (maximum - target_mean) / positive_maximum)
    scale = float(np.clip(scale, 0.0, 1.0))

    candidate = target_mean + scale * fluctuation
    tolerance = 256.0 * np.finfo(np.float64).eps * max(
        1.0, abs(minimum), abs(maximum), abs(target_mean)
    )
    if float(np.min(candidate)) < minimum - tolerance:
        raise RuntimeError("Uniform lower-bound scaling failed.")
    if float(np.max(candidate)) > maximum + tolerance:
        raise RuntimeError("Uniform upper-bound scaling failed.")
    result = np.clip(candidate, minimum, maximum)
    rescaled = bool(scale < 1.0 - 64.0 * np.finfo(np.float64).eps)
    return result, scale, rescaled


class MaternUSampler:
    """Sample non-negative, bounded physical reaction coefficients directly.

    Each sample has the form

        u(x) = mean_u + scale * std_u * z_{nu,ell}(x),

    with a centred/unit-standard-deviation Matérn pattern.  The mean, standard
    deviation, smoothness and length scale all vary independently according to
    ``DistributionConfig``.
    """

    def __init__(
        self,
        grid: RegularGrid2D,
        distribution: DistributionConfig,
        *,
        precompute_matern_bank: bool = False,
        max_cached_samplers: int | None = None,
        fft_workers: int | None = None,
    ) -> None:
        distribution.validate()
        self.grid = grid
        self.distribution = distribution
        self.matern_bank = MaternSamplerBank2D(
            grid.x,
            grid.y,
            nu_values=distribution.nu_values,
            length_scale_values=distribution.length_scales,
            variance=1.0,
            mean=0.0,
            precompute=precompute_matern_bank,
            max_cached_samplers=max_cached_samplers,
            fft_workers=fft_workers,
        )

    def sample(self, seed: int) -> SampledU:
        rng = np.random.default_rng(int(seed))
        raw, nu, length_scale, field_seed = self.matern_bank.sample_random(
            rng,
            dtype=np.float64,
        )
        z = _standardise(raw)

        target_mean = float(rng.uniform(*self.distribution.mean_u_range))
        target_std = float(rng.uniform(*self.distribution.std_u_range))
        u, fluctuation_scale, rescaled = _bounded_physical_field(
            z,
            target_mean=target_mean,
            target_std=target_std,
            minimum=self.distribution.u_min,
            maximum=self.distribution.u_max,
        )

        metadata = SampleMetadata(
            family="matern",
            family_id=FAMILY_TO_ID["matern"],
            seed=int(seed),
            field_seed=int(field_seed),
            nu=float(nu),
            length_scale=float(length_scale),
            target_mean_u=target_mean,
            actual_mean_u=float(np.mean(u)),
            minimum_u=float(np.min(u)),
            maximum_u=float(np.max(u)),
            target_std_u=target_std,
            actual_std_u=float(np.std(u)),
            fluctuation_scale=fluctuation_scale,
            rescaled=rescaled,
        )
        return SampledU(u=u, metadata=metadata)


# Compatibility name for early scripts while this new project is assembled.
MixedUSampler = MaternUSampler


__all__ = [
    "FAMILY_TO_ID",
    "ID_TO_FAMILY",
    "MaternUSampler",
    "MixedUSampler",
    "SampleMetadata",
    "SampledU",
]
