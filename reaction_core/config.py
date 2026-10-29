from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReactionConfig:
    """Physical and finite-element configuration for the reaction problem.

    The forward problem is

        -Delta p + u p = rhs           in (0, Lx) x (0, Ly),
        p = 0                          on the boundary.

    ``u`` is represented directly in physical coordinates; it is not a
    logarithmic network input.
    """

    Lx: float = 1.0
    Ly: float = 1.0
    fem_nx: int = 119
    fem_ny: int = 119
    state_degree: int = 1
    rhs: float = 1.0
    u_min: float = 10.0

    @property
    def source_value(self) -> float:
        """Compatibility alias for code that calls the forcing a source."""
        return self.rhs

    def validate(self) -> None:
        if self.Lx <= 0.0 or self.Ly <= 0.0:
            raise ValueError("Domain lengths must be positive.")
        if self.fem_nx < 2 or self.fem_ny < 2:
            raise ValueError("The FEM mesh must contain at least two cells per axis.")
        if self.state_degree not in (1, 2):
            raise ValueError("state_degree must be either 1 or 2.")
        if self.rhs <= 0.0:
            raise ValueError("rhs must be positive.")
        if self.u_min < 0.0:
            raise ValueError("u_min must be non-negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GridConfig:
    """Regular neural-operator grid.

    ``120 x 120`` coincides with the vertices of the default structured
    ``119 x 119`` triangular FEM mesh and matches the Darcy branch encoder.
    """

    nx: int = 120
    ny: int = 120

    def validate(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise ValueError("The regular grid must contain at least two points per axis.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DistributionConfig:
    """Full-domain Matérn distribution for the physical coefficient ``u``.

    A sample is formed as ``u = mean + std*z``, where ``z`` is a centred,
    unit-standard-deviation Matérn field.  If the requested fluctuation would
    cross the safeguards, its amplitude is reduced uniformly.  Consequently
    the spatial pattern and target arithmetic mean are retained without
    pointwise distortion.
    """

    family_names: tuple[str, ...] = ("matern",)
    family_probabilities: tuple[float, ...] = (1.0,)

    nu_values: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5)
    length_scales: tuple[float, ...] = (0.075, 0.10, 0.15, 0.20, 0.25)
    mean_u_range: tuple[float, float] = (95.0, 105.0)
    std_u_range: tuple[float, float] = (10.0, 30.0)
    u_min: float = 0.0
    u_max: float = 200.0

    @property
    def length_scales_m(self) -> tuple[float, ...]:
        """Compatibility alias for scripts ported from the MRE project."""
        return self.length_scales

    def validate(self) -> None:
        if self.family_names != ("matern",):
            raise ValueError("This project uses only the 'matern' family.")
        if self.family_probabilities != (1.0,):
            raise ValueError("The Matérn family probability must be exactly one.")
        if not self.nu_values or any(value <= 0.0 for value in self.nu_values):
            raise ValueError("All Matérn nu values must be positive.")
        if len(set(self.nu_values)) != len(self.nu_values):
            raise ValueError("Matérn nu values must be unique.")
        if not self.length_scales or any(value <= 0.0 for value in self.length_scales):
            raise ValueError("All Matérn length scales must be positive.")
        if len(set(self.length_scales)) != len(self.length_scales):
            raise ValueError("Matérn length scales must be unique.")

        mean_min, mean_max = self.mean_u_range
        if mean_min <= 0.0 or mean_max <= mean_min:
            raise ValueError("Invalid mean_u_range.")
        std_min, std_max = self.std_u_range
        if std_min < 0.0 or std_max < std_min:
            raise ValueError("Invalid std_u_range.")
        if self.u_min < 0.0 or self.u_max <= self.u_min:
            raise ValueError("Require 0 <= u_min < u_max.")
        if mean_min < self.u_min or mean_max > self.u_max:
            raise ValueError("mean_u_range must lie inside [u_min, u_max].")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetConfig:
    """HDF5 and derivative-direction settings."""

    n_kl_modes: int = 16
    compression: str | None = "lzf"
    output_dtype: str = "float32"

    def validate(self) -> None:
        if self.n_kl_modes < 1:
            raise ValueError("n_kl_modes must be positive.")
        if self.compression not in (None, "lzf", "gzip"):
            raise ValueError("compression must be None, 'lzf', or 'gzip'.")
        if self.output_dtype not in {"float32", "float64"}:
            raise ValueError("output_dtype must be float32 or float64.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectConfig:
    reaction: ReactionConfig = field(default_factory=ReactionConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    distribution: DistributionConfig = field(default_factory=DistributionConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def validate(self) -> None:
        self.reaction.validate()
        self.grid.validate()
        self.distribution.validate()
        self.dataset.validate()
        expected = (self.reaction.fem_nx + 1, self.reaction.fem_ny + 1)
        actual = (self.grid.nx, self.grid.ny)
        if actual != expected:
            raise ValueError(
                "The regular grid must match the structured FEM vertices: "
                f"expected {expected}, received {actual}."
            )
        if self.distribution.u_min < self.reaction.u_min:
            raise ValueError(
                "The sampling lower bound cannot be below ReactionConfig.u_min."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction": self.reaction.to_dict(),
            "grid": self.grid.to_dict(),
            "distribution": self.distribution.to_dict(),
            "dataset": self.dataset.to_dict(),
        }


__all__ = [
    "DatasetConfig",
    "DistributionConfig",
    "GridConfig",
    "ProjectConfig",
    "ReactionConfig",
]
