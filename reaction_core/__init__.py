"""Core utilities for the physical-coefficient reaction problem.

The public names are loaded lazily so an emulator trainer does not need
FEniCSx merely to import the package. KL construction uses NumPy and does not
depend on scikit-learn.
"""

from importlib import import_module

from .config import (
    DatasetConfig,
    DistributionConfig,
    GridConfig,
    ProjectConfig,
    ReactionConfig,
)
from .grid import RegularGrid2D


_LAZY_EXPORTS = {
    # Sampling and empirical KL construction.
    "MaternUSampler": ("sampling", "MaternUSampler"),
    "SampleMetadata": ("sampling", "SampleMetadata"),
    "SampledU": ("sampling", "SampledU"),
    "KLBasis2D": ("kl", "KLBasis2D"),
    "build_incremental_kl_basis": ("kl", "build_incremental_kl_basis"),
    # HDF5 datasets.
    "ReactionDataset": ("datasets", "ReactionDataset"),
    "ReactionH5Dataset": ("datasets", "ReactionH5Dataset"),
    "SelectiveReactionH5Dataset": (
        "datasets",
        "SelectiveReactionH5Dataset",
    ),
    "ReactionH5ShardWriter": ("datasets", "ReactionH5ShardWriter"),
    "ReactionShardWriter": ("datasets", "ReactionShardWriter"),
    "compute_scalar_channel_stats": ("datasets", "compute_scalar_channel_stats"),
    # FEniCSx-only maps, solver, and observations.
    "CoefficientGridMap": ("grid", "CoefficientGridMap"),
    "ParameterGridMap": ("grid", "ParameterGridMap"),
    "StructuredCoefficientGridMap": ("grid", "StructuredCoefficientGridMap"),
    "StructuredLagrangeStateGridMap": ("grid", "StructuredLagrangeStateGridMap"),
    "StructuredP1StateGridMap": ("grid", "StructuredP1StateGridMap"),
    "FullFieldObservationOperator": ("observations", "FullFieldObservationOperator"),
    "ForwardSolver": ("solver", "ForwardSolver"),
    "ReactionGradient": ("solver", "ReactionGradient"),
    "ReactionSolver2D": ("solver", "ReactionSolver2D"),
    "ReactionState": ("solver", "ReactionState"),
    # Exact-inversion helpers.  These remain lazy so importing reaction_core
    # for training does not initialise PETSc or eagerly import SciPy.
    "ScalarNoiseRealization": ("inversion", "ScalarNoiseRealization"),
    "ScalarVertexObservationAdjoint": (
        "inversion",
        "ScalarVertexObservationAdjoint",
    ),
    "StationaryMaternCovariance2D": (
        "inversion",
        "StationaryMaternCovariance2D",
    ),
    "TrapezoidalObservationAdjoint": (
        "inversion",
        "TrapezoidalObservationAdjoint",
    ),
    "add_scalar_heteroscedastic_noise": (
        "inversion",
        "add_scalar_heteroscedastic_noise",
    ),
    "explicit_dense_matern_covariance": (
        "inversion",
        "explicit_dense_matern_covariance",
    ),
    "interior_vertex_mask": ("inversion", "interior_vertex_mask"),
    "regular_sensor_mask": ("inversion", "regular_sensor_mask"),
    "resample_regular_grid_field": (
        "inversion",
        "resample_regular_grid_field",
    ),
    "resample_regular_grid_fields": (
        "inversion",
        "resample_regular_grid_fields",
    ),
    "trapezoid_weights": ("inversion", "trapezoid_weights"),
    "validate_matern_covariance_action": (
        "inversion",
        "validate_matern_covariance_action",
    ),
    "weighted_inner": ("inversion", "weighted_inner"),
    "weighted_norm": ("inversion", "weighted_norm"),
    "weighted_residual_covector": (
        "inversion",
        "weighted_residual_covector",
    ),
    # Shared truth/data archive contract (FEniCS- and Torch-free).
    "FineSyntheticArchive": ("synthetic_archive", "FineSyntheticArchive"),
    "load_fine_synthetic_archive": (
        "synthetic_archive",
        "load_fine_synthetic_archive",
    ),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(f".{module_name}", __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "DatasetConfig",
    "DistributionConfig",
    "GridConfig",
    "ProjectConfig",
    "ReactionConfig",
    "RegularGrid2D",
    *_LAZY_EXPORTS,
]
