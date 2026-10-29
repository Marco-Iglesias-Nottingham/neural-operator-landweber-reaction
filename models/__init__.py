"""Neural-operator models and physical-unit checkpoint wrappers."""

from .deeponet import (
    BranchNetReactionFieldOnly,
    DEFAULT_BRANCH_GRID_SIZE,
    DEFAULT_FOURIER_FREQUENCIES,
    DEFAULT_TRUNK_INPUT_DIM,
    DoubleConv,
    FieldUNetEncoderToVector,
    ReactionDeepONet,
    TrunkSharedGateAffinePreAct,
    add_coordinate_channels,
    fourier_encode_coordinates,
    regular_grid_coordinates,
)
from .emulator import (
    DEEPONET_CHECKPOINT_FORMAT,
    FNO_CHECKPOINT_FORMAT,
    ReactionDeepONetEmulator,
    ReactionFNOEmulator,
    load_reaction_emulator,
)
from .fno2d import FNO2d, FNOBlock2d, SpectralConv2d


__all__ = [
    "BranchNetReactionFieldOnly",
    "DEEPONET_CHECKPOINT_FORMAT",
    "DEFAULT_BRANCH_GRID_SIZE",
    "DEFAULT_FOURIER_FREQUENCIES",
    "DEFAULT_TRUNK_INPUT_DIM",
    "DoubleConv",
    "FNO2d",
    "FNOBlock2d",
    "FNO_CHECKPOINT_FORMAT",
    "FieldUNetEncoderToVector",
    "ReactionDeepONet",
    "ReactionDeepONetEmulator",
    "ReactionFNOEmulator",
    "SpectralConv2d",
    "TrunkSharedGateAffinePreAct",
    "add_coordinate_channels",
    "fourier_encode_coordinates",
    "load_reaction_emulator",
    "regular_grid_coordinates",
]
