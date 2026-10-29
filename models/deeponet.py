"""Darcy-encoder DeepONet for the stationary reaction equation.

The branch and shared-gate trunk intentionally preserve the production Darcy
``forward`` model.  Only the trunk coordinates change: the reaction equation
is stationary, so the trunk receives Fourier-encoded ``(x, y)`` and no time.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_FOURIER_FREQUENCIES = 6
DEFAULT_TRUNK_INPUT_DIM = 2 * (1 + 2 * DEFAULT_FOURIER_FREQUENCIES)
DEFAULT_BRANCH_GRID_SIZE = 120


def fourier_encode_coordinates(
    coordinates: torch.Tensor,
    *,
    num_frequencies: int = DEFAULT_FOURIER_FREQUENCIES,
    include_input: bool = True,
) -> torch.Tensor:
    """Encode physical ``(x,y)`` coordinates as in the Darcy trunk.

    Frequencies are ``1, 2, 4, ..., 2**(num_frequencies-1)``.  With the raw
    coordinates included and six frequencies, the final dimension is 26.
    Arbitrary leading dimensions are supported.
    """

    if coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have final dimension two: (x,y).")
    if num_frequencies < 0:
        raise ValueError("num_frequencies must be non-negative.")
    if not coordinates.is_floating_point():
        raise TypeError("coordinates must be floating point.")
    if num_frequencies == 0:
        return coordinates if include_input else coordinates[..., :0]

    frequencies = 2 ** torch.arange(
        num_frequencies,
        device=coordinates.device,
        dtype=coordinates.dtype,
    )
    projected = coordinates.unsqueeze(-1) * frequencies
    sine = torch.sin(2.0 * torch.pi * projected)
    cosine = torch.cos(2.0 * torch.pi * projected)
    encoded = torch.cat((sine, cosine), dim=-1).reshape(
        *coordinates.shape[:-1],
        -1,
    )
    if include_input:
        encoded = torch.cat((coordinates, encoded), dim=-1)
    return encoded


def regular_grid_coordinates(
    nx: int,
    ny: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one-dimensional physical grids on the unit square."""

    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must both be at least two.")
    return (
        torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, ny, device=device, dtype=dtype),
    )


def add_coordinate_channels(
    field: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
) -> torch.Tensor:
    """Append physical x/y channels to a normalized scalar field.

    ``field`` has shape ``[B,1,nx,ny]`` and the result has the exact Darcy
    branch layout ``[B,3,nx,ny]``.
    """

    if field.ndim != 4 or field.shape[1] != 1:
        raise ValueError("field must have shape [B,1,nx,ny].")
    batch, _, nx, ny = field.shape
    x = torch.as_tensor(x_grid, dtype=field.dtype, device=field.device)
    y = torch.as_tensor(y_grid, dtype=field.dtype, device=field.device)
    if x.ndim != 1 or x.numel() != nx:
        raise ValueError(f"x_grid must contain {nx} coordinates.")
    if y.ndim != 1 or y.numel() != ny:
        raise ValueError(f"y_grid must contain {ny} coordinates.")

    x_channel = x.reshape(1, 1, nx, 1).expand(batch, 1, nx, ny)
    y_channel = y.reshape(1, 1, 1, ny).expand(batch, 1, nx, ny)
    return torch.cat((field, x_channel, y_channel), dim=1)


class TrunkSharedGateAffinePreAct(nn.Module):
    """Production Darcy shared-gate trunk, unchanged except for xy input."""

    def __init__(
        self,
        trunk_input_dim: int,
        Dp: int,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        if trunk_input_dim < 1 or Dp < 1 or num_layers < 1:
            raise ValueError(
                "trunk_input_dim, Dp and num_layers must be positive."
            )
        self.Dp = int(Dp)
        self.output_dim = self.Dp
        self.num_layers = int(num_layers)
        self.encoder = nn.Linear(int(trunk_input_dim), self.output_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Linear(self.output_dim, self.output_dim)
                for _ in range(self.num_layers - 1)
            ]
        )

        # Per-layer affine transforms of the one shared branch gate.
        self.a = nn.ParameterList(
            [
                nn.Parameter(torch.ones(self.output_dim))
                for _ in range(self.num_layers)
            ]
        )
        self.b = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.output_dim))
                for _ in range(self.num_layers)
            ]
        )
        self.out_p = nn.Linear(self.Dp, 1)

    def _apply_gate_pre_act(
        self,
        hidden_pre_activation: torch.Tensor,
        shared_gate: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        layer_gate = (
            self.a[layer_index] * shared_gate
            + self.b[layer_index]
        )
        return hidden_pre_activation * layer_gate

    def forward(
        self,
        trunk_input: torch.Tensor,
        shared_gate: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(trunk_input)
        hidden = F.gelu(
            self._apply_gate_pre_act(hidden, shared_gate, 0)
        )
        for layer_index, layer in enumerate(self.blocks, start=1):
            hidden = F.gelu(
                self._apply_gate_pre_act(
                    layer(hidden),
                    shared_gate,
                    layer_index,
                )
            )
        return self.out_p(hidden)


class DoubleConv(nn.Module):
    """Exact convolution block from the production Darcy branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FieldUNetEncoderToVector(nn.Module):
    """Exact production Darcy 120x120 field encoder.

    Spatial progression: ``120 -> 60 -> 30 -> 15 -> 7`` followed by the
    ``512*7*7 -> output_dim`` head.
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 400,
    ) -> None:
        super().__init__()
        if input_channels != 3:
            raise ValueError(
                "The production reaction branch requires exactly three "
                "channels: standardized u, x and y."
            )
        if output_dim < 1:
            raise ValueError("output_dim must be positive.")

        self.input_channels = int(input_channels)
        self.output_dim = int(output_dim)
        self.enc1 = DoubleConv(self.input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.flatten = nn.Flatten()
        self.head = nn.Linear(512 * 7 * 7, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError("Branch input must have shape [B,3,120,120].")
        if tuple(x.shape[-2:]) != (
            DEFAULT_BRANCH_GRID_SIZE,
            DEFAULT_BRANCH_GRID_SIZE,
        ):
            raise ValueError(
                "The exact Darcy branch requires a 120x120 grid; received "
                f"{tuple(x.shape[-2:])}."
            )

        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        x = self.pool4(x)
        x = self.bottleneck(x)
        return self.head(self.flatten(x))


class BranchNetReactionFieldOnly(nn.Module):
    """Field-only branch producing the shared gate in R^Dp."""

    def __init__(
        self,
        input_channels: int = 3,
        Dp: int = 400,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.Dp = int(Dp)
        self.field_branch = FieldUNetEncoderToVector(
            input_channels=input_channels,
            output_dim=self.Dp,
        )
        self.normalize = bool(normalize)

    def forward(self, field_input: torch.Tensor) -> torch.Tensor:
        gate = self.field_branch(field_input)
        if self.normalize:
            gate = F.layer_norm(gate, gate.shape[-1:])
        return gate


class ReactionDeepONet(nn.Module):
    """Continuous-coordinate reaction surrogate with Darcy branch/trunk.

    Inputs are the already assembled branch tensor ``[u_normalized,x,y]`` and
    Fourier-encoded stationary coordinates.  Normalization and physical-unit
    conversion belong to :class:`ReactionDeepONetEmulator` or the trainer.
    """

    def __init__(
        self,
        *,
        branch_input_channels: int = 3,
        trunk_input_dim: int = DEFAULT_TRUNK_INPUT_DIM,
        Dp: int = 400,
        num_layers: int = 6,
        num_frequencies: int = DEFAULT_FOURIER_FREQUENCIES,
    ) -> None:
        super().__init__()
        expected_trunk_dim = 2 * (1 + 2 * int(num_frequencies))
        if int(trunk_input_dim) != expected_trunk_dim:
            raise ValueError(
                "trunk_input_dim is inconsistent with num_frequencies: "
                f"expected {expected_trunk_dim}, received {trunk_input_dim}."
            )

        self.Dp = int(Dp)
        self.output_dim = self.Dp
        self.num_layers = int(num_layers)
        self.num_frequencies = int(num_frequencies)
        self.trunk_input_dim = int(trunk_input_dim)
        self.branch_input_channels = int(branch_input_channels)

        self.branch_net = BranchNetReactionFieldOnly(
            input_channels=self.branch_input_channels,
            Dp=self.Dp,
        )
        self.trunk_net = TrunkSharedGateAffinePreAct(
            trunk_input_dim=self.trunk_input_dim,
            Dp=self.Dp,
            num_layers=self.num_layers,
        )

    def forward(
        self,
        branch_field_input: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> torch.Tensor:
        if branch_field_input.ndim != 4:
            raise ValueError("branch_field_input must have shape [B,3,120,120].")
        if trunk_input.ndim != 3:
            raise ValueError("trunk_input must have shape [B,N,Dtrunk].")
        if trunk_input.shape[-1] != self.trunk_input_dim:
            raise ValueError(
                f"Expected trunk dimension {self.trunk_input_dim}, "
                f"received {trunk_input.shape[-1]}."
            )
        if branch_field_input.shape[0] != trunk_input.shape[0]:
            raise ValueError("Branch and trunk batch dimensions differ.")

        batch_size, n_points, trunk_dimension = trunk_input.shape
        gate = self.branch_net(branch_field_input)
        gate = gate.unsqueeze(1).expand(-1, n_points, -1).reshape(
            batch_size * n_points,
            self.output_dim,
        )
        trunk_flat = trunk_input.reshape(
            batch_size * n_points,
            trunk_dimension,
        )
        pressure = self.trunk_net(trunk_flat, gate)
        return pressure.reshape(batch_size, n_points, 1)


__all__ = [
    "BranchNetReactionFieldOnly",
    "DEFAULT_BRANCH_GRID_SIZE",
    "DEFAULT_FOURIER_FREQUENCIES",
    "DEFAULT_TRUNK_INPUT_DIM",
    "DoubleConv",
    "FieldUNetEncoderToVector",
    "ReactionDeepONet",
    "TrunkSharedGateAffinePreAct",
    "add_coordinate_channels",
    "fourier_encode_coordinates",
    "regular_grid_coordinates",
]
