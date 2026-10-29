"""Two-dimensional Fourier neural operator for the reaction problem.

This is the current MRE FNO architecture with the physical input/output
channels made configurable.  The reaction operator has one coefficient input
and one scalar solution output, so both default to one channel.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


NormalisationType = Literal["none", "group"]
SpectralInitialisationType = Literal["legacy", "sqrt"]


class SpectralConv2d(nn.Module):
    """Two-dimensional Fourier layer with learned low-frequency weights."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int,
        *,
        spectral_initialisation: SpectralInitialisationType = "sqrt",
    ) -> None:
        super().__init__()

        if in_channels < 1 or out_channels < 1:
            raise ValueError("Spectral channel counts must be positive.")
        if modes_x < 1 or modes_y < 1:
            raise ValueError("Spectral mode counts must be positive.")
        if spectral_initialisation not in ("legacy", "sqrt"):
            raise ValueError(
                "spectral_initialisation must be 'legacy' or 'sqrt'."
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        self.spectral_initialisation = spectral_initialisation

        if spectral_initialisation == "legacy":
            scale = 1.0 / (self.in_channels * self.out_channels)
        else:
            scale = 1.0 / math.sqrt(
                self.in_channels * self.out_channels
            )

        shape = (
            self.in_channels,
            self.out_channels,
            self.modes_x,
            self.modes_y,
        )
        self.weight_positive = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.weight_negative = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )

    @staticmethod
    def _multiply(
        input_fft: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if weights.dtype != input_fft.dtype:
            weights = weights.to(dtype=input_fft.dtype)
        return torch.einsum("bixy,ioxy->boxy", input_fft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "SpectralConv2d expects (batch,channels,nx,ny)."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {x.shape[1]}."
            )
        if not x.is_floating_point():
            raise TypeError("SpectralConv2d expects real floating-point input.")

        batch, _, nx, ny = x.shape
        if nx < 2 or ny < 1:
            raise ValueError(f"Spatial grid is too small: ({nx}, {ny}).")

        x_fft = torch.fft.rfft2(x, norm="ortho")
        output_fft = torch.zeros(
            batch,
            self.out_channels,
            nx,
            x_fft.shape[-1],
            dtype=x_fft.dtype,
            device=x.device,
        )

        modes_x = min(self.modes_x, nx // 2)
        modes_y = min(self.modes_y, x_fft.shape[-1])
        output_fft[:, :, :modes_x, :modes_y] = self._multiply(
            x_fft[:, :, :modes_x, :modes_y],
            self.weight_positive[:, :, :modes_x, :modes_y],
        )
        output_fft[:, :, -modes_x:, :modes_y] = self._multiply(
            x_fft[:, :, -modes_x:, :modes_y],
            self.weight_negative[:, :, :modes_x, :modes_y],
        )

        return torch.fft.irfft2(
            output_fft,
            s=(nx, ny),
            norm="ortho",
        )


class ChannelMLP2d(nn.Module):
    """Pointwise nonlinear channel mixer inside a residual block."""

    def __init__(self, width: int, multiplier: int) -> None:
        super().__init__()
        if width < 1 or multiplier < 1:
            raise ValueError("width and multiplier must be positive.")
        self.net = nn.Sequential(
            nn.Conv2d(width, multiplier * width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(multiplier * width, width, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FNOBlock2d(nn.Module):
    """MRE spectral/local/channel-mixing operator block."""

    def __init__(
        self,
        width: int,
        modes_x: int,
        modes_y: int,
        *,
        normalisation: NormalisationType = "none",
        residual_blocks: bool = True,
        residual_scale: float = 0.5,
        local_kernel_size: int = 3,
        channel_mlp_multiplier: int = 2,
        spectral_initialisation: SpectralInitialisationType = "sqrt",
    ) -> None:
        super().__init__()

        if width < 1:
            raise ValueError("width must be positive.")
        if normalisation not in ("none", "group"):
            raise ValueError("normalisation must be 'none' or 'group'.")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive.")
        if local_kernel_size < 0:
            raise ValueError("local_kernel_size must be non-negative.")
        if local_kernel_size > 0 and local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be odd when enabled.")
        if channel_mlp_multiplier < 0:
            raise ValueError(
                "channel_mlp_multiplier must be non-negative."
            )

        self.residual_blocks = bool(residual_blocks)
        self.residual_scale = float(residual_scale)
        self.spectral = SpectralConv2d(
            width,
            width,
            modes_x,
            modes_y,
            spectral_initialisation=spectral_initialisation,
        )
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.local: nn.Module | None = (
            nn.Conv2d(
                width,
                width,
                kernel_size=local_kernel_size,
                padding=local_kernel_size // 2,
            )
            if local_kernel_size > 0
            else None
        )
        self.norm: nn.Module = (
            nn.GroupNorm(num_groups=1, num_channels=width)
            if normalisation == "group"
            else nn.Identity()
        )
        self.channel_mlp: nn.Module | None = (
            ChannelMLP2d(width, channel_mlp_multiplier)
            if channel_mlp_multiplier > 0
            else None
        )

    def _operator_update(self, x: torch.Tensor) -> torch.Tensor:
        update = self.spectral(x) + self.pointwise(x)
        if self.local is not None:
            update = update + self.local(x)
        return F.gelu(self.norm(update))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self._operator_update(x)
        if not self.residual_blocks:
            return update

        x = x + self.residual_scale * update
        if self.channel_mlp is not None:
            x = x + self.residual_scale * self.channel_mlp(x)
        return x


class FNO2d(nn.Module):
    """Grid-to-grid reaction operator with the current MRE FNO body.

    Defaults reproduce the body used by the current MRE trainer, with one
    reaction-specific output channel and isotropic 24x24 modes for the square
    domain.  Every setting is stored as a public attribute so it can be
    serialized in ``model_config``.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        modes_x: int = 24,
        modes_y: int = 24,
        n_layers: int = 6,
        padding: int = 8,
        append_coordinates: bool = True,
        normalisation: NormalisationType = "none",
        residual_blocks: bool = True,
        residual_scale: float = 0.5,
        local_kernel_size: int = 3,
        channel_mlp_multiplier: int = 2,
        spectral_initialisation: SpectralInitialisationType = "sqrt",
    ) -> None:
        super().__init__()

        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive.")
        if width < 1 or n_layers < 1:
            raise ValueError("width and n_layers must be positive.")
        if modes_x < 1 or modes_y < 1:
            raise ValueError("modes_x and modes_y must be positive.")
        if padding < 0:
            raise ValueError("padding must be non-negative.")
        if normalisation not in ("none", "group"):
            raise ValueError("normalisation must be 'none' or 'group'.")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive.")
        if local_kernel_size < 0:
            raise ValueError("local_kernel_size must be non-negative.")
        if local_kernel_size > 0 and local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be odd when enabled.")
        if channel_mlp_multiplier < 0:
            raise ValueError(
                "channel_mlp_multiplier must be non-negative."
            )
        if spectral_initialisation not in ("legacy", "sqrt"):
            raise ValueError(
                "spectral_initialisation must be 'legacy' or 'sqrt'."
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        self.n_layers = int(n_layers)
        self.padding = int(padding)
        self.append_coordinates = bool(append_coordinates)
        self.normalisation = normalisation
        self.residual_blocks = bool(residual_blocks)
        self.residual_scale = float(residual_scale)
        self.local_kernel_size = int(local_kernel_size)
        self.channel_mlp_multiplier = int(channel_mlp_multiplier)
        self.spectral_initialisation = spectral_initialisation

        lifted_channels = self.in_channels + (
            2 if self.append_coordinates else 0
        )
        self.lift = nn.Conv2d(lifted_channels, self.width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                FNOBlock2d(
                    self.width,
                    self.modes_x,
                    self.modes_y,
                    normalisation=self.normalisation,
                    residual_blocks=self.residual_blocks,
                    residual_scale=self.residual_scale,
                    local_kernel_size=self.local_kernel_size,
                    channel_mlp_multiplier=self.channel_mlp_multiplier,
                    spectral_initialisation=self.spectral_initialisation,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.project1 = nn.Conv2d(
            self.width,
            2 * self.width,
            kernel_size=1,
        )
        self.project2 = nn.Conv2d(
            2 * self.width,
            self.out_channels,
            kernel_size=1,
        )

    @staticmethod
    def coordinate_channels(x: torch.Tensor) -> torch.Tensor:
        batch, _, nx, ny = x.shape
        gx = torch.linspace(0.0, 1.0, nx, device=x.device, dtype=x.dtype)
        gy = torch.linspace(0.0, 1.0, ny, device=x.device, dtype=x.dtype)
        grid_x, grid_y = torch.meshgrid(gx, gy, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).expand(
            batch,
            -1,
            -1,
            -1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("FNO2d expects (batch,channels,nx,ny).")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {x.shape[1]}."
            )

        if self.append_coordinates:
            x = torch.cat((x, self.coordinate_channels(x)), dim=1)
        x = self.lift(x)

        if self.padding > 0:
            x = F.pad(x, (0, self.padding, 0, self.padding))
        for block in self.blocks:
            x = block(x)
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]

        return self.project2(F.gelu(self.project1(x)))


__all__ = [
    "ChannelMLP2d",
    "FNO2d",
    "FNOBlock2d",
    "NormalisationType",
    "SpectralConv2d",
    "SpectralInitialisationType",
]
