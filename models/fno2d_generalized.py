"""Generalized fully coupled Fourier neural operator for the Reaction problem.

This implements the architecture used in the theoretical GFNO result:

    u -- C_M (truncated Fourier analysis) --> c_M
      -- Phi_theta (dense globally coupled MLP) --> d_N
      -- S_N (Fourier synthesis) --> p.

The crucial difference from a standard FNO is that ``Phi_theta`` acts on the
*entire vector of retained Fourier coefficients*.  Distinct Fourier modes are
therefore allowed to couple directly.  A standard FNO instead applies a
mode-wise Fourier multiplier and only mixes channels within each frequency.

For real grid fields we use ``torch.fft.rfft2``.  The retained low-frequency
positive/negative x blocks are packed by separating real and imaginary parts,
so the learned network is an ordinary real-valued feed-forward network.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    """Two-layer GELU residual block in global Fourier coordinates."""

    def __init__(self, width: int, residual_scale: float = 0.5) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("width must be positive.")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive.")
        self.width = int(width)
        self.residual_scale = float(residual_scale)
        self.fc1 = nn.Linear(self.width, self.width)
        self.fc2 = nn.Linear(self.width, self.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.fc2(F.gelu(self.fc1(x)))
        return x + self.residual_scale * update


class GeneralizedFNO2d(nn.Module):
    """Fully coupled Fourier neural operator on a 2D regular grid.

    Parameters
    ----------
    input_modes_x, input_modes_y:
        Retained Fourier modes for the input analysis map C_M.  Along x we
        retain both the first ``input_modes_x`` and last ``input_modes_x``
        rFFT rows; along y we retain the first ``input_modes_y`` rFFT columns.
    output_modes_x, output_modes_y:
        Retained modes used by the output synthesis map S_N.
    hidden_width:
        Width of the globally coupled Fourier-coordinate MLP.
    residual_blocks:
        Number of two-layer residual MLP blocks.

    Notes
    -----
    For one real input channel, the real Fourier-coordinate input dimension is

        4 * input_modes_x * input_modes_y,

    because there are two x-frequency blocks and each complex coefficient is
    represented by real and imaginary parts.  The same formula applies to the
    output with ``out_channels`` and the output mode counts.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        input_modes_x: int = 16,
        input_modes_y: int = 16,
        output_modes_x: int = 12,
        output_modes_y: int = 12,
        hidden_width: int = 512,
        residual_blocks: int = 4,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()

        for name, value in {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "input_modes_x": input_modes_x,
            "input_modes_y": input_modes_y,
            "output_modes_x": output_modes_x,
            "output_modes_y": output_modes_y,
            "hidden_width": hidden_width,
        }.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive.")
        if int(residual_blocks) < 0:
            raise ValueError("residual_blocks cannot be negative.")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.input_modes_x = int(input_modes_x)
        self.input_modes_y = int(input_modes_y)
        self.output_modes_x = int(output_modes_x)
        self.output_modes_y = int(output_modes_y)
        self.hidden_width = int(hidden_width)
        self.residual_blocks = int(residual_blocks)
        self.residual_scale = float(residual_scale)

        self.input_dimension = (
            4 * self.in_channels * self.input_modes_x * self.input_modes_y
        )
        self.output_dimension = (
            4 * self.out_channels * self.output_modes_x * self.output_modes_y
        )

        self.input_layer = nn.Linear(self.input_dimension, self.hidden_width)
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    self.hidden_width,
                    residual_scale=self.residual_scale,
                )
                for _ in range(self.residual_blocks)
            ]
        )
        self.output_layer = nn.Linear(self.hidden_width, self.output_dimension)

    @staticmethod
    def _validate_grid_modes(
        nx: int,
        ny: int,
        modes_x: int,
        modes_y: int,
        *,
        label: str,
    ) -> None:
        # Requiring 2*modes_x <= nx prevents positive/negative x blocks from
        # overlapping.  rfft2 stores ny//2+1 non-negative y frequencies.
        if 2 * modes_x > nx:
            raise ValueError(
                f"{label} modes_x={modes_x} is too large for nx={nx}; "
                "require 2*modes_x <= nx."
            )
        if modes_y > ny // 2 + 1:
            raise ValueError(
                f"{label} modes_y={modes_y} is too large for ny={ny}; "
                f"rfft2 provides only {ny // 2 + 1} y modes."
            )

    @staticmethod
    def _pack_complex(coefficients: torch.Tensor) -> torch.Tensor:
        # (..., complex) -> (..., 2 real coordinates) -> one real vector.
        return torch.view_as_real(coefficients).reshape(coefficients.shape[0], -1)

    @staticmethod
    def _unpack_complex(
        vector: torch.Tensor,
        *,
        channels: int,
        modes_x: int,
        modes_y: int,
    ) -> torch.Tensor:
        expected = 4 * channels * modes_x * modes_y
        if vector.ndim != 2 or vector.shape[1] != expected:
            raise ValueError(
                f"Expected Fourier vector [B,{expected}], got {tuple(vector.shape)}."
            )
        real_pair = vector.reshape(
            vector.shape[0],
            channels,
            2 * modes_x,
            modes_y,
            2,
        ).contiguous()
        return torch.view_as_complex(real_pair)

    def encode_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply C_M: truncated Fourier analysis and real/imaginary packing."""
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                "GeneralizedFNO2d expects input [B,C,nx,ny] with "
                f"C={self.in_channels}; got {tuple(x.shape)}."
            )
        if not x.is_floating_point():
            raise TypeError("GeneralizedFNO2d expects real floating-point input.")
        if not torch.all(torch.isfinite(x)):
            raise ValueError("GeneralizedFNO2d input contains non-finite values.")

        _, _, nx, ny = x.shape
        self._validate_grid_modes(
            nx,
            ny,
            self.input_modes_x,
            self.input_modes_y,
            label="input",
        )
        x_fft = torch.fft.rfft2(x, norm="ortho")
        positive = x_fft[
            :, :, : self.input_modes_x, : self.input_modes_y
        ]
        negative = x_fft[
            :, :, -self.input_modes_x :, : self.input_modes_y
        ]
        retained = torch.cat((positive, negative), dim=2)
        return self._pack_complex(retained)

    def reduced_map(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Apply Phi_theta in the globally coupled Fourier coordinates."""
        if coordinates.ndim != 2 or coordinates.shape[1] != self.input_dimension:
            raise ValueError(
                f"Expected coordinates [B,{self.input_dimension}], "
                f"got {tuple(coordinates.shape)}."
            )
        z = F.gelu(self.input_layer(coordinates))
        for block in self.blocks:
            z = block(z)
        return self.output_layer(z)

    def decode_output(
        self,
        coordinates: torch.Tensor,
        *,
        nx: int,
        ny: int,
    ) -> torch.Tensor:
        """Apply S_N: unpack coefficients and synthesize a real grid field."""
        self._validate_grid_modes(
            nx,
            ny,
            self.output_modes_x,
            self.output_modes_y,
            label="output",
        )
        retained = self._unpack_complex(
            coordinates,
            channels=self.out_channels,
            modes_x=self.output_modes_x,
            modes_y=self.output_modes_y,
        )
        positive = retained[:, :, : self.output_modes_x, :]
        negative = retained[:, :, self.output_modes_x :, :]

        output_fft = torch.zeros(
            coordinates.shape[0],
            self.out_channels,
            nx,
            ny // 2 + 1,
            dtype=retained.dtype,
            device=coordinates.device,
        )
        output_fft[
            :, :, : self.output_modes_x, : self.output_modes_y
        ] = positive
        output_fft[
            :, :, -self.output_modes_x :, : self.output_modes_y
        ] = negative
        return torch.fft.irfft2(output_fft, s=(nx, ny), norm="ortho")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, nx, ny = x.shape
        input_coordinates = self.encode_input(x)
        output_coordinates = self.reduced_map(input_coordinates)
        return self.decode_output(output_coordinates, nx=nx, ny=ny)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = [
    "GeneralizedFNO2d",
    "ResidualMLPBlock",
    "count_trainable_parameters",
]
