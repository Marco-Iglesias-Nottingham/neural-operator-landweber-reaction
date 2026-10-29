"""Classical 2D Fourier Neural Operator for the Reaction problem.

This model deliberately removes the extra machinery used by the enhanced
Reaction FNO: there are no residual FNO blocks, no local 3x3 convolutions,
and no channel MLPs.  Each hidden layer is the classical

    spectral convolution + 1x1 pointwise convolution

followed by GELU between Fourier layers.  The default 32/12/4 configuration
with projection width 128 has 1,188,353 trainable PyTorch parameters.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class StandardSpectralConv2d(nn.Module):
    """Low-frequency 2D Fourier convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int,
        *,
        spectral_initialisation: str = "sqrt",
    ) -> None:
        super().__init__()

        if in_channels < 1 or out_channels < 1:
            raise ValueError("Channel counts must be positive.")
        if modes_x < 1 or modes_y < 1:
            raise ValueError("Fourier mode counts must be positive.")
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
                "StandardSpectralConv2d expects (batch,channels,nx,ny)."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"received {x.shape[1]}."
            )
        if not x.is_floating_point():
            raise TypeError("StandardSpectralConv2d expects real input.")

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


class StandardFNOBlock2d(nn.Module):
    """One classical FNO layer: Fourier + pointwise linear operator."""

    def __init__(
        self,
        width: int,
        modes_x: int,
        modes_y: int,
        *,
        spectral_initialisation: str = "sqrt",
    ) -> None:
        super().__init__()
        self.spectral = StandardSpectralConv2d(
            width,
            width,
            modes_x,
            modes_y,
            spectral_initialisation=spectral_initialisation,
        )
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spectral(x) + self.pointwise(x)


class StandardFNO2d(nn.Module):
    """Classical grid-to-grid FNO for ``u -> p`` on the unit square."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        modes_x: int = 12,
        modes_y: int = 12,
        n_layers: int = 4,
        padding: int = 8,
        append_coordinates: bool = True,
        projection_hidden_width: int = 128,
        spectral_initialisation: str = "sqrt",
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
        if projection_hidden_width < 1:
            raise ValueError("projection_hidden_width must be positive.")
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
        self.projection_hidden_width = int(projection_hidden_width)
        self.spectral_initialisation = spectral_initialisation

        lifted_channels = self.in_channels + (
            2 if self.append_coordinates else 0
        )
        self.lift = nn.Conv2d(lifted_channels, self.width, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                StandardFNOBlock2d(
                    self.width,
                    self.modes_x,
                    self.modes_y,
                    spectral_initialisation=self.spectral_initialisation,
                )
                for _ in range(self.n_layers)
            ]
        )

        # Classical FNO projection Q: width -> 128 -> scalar output.
        self.project1 = nn.Conv2d(
            self.width,
            self.projection_hidden_width,
            kernel_size=1,
        )
        self.project2 = nn.Conv2d(
            self.projection_hidden_width,
            self.out_channels,
            kernel_size=1,
        )

    @staticmethod
    def coordinate_channels(x: torch.Tensor) -> torch.Tensor:
        batch, _, nx, ny = x.shape
        gx = torch.linspace(
            0.0, 1.0, nx, device=x.device, dtype=x.dtype
        )
        gy = torch.linspace(
            0.0, 1.0, ny, device=x.device, dtype=x.dtype
        )
        grid_x, grid_y = torch.meshgrid(gx, gy, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).expand(
            batch, -1, -1, -1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "StandardFNO2d expects (batch,channels,nx,ny)."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {x.shape[1]}."
            )

        if self.append_coordinates:
            x = torch.cat((x, self.coordinate_channels(x)), dim=1)
        x = self.lift(x)

        if self.padding > 0:
            x = F.pad(x, (0, self.padding, 0, self.padding))

        # Classical FNO convention: activation between Fourier layers,
        # but not after the final Fourier layer.
        for layer_index, block in enumerate(self.blocks):
            x = block(x)
            if layer_index < self.n_layers - 1:
                x = F.gelu(x)

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]

        return self.project2(F.gelu(self.project1(x)))


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    model = StandardFNO2d()
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    test_input = torch.randn(2, 1, 120, 120)
    test_output = model(test_input)
    print("Input shape: ", tuple(test_input.shape))
    print("Output shape:", tuple(test_output.shape))
