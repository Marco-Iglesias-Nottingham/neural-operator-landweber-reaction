"""Generalized / reduced-basis DeepONet for the Reaction operator.

Architecture
------------
This implements

    F_theta(u) = p_bar + B_N phi_theta(F_M u),

with fixed linear input and output maps.

Input:
    F_M is a weighted PCA/KL-style encoder of the physical reaction field u.

Middle map:
    phi_theta : R^M -> R^N is a residual MLP.

Output:
    B_N is a fixed weighted-POD decoder of the pressure field p.

Both the input and output reduced coordinates are standardized mode-by-mode
for numerical conditioning.  The forward/JVP training loss should therefore
weight output-coordinate errors by sigma_out^2 so that it is equivalent, up
to one global factor, to physical W-L2 error.

For a physical perturbation h,

    D F_theta(u) h
      = B_N D phi_theta(F_M u) [F_M h],

so the input perturbation is encoded exactly by the fixed linear map F_M.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_field(values: torch.Tensor, n_points: int, name: str) -> torch.Tensor:
    if values.ndim == 4:
        if values.shape[1] != 1:
            raise ValueError(f"{name}: expected one channel, got {values.shape}.")
        values = values[:, 0].reshape(values.shape[0], -1)
    elif values.ndim == 3:
        values = values.reshape(values.shape[0], -1)
    elif values.ndim != 2:
        raise ValueError(
            f"{name}: expected [B,N], [B,H,W], or [B,1,H,W], got {values.shape}."
        )
    if values.shape[1] != n_points:
        raise ValueError(
            f"{name}: expected {n_points} points, got {values.shape[1]}."
        )
    return values


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, residual_scale: float = 0.5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.fc1(x))
        y = self.fc2(y)
        return F.gelu(x + self.residual_scale * y)


class ReducedResidualMap(nn.Module):
    """Smooth finite-dimensional map phi_theta : R^M -> R^N."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_width: int = 1024,
        residual_blocks: int = 4,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if input_dim < 1 or output_dim < 1 or hidden_width < 1:
            raise ValueError("Reduced-map dimensions must be positive.")
        if residual_blocks < 0:
            raise ValueError("residual_blocks must be nonnegative.")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_width = int(hidden_width)
        self.residual_blocks = int(residual_blocks)
        self.residual_scale = float(residual_scale)

        self.input_layer = nn.Linear(self.input_dim, self.hidden_width)
        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    self.hidden_width,
                    residual_scale=self.residual_scale,
                )
                for _ in range(self.residual_blocks)
            ]
        )
        self.output_layer = nn.Linear(self.hidden_width, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.input_layer(x))
        for block in self.blocks:
            z = block(z)
        return self.output_layer(z)


class ReactionGeneralizedDeepONet(nn.Module):
    """Fixed linear input encoder + residual MLP + fixed POD decoder."""

    architecture = "generalized"

    def __init__(
        self,
        *,
        input_mean_field: torch.Tensor,
        input_sqrt_weights: torch.Tensor,
        input_weighted_modes: torch.Tensor,
        input_coefficient_mean: torch.Tensor,
        input_coefficient_std: torch.Tensor,
        output_mean_field: torch.Tensor,
        output_sqrt_weights: torch.Tensor,
        output_weighted_modes: torch.Tensor,
        output_coefficient_mean: torch.Tensor,
        output_coefficient_std: torch.Tensor,
        hidden_width: int = 1024,
        residual_blocks: int = 4,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()

        input_mean_field = torch.as_tensor(
            input_mean_field, dtype=torch.float32
        ).reshape(-1)
        input_sqrt_weights = torch.as_tensor(
            input_sqrt_weights, dtype=torch.float32
        ).reshape(-1)
        input_weighted_modes = torch.as_tensor(
            input_weighted_modes, dtype=torch.float32
        )
        input_coefficient_mean = torch.as_tensor(
            input_coefficient_mean, dtype=torch.float32
        ).reshape(-1)
        input_coefficient_std = torch.as_tensor(
            input_coefficient_std, dtype=torch.float32
        ).reshape(-1)

        output_mean_field = torch.as_tensor(
            output_mean_field, dtype=torch.float32
        ).reshape(-1)
        output_sqrt_weights = torch.as_tensor(
            output_sqrt_weights, dtype=torch.float32
        ).reshape(-1)
        output_weighted_modes = torch.as_tensor(
            output_weighted_modes, dtype=torch.float32
        )
        output_coefficient_mean = torch.as_tensor(
            output_coefficient_mean, dtype=torch.float32
        ).reshape(-1)
        output_coefficient_std = torch.as_tensor(
            output_coefficient_std, dtype=torch.float32
        ).reshape(-1)

        if input_weighted_modes.ndim != 2:
            raise ValueError("input_weighted_modes must have shape [Nu,M].")
        if output_weighted_modes.ndim != 2:
            raise ValueError("output_weighted_modes must have shape [Np,N].")

        n_input_points, input_rank = input_weighted_modes.shape
        n_output_points, output_rank = output_weighted_modes.shape

        if input_mean_field.numel() != n_input_points:
            raise ValueError("Input mean/basis point counts differ.")
        if input_sqrt_weights.numel() != n_input_points:
            raise ValueError("Input weights/basis point counts differ.")
        if input_coefficient_mean.numel() != input_rank:
            raise ValueError("Input coefficient mean length differs from M.")
        if input_coefficient_std.numel() != input_rank:
            raise ValueError("Input coefficient std length differs from M.")

        if output_mean_field.numel() != n_output_points:
            raise ValueError("Output mean/basis point counts differ.")
        if output_sqrt_weights.numel() != n_output_points:
            raise ValueError("Output weights/basis point counts differ.")
        if output_coefficient_mean.numel() != output_rank:
            raise ValueError("Output coefficient mean length differs from N.")
        if output_coefficient_std.numel() != output_rank:
            raise ValueError("Output coefficient std length differs from N.")

        if torch.any(input_sqrt_weights <= 0) or torch.any(output_sqrt_weights <= 0):
            raise ValueError("Quadrature sqrt weights must be positive.")
        if torch.any(input_coefficient_std <= 0):
            raise ValueError("Input reduced-coordinate std must be positive.")
        if torch.any(output_coefficient_std <= 0):
            raise ValueError("Output reduced-coordinate std must be positive.")

        self.n_input_points = int(n_input_points)
        self.n_output_points = int(n_output_points)
        self.input_rank = int(input_rank)
        self.output_rank = int(output_rank)
        self.Dp = int(output_rank)

        # Fixed basis data is stored separately in the checkpoint.
        self.register_buffer(
            "input_mean_field", input_mean_field, persistent=False
        )
        self.register_buffer(
            "input_sqrt_weights", input_sqrt_weights, persistent=False
        )
        self.register_buffer(
            "input_weighted_modes", input_weighted_modes, persistent=False
        )
        self.register_buffer(
            "input_coefficient_mean", input_coefficient_mean, persistent=False
        )
        self.register_buffer(
            "input_coefficient_std", input_coefficient_std, persistent=False
        )

        self.register_buffer(
            "output_mean_field", output_mean_field, persistent=False
        )
        self.register_buffer(
            "output_sqrt_weights", output_sqrt_weights, persistent=False
        )
        self.register_buffer(
            "output_weighted_modes", output_weighted_modes, persistent=False
        )
        self.register_buffer(
            "output_coefficient_mean", output_coefficient_mean, persistent=False
        )
        self.register_buffer(
            "output_coefficient_std", output_coefficient_std, persistent=False
        )

        # Phi_out = W^{-1/2} V_out, so Phi_out^T W Phi_out = I.
        self.register_buffer(
            "output_physical_basis",
            output_weighted_modes / output_sqrt_weights[:, None],
            persistent=False,
        )

        self.phi = ReducedResidualMap(
            input_dim=self.input_rank,
            output_dim=self.output_rank,
            hidden_width=int(hidden_width),
            residual_blocks=int(residual_blocks),
            residual_scale=float(residual_scale),
        )

    # ------------------------------------------------------------------
    # Fixed linear input encoder F_M
    # ------------------------------------------------------------------
    def encode_input(self, physical_u: torch.Tensor) -> torch.Tensor:
        """Encode u into standardized fixed input PCA/KL coordinates."""
        values = _flatten_field(
            physical_u, self.n_input_points, "physical_u"
        )
        centered = values - self.input_mean_field[None, :]
        coeff = (
            centered * self.input_sqrt_weights[None, :]
        ) @ self.input_weighted_modes
        return (
            coeff - self.input_coefficient_mean[None, :]
        ) / self.input_coefficient_std[None, :]

    def encode_direction(self, physical_h: torch.Tensor) -> torch.Tensor:
        """Encode a physical perturbation h by the derivative of F_M.

        The mean is NOT subtracted because this is a tangent direction.
        """
        values = _flatten_field(
            physical_h, self.n_input_points, "physical_h"
        )
        coeff = (
            values * self.input_sqrt_weights[None, :]
        ) @ self.input_weighted_modes
        return coeff / self.input_coefficient_std[None, :]

    # ------------------------------------------------------------------
    # Finite-dimensional nonlinear map phi_theta
    # ------------------------------------------------------------------
    def reduced_map(self, standardized_input_coefficients: torch.Tensor) -> torch.Tensor:
        return self.phi(standardized_input_coefficients)

    # ------------------------------------------------------------------
    # Fixed output POD projection/decoder B_N
    # ------------------------------------------------------------------
    def project_output_field(
        self,
        physical_p: torch.Tensor,
        *,
        standardize: bool = True,
    ) -> torch.Tensor:
        values = _flatten_field(
            physical_p, self.n_output_points, "physical_p"
        )
        centered = values - self.output_mean_field[None, :]
        coeff = (
            centered * self.output_sqrt_weights[None, :]
        ) @ self.output_weighted_modes
        if standardize:
            coeff = (
                coeff - self.output_coefficient_mean[None, :]
            ) / self.output_coefficient_std[None, :]
        return coeff

    def project_output_jvp(
        self,
        physical_jvp: torch.Tensor,
        *,
        standardize: bool = True,
    ) -> torch.Tensor:
        values = _flatten_field(
            physical_jvp, self.n_output_points, "physical_jvp"
        )
        coeff = (
            values * self.output_sqrt_weights[None, :]
        ) @ self.output_weighted_modes
        if standardize:
            coeff = coeff / self.output_coefficient_std[None, :]
        return coeff

    def decode_output(
        self,
        standardized_output_coefficients: torch.Tensor,
    ) -> torch.Tensor:
        coeff = (
            self.output_coefficient_mean[None, :]
            + self.output_coefficient_std[None, :]
            * standardized_output_coefficients
        )
        values = (
            self.output_mean_field[None, :]
            + coeff @ self.output_physical_basis.T
        )
        return values.unsqueeze(-1)

    def decode_output_jvp(
        self,
        standardized_output_coefficient_jvp: torch.Tensor,
    ) -> torch.Tensor:
        dcoeff = (
            self.output_coefficient_std[None, :]
            * standardized_output_coefficient_jvp
        )
        values = dcoeff @ self.output_physical_basis.T
        return values.unsqueeze(-1)

    def output_energy_weights(self) -> torch.Tensor:
        w = self.output_coefficient_std.square()
        return w / w.sum().clamp_min(torch.finfo(w.dtype).tiny)

    def forward(self, physical_u: torch.Tensor) -> torch.Tensor:
        x = self.encode_input(physical_u)
        z = self.reduced_map(x)
        return self.decode_output(z)


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


__all__ = [
    "ResidualMLPBlock",
    "ReducedResidualMap",
    "ReactionGeneralizedDeepONet",
    "count_trainable_parameters",
]
