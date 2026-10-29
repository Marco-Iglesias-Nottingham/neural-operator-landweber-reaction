"""POD-DeepONet for the Reaction operator.

The branch is the same CNN encoder used by the standard Reaction DeepONet.
The learned coordinate trunk is replaced by a fixed weighted-POD basis.

Let W denote the trapezoidal quadrature matrix, and let Phi satisfy

    Phi^T W Phi = I.

The model predicts standardized POD coefficients

    z_r(u) = (c_r(u) - c_mean_r) / c_std_r,

where

    c_r(u) = <p(u) - p_bar, phi_r>_W.

The physical field is reconstructed as

    p_hat(u) = p_bar + Phi [c_mean + c_std * z_theta(u)].

For a direction h,

    D p_hat(u)[h] = Phi [c_std * D z_theta(u)[h]].

This gives a fixed, orthogonal spatial representation and removes the learned
trunk-conditioning issue from the canonical DeepONet.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .deeponet import FieldUNetEncoderToVector


class ReactionPODDeepONet(nn.Module):
    architecture = "pod"

    def __init__(
        self,
        *,
        mean_field: torch.Tensor,
        sqrt_weights: torch.Tensor,
        weighted_modes: torch.Tensor,
        coefficient_mean: torch.Tensor,
        coefficient_std: torch.Tensor,
        branch_input_channels: int = 3,
    ) -> None:
        super().__init__()
        mean_field = torch.as_tensor(mean_field, dtype=torch.float32).reshape(-1)
        sqrt_weights = torch.as_tensor(sqrt_weights, dtype=torch.float32).reshape(-1)
        weighted_modes = torch.as_tensor(weighted_modes, dtype=torch.float32)
        coefficient_mean = torch.as_tensor(
            coefficient_mean, dtype=torch.float32
        ).reshape(-1)
        coefficient_std = torch.as_tensor(
            coefficient_std, dtype=torch.float32
        ).reshape(-1)

        if weighted_modes.ndim != 2:
            raise ValueError("weighted_modes must have shape [N,R].")
        n_points, rank = weighted_modes.shape
        if mean_field.numel() != n_points or sqrt_weights.numel() != n_points:
            raise ValueError("POD mean/weights/modes use inconsistent point counts.")
        if coefficient_mean.numel() != rank or coefficient_std.numel() != rank:
            raise ValueError("POD coefficient statistics must have length R.")
        if torch.any(sqrt_weights <= 0):
            raise ValueError("All quadrature sqrt_weights must be positive.")
        if torch.any(coefficient_std <= 0):
            raise ValueError("All POD coefficient standard deviations must be positive.")
        if int(branch_input_channels) != 3:
            raise ValueError("Reaction POD branch expects normalized u plus x/y channels.")

        self.branch_input_channels = int(branch_input_channels)
        self.Dp = int(rank)
        self.output_dim = int(rank)
        self.pod_rank = int(rank)
        self.n_points = int(n_points)

        # These are fixed data, stored separately in checkpoint['pod_data'].
        # persistent=False avoids duplicating ~N*R numbers in model_state.
        self.register_buffer("mean_field", mean_field, persistent=False)
        self.register_buffer("sqrt_weights", sqrt_weights, persistent=False)
        self.register_buffer("weighted_modes", weighted_modes, persistent=False)
        self.register_buffer("coefficient_mean", coefficient_mean, persistent=False)
        self.register_buffer("coefficient_std", coefficient_std, persistent=False)

        # Phi = W^{-1/2} V, so Phi^T W Phi = I.
        physical_basis = weighted_modes / sqrt_weights[:, None]
        self.register_buffer("physical_basis", physical_basis, persistent=False)

        self.branch_net = FieldUNetEncoderToVector(
            input_channels=self.branch_input_channels,
            output_dim=self.pod_rank,
        )

    def normalized_coefficients(
        self,
        branch_field_input: torch.Tensor,
    ) -> torch.Tensor:
        if branch_field_input.ndim != 4:
            raise ValueError("branch_field_input must have shape [B,3,120,120].")
        return self.branch_net(branch_field_input)

    def physical_coefficients(
        self,
        normalized_coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.coefficient_mean[None, :]
            + self.coefficient_std[None, :] * normalized_coefficients
        )

    def decode_normalized_coefficients(
        self,
        normalized_coefficients: torch.Tensor,
    ) -> torch.Tensor:
        coeff = self.physical_coefficients(normalized_coefficients)
        values = self.mean_field[None, :] + coeff @ self.physical_basis.T
        return values.unsqueeze(-1)

    def decode_normalized_coefficient_jvp(
        self,
        normalized_coefficient_jvp: torch.Tensor,
    ) -> torch.Tensor:
        dcoeff = self.coefficient_std[None, :] * normalized_coefficient_jvp
        values = dcoeff @ self.physical_basis.T
        return values.unsqueeze(-1)

    def project_physical_field(
        self,
        physical_field: torch.Tensor,
        *,
        subtract_mean: bool = True,
        standardize: bool = True,
    ) -> torch.Tensor:
        """Project physical fields [B,N] or [B,1,H,W] onto POD coordinates."""
        values = physical_field
        if values.ndim == 4:
            values = values[:, 0].reshape(values.shape[0], -1)
        elif values.ndim == 3:
            values = values.reshape(values.shape[0], -1)
        elif values.ndim != 2:
            raise ValueError("physical_field must be [B,N], [B,H,W] or [B,1,H,W].")
        if values.shape[1] != self.n_points:
            raise ValueError(
                f"Expected {self.n_points} output points, got {values.shape[1]}."
            )

        centered = values
        if subtract_mean:
            centered = centered - self.mean_field[None, :]
        coeff = (centered * self.sqrt_weights[None, :]) @ self.weighted_modes
        if standardize:
            coeff = (
                coeff - self.coefficient_mean[None, :]
            ) / self.coefficient_std[None, :]
        return coeff

    def project_physical_jvp(
        self,
        physical_jvp: torch.Tensor,
        *,
        standardize: bool = True,
    ) -> torch.Tensor:
        """Project derivative fields; no mean-field subtraction is applied."""
        values = physical_jvp
        if values.ndim == 4:
            values = values[:, 0].reshape(values.shape[0], -1)
        elif values.ndim == 3:
            values = values.reshape(values.shape[0], -1)
        elif values.ndim != 2:
            raise ValueError("physical_jvp must be [B,N], [B,H,W] or [B,1,H,W].")
        coeff = (values * self.sqrt_weights[None, :]) @ self.weighted_modes
        if standardize:
            coeff = coeff / self.coefficient_std[None, :]
        return coeff

    def forward(self, branch_field_input: torch.Tensor) -> torch.Tensor:
        z = self.normalized_coefficients(branch_field_input)
        return self.decode_normalized_coefficients(z)


def count_trainable_parameters(model: nn.Module) -> int:
    return int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )


__all__ = ["ReactionPODDeepONet", "count_trainable_parameters"]
