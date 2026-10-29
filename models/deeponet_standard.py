"""Standard separable DeepONet for the Reaction operator.

This module deliberately reuses the same 120x120 convolutional branch encoder
as the legacy gated Reaction DeepONet, but replaces the branch-conditioned
trunk by a conventional separable DeepONet:

    b(u) in R^Dp,
    tau(x,y) in R^Dp,
    p_hat(u;x,y) = <b(u), tau(x,y)> / sqrt(Dp) + bias.

The trunk is independent of u.  Therefore, when all samples in a batch use the
same coordinates, the trunk basis is evaluated only once as [N,Dp] rather than
as a [B,N,Dp] hidden state through every trunk layer.  This is the main reason
full 120x120 (=14400 point) supervision is practical here.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deeponet import FieldUNetEncoderToVector


class StandardReactionTrunk(nn.Module):
    """Coordinate-only MLP producing the DeepONet basis tau(x,y)."""

    def __init__(
        self,
        *,
        trunk_input_dim: int,
        Dp: int = 400,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        if int(trunk_input_dim) < 1:
            raise ValueError("trunk_input_dim must be positive.")
        if int(Dp) < 1:
            raise ValueError("Dp must be positive.")
        if int(num_layers) < 1:
            raise ValueError("num_layers must be at least one.")

        self.trunk_input_dim = int(trunk_input_dim)
        self.Dp = int(Dp)
        self.num_layers = int(num_layers)

        self.encoder = nn.Linear(self.trunk_input_dim, self.Dp)
        self.blocks = nn.ModuleList(
            nn.Linear(self.Dp, self.Dp) for _ in range(self.num_layers - 1)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        for layer in self.blocks:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, encoded_coordinates: torch.Tensor) -> torch.Tensor:
        if encoded_coordinates.ndim not in (2, 3):
            raise ValueError(
                "encoded_coordinates must have shape [N,Dtrunk] or [B,N,Dtrunk]."
            )
        if encoded_coordinates.shape[-1] != self.trunk_input_dim:
            raise ValueError(
                f"Expected trunk input dimension {self.trunk_input_dim}; "
                f"received {encoded_coordinates.shape[-1]}."
            )
        hidden = F.gelu(self.encoder(encoded_coordinates))
        for layer in self.blocks:
            hidden = F.gelu(layer(hidden))
        return hidden


class ReactionStandardDeepONet(nn.Module):
    """Canonical/separable branch-trunk DeepONet for the Reaction PDE.

    The branch uses exactly the same convolutional 120x120 encoder as the
    legacy gated model.  The trunk is independent of the branch until the
    final inner product.

    ``trunk_input`` may be either
      * [N,Dtrunk] when coordinates are shared across the batch (preferred), or
      * [B,N,Dtrunk] for general batched coordinate sets.
    """

    architecture = "standard"

    def __init__(
        self,
        *,
        branch_input_channels: int = 3,
        trunk_input_dim: int,
        Dp: int = 400,
        num_layers: int = 6,
        num_frequencies: int = 6,
    ) -> None:
        super().__init__()
        expected_trunk_dim = 2 * (1 + 2 * int(num_frequencies))
        if int(trunk_input_dim) != expected_trunk_dim:
            raise ValueError(
                "trunk_input_dim is inconsistent with num_frequencies: "
                f"expected {expected_trunk_dim}, received {trunk_input_dim}."
            )
        if int(branch_input_channels) != 3:
            raise ValueError(
                "The production Reaction branch uses three channels: normalized u, x, y."
            )

        self.branch_input_channels = int(branch_input_channels)
        self.trunk_input_dim = int(trunk_input_dim)
        self.Dp = int(Dp)
        self.output_dim = self.Dp
        self.num_layers = int(num_layers)
        self.num_frequencies = int(num_frequencies)
        self.inner_product_scale = 1.0 / math.sqrt(float(self.Dp))

        self.branch_net = FieldUNetEncoderToVector(
            input_channels=self.branch_input_channels,
            output_dim=self.Dp,
        )
        self.trunk_net = StandardReactionTrunk(
            trunk_input_dim=self.trunk_input_dim,
            Dp=self.Dp,
            num_layers=self.num_layers,
        )
        self.output_bias = nn.Parameter(torch.zeros(1))

    def combine(
        self,
        branch_coefficients: torch.Tensor,
        trunk_basis: torch.Tensor,
        *,
        include_bias: bool = True,
    ) -> torch.Tensor:
        """Combine branch coefficients and trunk basis without recomputation."""
        if branch_coefficients.ndim != 2 or branch_coefficients.shape[-1] != self.Dp:
            raise ValueError("branch_coefficients must have shape [B,Dp].")

        if trunk_basis.ndim == 2:
            if trunk_basis.shape[-1] != self.Dp:
                raise ValueError("trunk_basis must have final dimension Dp.")
            values = torch.einsum("bd,nd->bn", branch_coefficients, trunk_basis)
        elif trunk_basis.ndim == 3:
            if trunk_basis.shape[0] != branch_coefficients.shape[0]:
                raise ValueError("Branch and trunk batch dimensions differ.")
            if trunk_basis.shape[-1] != self.Dp:
                raise ValueError("trunk_basis must have final dimension Dp.")
            values = torch.einsum("bd,bnd->bn", branch_coefficients, trunk_basis)
        else:
            raise ValueError("trunk_basis must have shape [N,Dp] or [B,N,Dp].")

        values = self.inner_product_scale * values
        if include_bias:
            values = values + self.output_bias
        return values.unsqueeze(-1)

    def forward(
        self,
        branch_field_input: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> torch.Tensor:
        if branch_field_input.ndim != 4:
            raise ValueError("branch_field_input must have shape [B,3,120,120].")
        branch_coefficients = self.branch_net(branch_field_input)
        trunk_basis = self.trunk_net(trunk_input)
        return self.combine(branch_coefficients, trunk_basis, include_bias=True)


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


__all__ = [
    "ReactionStandardDeepONet",
    "StandardReactionTrunk",
    "count_trainable_parameters",
]
