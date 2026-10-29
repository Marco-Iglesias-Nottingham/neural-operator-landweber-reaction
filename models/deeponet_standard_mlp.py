"""Standard/separable DeepONet with a non-downsampling MLP branch.

The branch sees all 120x120 coefficient sensor values directly:

    u in R^(120*120)
      -> hidden_width
      -> ... hidden layers ...
      -> b(u) in R^Dp.

There is no convolution, pooling, stride, or spatial downsampling.

The coordinate trunk is independent of u and the final operator is the
canonical DeepONet inner product

    p_hat(u;x,y) = <b(u), tau(x,y)> / sqrt(Dp) + bias.

Because the input sensor grid is fixed, x/y coordinates are not appended to
the branch input: their positions are already encoded by the fixed ordering of
the 14,400 sensor entries.  Fourier features are retained in the coordinate
trunk.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deeponet_standard import StandardReactionTrunk


class ReactionMLPBranch(nn.Module):
    """Fully connected branch using every coefficient-grid sensor."""

    def __init__(
        self,
        *,
        grid_size: int = 120,
        hidden_width: int = 1024,
        hidden_layers: int = 2,
        output_dim: int = 400,
    ) -> None:
        super().__init__()
        if int(grid_size) < 2:
            raise ValueError("grid_size must be at least 2.")
        if int(hidden_width) < 1:
            raise ValueError("hidden_width must be positive.")
        if int(hidden_layers) < 1:
            raise ValueError("hidden_layers must be at least 1.")
        if int(output_dim) < 1:
            raise ValueError("output_dim must be positive.")

        self.grid_size = int(grid_size)
        self.input_dim = self.grid_size * self.grid_size
        self.hidden_width = int(hidden_width)
        self.hidden_layers = int(hidden_layers)
        self.output_dim = int(output_dim)

        self.input_layer = nn.Linear(self.input_dim, self.hidden_width)
        self.hidden = nn.ModuleList(
            nn.Linear(self.hidden_width, self.hidden_width)
            for _ in range(self.hidden_layers - 1)
        )
        self.output_layer = nn.Linear(self.hidden_width, self.output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        for layer in self.hidden:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, normalized_u: torch.Tensor) -> torch.Tensor:
        """Return branch coefficients [B,Dp].

        Accepted input shapes:
          [B,1,120,120]
          [B,120,120]
          [B,14400]
        """
        if normalized_u.ndim == 4:
            if normalized_u.shape[1] != 1:
                raise ValueError(
                    "MLP branch expects one physical-u channel."
                )
            values = normalized_u[:, 0]
        elif normalized_u.ndim == 3:
            values = normalized_u
        elif normalized_u.ndim == 2:
            values = normalized_u
        else:
            raise ValueError(
                "normalized_u must have shape [B,1,nx,ny], [B,nx,ny], "
                "or [B,nx*ny]."
            )

        values = values.reshape(values.shape[0], -1)
        if values.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} branch sensors, "
                f"received {values.shape[1]}."
            )

        hidden = F.gelu(self.input_layer(values))
        for layer in self.hidden:
            hidden = F.gelu(layer(hidden))
        return self.output_layer(hidden)


class ReactionStandardMLPDeepONet(nn.Module):
    """Canonical DeepONet with an all-sensor MLP branch."""

    architecture = "standard_mlp"

    def __init__(
        self,
        *,
        branch_grid_size: int = 120,
        branch_hidden_width: int = 1024,
        branch_hidden_layers: int = 2,
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
                f"expected {expected_trunk_dim}, got {trunk_input_dim}."
            )

        self.branch_grid_size = int(branch_grid_size)
        self.branch_hidden_width = int(branch_hidden_width)
        self.branch_hidden_layers = int(branch_hidden_layers)
        self.trunk_input_dim = int(trunk_input_dim)
        self.Dp = int(Dp)
        self.output_dim = self.Dp
        self.num_layers = int(num_layers)
        self.num_frequencies = int(num_frequencies)
        self.inner_product_scale = 1.0 / math.sqrt(float(self.Dp))

        self.branch_net = ReactionMLPBranch(
            grid_size=self.branch_grid_size,
            hidden_width=self.branch_hidden_width,
            hidden_layers=self.branch_hidden_layers,
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
        normalized_u: torch.Tensor,
        trunk_input: torch.Tensor,
    ) -> torch.Tensor:
        coefficients = self.branch_net(normalized_u)
        trunk_basis = self.trunk_net(trunk_input)
        return self.combine(coefficients, trunk_basis, include_bias=True)


def count_trainable_parameters(model: nn.Module) -> int:
    return int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )


__all__ = [
    "ReactionMLPBranch",
    "ReactionStandardMLPDeepONet",
    "count_trainable_parameters",
]
