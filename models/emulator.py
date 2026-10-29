"""Physical-unit checkpoint wrappers for the reaction surrogates.

Canonical checkpoint formats
----------------------------
FNO checkpoints use::

    {
        "checkpoint_format": "reaction-fno-v1",
        "model_type": "fno",
        "epoch": int,
        "model_config": {... arguments accepted by FNO2d ...},
        "model_state": model.state_dict(),
        "stats": {
            "u_mean": scalar or [in_channels],
            "u_std": scalar or [in_channels],
            "p_mean": scalar or [out_channels],
            "p_std": scalar or [out_channels],
        },
    }

DeepONet checkpoints use::

    {
        "checkpoint_format": "reaction-deeponet-v1",
        "model_type": "deeponet",
        "epoch": int,
        "model_config": {... arguments accepted by ReactionDeepONet ...},
        "model_state": model.state_dict(),
        "stats": {"u_mean": scalar, "u_std": scalar,
                  "p_mean": scalar, "p_std": scalar},
        "grid": {
            "x_grid": array[120],
            "y_grid": array[120],
            "output_coordinates": array[n_points, 2],
        },
    }

All public operations accept the *physical reaction coefficient* ``u``.  No
logarithm, exponential or other parameter transformation is performed.  JVPs
and Euclidean VJPs therefore also use physical-u directions/pairings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np
import torch
from torch.func import jvp

from .deeponet import (
    DEFAULT_BRANCH_GRID_SIZE,
    DEFAULT_FOURIER_FREQUENCIES,
    ReactionDeepONet,
    add_coordinate_channels,
    fourier_encode_coordinates,
    regular_grid_coordinates,
)
from .deeponet_standard import ReactionStandardDeepONet
from .deeponet_standard_mlp import ReactionStandardMLPDeepONet
from .deeponet_generalized import ReactionGeneralizedDeepONet
from .fno2d import FNO2d
from .fno2d_standard import StandardFNO2d
from .fno2d_generalized import GeneralizedFNO2d


FNO_CHECKPOINT_FORMAT = "reaction-fno-v1"
DEEPONET_CHECKPOINT_FORMAT = "reaction-deeponet-v1"


def _resolve_device(
    requested: str | torch.device | None,
) -> torch.device:
    """Resolve ``auto`` and safely fall back when CUDA is unavailable."""

    if requested is None or str(requested).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            f"CUDA device {device} was requested but CUDA is unavailable; "
            "using CPU instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return device


def _load_checkpoint(
    path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:  # PyTorch versions before weights_only was introduced.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Reaction checkpoint must contain a dictionary.")
    return checkpoint


def _model_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get(
        "model_state",
        checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict"),
        ),
    )
    if not isinstance(state, Mapping):
        raise KeyError(
            "Checkpoint must contain model_state (or model_state_dict/state_dict)."
        )
    return {
        (
            str(key)[len("module.") :]
            if str(key).startswith("module.")
            else str(key)
        ): value
        for key, value in state.items()
    }


def _stats(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    values = checkpoint.get("stats")
    if not isinstance(values, Mapping):
        raise KeyError("Checkpoint is missing a dictionary-valued 'stats'.")
    return values


def _stat_value(
    stats: Mapping[str, Any],
    canonical: str,
    *aliases: str,
) -> Any:
    for key in (canonical, *aliases):
        if key in stats:
            return stats[key]
    choices = ", ".join(repr(key) for key in (canonical, *aliases))
    raise KeyError(f"Checkpoint stats require one of: {choices}.")


def _grid_value(
    checkpoint: Mapping[str, Any],
    stats: Mapping[str, Any],
    *names: str,
) -> Any | None:
    grid = checkpoint.get("grid")
    containers = (
        checkpoint,
        grid if isinstance(grid, Mapping) else {},
        stats,
    )
    for container in containers:
        for name in names:
            if name in container:
                return container[name]
    return None


def _normalisation_tensor(
    value: Any,
    *,
    channels: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Return a scalar/channel/spatial statistic in NCHW broadcast form."""

    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} contains non-finite values.")
    if tensor.numel() == 1:
        return tensor.reshape(1, 1, 1, 1)
    if tensor.ndim == 1 and tensor.numel() == channels:
        return tensor.reshape(1, channels, 1, 1)
    if tensor.ndim == 3 and tensor.shape[0] in (1, channels):
        return tensor.unsqueeze(0)
    if tensor.ndim == 4 and tensor.shape[0] == 1 and tensor.shape[1] in (
        1,
        channels,
    ):
        return tensor
    raise ValueError(
        f"{name} must be scalar, [{channels}], [{channels},nx,ny] or "
        f"[1,{channels},nx,ny]; received {tuple(tensor.shape)}."
    )


def _scalar_stat(
    value: Any,
    *,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() != 1 or not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} must be one finite scalar.")
    return tensor.reshape(1)


def _fno_input_tensor(
    value: Any,
    *,
    channels: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 2 and channels == 1:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        if tensor.shape[0] == channels:
            tensor = tensor[None]
        elif channels == 1:
            tensor = tensor[:, None]
    if tensor.ndim != 4 or tensor.shape[1] != channels:
        raise ValueError(
            f"{name} must have shape [nx,ny], [C,nx,ny], [B,nx,ny] "
            f"for C=1, or [B,C,nx,ny], with C={channels}."
        )
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} contains non-finite values.")
    return tensor


def _broadcast_like_input(
    value: Any,
    reference: torch.Tensor,
    *,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    tensor = _fno_input_tensor(
        value,
        channels=int(reference.shape[1]),
        device=device,
        name=name,
    )
    if tuple(tensor.shape[-2:]) != tuple(reference.shape[-2:]):
        raise ValueError(
            f"{name} grid {tuple(tensor.shape[-2:])} does not match "
            f"input grid {tuple(reference.shape[-2:])}."
        )
    if tensor.shape[0] not in (1, reference.shape[0]):
        raise ValueError(
            f"{name} batch must be 1 or {reference.shape[0]}, "
            f"received {tensor.shape[0]}."
        )
    if tensor.shape[0] == 1 and reference.shape[0] > 1:
        tensor = tensor.expand(reference.shape[0], -1, -1, -1)
    return tensor


def _fno_output_dual(
    value: Any,
    output: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    return _broadcast_like_input(
        value,
        output,
        device=device,
        name="output dual",
    )


def _numpy_fno_output(tensor: torch.Tensor) -> np.ndarray:
    values = tensor.detach().cpu().numpy()
    if values.shape[0] == 1:
        values = values[0]
    if values.ndim >= 3 and values.shape[-3] == 1:
        values = values[..., 0, :, :]
    return values


class ReactionFNOEmulator:
    """FNO forward, physical-u JVP and Euclidean VJP checkpoint wrapper."""

    checkpoint_format = FNO_CHECKPOINT_FORMAT
    model_type = "fno"

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = _resolve_device(device)
        self.checkpoint = _load_checkpoint(self.checkpoint_path, self.device)
        self.epoch = int(self.checkpoint.get("epoch", -1))

        stored_type = str(self.checkpoint.get("model_type", "fno")).lower()
        if stored_type not in ("fno", "reaction_fno"):
            raise ValueError(
                f"Checkpoint model_type={stored_type!r} is not an FNO."
            )
        if "model_config" not in self.checkpoint:
            raise KeyError("FNO checkpoint is missing 'model_config'.")
        self.model_config = dict(self.checkpoint["model_config"])
        constructor_config = dict(self.model_config)
        architecture = str(
            constructor_config.pop("architecture", "enhanced")
        ).lower()
        if architecture in ("standard", "classical", "canonical"):
            self.model = StandardFNO2d(**constructor_config).to(self.device)
        elif architecture in (
            "generalized_fourier",
            "generalized_fno",
            "generalised_fourier",
            "gfno",
        ):
            self.model = GeneralizedFNO2d(**constructor_config).to(self.device)
        elif architecture in ("enhanced", "reaction", "legacy"):
            self.model = FNO2d(**constructor_config).to(self.device)
        else:
            raise ValueError(
                f"Unknown FNO architecture={architecture!r} in checkpoint."
            )
        self.model.load_state_dict(_model_state(self.checkpoint), strict=True)
        self.model.eval()

        values = _stats(self.checkpoint)
        self.u_mean = _normalisation_tensor(
            _stat_value(values, "u_mean", "x_mean", "branch_mean"),
            channels=self.model.in_channels,
            device=self.device,
            name="u_mean",
        )
        self.u_std = _normalisation_tensor(
            _stat_value(values, "u_std", "x_std", "branch_std"),
            channels=self.model.in_channels,
            device=self.device,
            name="u_std",
        )
        self.p_mean = _normalisation_tensor(
            _stat_value(values, "p_mean", "y_mean", "target_mean"),
            channels=self.model.out_channels,
            device=self.device,
            name="p_mean",
        )
        self.p_std = _normalisation_tensor(
            _stat_value(values, "p_std", "y_std", "target_std"),
            channels=self.model.out_channels,
            device=self.device,
            name="p_std",
        )
        if torch.any(self.u_std <= 0) or torch.any(self.p_std <= 0):
            raise ValueError("u_std and p_std must be strictly positive.")

    def _input(self, physical_u: Any) -> torch.Tensor:
        return _fno_input_tensor(
            physical_u,
            channels=self.model.in_channels,
            device=self.device,
            name="physical_u",
        )

    def forward_tensor(self, physical_u: torch.Tensor) -> torch.Tensor:
        normalized_u = (physical_u - self.u_mean) / self.u_std
        normalized_p = self.model(normalized_u)
        return self.p_mean + self.p_std * normalized_p

    @torch.inference_mode()
    def forward(self, physical_u: Any) -> np.ndarray:
        return _numpy_fno_output(self.forward_tensor(self._input(physical_u)))

    def jacobian_vector_product(
        self,
        physical_u: Any,
        direction: Any,
    ) -> np.ndarray:
        u = self._input(physical_u)
        h = _broadcast_like_input(
            direction,
            u,
            device=self.device,
            name="direction",
        )
        _, derivative = jvp(
            self.forward_tensor,
            (u,),
            (h,),
            strict=False,
        )
        return _numpy_fno_output(derivative)

    def jacobian_transpose_vector_product(
        self,
        physical_u: Any,
        output_dual: Any,
    ) -> np.ndarray:
        u = self._input(physical_u).detach().requires_grad_(True)
        p = self.forward_tensor(u)
        dual = _fno_output_dual(output_dual, p, device=self.device)
        gradient = torch.autograd.grad(
            torch.sum(p * dual),
            u,
            create_graph=False,
            retain_graph=False,
        )[0]
        return _numpy_fno_output(gradient)

    jvp = jacobian_vector_product
    vjp = jacobian_transpose_vector_product


def _deeponet_input_tensor(
    value: Any,
    *,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[None] if tensor.shape[0] == 1 else tensor[:, None]
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError(
            f"{name} must have shape [120,120], [1,120,120], "
            "[B,120,120] or [B,1,120,120]."
        )
    if tuple(tensor.shape[-2:]) != (
        DEFAULT_BRANCH_GRID_SIZE,
        DEFAULT_BRANCH_GRID_SIZE,
    ):
        raise ValueError(
            f"{name} must use the production Darcy 120x120 branch grid; "
            f"received {tuple(tensor.shape[-2:])}."
        )
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} contains non-finite values.")
    return tensor


def _deeponet_direction(
    value: Any,
    reference: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    direction = _deeponet_input_tensor(
        value,
        device=device,
        name="direction",
    )
    if direction.shape[0] not in (1, reference.shape[0]):
        raise ValueError(
            f"Direction batch must be 1 or {reference.shape[0]}, "
            f"received {direction.shape[0]}."
        )
    if direction.shape[0] == 1 and reference.shape[0] > 1:
        direction = direction.expand(reference.shape[0], -1, -1, -1)
    return direction


def _coordinate_tensor(
    value: Any,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    coordinates = torch.as_tensor(value, dtype=torch.float32, device=device)
    if coordinates.ndim == 2:
        coordinates = coordinates.unsqueeze(0)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape [N,2] or [B,N,2].")
    if coordinates.shape[0] not in (1, batch_size):
        raise ValueError(
            f"Coordinate batch must be 1 or {batch_size}, "
            f"received {coordinates.shape[0]}."
        )
    if coordinates.shape[0] == 1 and batch_size > 1:
        coordinates = coordinates.expand(batch_size, -1, -1)
    if not torch.all(torch.isfinite(coordinates)):
        raise ValueError("coordinates contain non-finite values.")
    if torch.any(coordinates < 0.0) or torch.any(coordinates > 1.0):
        raise ValueError("Reaction coordinates must lie in the unit square.")
    return coordinates


def _deeponet_output_dual(
    value: Any,
    output: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    dual = torch.as_tensor(value, dtype=torch.float32, device=device)
    batch_size, n_points, _ = output.shape
    if dual.ndim == 1:
        dual = dual[None, :, None]
    elif dual.ndim == 2:
        if tuple(dual.shape) == (n_points, 1):
            dual = dual[None]
        elif dual.shape[-1] == n_points:
            dual = dual[..., None]
    if dual.ndim != 3 or dual.shape[1:] != (n_points, 1):
        raise ValueError(
            "output_dual must have shape [N], [N,1], [B,N] or [B,N,1]."
        )
    if dual.shape[0] not in (1, batch_size):
        raise ValueError(
            f"Output-dual batch must be 1 or {batch_size}, "
            f"received {dual.shape[0]}."
        )
    if dual.shape[0] == 1 and batch_size > 1:
        dual = dual.expand(batch_size, -1, -1)
    if not torch.all(torch.isfinite(dual)):
        raise ValueError("output_dual contains non-finite values.")
    return dual


def _numpy_deeponet_output(tensor: torch.Tensor) -> np.ndarray:
    values = tensor.detach().cpu().numpy()[..., 0]
    return values[0] if values.shape[0] == 1 else values


def _numpy_deeponet_gradient(tensor: torch.Tensor) -> np.ndarray:
    values = tensor.detach().cpu().numpy()[:, 0]
    return values[0] if values.shape[0] == 1 else values


class ReactionDeepONetEmulator:
    """Continuous reaction DeepONet wrapper in physical coefficient units."""

    checkpoint_format = DEEPONET_CHECKPOINT_FORMAT
    model_type = "deeponet"

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = _resolve_device(device)
        self.checkpoint = _load_checkpoint(self.checkpoint_path, self.device)
        self.epoch = int(self.checkpoint.get("epoch", -1))

        stored_type = str(
            self.checkpoint.get("model_type", "deeponet")
        ).lower()
        if stored_type not in ("deeponet", "reaction_deeponet"):
            raise ValueError(
                f"Checkpoint model_type={stored_type!r} is not a DeepONet."
            )
        if "model_config" not in self.checkpoint:
            raise KeyError("DeepONet checkpoint is missing 'model_config'.")
        raw_config = dict(self.checkpoint["model_config"])
        architecture = str(raw_config.get("architecture", "gated")).lower()
        config = dict(raw_config)
        config.pop("architecture", None)
        if "D" in config and "Dp" not in config:
            config["Dp"] = config.pop("D")
        if "num_spatial_frequencies" in config and "num_frequencies" not in config:
            config["num_frequencies"] = config.pop(
                "num_spatial_frequencies"
            )
        config.setdefault("num_frequencies", DEFAULT_FOURIER_FREQUENCIES)
        config.setdefault(
            "trunk_input_dim",
            2 * (1 + 2 * int(config["num_frequencies"])),
        )
        config.setdefault("branch_input_channels", 3)
        config.setdefault("Dp", 400)
        config.setdefault("num_layers", 6)
        self.model_config = dict(raw_config)
        self.architecture = architecture
        if architecture in ("standard", "separable", "canonical"):
            self.model = ReactionStandardDeepONet(**config).to(self.device)
        elif architecture in ("standard_mlp", "separable_mlp", "canonical_mlp"):
            config.pop("branch_input_channels", None)
            config.setdefault("branch_grid_size", 120)
            config.setdefault("branch_hidden_width", 1024)
            config.setdefault("branch_hidden_layers", 2)
            self.model = ReactionStandardMLPDeepONet(**config).to(self.device)
        elif architecture in (
            "generalized",
            "generalised",
            "reduced_basis",
            "rb_dino",
        ):
            input_basis = self.checkpoint.get("input_basis_data")
            output_basis = self.checkpoint.get("pod_data")
            if not isinstance(input_basis, Mapping):
                raise KeyError(
                    "Generalized DeepONet checkpoint is missing dictionary-valued "
                    "'input_basis_data'."
                )
            if not isinstance(output_basis, Mapping):
                raise KeyError(
                    "Generalized DeepONet checkpoint is missing dictionary-valued "
                    "'pod_data'."
                )

            required_basis_keys = (
                "mean_field",
                "sqrt_weights",
                "weighted_modes",
                "coefficient_mean",
                "coefficient_std",
            )
            for label, basis in (
                ("input_basis_data", input_basis),
                ("pod_data", output_basis),
            ):
                missing = [key for key in required_basis_keys if key not in basis]
                if missing:
                    raise KeyError(
                        f"Generalized DeepONet {label} is missing keys {missing}."
                    )

            self.model = ReactionGeneralizedDeepONet(
                input_mean_field=input_basis["mean_field"],
                input_sqrt_weights=input_basis["sqrt_weights"],
                input_weighted_modes=input_basis["weighted_modes"],
                input_coefficient_mean=input_basis["coefficient_mean"],
                input_coefficient_std=input_basis["coefficient_std"],
                output_mean_field=output_basis["mean_field"],
                output_sqrt_weights=output_basis["sqrt_weights"],
                output_weighted_modes=output_basis["weighted_modes"],
                output_coefficient_mean=output_basis["coefficient_mean"],
                output_coefficient_std=output_basis["coefficient_std"],
                hidden_width=int(config.get("hidden_width", 1024)),
                residual_blocks=int(config.get("residual_blocks", 4)),
                residual_scale=float(config.get("residual_scale", 0.5)),
            ).to(self.device)

            input_rank = int(config.get("input_rank", self.model.input_rank))
            output_rank = int(config.get("output_rank", self.model.output_rank))
            if input_rank != int(self.model.input_rank):
                raise ValueError(
                    "Generalized DeepONet input rank in model_config does not "
                    "match embedded input_basis_data: "
                    f"{input_rank} != {self.model.input_rank}."
                )
            if output_rank != int(self.model.output_rank):
                raise ValueError(
                    "Generalized DeepONet output rank in model_config does not "
                    "match embedded pod_data: "
                    f"{output_rank} != {self.model.output_rank}."
                )
        elif architecture in ("gated", "legacy", "darcy_gated"):
            self.model = ReactionDeepONet(**config).to(self.device)
        else:
            raise ValueError(
                f"Unsupported DeepONet architecture={architecture!r}; "
                "expected 'gated', 'standard', 'standard_mlp', or 'generalized'."
            )
        self.model.load_state_dict(_model_state(self.checkpoint), strict=True)
        self.model.eval()

        values = _stats(self.checkpoint)
        self.u_mean = _scalar_stat(
            _stat_value(values, "u_mean", "x_mean", "branch_mean"),
            device=self.device,
            name="u_mean",
        )
        self.u_std = _scalar_stat(
            _stat_value(values, "u_std", "x_std", "branch_std"),
            device=self.device,
            name="u_std",
        )
        self.p_mean = _scalar_stat(
            _stat_value(values, "p_mean", "y_mean", "target_mean"),
            device=self.device,
            name="p_mean",
        )
        self.p_std = _scalar_stat(
            _stat_value(values, "p_std", "y_std", "target_std"),
            device=self.device,
            name="p_std",
        )
        if bool(self.u_std <= 0) or bool(self.p_std <= 0):
            raise ValueError("u_std and p_std must be strictly positive.")

        x_stored = _grid_value(self.checkpoint, values, "x_grid")
        y_stored = _grid_value(self.checkpoint, values, "y_grid")
        if x_stored is None and y_stored is None:
            x_grid, y_grid = regular_grid_coordinates(
                DEFAULT_BRANCH_GRID_SIZE,
                DEFAULT_BRANCH_GRID_SIZE,
                device=self.device,
            )
        elif x_stored is None or y_stored is None:
            raise KeyError("Checkpoint must store both x_grid and y_grid.")
        else:
            x_grid = torch.as_tensor(
                x_stored,
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1)
            y_grid = torch.as_tensor(
                y_stored,
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1)
        if x_grid.numel() != 120 or y_grid.numel() != 120:
            raise ValueError("DeepONet branch x_grid and y_grid must have 120 entries.")
        if not torch.all(torch.isfinite(x_grid)) or not torch.all(
            torch.isfinite(y_grid)
        ):
            raise ValueError("Branch coordinate grids contain non-finite values.")
        self.x_grid = x_grid
        self.y_grid = y_grid

        stored_output_coordinates = _grid_value(
            self.checkpoint,
            values,
            "output_coordinates",
            "node_coords",
        )
        if stored_output_coordinates is None:
            grid_x, grid_y = torch.meshgrid(
                self.x_grid,
                self.y_grid,
                indexing="ij",
            )
            stored_output_coordinates = torch.stack(
                (grid_x.reshape(-1), grid_y.reshape(-1)),
                dim=1,
            )
        default_coordinates = torch.as_tensor(
            stored_output_coordinates,
            dtype=torch.float32,
            device=self.device,
        )
        if default_coordinates.ndim != 2 or default_coordinates.shape[1] != 2:
            raise ValueError(
                "Checkpoint output_coordinates must have shape [N,2]."
            )
        self.output_coordinates = default_coordinates

        if self._is_generalized:
            n_branch_points = int(self.x_grid.numel() * self.y_grid.numel())
            if int(self.model.n_input_points) != n_branch_points:
                raise ValueError(
                    "Generalized DeepONet embedded input basis point count does "
                    "not match checkpoint branch grid: "
                    f"{self.model.n_input_points} != {n_branch_points}."
                )
            if int(self.model.n_output_points) != int(
                self.output_coordinates.shape[0]
            ):
                raise ValueError(
                    "Generalized DeepONet embedded POD point count does not match "
                    "checkpoint output_coordinates: "
                    f"{self.model.n_output_points} != "
                    f"{self.output_coordinates.shape[0]}."
                )

    @property
    def _is_generalized(self) -> bool:
        return self.architecture in (
            "generalized",
            "generalised",
            "reduced_basis",
            "rb_dino",
        )

    def _require_generalized_full_grid(
        self,
        coordinates: torch.Tensor,
    ) -> None:
        """Require the fixed POD output grid used by the reduced model.

        Generalized DeepONet has a fixed POD decoder, unlike the coordinate
        trunk models.  It therefore cannot evaluate arbitrary coordinate
        subsets.  The broad Landweber launcher already uses a 14400-point
        DeepONet chunk, so it enters this wrapper once on the complete grid.
        """

        expected = self.output_coordinates.unsqueeze(0)
        if coordinates.shape[0] > 1:
            expected = expected.expand(coordinates.shape[0], -1, -1)
        if coordinates.shape != expected.shape or not torch.allclose(
            coordinates,
            expected,
            rtol=0.0,
            atol=2.0e-7,
        ):
            raise ValueError(
                "Generalized DeepONet uses a fixed POD output grid and must be "
                "called on the complete checkpoint output_coordinates. For the "
                "120x120 Landweber run set DEEPONET_QUERY_CHUNK=14400."
            )

    def _input(self, physical_u: Any) -> torch.Tensor:
        return _deeponet_input_tensor(
            physical_u,
            device=self.device,
            name="physical_u",
        )

    def _coordinates(
        self,
        coordinates: Any | None,
        batch_size: int,
    ) -> torch.Tensor:
        return _coordinate_tensor(
            self.output_coordinates if coordinates is None else coordinates,
            batch_size=batch_size,
            device=self.device,
        )

    def forward_tensor(
        self,
        physical_u: torch.Tensor,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        if self._is_generalized:
            self._require_generalized_full_grid(coordinates)
            # The reduced model owns its physical-unit input PCA centering and
            # its physical-unit POD output mean/scaling.  Do not apply the
            # scalar training statistics used by the other DeepONets here.
            return self.model(physical_u)

        normalized_u = (physical_u - self.u_mean) / self.u_std
        if self.architecture in ("standard_mlp", "separable_mlp", "canonical_mlp"):
            branch = normalized_u
        else:
            branch = add_coordinate_channels(
                normalized_u,
                self.x_grid,
                self.y_grid,
            )
        trunk = fourier_encode_coordinates(
            coordinates,
            num_frequencies=self.model.num_frequencies,
        )
        normalized_p = self.model(branch, trunk)
        return self.p_mean + self.p_std * normalized_p

    @torch.inference_mode()
    def forward(
        self,
        physical_u: Any,
        coordinates: Any | None = None,
    ) -> np.ndarray:
        u = self._input(physical_u)
        xy = self._coordinates(coordinates, u.shape[0])
        return _numpy_deeponet_output(self.forward_tensor(u, xy))

    def jacobian_vector_product(
        self,
        physical_u: Any,
        direction: Any,
        coordinates: Any | None = None,
    ) -> np.ndarray:
        u = self._input(physical_u)
        h = _deeponet_direction(direction, u, device=self.device)
        xy = self._coordinates(coordinates, u.shape[0])

        if self._is_generalized:
            self._require_generalized_full_grid(xy)
            # Exact reduced derivative
            #   h -> F_M h -> D phi(F_M u)[F_M h] -> B_N(...).
            x = self.model.encode_input(u)
            dx = self.model.encode_direction(h)
            _, dz = jvp(
                self.model.reduced_map,
                (x,),
                (dx,),
                strict=False,
            )
            derivative = self.model.decode_output_jvp(dz)
            return _numpy_deeponet_output(derivative)

        def physical_forward(argument: torch.Tensor) -> torch.Tensor:
            return self.forward_tensor(argument, xy)

        _, derivative = jvp(
            physical_forward,
            (u,),
            (h,),
            strict=False,
        )
        return _numpy_deeponet_output(derivative)

    def jacobian_transpose_vector_product(
        self,
        physical_u: Any,
        output_dual: Any,
        coordinates: Any | None = None,
    ) -> np.ndarray:
        u = self._input(physical_u)
        xy = self._coordinates(coordinates, u.shape[0])

        if self._is_generalized:
            self._require_generalized_full_grid(xy)

            # The POD decoder is
            #   p = p_bar + (c_bar + sigma_out * z) Phi_out^T.
            # Therefore an Euclidean physical-output covector q induces
            #   g_z = sigma_out * (q Phi_out).
            # Reverse-mode AD is then required only through the small reduced
            # nonlinear map phi, after which F_M^T is applied analytically.
            template = torch.empty(
                (
                    u.shape[0],
                    int(self.model.n_output_points),
                    1,
                ),
                dtype=u.dtype,
                device=self.device,
            )
            dual = _deeponet_output_dual(
                output_dual,
                template,
                device=self.device,
            )[..., 0]

            x = self.model.encode_input(u).detach().requires_grad_(True)
            z = self.model.reduced_map(x)
            reduced_dual = (
                dual @ self.model.output_physical_basis
            ) * self.model.output_coefficient_std[None, :]
            grad_x = torch.autograd.grad(
                torch.sum(z * reduced_dual),
                x,
                create_graph=False,
                retain_graph=False,
            )[0]

            grad_flat = (
                (
                    grad_x
                    / self.model.input_coefficient_std[None, :]
                )
                @ self.model.input_weighted_modes.T
            ) * self.model.input_sqrt_weights[None, :]
            gradient = grad_flat.reshape(
                u.shape[0],
                1,
                int(self.x_grid.numel()),
                int(self.y_grid.numel()),
            )
            return _numpy_deeponet_gradient(gradient)

        u = u.detach().requires_grad_(True)
        p = self.forward_tensor(u, xy)
        dual = _deeponet_output_dual(
            output_dual,
            p,
            device=self.device,
        )
        gradient = torch.autograd.grad(
            torch.sum(p * dual),
            u,
            create_graph=False,
            retain_graph=False,
        )[0]
        return _numpy_deeponet_gradient(gradient)

    jvp = jacobian_vector_product
    vjp = jacobian_transpose_vector_product


def load_reaction_emulator(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = "auto",
) -> ReactionFNOEmulator | ReactionDeepONetEmulator:
    """Load the appropriate wrapper using the checkpoint ``model_type``."""

    resolved_device = _resolve_device(device)
    checkpoint = _load_checkpoint(checkpoint_path, resolved_device)
    model_type = str(checkpoint.get("model_type", "")).lower()
    checkpoint_format = str(checkpoint.get("checkpoint_format", "")).lower()
    if model_type in ("fno", "reaction_fno") or checkpoint_format == (
        FNO_CHECKPOINT_FORMAT
    ):
        return ReactionFNOEmulator(checkpoint_path, device=resolved_device)
    if model_type in ("deeponet", "reaction_deeponet") or checkpoint_format == (
        DEEPONET_CHECKPOINT_FORMAT
    ):
        return ReactionDeepONetEmulator(
            checkpoint_path,
            device=resolved_device,
        )
    raise ValueError(
        "Cannot infer emulator type. Store model_type='fno' or 'deeponet' "
        "and the corresponding checkpoint_format."
    )


__all__ = [
    "DEEPONET_CHECKPOINT_FORMAT",
    "FNO_CHECKPOINT_FORMAT",
    "ReactionDeepONetEmulator",
    "ReactionFNOEmulator",
    "load_reaction_emulator",
]
