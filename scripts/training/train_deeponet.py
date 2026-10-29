#!/usr/bin/env python3
"""Train a standard/separable DeepONet with full-grid DINO supervision.

Compared with the legacy gated DeepONet trainer, this script uses the
canonical DeepONet factorisation

    p_hat(u;x) = <b(u), tau(x)> / sqrt(Dp) + bias,

where the coordinate trunk tau is independent of u.  The full 120x120 output
grid (14400 points) is used by default for both forward and JVP losses.

For DINO, the JVP is taken only through the branch coefficients b(u), which is
mathematically exact because tau(x) is independent of u.  This avoids carrying
a [B,N,Dp] branch-conditioned hidden state through every trunk layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.func import jvp
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
for search_path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from models.deeponet import add_coordinate_channels, fourier_encode_coordinates  # noqa: E402
from models.deeponet_standard import (  # noqa: E402
    ReactionStandardDeepONet,
    count_trainable_parameters,
)
from reaction_core.datasets import (  # noqa: E402
    ReactionH5Dataset,
    SelectiveReactionH5Dataset,
)
from training_utils import (  # noqa: E402
    BalancedValidationModeSampler,
    EpochRandomModeSampler,
    EpochShuffleSampler,
    assert_disjoint_splits,
    assert_expected_independent_layout,
    audit_shards,
    dino_validation_is_due,
    expand_shard_patterns,
    prepare_jvp_overlays,
    seed_everything,
    seed_worker,
)


METRIC_NAMES = ("loss", "forward", "dino", "relative_l2")
CHECKPOINT_FORMAT = "reaction-deeponet-v1"
ARCHITECTURE = "standard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train the standard/separable full-grid Reaction DeepONet.",
    )
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--val-shards", nargs="+", required=True)
    parser.add_argument("--train-jvp-sidecars", nargs="+", default=None)
    parser.add_argument("--val-jvp-sidecars", nargs="+", default=None)
    parser.add_argument("--expected-jvp-basis-hash", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/reaction_deeponet_standard_dino.pt",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stats-samples", type=int, default=0)
    parser.add_argument(
        "--train-samples-limit",
        type=int,
        default=0,
        help="Use only the first N samples across the supplied train shards; 0 uses all.",
    )
    parser.add_argument("--Dp", type=int, default=400)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-frequencies", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-9)
    parser.add_argument("--dino-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.5)
    parser.add_argument("--dino-validation-every", type=int, default=10)
    parser.add_argument("--early-stop-min-epochs", type=int, default=120)
    parser.add_argument("--early-stop-patience", type=int, default=60)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--snapshot-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--expected-train-shards", type=int, required=True)
    parser.add_argument("--expected-val-shards", type=int, default=4)
    parser.add_argument("--expected-samples-per-shard", type=int, default=625)
    parser.add_argument("--expected-train-base-seed", type=int, default=240000)
    parser.add_argument("--expected-val-base-seed", type=int, default=1240000)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "Dp": args.Dp,
        "num_layers": args.num_layers,
        "num_frequencies": args.num_frequencies,
        "dino_validation_every": args.dino_validation_every,
        "early_stop_min_epochs": args.early_stop_min_epochs,
        "expected_train_shards": args.expected_train_shards,
        "expected_val_shards": args.expected_val_shards,
        "expected_samples_per_shard": args.expected_samples_per_shard,
    }
    invalid = [name for name, value in positive.items() if int(value) < 1]
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}.")
    if args.num_workers < 0 or args.save_every < 0 or args.snapshot_every < 0:
        raise ValueError("Worker/save counts cannot be negative.")
    if args.train_samples_limit < 0:
        raise ValueError("--train-samples-limit cannot be negative.")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("Invalid optimizer hyperparameters.")
    if args.dino_weight < 0.0 or args.grad_clip <= 0.0:
        raise ValueError("Invalid DINO weight or gradient clipping threshold.")
    if args.early_stop_patience < 0 or args.early_stop_min_delta < 0.0:
        raise ValueError("Early-stopping patience/delta cannot be negative.")


def compute_stats(dataset: ReactionH5Dataset, max_samples: int) -> dict[str, np.ndarray]:
    count = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)
    if count < 1:
        raise ValueError("Training dataset is empty.")
    sums = np.zeros(2, dtype=np.float64)
    squares = np.zeros(2, dtype=np.float64)
    elements = np.zeros(2, dtype=np.int64)
    for index in range(count):
        u, p = dataset[index][:2]
        for channel, values in enumerate((u.numpy(), p.numpy())):
            array = np.asarray(values, dtype=np.float64)
            sums[channel] += np.sum(array)
            squares[channel] += np.sum(array * array)
            elements[channel] += array.size
        if index and index % 1000 == 0:
            print(f"statistics: {index}/{count}", flush=True)
    means = sums / elements
    variances = np.maximum(squares / elements - means**2, 1.0e-20)
    return {
        "u_mean": np.asarray([means[0]], dtype=np.float32),
        "u_std": np.asarray([np.sqrt(variances[0])], dtype=np.float32),
        "p_mean": np.asarray([means[1]], dtype=np.float32),
        "p_std": np.asarray([np.sqrt(variances[1])], dtype=np.float32),
    }


def checkpoint_stats(raw: object) -> dict[str, np.ndarray]:
    if not isinstance(raw, dict):
        raise ValueError("Resume checkpoint has invalid normalization stats.")
    result: dict[str, np.ndarray] = {}
    for key in ("u_mean", "u_std", "p_mean", "p_std"):
        if key not in raw:
            raise ValueError(f"Resume checkpoint stats are missing {key!r}.")
        value = np.asarray(raw[key], dtype=np.float32).reshape(-1)
        if value.size != 1 or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid resume statistic {key!r}: {value}.")
        result[key] = value
    if result["u_std"][0] <= 0.0 or result["p_std"][0] <= 0.0:
        raise ValueError("Resume standard deviations must be positive.")
    return result


def scalar_stat(stats: dict[str, np.ndarray], key: str, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(stats[key], dtype=torch.float32, device=device).reshape(1, 1, 1, 1)


def grid_coordinates(x: np.ndarray, y: np.ndarray, device: torch.device) -> torch.Tensor:
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y, dtype=torch.float32, device=device)
    grid_x, grid_y = torch.meshgrid(x_tensor, y_tensor, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)


def averaged(totals: dict[str, float], count: int) -> dict[str, float]:
    if count < 1:
        raise RuntimeError("DataLoader yielded no samples.")
    return {name: totals[name] / count for name in METRIC_NAMES}


def branch_coefficients(
    model: ReactionStandardDeepONet,
    normalized_u: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
) -> torch.Tensor:
    branch_input = add_coordinate_channels(normalized_u, x_grid, y_grid)
    return model.branch_net(branch_input)


def train_epoch(
    *,
    model: ReactionStandardDeepONet,
    loader: DataLoader,
    directions: torch.Tensor,
    encoded_coordinates: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    stats: dict[str, np.ndarray],
    device: torch.device,
    optimizer: AdamW,
    dino_weight: float,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    u_mean = scalar_stat(stats, "u_mean", device)
    u_std = scalar_stat(stats, "u_std", device)
    p_mean = scalar_stat(stats, "p_mean", device).reshape(())
    p_std = scalar_stat(stats, "p_std", device).reshape(())
    totals = {name: 0.0 for name in METRIC_NAMES}
    seen = 0
    use_dino = dino_weight > 0.0

    for batch in loader:
        if use_dino:
            u, p, mode_indices, exact_jvp = batch
            mode_indices = mode_indices.to(device=device, dtype=torch.long, non_blocking=True)
            exact_jvp = exact_jvp.to(device, non_blocking=True)
        else:
            u, p = batch
        u = u.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)
        batch_size = int(u.shape[0])
        normalized_u = (u - u_mean) / u_std
        target_p = ((p[:, 0].reshape(batch_size, -1) - p_mean) / p_std).unsqueeze(-1)

        optimizer.zero_grad(set_to_none=True)

        # The trunk is independent of u and shared by the whole batch.
        trunk_basis = model.trunk_net(encoded_coordinates)  # [14400,Dp]

        if use_dino:
            direction = torch.index_select(directions, 0, mode_indices).unsqueeze(1)

            def branch_map(argument: torch.Tensor) -> torch.Tensor:
                return branch_coefficients(model, argument, x_grid, y_grid)

            coefficients, coefficient_jvp = jvp(
                branch_map,
                (normalized_u,),
                (direction / u_std,),
                strict=False,
            )
            prediction = model.combine(coefficients, trunk_basis, include_bias=True)
            predicted_jvp = model.combine(
                coefficient_jvp,
                trunk_basis,
                include_bias=False,
            )
            target_jvp = (exact_jvp[:, 0].reshape(batch_size, -1) / p_std).unsqueeze(-1)
            dino_loss = torch.mean((predicted_jvp - target_jvp) ** 2)
        else:
            coefficients = branch_coefficients(model, normalized_u, x_grid, y_grid)
            prediction = model.combine(coefficients, trunk_basis, include_bias=True)
            dino_loss = prediction.new_zeros(())

        forward_loss = torch.mean((prediction - target_p) ** 2)
        loss = forward_loss + dino_weight * dino_loss
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            prediction_physical = p_mean + p_std * prediction[..., 0]
            target_physical = p[:, 0].reshape(batch_size, -1)
            numerator = torch.linalg.vector_norm(prediction_physical - target_physical, dim=1)
            denominator = torch.linalg.vector_norm(target_physical, dim=1).clamp_min(1.0e-12)
            rel = torch.mean(numerator / denominator)

        values = {
            "loss": loss,
            "forward": forward_loss,
            "dino": dino_loss,
            "relative_l2": rel,
        }
        for name, value in values.items():
            totals[name] += float(value.detach()) * batch_size
        seen += batch_size

    return averaged(totals, seen)


def evaluate_forward(
    *,
    model: ReactionStandardDeepONet,
    loader: DataLoader,
    encoded_coordinates: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    stats: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    u_mean = scalar_stat(stats, "u_mean", device)
    u_std = scalar_stat(stats, "u_std", device)
    p_mean = scalar_stat(stats, "p_mean", device).reshape(())
    p_std = scalar_stat(stats, "p_std", device).reshape(())
    totals = {name: 0.0 for name in METRIC_NAMES}
    seen = 0

    with torch.inference_mode():
        trunk_basis = model.trunk_net(encoded_coordinates)
        for u, p in loader:
            u = u.to(device, non_blocking=True)
            p = p.to(device, non_blocking=True)
            batch_size = int(u.shape[0])
            coefficients = branch_coefficients(model, (u - u_mean) / u_std, x_grid, y_grid)
            prediction = model.combine(coefficients, trunk_basis, include_bias=True)
            target = ((p[:, 0].reshape(batch_size, -1) - p_mean) / p_std).unsqueeze(-1)
            forward_loss = torch.mean((prediction - target) ** 2)
            prediction_physical = p_mean + p_std * prediction[..., 0]
            target_physical = p[:, 0].reshape(batch_size, -1)
            numerator = torch.linalg.vector_norm(prediction_physical - target_physical, dim=1)
            denominator = torch.linalg.vector_norm(target_physical, dim=1).clamp_min(1.0e-12)
            rel = torch.mean(numerator / denominator)
            values = {
                "loss": forward_loss,
                "forward": forward_loss,
                "dino": prediction.new_zeros(()),
                "relative_l2": rel,
            }
            for name, value in values.items():
                totals[name] += float(value.detach()) * batch_size
            seen += batch_size
    return averaged(totals, seen)


def evaluate_dino_balanced(
    *,
    model: ReactionStandardDeepONet,
    loader: DataLoader,
    directions: torch.Tensor,
    encoded_coordinates: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    stats: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    u_mean = scalar_stat(stats, "u_mean", device)
    u_std = scalar_stat(stats, "u_std", device)
    p_std = scalar_stat(stats, "p_std", device).reshape(())
    n_modes = int(directions.shape[0])
    mode_sums = np.zeros(n_modes, dtype=np.float64)
    mode_counts = np.zeros(n_modes, dtype=np.int64)

    with torch.no_grad():
        trunk_basis = model.trunk_net(encoded_coordinates)
        for u, mode_indices, exact_jvp in loader:
            u = u.to(device, non_blocking=True)
            mode_indices = mode_indices.to(device=device, dtype=torch.long, non_blocking=True)
            exact_jvp = exact_jvp.to(device, non_blocking=True)
            batch_size = int(u.shape[0])
            normalized_u = (u - u_mean) / u_std
            direction = torch.index_select(directions, 0, mode_indices).unsqueeze(1)

            def branch_map(argument: torch.Tensor) -> torch.Tensor:
                return branch_coefficients(model, argument, x_grid, y_grid)

            _, coefficient_jvp = jvp(
                branch_map,
                (normalized_u,),
                (direction / u_std,),
                strict=False,
            )
            predicted_jvp = model.combine(
                coefficient_jvp,
                trunk_basis,
                include_bias=False,
            )
            target_jvp = (exact_jvp[:, 0].reshape(batch_size, -1) / p_std).unsqueeze(-1)
            per_sample = torch.mean((predicted_jvp - target_jvp) ** 2, dim=(1, 2))
            for mode, error in zip(
                mode_indices.detach().cpu().tolist(),
                per_sample.detach().cpu().tolist(),
            ):
                mode_sums[int(mode)] += float(error)
                mode_counts[int(mode)] += 1

    if int(np.sum(mode_counts)) < 1:
        raise RuntimeError("Derivative validation DataLoader yielded no samples.")
    per_mode = np.full(n_modes, np.nan, dtype=np.float64)
    populated = mode_counts > 0
    per_mode[populated] = mode_sums[populated] / mode_counts[populated]
    return float(np.sum(mode_sums) / np.sum(mode_counts)), per_mode, mode_counts


def compact_audit(audit: dict[str, object]) -> dict[str, object]:
    return {
        "paths": [str(path) for path in audit["paths"]],
        "n_shards": int(audit["n_shards"]),
        "n_samples": int(audit["n_samples"]),
        "sample_counts": [int(value) for value in audit["sample_counts"]],
        "shape": list(audit["shape"]),
        "seed_records": list(audit["seed_records"]),
        "project_config": audit["project_config"],
    }


def inference_checkpoint_payload(full_payload: dict[str, object]) -> dict[str, object]:
    keys = (
        "checkpoint_format",
        "model_type",
        "epoch",
        "model_config",
        "model_state",
        "stats",
        "directions_u",
        "grid",
        "train_shards",
        "val_shards",
        "training_state",
        "training_args",
        "shard_audit",
        "dino_sampling",
        "validation_query_indices",
    )
    result = {key: full_payload[key] for key in keys}
    history = full_payload.get("history", [])
    result["last_history_row"] = history[-1] if history else None
    result["inference_only"] = True
    return result


def scan_history(history: list[dict[str, object]]) -> tuple[float, float]:
    best_forward = float("inf")
    best_composite = float("inf")
    for row in history:
        val = row.get("val")
        if not isinstance(val, dict):
            continue
        if val.get("forward") is not None:
            best_forward = min(best_forward, float(val["forward"]))
        composite = val.get("composite")
        if composite is not None:
            best_composite = min(best_composite, float(composite))
    return best_forward, best_composite


def validate_resume_checkpoint(checkpoint: dict[str, Any], args: argparse.Namespace) -> None:
    if checkpoint.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError("Resume checkpoint has the wrong format.")
    if checkpoint.get("model_type") != "deeponet":
        raise ValueError("Resume checkpoint is not a DeepONet.")
    config = checkpoint.get("model_config")
    if not isinstance(config, dict) or config.get("architecture") != ARCHITECTURE:
        raise ValueError("Resume checkpoint is not a standard/separable DeepONet.")
    saved_args = checkpoint.get("training_args")
    if not isinstance(saved_args, dict):
        raise ValueError("Resume checkpoint has invalid training_args.")
    critical = (
        "Dp",
        "num_layers",
        "num_frequencies",
        "learning_rate",
        "weight_decay",
        "dino_weight",
        "seed",
        "train_samples_limit",
        "expected_train_shards",
        "expected_val_shards",
        "expected_samples_per_shard",
        "expected_train_base_seed",
        "expected_val_base_seed",
    )
    for name in critical:
        if name not in saved_args:
            raise ValueError(f"Resume checkpoint training_args omit {name!r}.")
        current = getattr(args, name)
        saved = saved_args[name]
        equal = (
            bool(np.isclose(float(current), float(saved), rtol=0.0, atol=0.0))
            if isinstance(current, float)
            else current == saved
        )
        if not equal:
            raise ValueError(
                f"Resume changes critical argument {name}: checkpoint={saved!r}, requested={current!r}."
            )


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    train_paths = expand_shard_patterns(args.train_shards)
    val_paths = expand_shard_patterns(args.val_shards)
    train_audit = audit_shards(train_paths, label="training")
    val_audit = audit_shards(val_paths, label="validation")
    assert_disjoint_splits(train_audit, val_audit)
    layout = assert_expected_independent_layout(
        train_audit,
        val_audit,
        expected_train_shards=args.expected_train_shards,
        expected_val_shards=args.expected_val_shards,
        expected_samples_per_shard=args.expected_samples_per_shard,
        expected_train_base_seed=args.expected_train_base_seed,
        expected_val_base_seed=args.expected_val_base_seed,
    )
    if tuple(train_audit["shape"]) != (120, 120):
        raise ValueError(f"Standard Reaction DeepONet requires 120x120 inputs; got {train_audit['shape']}.")

    available_train_samples = int(layout["train_samples"])
    active_train_samples = (
        available_train_samples
        if int(args.train_samples_limit) <= 0
        else int(args.train_samples_limit)
    )
    if not 1 <= active_train_samples <= available_train_samples:
        raise ValueError(
            f"--train-samples-limit must be in [1,{available_train_samples}]; got {args.train_samples_limit}."
        )
    layout = dict(layout)
    layout["available_train_samples"] = available_train_samples
    layout["active_train_samples"] = active_train_samples

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(requested_device)
    use_dino = args.dino_weight > 0.0
    train_jvp_paths, val_jvp_paths, derivative_contract = prepare_jvp_overlays(
        args, train_paths, val_paths, use_dino=use_dino
    )

    resume_checkpoint: dict[str, Any] | None = None
    if args.resume is not None:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_checkpoint(resume_checkpoint, args)
        stats = checkpoint_stats(resume_checkpoint.get("stats"))
    else:
        stats_dataset = ReactionH5Dataset(train_paths, include_jvp=False)
        try:
            stats_count = (
                active_train_samples
                if int(args.stats_samples) <= 0
                else min(active_train_samples, int(args.stats_samples))
            )
            stats = compute_stats(stats_dataset, stats_count)
        finally:
            stats_dataset.close()

    train_dataset = SelectiveReactionH5Dataset(
        train_paths,
        include_jvp=use_dino,
        include_output=True,
        jvp_sidecar_paths=train_jvp_paths,
    )
    val_forward_dataset = SelectiveReactionH5Dataset(
        val_paths,
        include_jvp=False,
        include_output=True,
    )
    val_dino_dataset = (
        SelectiveReactionH5Dataset(
            val_paths, include_jvp=True, include_output=False,
            jvp_sidecar_paths=val_jvp_paths,
        )
        if use_dino
        else None
    )
    n_modes = int(train_dataset.directions.shape[0])
    train_sampler = (
        EpochRandomModeSampler(active_train_samples, n_modes, seed=args.seed)
        if use_dino
        else EpochShuffleSampler(active_train_samples, seed=args.seed)
    )

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=loader_generator,
        persistent_workers=args.num_workers > 0,
    )
    validation_workers = min(args.num_workers, 2)
    val_forward_loader = DataLoader(
        val_forward_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=validation_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=validation_workers > 0,
    )
    val_dino_sampler = None
    val_dino_loader = None
    if use_dino:
        assert val_dino_dataset is not None
        val_dino_sampler = BalancedValidationModeSampler(len(val_dino_dataset), n_modes)
        val_dino_loader = DataLoader(
            val_dino_dataset,
            batch_size=args.batch_size,
            sampler=val_dino_sampler,
            num_workers=validation_workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            persistent_workers=validation_workers > 0,
        )

    x_grid_np = np.asarray(train_audit["x_grid"], dtype=np.float32)
    y_grid_np = np.asarray(train_audit["y_grid"], dtype=np.float32)
    x_grid = torch.as_tensor(x_grid_np, device=device)
    y_grid = torch.as_tensor(y_grid_np, device=device)
    coordinates = grid_coordinates(x_grid_np, y_grid_np, device)
    encoded_coordinates = fourier_encode_coordinates(
        coordinates,
        num_frequencies=args.num_frequencies,
    )
    n_total_queries = int(coordinates.shape[0])
    if n_total_queries != 120 * 120:
        raise ValueError(f"Expected 14400 output vertices; got {n_total_queries}.")

    directions_np = np.asarray(
        derivative_contract["directions"] if derivative_contract else train_audit["directions"],
        dtype=np.float32,
    )
    directions = torch.as_tensor(directions_np, device=device)

    model_config: dict[str, object] = {
        "architecture": ARCHITECTURE,
        "branch_input_channels": 3,
        "trunk_input_dim": int(encoded_coordinates.shape[-1]),
        "Dp": args.Dp,
        "num_layers": args.num_layers,
        "num_frequencies": args.num_frequencies,
    }
    model = ReactionStandardDeepONet(
        branch_input_channels=3,
        trunk_input_dim=int(encoded_coordinates.shape[-1]),
        Dp=args.Dp,
        num_layers=args.num_layers,
        num_frequencies=args.num_frequencies,
    ).to(device)
    parameter_count = count_trainable_parameters(model)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=15,
        min_lr=1.0e-6,
    )

    start_epoch = 1
    history: list[dict[str, object]] = []
    best_forward = float("inf")
    best_composite = float("inf")
    epochs_without_improvement = 0
    dino_validation_round = 0
    if resume_checkpoint is not None:
        if resume_checkpoint.get("model_config") != model_config:
            raise ValueError("Resume checkpoint model_config differs from requested standard DeepONet.")
        if not np.array_equal(np.asarray(resume_checkpoint.get("directions_u")), directions_np):
            raise ValueError("Resume checkpoint DINO directions differ from the shards.")
        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        history = list(resume_checkpoint.get("history", []))
        best_forward, best_composite = scan_history(history)
        state = resume_checkpoint.get("training_state", {})
        if isinstance(state, dict):
            best_forward = float(state.get("best_forward", best_forward))
            best_composite = float(state.get("best_composite", best_composite))
            epochs_without_improvement = int(state.get("epochs_without_forward_improvement", 0))
            dino_validation_round = int(state.get("dino_validation_round", 0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem = args.output.with_suffix("")
    best_forward_path = Path(f"{stem}.best_forward.pt")
    best_composite_path = Path(f"{stem}.best_composite.pt")
    history_path = Path(f"{stem}.history.json")
    output_coordinates = coordinates.detach().cpu().numpy().astype(np.float32)
    validation_query_indices = np.arange(n_total_queries, dtype=np.int64)
    training_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    shard_audit = {
        "layout": layout,
        "training": compact_audit(train_audit),
        "validation": compact_audit(val_audit),
        "derivative_supervision": (
            None if derivative_contract is None else {
                key: value for key, value in derivative_contract.items() if key != "directions"
            }
        ),
    }

    print(f"Device:               {device}")
    print(f"Architecture:         standard/separable DeepONet")
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Shard layout:         {layout}")
    print(f"Training batches:     {len(train_loader)} per epoch")
    print(f"Full-grid queries:    {n_total_queries} per sample")
    print(f"Dp / trunk layers:    {args.Dp} / {args.num_layers}")
    print(f"DINO weight:          {args.dino_weight:g}")
    print(
        "Derivative validation: "
        f"epoch 1, every {args.dino_validation_every}, and final"
    )
    print("Normalization:", {key: value.tolist() for key, value in stats.items()})

    start = perf_counter()
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = perf_counter()
            train_sampler.set_epoch(epoch)
            train_metrics = train_epoch(
                model=model,
                loader=train_loader,
                directions=directions,
                encoded_coordinates=encoded_coordinates,
                x_grid=x_grid,
                y_grid=y_grid,
                stats=stats,
                device=device,
                optimizer=optimizer,
                dino_weight=args.dino_weight,
                grad_clip=args.grad_clip,
            )
            val_metrics = evaluate_forward(
                model=model,
                loader=val_forward_loader,
                encoded_coordinates=encoded_coordinates,
                x_grid=x_grid,
                y_grid=y_grid,
                stats=stats,
                device=device,
            )
            forward_objective = float(val_metrics["forward"])
            scheduler.step(forward_objective)

            improved_forward = forward_objective < best_forward - args.early_stop_min_delta
            if improved_forward:
                best_forward = forward_objective
                epochs_without_improvement = 0
            elif epoch >= args.early_stop_min_epochs:
                epochs_without_improvement += 1

            stop_requested = (
                args.early_stop_patience > 0
                and epoch >= args.early_stop_min_epochs
                and epochs_without_improvement >= args.early_stop_patience
            )

            val_dino: float | None = None
            val_dino_per_mode: list[float | None] | None = None
            val_dino_counts: list[int] | None = None
            if use_dino and (
                dino_validation_is_due(epoch, args.epochs, args.dino_validation_every)
                or stop_requested
            ):
                assert val_dino_sampler is not None and val_dino_loader is not None
                val_dino_sampler.set_round(0)
                value, per_mode, counts = evaluate_dino_balanced(
                    model=model,
                    loader=val_dino_loader,
                    directions=directions,
                    encoded_coordinates=encoded_coordinates,
                    x_grid=x_grid,
                    y_grid=y_grid,
                    stats=stats,
                    device=device,
                )
                val_dino = float(value)
                val_dino_per_mode = [
                    None if not np.isfinite(item) else float(item) for item in per_mode
                ]
                val_dino_counts = [int(item) for item in counts]
                dino_validation_round += 1

            composite = (
                forward_objective
                if not use_dino
                else (
                    None
                    if val_dino is None
                    else forward_objective + args.dino_weight * val_dino
                )
            )
            improved_composite = composite is not None and composite < best_composite
            if improved_composite:
                best_composite = float(composite)

            val_history = dict(val_metrics)
            val_history.update(
                {
                    "forward_objective": forward_objective,
                    "dino": val_dino,
                    "dino_per_mode": val_dino_per_mode,
                    "dino_mode_counts": val_dino_counts,
                    "composite": composite,
                    "loss": composite if composite is not None else forward_objective,
                }
            )
            elapsed = perf_counter() - epoch_start
            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_history,
                "seconds": elapsed,
            }
            history.append(row)
            state = {
                "best_forward": best_forward,
                "best_composite": best_composite,
                "epochs_without_forward_improvement": epochs_without_improvement,
                "dino_validation_round": dino_validation_round,
                "stopped_early": stop_requested,
                "trainable_parameters": parameter_count,
            }
            payload: dict[str, object] = {
                "checkpoint_format": CHECKPOINT_FORMAT,
                "model_type": "deeponet",
                "epoch": epoch,
                "model_config": model_config,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "stats": stats,
                "directions_u": directions_np,
                "grid": {
                    "x_grid": x_grid_np,
                    "y_grid": y_grid_np,
                    "output_coordinates": output_coordinates,
                },
                "train_shards": [str(path) for path in train_paths],
                "val_shards": [str(path) for path in val_paths],
                "history": history,
                "training_state": state,
                "training_args": training_args,
                "shard_audit": shard_audit,
                "dino_sampling": {
                    "training": "one uniform random stored KL/JVP mode per sample per epoch",
                    "validation": "one fixed balanced stored mode per validation sample",
                    "spatial_supervision": "all 120x120 = 14400 output vertices",
                    "jvp_implementation": "exact branch-only JVP; trunk is independent of u",
                    "hdf5_jvp_modes_read_per_sample": 1,
                },
                "validation_query_indices": validation_query_indices,
            }

            if improved_forward:
                torch.save(inference_checkpoint_payload(payload), best_forward_path)
            if improved_composite:
                torch.save(inference_checkpoint_payload(payload), best_composite_path)
            if (
                (args.save_every > 0 and epoch % args.save_every == 0)
                or epoch == args.epochs
                or stop_requested
            ):
                torch.save(payload, args.output)
            if (
                args.snapshot_every > 0
                and (epoch % args.snapshot_every == 0 or stop_requested)
            ):
                torch.save(
                    inference_checkpoint_payload(payload),
                    Path(f"{stem}.epoch{epoch:04d}.pt"),
                )
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

            dino_text = "not-evaluated" if val_dino is None else f"{val_dino:.4e}"
            print(
                f"epoch {epoch:04d} | train={train_metrics['loss']:.4e} | "
                f"val_fwd={forward_objective:.4e} | val_jvp={dino_text} | "
                f"relL2={val_metrics['relative_l2']:.4e} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s",
                flush=True,
            )
            if stop_requested:
                print(
                    "Early stopping: no forward-validation improvement for "
                    f"{epochs_without_improvement} epochs.",
                    flush=True,
                )
                break
    finally:
        train_dataset.close()
        val_forward_dataset.close()
        if val_dino_dataset is not None:
            val_dino_dataset.close()

    print(f"Training completed in {(perf_counter() - start) / 3600.0:.2f} h")


if __name__ == "__main__":
    main()
