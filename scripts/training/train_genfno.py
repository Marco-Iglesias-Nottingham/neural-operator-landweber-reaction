"""Train the generalized fully coupled Fourier neural operator.

Architecture:
    normalized u
      -> truncated rFFT coefficients C_M
      -> real/imaginary packing
      -> globally fully connected residual MLP Phi_theta
      -> output Fourier coefficients
      -> inverse rFFT synthesis S_N
      -> normalized p.

Unlike a standard FNO, the learned spectral network couples *all retained
Fourier modes directly*.  The default broad-amplitude experiment uses 1000
active training samples, 16x16 input modes, 12x12 output modes, hidden width
512, four residual MLP blocks, DINO weight 0.01, and no Sobolev term.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
for search_path in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from models.fno2d_generalized import (  # noqa: E402
    GeneralizedFNO2d,
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
from train_reaction_fno_dino import (  # noqa: E402
    checkpoint_payload,
    checkpoint_stats,
    compact_audit,
    compute_stats,
    evaluate_dino_balanced,
    evaluate_forward,
    inference_checkpoint_payload,
    scan_history,
    train_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--val-shards", nargs="+", required=True)
    parser.add_argument("--train-jvp-sidecars", nargs="+", default=None)
    parser.add_argument("--val-jvp-sidecars", nargs="+", default=None)
    parser.add_argument("--expected-jvp-basis-hash", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "checkpoints/broad_amp/"
            "reaction_fno_generalized_broad_1000_im16_om12_h512_rb4.pt"
        ),
    )

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-samples-limit", type=int, default=1000)
    parser.add_argument("--stats-samples", type=int, default=0)

    parser.add_argument("--input-modes-x", type=int, default=16)
    parser.add_argument("--input-modes-y", type=int, default=16)
    parser.add_argument("--output-modes-x", type=int, default=12)
    parser.add_argument("--output-modes-y", type=int, default=12)
    parser.add_argument("--hidden-width", type=int, default=512)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--residual-scale", type=float, default=0.5)

    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-9)
    parser.add_argument("--sobolev-weight", type=float, default=0.0)
    parser.add_argument("--dino-weight", type=float, default=0.01)
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

    parser.add_argument("--expected-train-shards", type=int, default=2)
    parser.add_argument("--expected-val-shards", type=int, default=4)
    parser.add_argument("--expected-samples-per-shard", type=int, default=625)
    parser.add_argument("--expected-train-base-seed", type=int, default=240000)
    parser.add_argument("--expected-val-base-seed", type=int, default=1240000)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "input_modes_x": args.input_modes_x,
        "input_modes_y": args.input_modes_y,
        "output_modes_x": args.output_modes_x,
        "output_modes_y": args.output_modes_y,
        "hidden_width": args.hidden_width,
        "dino_validation_every": args.dino_validation_every,
        "early_stop_min_epochs": args.early_stop_min_epochs,
        "expected_train_shards": args.expected_train_shards,
        "expected_val_shards": args.expected_val_shards,
        "expected_samples_per_shard": args.expected_samples_per_shard,
    }
    invalid = [name for name, value in positive.items() if int(value) < 1]
    if invalid:
        raise ValueError(f"These arguments must be positive: {invalid}.")
    if args.num_workers < 0 or args.residual_blocks < 0:
        raise ValueError("Worker/residual-block counts cannot be negative.")
    if args.residual_scale <= 0.0:
        raise ValueError("--residual-scale must be positive.")
    if args.save_every < 0 or args.snapshot_every < 0:
        raise ValueError("Save/snapshot intervals cannot be negative.")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("Invalid optimizer hyperparameters.")
    if args.sobolev_weight < 0.0 or args.dino_weight < 0.0:
        raise ValueError("Loss weights cannot be negative.")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad-clip must be positive.")
    if args.early_stop_patience < 0 or args.early_stop_min_delta < 0.0:
        raise ValueError("Early-stopping patience/delta cannot be negative.")
    if args.train_samples_limit == 0 or args.train_samples_limit < -1:
        raise ValueError("--train-samples-limit must be -1 or positive.")


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

    available = int(layout["train_samples"])
    active = (
        available
        if args.train_samples_limit < 0
        else int(args.train_samples_limit)
    )
    if not 1 <= active <= available:
        raise ValueError(
            f"Invalid active training sample count {active}; "
            f"available={available}."
        )

    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(requested_device)

    resume_checkpoint: dict[str, object] | None = None
    if args.resume is not None:
        resume_checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )
        if resume_checkpoint.get("checkpoint_format") != "reaction-fno-v1":
            raise ValueError("Resume checkpoint is not a Reaction FNO.")
        if resume_checkpoint.get("model_type") != "fno":
            raise ValueError("Resume checkpoint model_type is not 'fno'.")
        config = resume_checkpoint.get("model_config", {})
        if not isinstance(config, dict) or config.get("architecture") not in ("generalized_fourier", "generalized_fno", "gfno"):
            raise ValueError("Resume checkpoint is not a generalized FNO.")
        for required in (
            "optimizer_state",
            "scheduler_state",
            "training_state",
        ):
            if required not in resume_checkpoint:
                raise ValueError(
                    "Resume requires the full base checkpoint, not an "
                    f"inference-only checkpoint; missing {required!r}."
                )
        stats = checkpoint_stats(resume_checkpoint.get("stats"))
    else:
        stats_dataset = ReactionH5Dataset(train_paths, include_jvp=False)
        try:
            stats_count = (
                active
                if args.stats_samples <= 0
                else min(active, args.stats_samples)
            )
            stats = compute_stats(stats_dataset, stats_count)
        finally:
            stats_dataset.close()

    use_dino = args.dino_weight > 0.0
    train_jvp_paths, val_jvp_paths, derivative_contract = prepare_jvp_overlays(
        args, train_paths, val_paths, use_dino=use_dino
    )
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
            val_paths,
            include_jvp=True,
            include_output=False,
            jvp_sidecar_paths=val_jvp_paths,
        )
        if use_dino
        else None
    )

    n_modes = int(train_dataset.directions.shape[0])
    train_sampler = (
        EpochRandomModeSampler(active, n_modes, seed=args.seed)
        if use_dino
        else EpochShuffleSampler(active, seed=args.seed)
    )

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
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
        val_dino_sampler = BalancedValidationModeSampler(
            len(val_dino_dataset), n_modes
        )
        val_dino_loader = DataLoader(
            val_dino_dataset,
            batch_size=args.batch_size,
            sampler=val_dino_sampler,
            num_workers=validation_workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            persistent_workers=validation_workers > 0,
        )

    directions_np = np.asarray(
        derivative_contract["directions"] if derivative_contract else train_audit["directions"],
        dtype=np.float32,
    )
    directions = torch.as_tensor(
        directions_np, dtype=torch.float32, device=device
    )

    constructor_config = {
        "in_channels": 1,
        "out_channels": 1,
        "input_modes_x": args.input_modes_x,
        "input_modes_y": args.input_modes_y,
        "output_modes_x": args.output_modes_x,
        "output_modes_y": args.output_modes_y,
        "hidden_width": args.hidden_width,
        "residual_blocks": args.residual_blocks,
        "residual_scale": args.residual_scale,
    }
    model_config: dict[str, object] = {
        "architecture": "generalized_fourier",
        **constructor_config,
    }
    model = GeneralizedFNO2d(**constructor_config).to(device)
    params = count_trainable_parameters(model)

    if resume_checkpoint is not None:
        if resume_checkpoint.get("model_config") != model_config:
            raise ValueError(
                "Resume checkpoint architecture/configuration differs from "
                "the requested generalized FNO."
            )
        if not np.array_equal(
            np.asarray(resume_checkpoint.get("directions_u")), directions_np
        ):
            raise ValueError("Resume checkpoint derivative directions differ from this campaign.")
        model.load_state_dict(resume_checkpoint["model_state"])

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
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
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        history = list(resume_checkpoint.get("history", []))
        best_forward, best_composite = scan_history(history)
        state = resume_checkpoint.get("training_state", {})
        if isinstance(state, dict):
            best_forward = float(state.get("best_forward", best_forward))
            best_composite = float(
                state.get("best_composite", best_composite)
            )
            epochs_without_improvement = int(
                state.get("epochs_without_forward_improvement", 0)
            )
            dino_validation_round = int(
                state.get("dino_validation_round", 0)
            )

    if start_epoch > args.epochs:
        raise ValueError(
            f"Resume starts at epoch {start_epoch}, beyond "
            f"--epochs={args.epochs}."
        )

    output = args.output
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    best_forward_path = Path(f"{stem}.best_forward.pt")
    best_composite_path = Path(f"{stem}.best_composite.pt")
    history_path = Path(f"{stem}.history.json")

    training_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    shard_audit = {
        "layout": {
            **dict(layout),
            "available_train_samples": available,
            "active_train_samples": active,
        },
        "training": compact_audit(train_audit),
        "validation": compact_audit(val_audit),
        "derivative_supervision": (
            None if derivative_contract is None else {
                key: value for key, value in derivative_contract.items() if key != "directions"
            }
        ),
    }

    print("=" * 72)
    print("Reaction GENERALIZED FNO DINO training")
    print(f"Training samples   : {active} active")
    print(f"Output             : {output}")
    print(f"Max epochs         : {args.epochs}")
    print(f"Snapshots          : every {args.snapshot_every} epochs")
    print(
        "Architecture       : GENERALIZED fully coupled Fourier MLP; "
        f"input modes={args.input_modes_x}x{args.input_modes_y}, "
        f"output modes={args.output_modes_x}x{args.output_modes_y}"
    )
    print(
        f"Global MLP         : hidden={args.hidden_width}, "
        f"residual blocks={args.residual_blocks}, scale={args.residual_scale:g}"
    )
    print(
        f"Fourier dimensions : M={model.input_dimension}, "
        f"N={model.output_dimension}"
    )
    print(f"Trainable params    : {params:,}")
    print("=" * 72)
    print(f"Device:             {device}")
    print(f"Shard layout:       {shard_audit['layout']}")
    print(f"Grid:               {train_audit['shape']}")
    print(f"DINO directions:    {directions.shape[0]}")
    print(f"Training batches:   {len(train_loader)} per epoch")
    print(f"Forward-val batches:{len(val_forward_loader)} per epoch")
    print(
        "Derivative validation: "
        f"epoch 1, every {args.dino_validation_every}, and final"
    )
    print("Normalization:", {k: v.tolist() for k, v in stats.items()})

    start = perf_counter()
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = perf_counter()
            train_sampler.set_epoch(epoch)

            train_metrics = train_epoch(
                model=model,
                loader=train_loader,
                directions=directions,
                stats=stats,
                device=device,
                optimizer=optimizer,
                sobolev_weight=args.sobolev_weight,
                dino_weight=args.dino_weight,
                grad_clip=args.grad_clip,
            )
            val_metrics = evaluate_forward(
                model=model,
                loader=val_forward_loader,
                stats=stats,
                device=device,
                sobolev_weight=args.sobolev_weight,
            )
            forward_objective = float(val_metrics["loss"])
            scheduler.step(forward_objective)

            improved_forward = (
                forward_objective
                < best_forward - args.early_stop_min_delta
            )
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
                dino_validation_is_due(
                    epoch,
                    args.epochs,
                    args.dino_validation_every,
                )
                or stop_requested
            ):
                assert val_dino_sampler is not None
                assert val_dino_loader is not None
                val_dino_sampler.set_round(0)
                value, per_mode, counts = evaluate_dino_balanced(
                    model=model,
                    loader=val_dino_loader,
                    directions=directions,
                    stats=stats,
                    device=device,
                )
                val_dino = float(value)
                val_dino_per_mode = [
                    None if not np.isfinite(item) else float(item)
                    for item in per_mode
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
            improved_composite = (
                composite is not None and composite < best_composite
            )
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
                    "loss": (
                        composite
                        if composite is not None
                        else forward_objective
                    ),
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
                "epochs_without_forward_improvement": (
                    epochs_without_improvement
                ),
                "dino_validation_round": dino_validation_round,
                "stopped_early": stop_requested,
            }
            payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                stats=stats,
                model_config=model_config,
                directions=directions_np,
                train_paths=train_paths,
                val_paths=val_paths,
                history=history,
                training_state=state,
                training_args=training_args,
                shard_audit=shard_audit,
            )

            if improved_forward:
                torch.save(
                    inference_checkpoint_payload(payload),
                    best_forward_path,
                )
            if improved_composite:
                torch.save(
                    inference_checkpoint_payload(payload),
                    best_composite_path,
                )
            if (
                (args.save_every > 0 and epoch % args.save_every == 0)
                or epoch == args.epochs
                or stop_requested
            ):
                torch.save(payload, output)
            if (
                args.snapshot_every > 0
                and (
                    epoch % args.snapshot_every == 0
                    or epoch == args.epochs
                    or stop_requested
                )
            ):
                torch.save(
                    inference_checkpoint_payload(payload),
                    Path(f"{stem}.epoch{epoch:04d}.pt"),
                )

            history_path.write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )

            dino_text = (
                "not-evaluated"
                if val_dino is None
                else f"{val_dino:.4e}"
            )
            print(
                f"epoch {epoch:04d} | "
                f"train={train_metrics['loss']:.4e} | "
                f"val_fwd_obj={forward_objective:.4e} | "
                f"val_fwd={val_metrics['forward']:.4e} | "
                f"val_jvp={dino_text} | "
                f"relL2={val_metrics['relative_l2']:.4e} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.1f}s",
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

    print(
        f"Training completed in {(perf_counter() - start) / 3600.0:.2f} h"
    )


if __name__ == "__main__":
    main()
