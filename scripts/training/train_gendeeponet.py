#!/usr/bin/env python3
"""Train the generalized / reduced-basis DeepONet.

Architecture:
    u --fixed input PCA/KL encoder F_M--> x in R^M
      --residual MLP phi_theta--> z in R^N
      --fixed output POD decoder B_N--> p.

DINO is trained entirely in reduced coordinates:

    dx = F_M h                         (exact linear projection)
    dz_theta = D phi_theta(x)[dx]
    dz_exact = projected FEM JVP.

The output coefficient loss is energy weighted so that it is equivalent, up
to one global constant, to physical W-L2 error in the retained POD subspace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from torch.func import jvp
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
for p in (PROJECT_ROOT, SCRIPT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models.deeponet_generalized import (  # noqa: E402
    ReactionGeneralizedDeepONet,
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

CHECKPOINT_FORMAT = "reaction-deeponet-v1"


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--train-shards", nargs="+", required=True)
    p.add_argument("--val-shards", nargs="+", required=True)
    p.add_argument("--train-jvp-sidecars", nargs="+", default=None)
    p.add_argument("--val-jvp-sidecars", nargs="+", default=None)
    p.add_argument("--expected-jvp-basis-hash", default=None)

    p.add_argument("--input-basis", type=Path, required=True)
    p.add_argument("--input-rank", type=int, default=200)

    p.add_argument("--pod-basis", type=Path, required=True)
    p.add_argument("--output-rank", type=int, default=150)

    p.add_argument("--hidden-width", type=int, default=1024)
    p.add_argument("--residual-blocks", type=int, default=4)
    p.add_argument("--residual-scale", type=float, default=0.5)

    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-samples-limit", type=int, default=10000)
    p.add_argument("--stats-samples", type=int, default=0)

    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-9)
    p.add_argument("--dino-weight", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=1.5)

    p.add_argument("--dino-validation-every", type=int, default=10)
    p.add_argument("--early-stop-min-epochs", type=int, default=120)
    p.add_argument("--early-stop-patience", type=int, default=60)
    p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--snapshot-every", type=int, default=50)

    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", type=Path, default=None)

    p.add_argument("--expected-train-shards", type=int, required=True)
    p.add_argument("--expected-val-shards", type=int, default=4)
    p.add_argument("--expected-samples-per-shard", type=int, default=625)
    p.add_argument("--expected-train-base-seed", type=int, default=240000)
    p.add_argument("--expected-val-base-seed", type=int, default=1240000)
    return p.parse_args()


def load_basis(path: Path, rank: int, kind: str):
    path = path.expanduser().resolve()
    d = np.load(path, allow_pickle=False)
    V = np.asarray(d["weighted_modes"], dtype=np.float32)
    if rank > V.shape[1]:
        raise ValueError(
            f"{kind} rank {rank} exceeds saved max rank {V.shape[1]}."
        )
    return {
        "mean_field": np.asarray(
            d["mean_field"], dtype=np.float32
        ).reshape(-1),
        "sqrt_weights": np.asarray(
            d["sqrt_weights"], dtype=np.float32
        ).reshape(-1),
        "weighted_modes": V[:, :rank].copy(),
        "coefficient_mean": np.asarray(
            d["coefficient_mean"], dtype=np.float32
        )[:rank].copy(),
        "coefficient_std": np.asarray(
            d["coefficient_std"], dtype=np.float32
        )[:rank].copy(),
        "x_grid": np.asarray(
            d["x_grid"], dtype=np.float32
        ).reshape(-1),
        "y_grid": np.asarray(
            d["y_grid"], dtype=np.float32
        ).reshape(-1),
        "source": str(path),
    }


def compute_stats(dataset, count):
    n = len(dataset) if count <= 0 else min(len(dataset), count)
    sums = np.zeros(2, dtype=np.float64)
    squares = np.zeros(2, dtype=np.float64)
    elements = np.zeros(2, dtype=np.int64)
    for i in range(n):
        u, p = dataset[i][:2]
        for k, a in enumerate((u.numpy(), p.numpy())):
            a = np.asarray(a, dtype=np.float64)
            sums[k] += a.sum()
            squares[k] += np.sum(a * a)
            elements[k] += a.size
        if i and i % 1000 == 0:
            print(f"statistics: {i}/{n}", flush=True)
    mean = sums / elements
    var = np.maximum(
        squares / elements - mean * mean, 1e-20
    )
    return {
        "u_mean": np.asarray([mean[0]], dtype=np.float32),
        "u_std": np.asarray([np.sqrt(var[0])], dtype=np.float32),
        "p_mean": np.asarray([mean[1]], dtype=np.float32),
        "p_std": np.asarray([np.sqrt(var[1])], dtype=np.float32),
    }


def pstd_tensor(stats, device):
    return torch.as_tensor(
        stats["p_std"], dtype=torch.float32, device=device
    ).reshape(())


def energy_weighted_mse(pred, target, weights):
    return torch.mean(
        torch.sum(
            weights[None, :] * (pred - target).square(),
            dim=1,
        )
    )


def compact_audit(audit):
    return {
        "paths": [str(p) for p in audit["paths"]],
        "n_shards": int(audit["n_shards"]),
        "n_samples": int(audit["n_samples"]),
        "sample_counts": [int(v) for v in audit["sample_counts"]],
        "shape": list(audit["shape"]),
        "seed_records": list(audit["seed_records"]),
        "project_config": audit["project_config"],
    }


def inference_payload(full):
    keys = (
        "checkpoint_format",
        "model_type",
        "epoch",
        "model_config",
        "model_state",
        "stats",
        "input_basis_data",
        "pod_data",
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
    out = {k: full[k] for k in keys}
    hist = full.get("history", [])
    out["last_history_row"] = hist[-1] if hist else None
    out["inference_only"] = True
    return out


def train_epoch(
    model,
    loader,
    directions,
    optimizer,
    dino_weight,
    grad_clip,
    device,
):
    model.train()
    w = model.output_energy_weights()

    totals = {
        "loss": 0.0,
        "forward_energy": 0.0,
        "dino_energy": 0.0,
        "relative_l2": 0.0,
    }
    seen = 0
    use_dino = dino_weight > 0

    for batch in loader:
        if use_dino:
            u, p, mode_indices, exact_jvp = batch
            mode_indices = mode_indices.to(
                device=device, dtype=torch.long, non_blocking=True
            )
            exact_jvp = exact_jvp.to(device, non_blocking=True)
        else:
            u, p = batch

        u = u.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)
        B = u.shape[0]

        x = model.encode_input(u)
        target_z = model.project_output_field(p)

        optimizer.zero_grad(set_to_none=True)

        if use_dino:
            h = torch.index_select(
                directions, 0, mode_indices
            ).unsqueeze(1)
            dx = model.encode_direction(h)
            pred_z, pred_dz = jvp(
                model.reduced_map,
                (x,),
                (dx,),
                strict=False,
            )
            target_dz = model.project_output_jvp(exact_jvp)
            dino_loss = energy_weighted_mse(
                pred_dz, target_dz, w
            )
        else:
            pred_z = model.reduced_map(x)
            dino_loss = pred_z.new_zeros(())

        forward_loss = energy_weighted_mse(
            pred_z, target_z, w
        )
        loss = forward_loss + dino_weight * dino_loss

        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            pred_p = model.decode_output(pred_z)[..., 0]
            target_p = p[:, 0].reshape(B, -1)
            rel = torch.mean(
                torch.linalg.vector_norm(
                    pred_p - target_p, dim=1
                )
                / torch.linalg.vector_norm(
                    target_p, dim=1
                ).clamp_min(1e-12)
            )

        totals["loss"] += float(loss.detach()) * B
        totals["forward_energy"] += float(forward_loss.detach()) * B
        totals["dino_energy"] += float(dino_loss.detach()) * B
        totals["relative_l2"] += float(rel.detach()) * B
        seen += B

    return {k: v / seen for k, v in totals.items()}


@torch.inference_mode()
def evaluate_forward(model, loader, stats, device):
    model.eval()
    p_std = pstd_tensor(stats, device)
    w = model.output_energy_weights()

    totals = {
        "forward": 0.0,
        "forward_energy": 0.0,
        "relative_l2": 0.0,
    }
    seen = 0

    for u, p in loader:
        u = u.to(device, non_blocking=True)
        p = p.to(device, non_blocking=True)
        B = u.shape[0]

        x = model.encode_input(u)
        pred_z = model.reduced_map(x)
        target_z = model.project_output_field(p)
        pred_p = model.decode_output(pred_z)[..., 0]
        target_p = p[:, 0].reshape(B, -1)

        field_loss = torch.mean(
            ((pred_p - target_p) / p_std).square()
        )
        coeff_loss = energy_weighted_mse(
            pred_z, target_z, w
        )
        rel = torch.mean(
            torch.linalg.vector_norm(
                pred_p - target_p, dim=1
            )
            / torch.linalg.vector_norm(
                target_p, dim=1
            ).clamp_min(1e-12)
        )

        totals["forward"] += float(field_loss) * B
        totals["forward_energy"] += float(coeff_loss) * B
        totals["relative_l2"] += float(rel) * B
        seen += B

    return {k: v / seen for k, v in totals.items()}


def evaluate_dino_balanced(
    model,
    loader,
    directions,
    stats,
    device,
):
    model.eval()
    p_std = pstd_tensor(stats, device)
    w = model.output_energy_weights()
    n_modes = int(directions.shape[0])

    field_sums = np.zeros(n_modes)
    coeff_sums = np.zeros(n_modes)
    counts = np.zeros(n_modes, dtype=np.int64)

    with torch.no_grad():
        for u, mode_indices, exact_jvp in loader:
            u = u.to(device, non_blocking=True)
            mode_indices = mode_indices.to(
                device=device, dtype=torch.long, non_blocking=True
            )
            exact_jvp = exact_jvp.to(device, non_blocking=True)
            B = u.shape[0]

            x = model.encode_input(u)
            h = torch.index_select(
                directions, 0, mode_indices
            ).unsqueeze(1)
            dx = model.encode_direction(h)

            _, pred_dz = jvp(
                model.reduced_map,
                (x,),
                (dx,),
                strict=False,
            )
            target_dz = model.project_output_jvp(exact_jvp)

            pred_j = model.decode_output_jvp(pred_dz)[..., 0]
            exact_flat = exact_jvp[:, 0].reshape(B, -1)

            field_per = torch.mean(
                ((pred_j - exact_flat) / p_std).square(),
                dim=1,
            )
            coeff_per = torch.sum(
                w[None, :] * (pred_dz - target_dz).square(),
                dim=1,
            )

            for m, fe, ce in zip(
                mode_indices.cpu().tolist(),
                field_per.cpu().tolist(),
                coeff_per.cpu().tolist(),
            ):
                field_sums[m] += float(fe)
                coeff_sums[m] += float(ce)
                counts[m] += 1

    pop = counts > 0
    field_mode = np.full(n_modes, np.nan)
    coeff_mode = np.full(n_modes, np.nan)
    field_mode[pop] = field_sums[pop] / counts[pop]
    coeff_mode[pop] = coeff_sums[pop] / counts[pop]

    return (
        float(field_sums.sum() / counts.sum()),
        float(coeff_sums.sum() / counts.sum()),
        field_mode,
        coeff_mode,
        counts,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)

    train_paths = expand_shard_patterns(args.train_shards)
    val_paths = expand_shard_patterns(args.val_shards)
    ta = audit_shards(train_paths, label="training")
    va = audit_shards(val_paths, label="validation")
    assert_disjoint_splits(ta, va)

    layout = assert_expected_independent_layout(
        ta,
        va,
        expected_train_shards=args.expected_train_shards,
        expected_val_shards=args.expected_val_shards,
        expected_samples_per_shard=args.expected_samples_per_shard,
        expected_train_base_seed=args.expected_train_base_seed,
        expected_val_base_seed=args.expected_val_base_seed,
    )

    available = int(layout["train_samples"])
    active = (
        available
        if args.train_samples_limit <= 0
        else int(args.train_samples_limit)
    )
    if not 1 <= active <= available:
        raise ValueError("Invalid active training sample count.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    inp = load_basis(
        args.input_basis, args.input_rank, "input"
    )
    out = load_basis(
        args.pod_basis, args.output_rank, "output"
    )

    shard_x = np.asarray(ta["x_grid"], dtype=np.float32)
    shard_y = np.asarray(ta["y_grid"], dtype=np.float32)
    for name, basis in (("input", inp), ("output", out)):
        if not np.allclose(basis["x_grid"], shard_x):
            raise ValueError(f"{name} basis x-grid differs from shard grid.")
        if not np.allclose(basis["y_grid"], shard_y):
            raise ValueError(f"{name} basis y-grid differs from shard grid.")

    stats_ds = ReactionH5Dataset(
        train_paths, include_jvp=False
    )
    try:
        stats_count = (
            active
            if args.stats_samples <= 0
            else min(active, args.stats_samples)
        )
        stats = compute_stats(stats_ds, stats_count)
    finally:
        stats_ds.close()

    use_dino = args.dino_weight > 0
    train_jvp_paths, val_jvp_paths, derivative_contract = prepare_jvp_overlays(
        args, train_paths, val_paths, use_dino=use_dino
    )
    train_ds = SelectiveReactionH5Dataset(
        train_paths,
        include_jvp=use_dino,
        include_output=True,
        jvp_sidecar_paths=train_jvp_paths,
    )
    vf_ds = SelectiveReactionH5Dataset(
        val_paths,
        include_jvp=False,
        include_output=True,
    )
    vj_ds = (
        SelectiveReactionH5Dataset(
            val_paths,
            include_jvp=True,
            include_output=False,
            jvp_sidecar_paths=val_jvp_paths,
        )
        if use_dino
        else None
    )

    n_modes = int(train_ds.directions.shape[0])
    sampler = (
        EpochRandomModeSampler(
            active, n_modes, seed=args.seed
        )
        if use_dino
        else EpochShuffleSampler(
            active, seed=args.seed
        )
    )

    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=gen,
        persistent_workers=args.num_workers > 0,
    )

    vw = min(args.num_workers, 2)
    vf_loader = DataLoader(
        vf_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=vw,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=vw > 0,
    )

    vj_sampler = None
    vj_loader = None
    if use_dino:
        vj_sampler = BalancedValidationModeSampler(
            len(vj_ds), n_modes
        )
        vj_loader = DataLoader(
            vj_ds,
            batch_size=args.batch_size,
            sampler=vj_sampler,
            num_workers=vw,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            persistent_workers=vw > 0,
        )

    directions_np = np.asarray(
        derivative_contract["directions"] if derivative_contract else ta["directions"],
        dtype=np.float32,
    )
    directions = torch.as_tensor(
        directions_np, dtype=torch.float32, device=device
    )

    model = ReactionGeneralizedDeepONet(
        input_mean_field=torch.as_tensor(
            inp["mean_field"], device=device
        ),
        input_sqrt_weights=torch.as_tensor(
            inp["sqrt_weights"], device=device
        ),
        input_weighted_modes=torch.as_tensor(
            inp["weighted_modes"], device=device
        ),
        input_coefficient_mean=torch.as_tensor(
            inp["coefficient_mean"], device=device
        ),
        input_coefficient_std=torch.as_tensor(
            inp["coefficient_std"], device=device
        ),
        output_mean_field=torch.as_tensor(
            out["mean_field"], device=device
        ),
        output_sqrt_weights=torch.as_tensor(
            out["sqrt_weights"], device=device
        ),
        output_weighted_modes=torch.as_tensor(
            out["weighted_modes"], device=device
        ),
        output_coefficient_mean=torch.as_tensor(
            out["coefficient_mean"], device=device
        ),
        output_coefficient_std=torch.as_tensor(
            out["coefficient_std"], device=device
        ),
        hidden_width=args.hidden_width,
        residual_blocks=args.residual_blocks,
        residual_scale=args.residual_scale,
    ).to(device)

    params = count_trainable_parameters(model)
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
        min_lr=1e-7,
    )

    start_epoch = 1
    history = []
    best_forward = float("inf")
    best_composite = float("inf")
    no_improve = 0
    dino_round = 0

    if args.resume is not None:
        ck = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )
        if ck.get("model_config", {}).get("architecture") != "generalized":
            raise ValueError("Resume checkpoint is not generalized DeepONet.")
        if not np.array_equal(np.asarray(ck.get("directions_u")), directions_np):
            raise ValueError("Resume checkpoint derivative directions differ from this campaign.")
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch = int(ck["epoch"]) + 1
        history = list(ck.get("history", []))
        state = ck.get("training_state", {})
        best_forward = float(
            state.get("best_forward", best_forward)
        )
        best_composite = float(
            state.get("best_composite", best_composite)
        )
        no_improve = int(
            state.get("epochs_without_forward_improvement", 0)
        )
        dino_round = int(
            state.get("dino_validation_round", 0)
        )

    output = args.output
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    bf = Path(f"{stem}.best_forward.pt")
    bc = Path(f"{stem}.best_composite.pt")
    histpath = Path(f"{stem}.history.json")

    x_np = np.asarray(ta["x_grid"], dtype=np.float32)
    y_np = np.asarray(ta["y_grid"], dtype=np.float32)
    gx, gy = np.meshgrid(x_np, y_np, indexing="ij")
    output_coordinates = np.stack(
        (gx.reshape(-1), gy.reshape(-1)), axis=1
    ).astype(np.float32)
    val_q = np.arange(120 * 120, dtype=np.int64)

    input_ck = {
        "mean_field": inp["mean_field"],
        "sqrt_weights": inp["sqrt_weights"],
        "weighted_modes": inp["weighted_modes"],
        "coefficient_mean": inp["coefficient_mean"],
        "coefficient_std": inp["coefficient_std"],
        "source_basis": inp["source"],
    }
    pod_ck = {
        "mean_field": out["mean_field"],
        "sqrt_weights": out["sqrt_weights"],
        "weighted_modes": out["weighted_modes"],
        "coefficient_mean": out["coefficient_mean"],
        "coefficient_std": out["coefficient_std"],
        "source_basis": out["source"],
    }

    model_config = {
        "architecture": "generalized",
        "input_rank": int(args.input_rank),
        "output_rank": int(args.output_rank),
        "Dp": int(args.output_rank),
        "hidden_width": int(args.hidden_width),
        "residual_blocks": int(args.residual_blocks),
        "residual_scale": float(args.residual_scale),
    }

    training_args = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
    }

    shard_audit = {
        "layout": {
            **dict(layout),
            "available_train_samples": available,
            "active_train_samples": active,
        },
        "training": compact_audit(ta),
        "validation": compact_audit(va),
        "derivative_supervision": (
            None if derivative_contract is None else {
                key: value for key, value in derivative_contract.items() if key != "directions"
            }
        ),
    }

    print("==============================================================")
    print("Reaction generalized / reduced-basis DeepONet")
    print("==============================================================")
    print(f"Device:               {device}")
    print(f"Input rank M:         {args.input_rank}")
    print(f"Output POD rank N:    {args.output_rank}")
    print(f"Hidden width:         {args.hidden_width}")
    print(f"Residual blocks:      {args.residual_blocks}")
    print(f"Trainable parameters: {params:,}")
    print(f"DINO weight:          {args.dino_weight:g}")
    print(f"Batch size:           {args.batch_size}")
    print(f"Training batches:     {len(train_loader)}")
    print(
        "Derivative path:      fixed linear F_M h -> "
        "JVP of reduced residual MLP -> fixed POD B_N"
    )
    print("==============================================================")

    start = perf_counter()

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = perf_counter()
            sampler.set_epoch(epoch)

            tr = train_epoch(
                model,
                train_loader,
                directions,
                optimizer,
                args.dino_weight,
                args.grad_clip,
                device,
            )

            vf = evaluate_forward(
                model, vf_loader, stats, device
            )
            fobj = float(vf["forward"])
            scheduler.step(fobj)

            improved = (
                fobj
                < best_forward
                - args.early_stop_min_delta
            )
            if improved:
                best_forward = fobj
                no_improve = 0
            elif epoch >= args.early_stop_min_epochs:
                no_improve += 1

            stop = (
                args.early_stop_patience > 0
                and epoch >= args.early_stop_min_epochs
                and no_improve >= args.early_stop_patience
            )

            vj_field = None
            vj_coeff = None
            vj_field_mode = None
            vj_coeff_mode = None
            counts = None

            if use_dino and (
                dino_validation_is_due(
                    epoch,
                    args.epochs,
                    args.dino_validation_every,
                )
                or stop
            ):
                vj_sampler.set_round(0)
                (
                    vj_field,
                    vj_coeff,
                    vj_field_mode,
                    vj_coeff_mode,
                    counts,
                ) = evaluate_dino_balanced(
                    model,
                    vj_loader,
                    directions,
                    stats,
                    device,
                )
                dino_round += 1

            composite = (
                None
                if use_dino and vj_field is None
                else (
                    fobj
                    if not use_dino
                    else fobj
                    + args.dino_weight * float(vj_field)
                )
            )
            imp_comp = (
                composite is not None
                and composite < best_composite
            )
            if imp_comp:
                best_composite = float(composite)

            elapsed = perf_counter() - t0
            valhist = dict(vf)
            valhist.update(
                {
                    "dino": vj_field,
                    "dino_coeff_energy": vj_coeff,
                    "dino_per_mode": (
                        None
                        if vj_field_mode is None
                        else [float(v) for v in vj_field_mode]
                    ),
                    "dino_coeff_energy_per_mode": (
                        None
                        if vj_coeff_mode is None
                        else [float(v) for v in vj_coeff_mode]
                    ),
                    "dino_mode_counts": (
                        None
                        if counts is None
                        else [int(v) for v in counts]
                    ),
                    "composite": composite,
                }
            )

            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": tr,
                "val": valhist,
                "seconds": elapsed,
            }
            history.append(row)

            state = {
                "best_forward": best_forward,
                "best_composite": best_composite,
                "epochs_without_forward_improvement": no_improve,
                "dino_validation_round": dino_round,
                "stopped_early": stop,
                "trainable_parameters": params,
            }

            payload = {
                "checkpoint_format": CHECKPOINT_FORMAT,
                "model_type": "deeponet",
                "epoch": epoch,
                "model_config": model_config,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "stats": stats,
                "input_basis_data": input_ck,
                "pod_data": pod_ck,
                "directions_u": directions_np,
                "grid": {
                    "x_grid": x_np,
                    "y_grid": y_np,
                    "output_coordinates": output_coordinates,
                },
                "train_shards": [str(p) for p in train_paths],
                "val_shards": [str(p) for p in val_paths],
                "history": history,
                "training_state": state,
                "training_args": training_args,
                "shard_audit": shard_audit,
                "dino_sampling": {
                    "training": (
                        "one uniform random stored KL/JVP mode "
                        "per sample per epoch"
                    ),
                    "validation": (
                        "one fixed balanced stored mode per "
                        "validation sample"
                    ),
                    "input_encoder": (
                        "fixed weighted PCA/KL-style linear map"
                    ),
                    "output_decoder": (
                        "fixed weighted POD map"
                    ),
                    "training_loss": (
                        "output standardized POD coordinates "
                        "weighted by sigma_out^2"
                    ),
                    "jvp_implementation": (
                        "F_M h encoded exactly; torch.func.jvp "
                        "only through reduced residual MLP"
                    ),
                },
                "validation_query_indices": val_q,
            }

            if improved:
                torch.save(inference_payload(payload), bf)
            if imp_comp:
                torch.save(inference_payload(payload), bc)

            if (
                (args.save_every > 0 and epoch % args.save_every == 0)
                or epoch == args.epochs
                or stop
            ):
                torch.save(payload, output)

            if (
                args.snapshot_every > 0
                and (epoch % args.snapshot_every == 0 or stop)
            ):
                torch.save(
                    inference_payload(payload),
                    Path(f"{stem}.epoch{epoch:04d}.pt"),
                )

            histpath.write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )

            jt = (
                "not-evaluated"
                if vj_field is None
                else f"{vj_field:.4e}"
            )
            print(
                f"epoch {epoch:04d} | "
                f"train={tr['loss']:.4e} | "
                f"val_fwd={fobj:.4e} | "
                f"val_jvp={jt} | "
                f"coeff_fwd_energy={vf['forward_energy']:.4e} | "
                f"relL2={vf['relative_l2']:.4e} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.1f}s",
                flush=True,
            )

            if stop:
                print(
                    f"Early stopping after {no_improve} "
                    "non-improving epochs.",
                    flush=True,
                )
                break
    finally:
        train_ds.close()
        vf_ds.close()
        if vj_ds is not None:
            vj_ds.close()

    print(
        f"Training completed in "
        f"{(perf_counter()-start)/3600:.2f} h"
    )


if __name__ == "__main__":
    main()
