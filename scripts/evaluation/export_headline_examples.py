#!/usr/bin/env python3
"""Export compact FEM-versus-surrogate headline examples from test shards.

The HDF5 shards already contain FEM pressure fields and FEM tangent fields in
the stored KL directions.  This script only evaluates the four frozen
headline surrogates and writes a small, portable NPZ artifact for plotting.
It does not run FEniCS or regenerate test data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = ROOT / "scripts/training"
for search_path in (ROOT, TRAINING_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from models.emulator import load_reaction_emulator  # noqa: E402
from reaction_core.datasets import ReactionH5Dataset  # noqa: E402
from training_utils import audit_shards, expand_shard_patterns  # noqa: E402


FAMILIES = ("fno", "deeponet", "genfno", "gendeeponet")
LABELS = {
    "fno": "FNO",
    "deeponet": "DeepONet",
    "genfno": "GenFNO",
    "gendeeponet": "GenDeepONet",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Export deterministic headline pressure/JVP test examples.",
    )
    parser.add_argument(
        "--downstream-manifest",
        type=Path,
        default=Path("configs/paper/downstream_models.tsv"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT,
        help="Root used to resolve relative checkpoint paths in the manifest.",
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        default=["data/shards_broad_amp_final_test/reaction_shard_*.h5"],
    )
    parser.add_argument("--sample-indices", type=int, nargs="+", default=[0, 1249, 2499])
    parser.add_argument("--modes", type=int, nargs="+", default=[0, 7, 15])
    parser.add_argument(
        "--external-field",
        action="append",
        default=[],
        metavar="LABEL=NPZ",
        help=(
            "Append a labelled non-test field from a derivative-diagnostic NPZ. "
            "The archive must contain u, fem_forward, fem_jvp, and directions_u."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--expected-samples-per-shard", type=int, default=625)
    parser.add_argument("--expected-base-seed", type=int, default=3240000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/paper_T2_broad/headline_examples/headline_examples.npz"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_headline_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("downstream_role") == "headline_opt"
        ]
    by_family: dict[str, dict[str, str]] = {}
    for row in rows:
        family = row.get("model_family", "")
        if family not in FAMILIES:
            raise ValueError(f"Unexpected headline model family {family!r}.")
        if family in by_family:
            raise ValueError(f"Duplicate headline model family {family!r}.")
        by_family[family] = row
    missing = [family for family in FAMILIES if family not in by_family]
    if missing:
        raise ValueError(f"Missing headline models for: {', '.join(missing)}.")
    return [by_family[family] for family in FAMILIES]


def validate_pool(
    audit: dict[str, Any],
    *,
    expected_shards: int,
    expected_samples_per_shard: int,
    expected_base_seed: int,
) -> None:
    if int(audit["n_shards"]) != expected_shards:
        raise ValueError(
            f"Expected {expected_shards} shards, found {audit['n_shards']}."
        )
    counts = [int(value) for value in audit["sample_counts"]]
    if counts != [expected_samples_per_shard] * expected_shards:
        raise ValueError(
            f"Expected {expected_samples_per_shard} samples per shard; found {counts}."
        )
    offset = 0
    for index, record in enumerate(audit["seed_records"]):
        base_seed = int(record["base_seed"])
        sample_offset = int(record["sample_offset"])
        first_seed = int(record["first_seed"])
        last_seed = int(record["last_seed"])
        expected_first = expected_base_seed + offset
        if (
            base_seed != expected_base_seed
            or sample_offset != offset
            or first_seed != expected_first
            or last_seed != expected_first + counts[index] - 1
        ):
            raise ValueError(
                f"Shard {index} does not match the required final-test seed sequence."
            )
        offset += counts[index]


def resolve_checkpoint(root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def grid_weights(x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x_grid, dtype=np.float64)
    y = np.asarray(y_grid, dtype=np.float64)
    dx = np.diff(x)
    dy = np.diff(y)
    if not np.allclose(dx, dx[0]) or not np.allclose(dy, dy[0]):
        raise ValueError("Headline example grids must be uniform.")
    wx = np.full(x.size, float(dx[0]), dtype=np.float64)
    wy = np.full(y.size, float(dy[0]), dtype=np.float64)
    wx[[0, -1]] *= 0.5
    wy[[0, -1]] *= 0.5
    return wx[:, None] * wy[None, :]


def relative_l2(prediction: np.ndarray, exact: np.ndarray, weights: np.ndarray) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(exact, dtype=np.float64)
    numerator = float(np.sum(weights * error * error, dtype=np.float64))
    denominator = float(np.sum(weights * exact * exact, dtype=np.float64))
    return float(np.sqrt(numerator / max(denominator, 1.0e-300)))


def field(values: Any, shape: tuple[int, int], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == shape:
        return array
    if array.size == shape[0] * shape[1]:
        return array.reshape(shape)
    raise ValueError(f"{name} has shape {array.shape}; expected {shape}.")


def parse_external_field(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError(
            f"--external-field must have the form LABEL=NPZ; got {specification!r}."
        )
    label, raw_path = specification.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("--external-field requires a non-empty label.")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return label, path


def load_external_field(
    path: Path,
    *,
    label: str,
    shape: tuple[int, int],
    reference_directions: np.ndarray,
    mode_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"u", "fem_forward", "fem_jvp", "directions_u"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"External field {label!r} is missing: {', '.join(missing)}.")
        stored_directions = np.asarray(archive["directions_u"], dtype=np.float64)
        if stored_directions.shape != reference_directions.shape or not np.allclose(
            stored_directions,
            reference_directions,
            rtol=1.0e-6,
            atol=1.0e-7,
        ):
            raise ValueError(
                f"External field {label!r} does not use the final-test KL directions."
            )
        all_jvp = np.asarray(archive["fem_jvp"], dtype=np.float64)
        if all_jvp.shape[0] != reference_directions.shape[0]:
            raise ValueError(f"External field {label!r} has an invalid fem_jvp shape.")
        return (
            field(archive["u"], shape, name=f"{label} input u"),
            field(archive["fem_forward"], shape, name=f"{label} FEM pressure"),
            np.stack(
                [
                    field(all_jvp[mode], shape, name=f"{label} FEM JVP mode {mode}")
                    for mode in mode_indices
                ]
            ),
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.suffix != ".npz":
        raise ValueError("--output must end in .npz.")
    json_path = output.with_suffix(".json")
    if not args.overwrite and (output.exists() or json_path.exists()):
        raise FileExistsError(f"Refusing existing output {output} or {json_path}.")

    models = read_headline_manifest(args.downstream_manifest.resolve())
    shard_paths = expand_shard_patterns(args.shards)
    audit = audit_shards(shard_paths, label="final_test_headline_examples")
    validate_pool(
        audit,
        expected_shards=int(args.expected_shards),
        expected_samples_per_shard=int(args.expected_samples_per_shard),
        expected_base_seed=int(args.expected_base_seed),
    )

    dataset = ReactionH5Dataset(shard_paths, include_jvp=True)
    try:
        sample_indices = [int(value) for value in args.sample_indices]
        mode_indices = [int(value) for value in args.modes]
        if len(sample_indices) != len(set(sample_indices)):
            raise ValueError("--sample-indices contains duplicates.")
        if len(mode_indices) != len(set(mode_indices)):
            raise ValueError("--modes contains duplicates.")
        if any(index < 0 or index >= len(dataset) for index in sample_indices):
            raise ValueError(f"--sample-indices must lie in [0,{len(dataset)}).")
        n_modes = int(dataset.directions.shape[0])
        if any(mode < 0 or mode >= n_modes for mode in mode_indices):
            raise ValueError(f"--modes must lie in [0,{n_modes}).")

        shape = (len(dataset.x_grid), len(dataset.y_grid))
        weights = grid_weights(dataset.x_grid, dataset.y_grid)
        inputs: list[np.ndarray] = []
        exact_pressure: list[np.ndarray] = []
        exact_jvp: list[np.ndarray] = []
        field_labels = [f"Test {index}" for index in sample_indices]
        field_origins = ["independent test" for _ in sample_indices]
        field_test_indices = list(sample_indices)
        for index in sample_indices:
            u_tensor, p_tensor, jvp_tensor = dataset[index]
            inputs.append(field(u_tensor.numpy(), shape, name="input u"))
            exact_pressure.append(field(p_tensor.numpy(), shape, name="exact pressure"))
            all_jvp = np.asarray(jvp_tensor.numpy(), dtype=np.float64)
            exact_jvp.append(
                np.stack(
                    [field(all_jvp[mode], shape, name=f"exact JVP mode {mode}") for mode in mode_indices]
                )
            )

        directions = np.stack(
            [np.asarray(dataset.directions[mode], dtype=np.float64) for mode in mode_indices]
        )
        external_records: list[dict[str, Any]] = []
        for specification in args.external_field:
            label, external_path = parse_external_field(specification)
            if label in field_labels:
                raise ValueError(f"Duplicate field label {label!r}.")
            external_u, external_p, external_jvp = load_external_field(
                external_path,
                label=label,
                shape=shape,
                reference_directions=np.asarray(dataset.directions, dtype=np.float64),
                mode_indices=mode_indices,
            )
            inputs.append(external_u)
            exact_pressure.append(external_p)
            exact_jvp.append(external_jvp)
            field_labels.append(label)
            field_origins.append("declared OOD")
            field_test_indices.append(-1)
            external_records.append(
                {
                    "label": label,
                    "source_name": external_path.name,
                    "source_sha256": sha256(external_path),
                }
            )

        u_values = np.stack(inputs)
        p_values = np.stack(exact_pressure)
        j_values = np.stack(exact_jvp)

        pressure_predictions: list[np.ndarray] = []
        jvp_predictions: list[np.ndarray] = []
        forward_errors: list[list[float]] = []
        derivative_errors: list[list[list[float]]] = []
        checkpoint_paths: list[Path] = []
        for model in models:
            checkpoint = resolve_checkpoint(args.checkpoint_root.resolve(), model["checkpoint"])
            checkpoint_paths.append(checkpoint)
            print(f"{LABELS[model['model_family']]}: {model['run_id']}", flush=True)
            emulator = load_reaction_emulator(checkpoint, device=args.device)
            expected_epoch = int(model["checkpoint_epoch"])
            if int(emulator.epoch) != expected_epoch:
                raise ValueError(
                    f"{model['run_id']} checkpoint reports epoch {emulator.epoch}, "
                    f"expected {expected_epoch}."
                )
            model_pressure: list[np.ndarray] = []
            model_jvp: list[np.ndarray] = []
            model_forward_errors: list[float] = []
            model_derivative_errors: list[list[float]] = []
            for sample_position, u in enumerate(u_values):
                prediction = field(emulator.forward(u), shape, name="surrogate pressure")
                model_pressure.append(prediction)
                model_forward_errors.append(
                    relative_l2(prediction, p_values[sample_position], weights)
                )
                sample_jvp: list[np.ndarray] = []
                sample_errors: list[float] = []
                for mode_position, direction in enumerate(directions):
                    derivative = field(
                        emulator.jvp(u, direction), shape, name="surrogate JVP"
                    )
                    sample_jvp.append(derivative)
                    sample_errors.append(
                        relative_l2(
                            derivative,
                            j_values[sample_position, mode_position],
                            weights,
                        )
                    )
                model_jvp.append(np.stack(sample_jvp))
                model_derivative_errors.append(sample_errors)
            pressure_predictions.append(np.stack(model_pressure))
            jvp_predictions.append(np.stack(model_jvp))
            forward_errors.append(model_forward_errors)
            derivative_errors.append(model_derivative_errors)
    finally:
        dataset.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format=np.asarray("reaction-headline-field-examples-v2"),
        x_grid=np.asarray(dataset.x_grid, dtype=np.float64),
        y_grid=np.asarray(dataset.y_grid, dtype=np.float64),
        sample_indices=np.asarray(field_test_indices, dtype=np.int64),
        field_labels=np.asarray(field_labels),
        field_origins=np.asarray(field_origins),
        mode_indices=np.asarray(mode_indices, dtype=np.int64),
        model_families=np.asarray([row["model_family"] for row in models]),
        architectures=np.asarray([LABELS[row["model_family"]] for row in models]),
        run_ids=np.asarray([row["run_id"] for row in models]),
        checkpoint_epochs=np.asarray([int(row["checkpoint_epoch"]) for row in models]),
        directions=np.asarray(directions, dtype=np.float32),
        input_u=np.asarray(u_values, dtype=np.float32),
        exact_pressure=np.asarray(p_values, dtype=np.float32),
        surrogate_pressure=np.asarray(pressure_predictions, dtype=np.float32),
        exact_jvp=np.asarray(j_values, dtype=np.float32),
        surrogate_jvp=np.asarray(jvp_predictions, dtype=np.float32),
        forward_relative_l2=np.asarray(forward_errors, dtype=np.float64),
        jvp_relative_l2=np.asarray(derivative_errors, dtype=np.float64),
    )
    metadata = {
        "format": "reaction-headline-field-examples-v2",
        "selection": {
            "rule": "fixed declared global indices; no field was selected by visual appearance",
            "sample_indices": sample_indices,
            "mode_indices": mode_indices,
        },
        "fields": [
            {
                "label": label,
                "origin": origin,
                "test_sample_index": index if index >= 0 else None,
            }
            for label, origin, index in zip(
                field_labels, field_origins, field_test_indices, strict=True
            )
        ],
        "external_fields": external_records,
        "test_pool": {
            "n_shards": int(audit["n_shards"]),
            "n_samples": int(audit["n_samples"]),
            "sample_counts": [int(value) for value in audit["sample_counts"]],
            "expected_base_seed": int(args.expected_base_seed),
            "shard_names": [path.name for path in shard_paths],
        },
        "models": [
            {
                "model_family": row["model_family"],
                "architecture": LABELS[row["model_family"]],
                "run_id": row["run_id"],
                "checkpoint_epoch": int(row["checkpoint_epoch"]),
                "checkpoint_name": checkpoint_paths[index].name,
                "checkpoint_sha256": sha256(checkpoint_paths[index]),
            }
            for index, row in enumerate(models)
        ],
        "npz_sha256": sha256(output),
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
