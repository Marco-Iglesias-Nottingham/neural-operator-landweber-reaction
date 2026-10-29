#!/usr/bin/env python3
"""FEM-versus-surrogate derivative audit at headline Landweber fields.

The diagnostic is deliberately split into two stages because the Sulis GPU
PyTorch environment and CPU FEniCSx environment are maintained separately.

``surrogate`` (GPU, no FEniCSx)
    Loads the four frozen headline checkpoints and their projection-diagnostic
    Landweber archives.  At the common initial field, the T2 truth, and each
    model's oracle minimum-u-error field it stores the surrogate forward map,
    JVPs in the shared DINO/KL directions, the residual VJP, and the
    Matérn-preconditioned Landweber direction.

``fem`` (CPU, no PyTorch)
    Replays those exact fields/directions with the 119x119-cell P1 FEM.  It
    compares JVPs, compares VJPs using the *same surrogate residual* (isolating
    derivative error), and compares the complete surrogate/FEM Landweber
    directions using their respective residuals.

The minimum-u-error field is an oracle synthetic diagnostic: selecting it
requires the known truth and is not a deployable stopping rule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


FORMAT_SURROGATE = "reaction-headline-derivative-surrogate-v1"
FORMAT_INDEX = "reaction-headline-derivative-index-v1"
FORMAT_COMPARISON = "reaction-headline-derivative-comparison-v1"
BASEPOINTS = ("initial", "truth", "selected")
EXPECTED_ARCHITECTURES = ("FNO", "DeepONet", "GenFNO", "GenDeepONet")


def find_project_root() -> Path:
    environment_root = os.environ.get("REACTION_PROJECT_ROOT")
    candidates: list[Path] = []
    if environment_root:
        candidates.append(Path(environment_root).expanduser())
    here = Path(__file__).resolve()
    candidates.extend((here.parent, *here.parents, Path.cwd(), *Path.cwd().parents))
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "reaction_core").is_dir() and (resolved / "models").is_dir():
            return resolved
    raise RuntimeError("Could not locate reaction_emulator_project_matern_only.")


PROJECT_ROOT = find_project_root()
SCRIPT_DIR = Path(__file__).resolve().parent
for search_path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))


def project_path(path: Path) -> Path:
    expanded = path.expanduser()
    return (expanded if expanded.is_absolute() else PROJECT_ROOT / expanded).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def prepare_output_directory(path: Path, *, overwrite: bool) -> Path:
    resolved = project_path(path)
    if resolved.exists() and not resolved.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing non-empty output directory {resolved}; choose a new path."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _finite_array(values: Any, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be non-empty and finite.")
    return result


def comparison_metrics(
    predicted: np.ndarray,
    exact: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Return norm, angle, and optimally-rescaled error diagnostics.

    ``weights`` describes the trailing two spatial axes and is broadcast over
    leading axes, allowing the same function to pool a stack of JVP fields.
    """

    predicted_array = _finite_array(predicted, name="predicted")
    exact_array = _finite_array(exact, name="exact")
    if predicted_array.shape != exact_array.shape:
        raise ValueError(
            f"predicted shape {predicted_array.shape} != exact {exact_array.shape}."
        )
    if weights is None:
        effective = np.ones(predicted_array.shape, dtype=np.float64)
    else:
        spatial_weights = _finite_array(weights, name="weights")
        if spatial_weights.ndim != 2 or predicted_array.shape[-2:] != spatial_weights.shape:
            raise ValueError("weights must match the trailing two spatial axes.")
        if np.any(spatial_weights < 0.0):
            raise ValueError("weights must be non-negative.")
        effective = np.broadcast_to(spatial_weights, predicted_array.shape)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != predicted_array.shape[-2:]:
            raise ValueError("mask must match the trailing two spatial axes.")
        effective = effective * np.broadcast_to(mask_array, predicted_array.shape)

    exact_sq = float(np.sum(effective * exact_array * exact_array, dtype=np.float64))
    predicted_sq = float(
        np.sum(effective * predicted_array * predicted_array, dtype=np.float64)
    )
    dot = float(np.sum(effective * predicted_array * exact_array, dtype=np.float64))
    error_sq = float(
        np.sum(effective * (predicted_array - exact_array) ** 2, dtype=np.float64)
    )
    exact_norm = math.sqrt(max(exact_sq, 0.0))
    predicted_norm = math.sqrt(max(predicted_sq, 0.0))
    relative_error = math.sqrt(max(error_sq, 0.0)) / max(exact_norm, 1.0e-300)
    norm_ratio = predicted_norm / max(exact_norm, 1.0e-300)
    cosine = dot / max(predicted_norm * exact_norm, 1.0e-300)
    scale = dot / predicted_sq if predicted_sq > 0.0 else float("nan")
    if np.isfinite(scale):
        scaled_error_sq = float(
            np.sum(
                effective * (scale * predicted_array - exact_array) ** 2,
                dtype=np.float64,
            )
        )
        relative_after_scale = math.sqrt(max(scaled_error_sq, 0.0)) / max(
            exact_norm, 1.0e-300
        )
    else:
        relative_after_scale = float("nan")
    return {
        "exact_norm": exact_norm,
        "predicted_norm": predicted_norm,
        "relative_error": relative_error,
        "norm_ratio": norm_ratio,
        "cosine": cosine,
        "optimal_scale_predicted_to_exact": float(scale),
        "relative_error_after_optimal_scale": relative_after_scale,
    }


def per_direction_metrics(
    predicted: np.ndarray,
    exact: np.ndarray,
    *,
    weights: np.ndarray,
    mask: np.ndarray,
) -> list[dict[str, float | int]]:
    if predicted.ndim != 3 or predicted.shape != exact.shape:
        raise ValueError("JVP stacks must have shape (directions,nx,ny).")
    rows: list[dict[str, float | int]] = []
    for index in range(predicted.shape[0]):
        rows.append(
            {
                "direction_index": index,
                **comparison_metrics(
                    predicted[index], exact[index], weights=weights, mask=mask
                ),
            }
        )
    return rows


def summarize_direction_rows(
    rows: list[dict[str, float | int]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in ("relative_error", "norm_ratio", "cosine"):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        result[f"{metric}_mean"] = float(np.mean(values))
        result[f"{metric}_median"] = float(np.median(values))
        result[f"{metric}_p95"] = float(np.quantile(values, 0.95))
        result[f"{metric}_minimum"] = float(np.min(values))
        result[f"{metric}_maximum"] = float(np.max(values))
    return result


def read_headline_manifest(path: Path) -> list[dict[str, str]]:
    manifest = project_path(path)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"architecture", "run_id", "checkpoint", "output_dir"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{manifest} lacks required columns {sorted(required)}.")
    selected = [row for row in rows if row.get("ablation_axis", "headline") == "headline"]
    architectures = [row["architecture"] for row in selected]
    if len(selected) != 4 or set(architectures) != set(EXPECTED_ARCHITECTURES):
        raise ValueError(
            "Headline manifest must contain exactly one of each architecture: "
            + ", ".join(EXPECTED_ARCHITECTURES)
        )
    if len(set(architectures)) != 4:
        raise ValueError("Headline manifest contains a duplicate architecture.")
    run_ids = [row["run_id"] for row in selected]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Headline manifest contains a duplicate run_id.")
    for run_id in run_ids:
        if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
            raise ValueError(f"Unsafe run_id in headline manifest: {run_id!r}.")
    return sorted(
        selected,
        key=lambda row: EXPECTED_ARCHITECTURES.index(row["architecture"]),
    )


def validate_checkpoint_identity(operator: Any, row: dict[str, str]) -> str:
    """Cross-check manifest identity against checkpoint-internal metadata."""

    raw_architecture = str(
        operator.emulator.model_config.get("architecture", "")
    ).lower()
    aliases = {
        "FNO": {"standard", "classical", "canonical"},
        "GenFNO": {
            "generalized_fourier",
            "generalized_fno",
            "generalised_fourier",
            "gfno",
        },
        "DeepONet": {"standard", "separable", "canonical"},
        "GenDeepONet": {
            "generalized",
            "generalised",
            "reduced_basis",
            "rb_dino",
        },
    }
    expected = row["architecture"]
    if raw_architecture not in aliases[expected]:
        raise ValueError(
            f"Manifest labels {row['run_id']} as {expected}, but its checkpoint "
            f"declares architecture={raw_architecture!r}."
        )
    manifest_epoch = row.get("checkpoint_epoch", "").strip()
    if manifest_epoch:
        try:
            expected_epoch = int(manifest_epoch)
        except ValueError as exc:
            raise ValueError(
                f"Invalid checkpoint_epoch={manifest_epoch!r} for {row['run_id']}."
            ) from exc
        if operator.epoch != expected_epoch:
            raise ValueError(
                f"Manifest expects epoch {expected_epoch} for {row['run_id']}, "
                f"but the checkpoint stores epoch {operator.epoch}."
            )
    return raw_architecture


def result_path_from_row(row: dict[str, str]) -> Path:
    output = Path(row["output_dir"]).expanduser()
    run_dir = output if output.is_absolute() else PROJECT_ROOT / "results" / output
    result = run_dir / "result.npz"
    if not result.is_file():
        raise FileNotFoundError(f"Missing headline result: {result}")
    return result.resolve()


def checkpoint_path_from_row(row: dict[str, str]) -> Path:
    checkpoint = project_path(Path(row["checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing headline checkpoint: {checkpoint}")
    return checkpoint


def load_landweber_fields(path: Path) -> dict[str, Any]:
    required = {
        "initial_u",
        "truth_u_resampled_model_grid",
        "recovered_u_minimum_u_error",
        "minimum_u_error_iteration",
        "data_observed_resampled",
        "observation_mask",
        "observation_weights",
        "model_x_grid",
        "model_y_grid",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"{path} lacks required arrays: {', '.join(missing)}")
        values = {name: np.asarray(archive[name]).copy() for name in required}
    shape = values["truth_u_resampled_model_grid"].shape
    if shape != (120, 120):
        raise ValueError(f"{path}: expected a 120x120 truth, got {shape}.")
    for name in (
        "initial_u",
        "recovered_u_minimum_u_error",
        "data_observed_resampled",
        "observation_mask",
        "observation_weights",
    ):
        if values[name].shape != shape:
            raise ValueError(
                f"{path}: {name} has shape {values[name].shape}, expected {shape}."
            )
    for name in ("model_x_grid", "model_y_grid"):
        if values[name].shape != (120,) or not np.all(np.isfinite(values[name])):
            raise ValueError(f"{path}: {name} must be a finite length-120 vector.")
    for name in (
        "initial_u",
        "truth_u_resampled_model_grid",
        "recovered_u_minimum_u_error",
        "data_observed_resampled",
        "observation_weights",
    ):
        if not np.all(np.isfinite(values[name])):
            raise ValueError(f"{path}: {name} contains non-finite values.")
    if np.any(values["observation_weights"] < 0.0):
        raise ValueError(f"{path}: observation_weights contains negative entries.")
    raw_mask = values["observation_mask"]
    if raw_mask.dtype != np.bool_ and not np.all(np.isin(raw_mask, (0, 1))):
        raise ValueError(f"{path}: observation_mask is not Boolean-valued.")
    values["observation_mask"] = raw_mask.astype(bool, copy=False)
    return values


def common_array_check(
    reference: dict[str, Any], current: dict[str, Any], path: Path
) -> None:
    for name in (
        "initial_u",
        "truth_u_resampled_model_grid",
        "data_observed_resampled",
        "observation_mask",
        "observation_weights",
        "model_x_grid",
        "model_y_grid",
    ):
        if not np.array_equal(reference[name], current[name]):
            raise ValueError(
                f"{path}: common input {name} differs between headline runs."
            )


def stage_surrogate(args: argparse.Namespace) -> None:
    from reaction_core.inversion import (  # lazy: no FEniCSx
        StationaryMaternCovariance2D,
        weighted_residual_covector,
    )
    from reaction_core.kl import KLBasis2D
    from landweber_surrogate import SurrogateReactionOperator

    manifest_path = project_path(args.manifest)
    rows = read_headline_manifest(manifest_path)
    basis_path = project_path(args.kl_basis)
    basis = KLBasis2D.load(basis_path)
    directions = np.asarray(
        basis.select_directions(args.num_directions), dtype=np.float32
    ).astype(np.float64)
    grid = basis.grid
    if grid.shape != (120, 120):
        raise ValueError(f"KL basis grid must be 120x120, got {grid.shape}.")
    output_dir = prepare_output_directory(args.output_dir, overwrite=args.overwrite)

    index_rows: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    with StationaryMaternCovariance2D(
        grid,
        nu=args.matern_nu,
        length_scale=args.matern_length_scale,
        variance=args.matern_variance,
        fft_workers=args.fft_workers,
    ) as covariance:
        for number, row in enumerate(rows, start=1):
            architecture = row["architecture"]
            run_id = row["run_id"]
            result_path = result_path_from_row(row)
            checkpoint_path = checkpoint_path_from_row(row)
            fields = load_landweber_fields(result_path)
            if reference is None:
                reference = fields
                if not (
                    np.allclose(grid.x, fields["model_x_grid"], rtol=0.0, atol=1.0e-13)
                    and np.allclose(grid.y, fields["model_y_grid"], rtol=0.0, atol=1.0e-13)
                ):
                    raise ValueError("KL basis grid differs from the Landweber model grid.")
            else:
                common_array_check(reference, fields, result_path)

            print(f"[{number}/4] {architecture}: {run_id}", flush=True)
            operator = SurrogateReactionOperator(
                checkpoint_path,
                grid=grid,
                device=args.device,
                deeponet_query_chunk=args.deeponet_query_chunk,
            )
            checkpoint_architecture = validate_checkpoint_identity(operator, row)
            basepoint_fields = {
                "initial": np.asarray(fields["initial_u"], dtype=np.float64),
                "truth": np.asarray(
                    fields["truth_u_resampled_model_grid"], dtype=np.float64
                ),
                "selected": np.asarray(
                    fields["recovered_u_minimum_u_error"], dtype=np.float64
                ),
            }
            observed = np.asarray(fields["data_observed_resampled"], dtype=np.float64)
            weights = np.asarray(fields["observation_weights"], dtype=np.float64)
            mask = np.asarray(fields["observation_mask"], dtype=bool)
            arrays: dict[str, np.ndarray] = {
                "x_grid": np.asarray(grid.x, dtype=np.float64),
                "y_grid": np.asarray(grid.y, dtype=np.float64),
                "directions_u": directions,
                "data_observed": observed,
                "observation_weights": weights,
                "observation_mask": mask,
                "minimum_u_error_iteration": np.asarray(
                    int(np.asarray(fields["minimum_u_error_iteration"]).reshape(())),
                    dtype=np.int64,
                ),
            }
            for basepoint, u in basepoint_fields.items():
                if np.min(u) < args.u_min - 1.0e-10 or np.max(u) > args.u_max + 1.0e-10:
                    raise ValueError(
                        f"{run_id}:{basepoint} lies outside "
                        f"[{args.u_min},{args.u_max}]."
                    )
                print(f"  {basepoint}: forward + {len(directions)} JVPs + VJP", flush=True)
                prediction = operator.forward(u)
                predicted_jvps = np.stack(
                    [operator.jvp(u, direction) for direction in directions], axis=0
                )
                residual = prediction - observed
                output_dual = weighted_residual_covector(residual, weights, mask=mask)
                covector = operator.vjp(u, output_dual)
                landweber_direction = covariance.apply_coordinate_covector(covector)
                arrays[f"u_{basepoint}"] = u
                arrays[f"surrogate_forward_{basepoint}"] = prediction
                arrays[f"surrogate_jvp_{basepoint}"] = predicted_jvps
                arrays[f"surrogate_residual_{basepoint}"] = residual
                arrays[f"surrogate_coordinate_covector_{basepoint}"] = covector
                arrays[f"surrogate_landweber_direction_{basepoint}"] = landweber_direction

            run_dir = output_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            artifact = run_dir / "surrogate_derivatives.npz"
            atomic_npz(artifact, arrays)
            metadata = {
                "format": FORMAT_SURROGATE,
                "architecture": architecture,
                "run_id": run_id,
                "checkpoint_epoch": int(operator.epoch),
                "checkpoint_architecture": checkpoint_architecture,
                "minimum_u_error_iteration": int(
                    np.asarray(arrays["minimum_u_error_iteration"])
                    .reshape(())
                    .item()
                ),
                "device": operator.device,
                "number_of_directions": int(directions.shape[0]),
                "basepoints": list(BASEPOINTS),
                "paths": {
                    "checkpoint": checkpoint_path,
                    "landweber_result": result_path,
                    "kl_basis": basis_path,
                    "manifest": manifest_path,
                },
                "sha256": {
                    "checkpoint": sha256_file(checkpoint_path),
                    "landweber_result": sha256_file(result_path),
                    "kl_basis": sha256_file(basis_path),
                    "directions_u": sha256_array(directions),
                    "truth": sha256_array(arrays["u_truth"]),
                    "data_observed": sha256_array(observed),
                    "artifact": sha256_file(artifact),
                },
            }
            atomic_json(run_dir / "metadata.json", metadata)
            index_rows.append(
                {
                    **metadata,
                    "artifact": str(artifact.relative_to(output_dir)),
                }
            )

    atomic_json(
        output_dir / "index.json",
        {
            "format": FORMAT_INDEX,
            "stage": "surrogate",
            "runs": index_rows,
            "matern": {
                "nu": args.matern_nu,
                "length_scale": args.matern_length_scale,
                "variance": args.matern_variance,
            },
            "inversion_bounds": {"u_min": args.u_min, "u_max": args.u_max},
        },
    )
    (output_dir / "COMPLETED.txt").write_text(
        "Surrogate derivative stage completed successfully.\n", encoding="utf-8"
    )
    print(f"Saved surrogate stage in {output_dir}", flush=True)


def _load_surrogate_index(input_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index_path = input_dir / "index.json"
    if not index_path.is_file() or not (input_dir / "COMPLETED.txt").is_file():
        raise FileNotFoundError(f"Incomplete surrogate stage: {input_dir}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT_INDEX or payload.get("stage") != "surrogate":
        raise ValueError(f"Unsupported surrogate index: {index_path}")
    rows = payload.get("runs")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("Surrogate index must contain exactly four runs.")
    if {row.get("architecture") for row in rows} != set(EXPECTED_ARCHITECTURES):
        raise ValueError("Surrogate index architecture set is incomplete.")
    if len({row.get("run_id") for row in rows}) != 4:
        raise ValueError("Surrogate index run_id values are not unique.")
    if len({row.get("artifact") for row in rows}) != 4:
        raise ValueError("Surrogate index artifact paths are not unique.")
    return payload, rows


def _load_surrogate_artifact(input_dir: Path, row: dict[str, Any]) -> dict[str, np.ndarray]:
    path = (input_dir / row["artifact"]).resolve()
    if input_dir not in path.parents or not path.is_file():
        raise FileNotFoundError(f"Invalid surrogate artifact: {path}")
    expected_hashes = row.get("sha256", {})
    if sha256_file(path) != expected_hashes.get("artifact"):
        raise ValueError(f"{path}: artifact file hash mismatch.")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    required = {
        "x_grid", "y_grid", "directions_u", "data_observed",
        "observation_weights", "observation_mask", "minimum_u_error_iteration",
    }
    for basepoint in BASEPOINTS:
        required.update(
            {
                f"u_{basepoint}", f"surrogate_forward_{basepoint}",
                f"surrogate_jvp_{basepoint}", f"surrogate_residual_{basepoint}",
                f"surrogate_coordinate_covector_{basepoint}",
                f"surrogate_landweber_direction_{basepoint}",
            }
        )
    missing = sorted(required.difference(arrays))
    if missing:
        raise KeyError(f"{path} lacks arrays: {', '.join(missing)}")
    for name, key in (
        ("directions_u", "directions_u"),
        ("u_truth", "truth"),
        ("data_observed", "data_observed"),
    ):
        if sha256_array(arrays[name]) != expected_hashes.get(key):
            raise ValueError(f"{path}: hash mismatch for {name}.")
    return arrays


def _fem_solver_options(args: argparse.Namespace) -> dict[str, Any]:
    if args.linear_solver == "lu":
        return {"ksp_type": "preonly", "pc_type": "lu"}
    return {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": args.ksp_rtol,
        "ksp_atol": 0.0,
        "ksp_max_it": args.ksp_max_it,
    }


def stage_fem(args: argparse.Namespace) -> None:
    from mpi4py import MPI
    from reaction_core import (
        ReactionConfig,
        ReactionSolver2D,
        RegularGrid2D,
        StructuredCoefficientGridMap,
        StructuredLagrangeStateGridMap,
    )
    from reaction_core.inversion import (
        ScalarVertexObservationAdjoint,
        StationaryMaternCovariance2D,
        weighted_norm,
    )

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Run the FEM stage with exactly one MPI rank.")
    input_dir = project_path(args.input_dir)
    index, rows = _load_surrogate_index(input_dir)
    artifacts = [(row, _load_surrogate_artifact(input_dir, row)) for row in rows]

    reference = artifacts[0][1]
    x = np.asarray(reference["x_grid"], dtype=np.float64)
    y = np.asarray(reference["y_grid"], dtype=np.float64)
    grid = RegularGrid2D(x=x, y=y)
    if grid.shape != (120, 120) or args.fem_nx + 1 != 120 or args.fem_ny + 1 != 120:
        raise ValueError("This headline diagnostic requires a 120x120 grid and 119x119 FEM cells.")
    directions = np.asarray(reference["directions_u"], dtype=np.float64)
    observed = np.asarray(reference["data_observed"], dtype=np.float64)
    weights = np.asarray(reference["observation_weights"], dtype=np.float64)
    mask = np.asarray(reference["observation_mask"], dtype=bool)
    if directions.ndim != 3 or directions.shape[1:] != grid.shape:
        raise ValueError("directions_u must have shape (number,120,120).")
    for name, values in (
        ("directions_u", directions),
        ("data_observed", observed),
        ("observation_weights", weights),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")
    if (
        observed.shape != grid.shape
        or weights.shape != grid.shape
        or mask.shape != grid.shape
    ):
        raise ValueError("Observation arrays must all have shape (120,120).")
    if np.any(weights < 0.0) or not np.any(mask):
        raise ValueError("Observation weights/mask are invalid.")
    bounds = index.get("inversion_bounds", {})
    u_min = float(bounds.get("u_min", float("nan")))
    u_max = float(bounds.get("u_max", float("nan")))
    if not np.isfinite(u_min) or not np.isfinite(u_max) or u_max <= u_min:
        raise ValueError("Surrogate index has invalid or missing inversion bounds.")
    for row, arrays in artifacts[1:]:
        for name in (
            "x_grid",
            "y_grid",
            "directions_u",
            "data_observed",
            "observation_weights",
            "observation_mask",
        ):
            if not np.array_equal(reference[name], arrays[name]):
                raise ValueError(f"{row['run_id']}: common array {name} differs.")

    cases_by_field: dict[str, list[tuple[dict[str, Any], dict[str, np.ndarray], str]]] = {}
    for row, arrays in artifacts:
        for basepoint in BASEPOINTS:
            field_hash = sha256_array(arrays[f"u_{basepoint}"])
            cases_by_field.setdefault(field_hash, []).append((row, arrays, basepoint))
    output_dir = prepare_output_directory(args.output_dir, overwrite=args.overwrite)

    reaction = ReactionConfig(
        fem_nx=args.fem_nx,
        fem_ny=args.fem_ny,
        state_degree=1,
        rhs=1.0,
        u_min=u_min,
    )
    csv_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    solver_options = _fem_solver_options(args)
    with ReactionSolver2D(
        reaction,
        comm=MPI.COMM_WORLD,
        minimum_u=u_min,
        initial_u=100.0,
        solver_options=solver_options,
    ) as solver, StationaryMaternCovariance2D(
        grid,
        nu=float(index["matern"]["nu"]),
        length_scale=float(index["matern"]["length_scale"]),
        variance=float(index["matern"]["variance"]),
        fft_workers=args.fft_workers,
    ) as covariance:
        coefficient_map = StructuredCoefficientGridMap(solver, grid)
        state_map = StructuredLagrangeStateGridMap(solver, grid)
        observation_adjoint = ScalarVertexObservationAdjoint(
            solver,
            state_map,
            solver_options=solver_options,
        )
        try:
            for field_number, (field_hash, cases) in enumerate(cases_by_field.items(), start=1):
                u = np.asarray(cases[0][1][f"u_{cases[0][2]}"], dtype=np.float64)
                if (
                    not np.all(np.isfinite(u))
                    or np.min(u) < u_min - 1.0e-10
                    or np.max(u) > u_max + 1.0e-10
                ):
                    raise ValueError(
                        f"Field {field_hash} is non-finite or outside [{u_min},{u_max}]."
                    )
                for row, arrays, basepoint in cases[1:]:
                    if not np.array_equal(u, arrays[f"u_{basepoint}"]):
                        raise ValueError(f"Internal field-hash collision for {row['run_id']}.")
                print(
                    f"FEM field {field_number}/{len(cases_by_field)}: "
                    + ", ".join(
                        f"{row['architecture']}:{basepoint}"
                        for row, _, basepoint in cases
                    ),
                    flush=True,
                )
                solver.set_coefficient(coefficient_map.grid_to_array(u))
                state = solver.forward()
                fem_forward = state_map.state_to_grid(state)
                fem_jvps = []
                for direction_index, direction in enumerate(directions):
                    direction_function = coefficient_map.grid_to_function(
                        direction, name=f"diagnostic_direction_{direction_index:02d}"
                    )
                    tangent = solver.tangent(state, direction_function)
                    fem_jvps.append(state_map.state_to_grid(tangent))
                fem_jvps_array = np.stack(fem_jvps, axis=0)

                fem_residual = fem_forward - observed
                fem_source = observation_adjoint.lift_weighted_residual(
                    fem_residual, weights, mask=mask, name="fem_residual_source"
                )
                fem_adjoint = solver.adjoint(fem_source)
                fem_native_covector = solver.parameter_coordinate_adjoint(state, fem_adjoint)
                fem_covector = coefficient_map.array_to_grid(fem_native_covector)
                fem_direction = covariance.apply_coordinate_covector(fem_covector)

                for row, arrays, basepoint in cases:
                    architecture = row["architecture"]
                    run_id = row["run_id"]
                    surrogate_forward = arrays[f"surrogate_forward_{basepoint}"]
                    surrogate_jvps = arrays[f"surrogate_jvp_{basepoint}"]
                    surrogate_residual = arrays[f"surrogate_residual_{basepoint}"]
                    surrogate_covector = arrays[
                        f"surrogate_coordinate_covector_{basepoint}"
                    ]
                    surrogate_direction = arrays[
                        f"surrogate_landweber_direction_{basepoint}"
                    ]

                    # Exact J_h(u)^* W r_theta uses the same residual as the
                    # surrogate VJP and therefore isolates derivative error.
                    shared_source = observation_adjoint.lift_weighted_residual(
                        surrogate_residual,
                        weights,
                        mask=mask,
                        name=f"surrogate_residual_source_{architecture}_{basepoint}",
                    )
                    shared_adjoint = solver.adjoint(shared_source)
                    shared_native = solver.parameter_coordinate_adjoint(state, shared_adjoint)
                    fem_covector_same_residual = coefficient_map.array_to_grid(shared_native)
                    fem_direction_same_residual = covariance.apply_coordinate_covector(
                        fem_covector_same_residual
                    )

                    forward_metrics = comparison_metrics(
                        surrogate_forward, fem_forward, weights=weights, mask=mask
                    )
                    jvp_pooled = comparison_metrics(
                        surrogate_jvps, fem_jvps_array, weights=weights, mask=mask
                    )
                    direction_rows = per_direction_metrics(
                        surrogate_jvps, fem_jvps_array, weights=weights, mask=mask
                    )
                    vjp_same = comparison_metrics(
                        surrogate_covector, fem_covector_same_residual
                    )
                    preconditioned_same = comparison_metrics(
                        surrogate_direction,
                        fem_direction_same_residual,
                        weights=weights,
                    )
                    operational_direction = comparison_metrics(
                        surrogate_direction, fem_direction, weights=weights
                    )
                    data_norm = weighted_norm(observed, weights, mask=mask)
                    surrogate_residual_norm = weighted_norm(
                        surrogate_residual, weights, mask=mask
                    )
                    fem_residual_norm = weighted_norm(fem_residual, weights, mask=mask)

                    detail = {
                        "format": FORMAT_COMPARISON,
                        "architecture": architecture,
                        "run_id": run_id,
                        "basepoint": basepoint,
                        "minimum_u_error_iteration": int(
                            np.asarray(arrays["minimum_u_error_iteration"])
                            .reshape(())
                            .item()
                        ),
                        "field_sha256": field_hash,
                        "field": {
                            "minimum": float(np.min(u)),
                            "maximum": float(np.max(u)),
                            "lower_active_count": int(
                                np.count_nonzero(np.isclose(u, u_min, rtol=0.0, atol=1.0e-12))
                            ),
                            "upper_active_count": int(
                                np.count_nonzero(np.isclose(u, u_max, rtol=0.0, atol=1.0e-12))
                            ),
                        },
                        "forward_surrogate_vs_fem": forward_metrics,
                        "jvp_surrogate_vs_fem_pooled": jvp_pooled,
                        "jvp_per_direction": direction_rows,
                        "jvp_per_direction_summary": summarize_direction_rows(direction_rows),
                        "vjp_same_surrogate_residual": vjp_same,
                        "preconditioned_direction_same_surrogate_residual": preconditioned_same,
                        "operational_landweber_direction": operational_direction,
                        "relative_data_residual": {
                            "surrogate": surrogate_residual_norm / max(data_norm, 1.0e-300),
                            "fem": fem_residual_norm / max(data_norm, 1.0e-300),
                        },
                    }
                    detailed_rows.append(detail)
                    row_dir = output_dir / run_id
                    row_dir.mkdir(parents=True, exist_ok=True)
                    atomic_json(row_dir / f"{basepoint}.json", detail)
                    atomic_npz(
                        row_dir / f"{basepoint}.npz",
                        {
                            "u": u,
                            "directions_u": directions,
                            "data_observed": observed,
                            "observation_weights": weights,
                            "observation_mask": mask,
                            "surrogate_forward": surrogate_forward,
                            "fem_forward": fem_forward,
                            "surrogate_jvp": surrogate_jvps,
                            "fem_jvp": fem_jvps_array,
                            "surrogate_coordinate_covector": surrogate_covector,
                            "fem_coordinate_covector_same_residual": fem_covector_same_residual,
                            "fem_coordinate_covector_physical_residual": fem_covector,
                            "surrogate_landweber_direction": surrogate_direction,
                            "fem_landweber_direction_same_residual": fem_direction_same_residual,
                            "fem_landweber_direction_physical_residual": fem_direction,
                        },
                    )
                    csv_rows.append(
                        {
                            "architecture": architecture,
                            "run_id": run_id,
                            "basepoint": basepoint,
                            "minimum_u_error_iteration": int(
                                np.asarray(arrays["minimum_u_error_iteration"])
                                .reshape(())
                                .item()
                            ),
                            "surrogate_relative_data_residual": detail[
                                "relative_data_residual"
                            ]["surrogate"],
                            "fem_relative_data_residual": detail[
                                "relative_data_residual"
                            ]["fem"],
                            "forward_relative_error": forward_metrics["relative_error"],
                            "jvp_pooled_relative_error": jvp_pooled["relative_error"],
                            "jvp_pooled_norm_ratio": jvp_pooled["norm_ratio"],
                            "jvp_pooled_cosine": jvp_pooled["cosine"],
                            "jvp_relative_error_p95": detail[
                                "jvp_per_direction_summary"
                            ]["relative_error_p95"],
                            "jvp_cosine_minimum": detail[
                                "jvp_per_direction_summary"
                            ]["cosine_minimum"],
                            "vjp_same_residual_relative_error": vjp_same[
                                "relative_error"
                            ],
                            "vjp_same_residual_cosine": vjp_same["cosine"],
                            "preconditioned_same_residual_relative_error": preconditioned_same[
                                "relative_error"
                            ],
                            "preconditioned_same_residual_cosine": preconditioned_same[
                                "cosine"
                            ],
                            "operational_direction_relative_error": operational_direction[
                                "relative_error"
                            ],
                            "operational_direction_cosine": operational_direction[
                                "cosine"
                            ],
                        }
                    )
                    print(
                        f"  {architecture:12s} {basepoint:8s} "
                        f"JVP={jvp_pooled['relative_error']:.3e} "
                        f"cos={jvp_pooled['cosine']:.4f} "
                        f"direction-cos={operational_direction['cosine']:.4f}",
                        flush=True,
                    )
        finally:
            observation_adjoint.close()

    csv_path = output_dir / "headline_derivative_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    atomic_json(
        output_dir / "headline_derivative_comparison.json",
        {
            "format": FORMAT_COMPARISON,
            "source_index": input_dir / "index.json",
            "fem": {
                "cells": [args.fem_nx, args.fem_ny],
                "state_degree": 1,
                "rhs": 1.0,
                "linear_solver": args.linear_solver,
                "ksp_rtol": args.ksp_rtol,
                "ksp_max_it": args.ksp_max_it,
            },
            "comparisons": detailed_rows,
        },
    )
    (output_dir / "COMPLETED.txt").write_text(
        "FEM-versus-surrogate derivative comparison completed successfully.\n",
        encoding="utf-8",
    )
    print(f"Saved FEM comparison in {output_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compare headline surrogate and FEM derivatives at inversion fields.",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    surrogate = subparsers.add_parser("surrogate", help="Run the GPU/PyTorch stage.")
    surrogate.add_argument("--manifest", type=Path, required=True)
    surrogate.add_argument("--kl-basis", type=Path, required=True)
    surrogate.add_argument("--output-dir", type=Path, required=True)
    surrogate.add_argument("--num-directions", type=int, default=16)
    surrogate.add_argument("--device", default="cuda")
    surrogate.add_argument("--deeponet-query-chunk", type=int, default=14400)
    surrogate.add_argument("--matern-nu", type=float, default=1.5)
    surrogate.add_argument("--matern-length-scale", type=float, default=0.1)
    surrogate.add_argument("--matern-variance", type=float, default=1.0)
    surrogate.add_argument("--u-min", type=float, default=0.0)
    surrogate.add_argument("--u-max", type=float, default=200.0)
    surrogate.add_argument("--fft-workers", type=int, default=8)
    surrogate.add_argument("--overwrite", action="store_true")

    fem = subparsers.add_parser("fem", help="Run the CPU/FEniCSx comparison stage.")
    fem.add_argument("--input-dir", type=Path, required=True)
    fem.add_argument("--output-dir", type=Path, required=True)
    fem.add_argument("--fem-nx", type=int, default=119)
    fem.add_argument("--fem-ny", type=int, default=119)
    fem.add_argument("--linear-solver", choices=("cg-hypre", "lu"), default="cg-hypre")
    fem.add_argument("--ksp-rtol", type=float, default=1.0e-11)
    fem.add_argument("--ksp-max-it", type=int, default=10000)
    fem.add_argument("--fft-workers", type=int, default=8)
    fem.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.stage == "surrogate":
        if args.num_directions < 1:
            raise ValueError("--num-directions must be positive.")
        if args.deeponet_query_chunk < 1:
            raise ValueError("--deeponet-query-chunk must be positive.")
        if args.matern_nu <= 0.0 or args.matern_length_scale <= 0.0:
            raise ValueError("Matérn nu and length scale must be positive.")
        if args.matern_variance <= 0.0:
            raise ValueError("Matérn variance must be positive.")
        if not np.isfinite(args.u_min) or not np.isfinite(args.u_max):
            raise ValueError("Inversion bounds must be finite.")
        if args.u_max <= args.u_min:
            raise ValueError("--u-max must be greater than --u-min.")
    else:
        if args.fem_nx < 2 or args.fem_ny < 2:
            raise ValueError("FEM cell counts must be at least two.")
        if args.ksp_rtol <= 0.0 or args.ksp_max_it < 1:
            raise ValueError("Invalid KSP tolerances.")
    if args.fft_workers == 0:
        raise ValueError("--fft-workers cannot be zero.")


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    validate_args(args)
    if args.stage == "surrogate":
        stage_surrogate(args)
    else:
        stage_fem(args)


if __name__ == "__main__":
    main()
