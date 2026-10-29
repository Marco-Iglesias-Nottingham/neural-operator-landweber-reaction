#!/usr/bin/env python3
"""Checks for the shared exact/surrogate synthetic-archive contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reaction_core.grid import RegularGrid2D
from reaction_core.inversion import interior_vertex_mask, trapezoid_weights
from reaction_core.synthetic_archive import (
    SYNTHETIC_FILE_FORMAT,
    load_fine_synthetic_archive,
)


grid = RegularGrid2D.from_domain(1.0, 1.0, 5, 5)
shape = grid.shape
mask = interior_vertex_mask(grid)
weights = trapezoid_weights(grid)
u = np.full(shape, 5.0)
clean = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) * 1.0e-4
noise = np.zeros(shape)
noise[mask] = 2.0e-6
noisy = clean + noise

with tempfile.TemporaryDirectory(prefix="reaction-synthetic-archive-") as directory:
    root = Path(directory)
    path = root / "truth.npz"
    np.savez_compressed(
        path,
        format=np.asarray(SYNTHETIC_FILE_FORMAT),
        parameterisation=np.asarray("direct physical u"),
        state_degree=np.asarray(1),
        fine_fem_x_grid=grid.x,
        fine_fem_y_grid=grid.y,
        fine_fem_nx_cells=np.asarray(4),
        fine_fem_ny_cells=np.asarray(4),
        rhs=np.asarray(1.0),
        u_fem_P1_vertices=u,
        data_noiseless=clean,
        data_noisy=noisy,
        noise_fem_P1_vertices=noise,
        observation_mask=mask,
        observation_weights_trapezoid=weights,
        truth_kind=np.asarray("test"),
        truth_seed=np.asarray(17),
        noise_seed=np.asarray(23),
        truth_metadata_json=np.asarray(json.dumps({"case": "fixture"})),
    )

    archive = load_fine_synthetic_archive(
        Path("truth"),
        project_root=root,
        supported_u_bounds=(0.0, 200.0),
    )
    assert archive.path == path.resolve()
    assert archive.fine_grid.shape == shape
    assert archive.fine_fem_nx == archive.fine_fem_ny == 4
    assert archive.truth_seed == 17
    assert archive.noise_seed == 23
    assert archive.metadata["truth"] == {"case": "fixture"}
    assert np.array_equal(archive.u_fine, u)
    assert np.array_equal(archive.data_noisy_fine, noisy)

    try:
        load_fine_synthetic_archive(
            path,
            supported_u_bounds=(10.0, 100.0),
        )
    except ValueError as exc:
        assert "below the emulator input support" in str(exc)
    else:
        raise AssertionError("Expected the narrower emulator support to reject u=5.")

print("Shared synthetic-archive checks passed.")
