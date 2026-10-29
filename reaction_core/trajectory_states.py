"""Optional streaming archive for Landweber parameter states.

The production drivers keep this feature disabled unless a positive sampling
interval is requested.  HDF5 is imported lazily so the historical/default
execution path does not acquire a new import or storage cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


TRAJECTORY_STATE_FORMAT = "reaction-landweber-trajectory-states-v1"


class TrajectoryStateWriter:
    """Stream selected parameter iterates to a compressed HDF5 sidecar."""

    def __init__(
        self,
        output_dir: Path,
        *,
        state_shape: tuple[int, int],
        save_every: int,
        filename: str = "trajectory_states.h5",
    ) -> None:
        if int(save_every) < 1:
            raise ValueError("save_every must be positive.")
        if len(state_shape) != 2 or min(int(v) for v in state_shape) < 1:
            raise ValueError("state_shape must contain two positive dimensions.")

        import h5py  # Lazy by design: disabled runs do not import HDF5.

        self._h5py = h5py
        self.save_every = int(save_every)
        self.state_shape = tuple(int(v) for v in state_shape)
        self.path = Path(output_dir) / filename
        self.partial_path = self.path.with_name(self.path.name + ".partial")
        if self.path.exists() or self.partial_path.exists():
            raise FileExistsError(
                f"Trajectory-state output already exists: {self.path} or "
                f"{self.partial_path}"
            )

        self._file: Any = h5py.File(self.partial_path, "x")
        self._file.attrs["format"] = TRAJECTORY_STATE_FORMAT
        self._file.attrs["save_every"] = self.save_every
        self._file.attrs["state_shape"] = self.state_shape
        self._iterations = self._file.create_dataset(
            "iteration",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(1024,),
            fletcher32=True,
        )
        self._states = self._file.create_dataset(
            "u",
            shape=(0, *self.state_shape),
            maxshape=(None, *self.state_shape),
            dtype=np.float64,
            chunks=(1, *self.state_shape),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fletcher32=True,
        )
        self._count = 0
        self._last_iteration = -1
        self._finalized = False

    def observe(self, iteration: int, state: np.ndarray) -> bool:
        """Append a state when ``iteration`` is on the requested cadence."""

        iteration = int(iteration)
        if iteration < 0:
            raise ValueError("iteration cannot be negative.")
        if iteration % self.save_every:
            return False
        if iteration <= self._last_iteration:
            raise ValueError("Trajectory iterations must be strictly increasing.")
        array = np.asarray(state, dtype=np.float64)
        if array.shape != self.state_shape:
            raise ValueError(
                f"Expected state shape {self.state_shape}, received {array.shape}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("Trajectory state contains non-finite values.")

        index = self._count
        self._iterations.resize((index + 1,))
        self._states.resize((index + 1, *self.state_shape))
        self._iterations[index] = iteration
        self._states[index, ...] = array
        self._count += 1
        self._last_iteration = iteration
        return True

    @property
    def count(self) -> int:
        return self._count

    def finalize(self) -> Path:
        """Close the valid archive and publish it atomically."""

        if self._finalized:
            return self.path
        if self._file is None:
            raise RuntimeError("Cannot finalize an aborted trajectory archive.")
        self._file.attrs["saved_state_count"] = self._count
        self._file.attrs["last_saved_iteration"] = self._last_iteration
        self._file.flush()
        self._file.close()
        self._file = None
        self.partial_path.replace(self.path)
        self._finalized = True
        return self.path

    def abort(self) -> None:
        """Close and remove this writer's unpublished partial file."""

        if self._file is not None:
            self._file.close()
            self._file = None
        if not self._finalized and self.partial_path.exists():
            self.partial_path.unlink()
