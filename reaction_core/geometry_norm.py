"""Matrix-free covariance-inverse solves for revision geometry diagnostics."""

from __future__ import annotations

from time import perf_counter
from typing import Protocol

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


class CovarianceAction(Protocol):
    def apply_coordinate_covector(self, values: np.ndarray) -> np.ndarray: ...


def solve_covariance_inverse(
    covariance: CovarianceAction,
    vector: np.ndarray,
    *,
    x0: np.ndarray | None,
    rtol: float,
    maxiter: int,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Solve ``C x = vector`` by CG and return strict residual diagnostics."""

    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 2 or not np.all(np.isfinite(vector)):
        raise ValueError("vector must be a finite two-dimensional field.")
    if rtol <= 0.0 or maxiter < 1:
        raise ValueError("rtol and maxiter must be positive.")
    shape = vector.shape
    size = vector.size
    operator = LinearOperator(
        (size, size),
        matvec=lambda flat: covariance.apply_coordinate_covector(
            np.asarray(flat, dtype=np.float64).reshape(shape)
        ).reshape(-1),
        dtype=np.float64,
    )
    iterations = 0

    def callback(_iterate: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    started = perf_counter()
    solution, info = cg(
        operator,
        vector.reshape(-1),
        x0=None if x0 is None else np.asarray(x0, dtype=np.float64).reshape(-1),
        rtol=rtol,
        atol=0.0,
        maxiter=maxiter,
        callback=callback,
    )
    seconds = perf_counter() - started
    solution_field = np.asarray(solution, dtype=np.float64).reshape(shape)
    residual = covariance.apply_coordinate_covector(solution_field) - vector
    relative_residual = float(
        np.linalg.norm(residual.reshape(-1))
        / max(np.linalg.norm(vector.reshape(-1)), 1.0e-30)
    )
    diagnostics: dict[str, float | int | bool] = {
        "converged": bool(info == 0),
        "info": int(info),
        "iterations": int(iterations),
        "relative_residual": relative_residual,
        "seconds": float(seconds),
    }
    if info != 0 or relative_residual > max(10.0 * rtol, 1.0e-10):
        raise RuntimeError(f"Covariance CG did not converge: {diagnostics}")
    return solution_field, diagnostics
