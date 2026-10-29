#!/usr/bin/env python3
"""Mathematical smoke tests for the authoritative reaction FEM core.

The default mesh is deliberately small so this script is quick enough for a
login-node/interactive check.  Production resolution is controlled by the
project configuration and shard generator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from mpi4py import MPI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reaction_core.config import ReactionConfig  # noqa: E402
from reaction_core.grid import (  # noqa: E402
    RegularGrid2D,
    StructuredCoefficientGridMap,
    StructuredLagrangeStateGridMap,
)
from reaction_core.inversion import (  # noqa: E402
    ScalarVertexObservationAdjoint,
    StationaryMaternCovariance2D,
    add_scalar_heteroscedastic_noise,
    interior_vertex_mask,
    trapezoid_weights,
    validate_matern_covariance_action,
    weighted_inner,
)
from reaction_core.observations import FullFieldObservationOperator  # noqa: E402
from reaction_core.solver import ReactionSolver2D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check grid permutations, positivity, cached LU reuse, the "
            "reaction tangent and its L2 adjoint."
        )
    )
    parser.add_argument("--fem-nx", type=int, default=12)
    parser.add_argument("--fem-ny", type=int, default=12)
    parser.add_argument("--state-degree", type=int, choices=(1, 2), default=1)
    parser.add_argument("--taylor-rate-min", type=float, default=1.75)
    parser.add_argument("--adjoint-rtol", type=float, default=2.0e-9)
    return parser.parse_args()


def difference_function(
    solver: ReactionSolver2D,
    first,
    second,
    tangent,
    epsilon: float,
):
    values = (
        first.x.array
        - second.x.array
        - float(epsilon) * tangent.x.array
    )
    return solver.new_state_function(values, name="Taylor_remainder")


def main() -> None:
    args = parse_args()
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError(
            "Run this structured-grid test with one MPI rank."
        )
    if args.fem_nx < 3 or args.fem_ny < 3:
        raise ValueError("Use at least three cells per axis for this test.")

    config = ReactionConfig(
        Lx=1.0,
        Ly=1.0,
        fem_nx=args.fem_nx,
        fem_ny=args.fem_ny,
        state_degree=args.state_degree,
        rhs=1.0,
    )
    grid = RegularGrid2D.from_domain(
        config.Lx,
        config.Ly,
        config.fem_nx + 1,
        config.fem_ny + 1,
    )
    X, Y = grid.mesh()

    metrics: dict[str, object] = {
        "fem_cells": [config.fem_nx, config.fem_ny],
        "state_degree": config.state_degree,
        "grid_shape": list(grid.shape),
        "matern_fft_dense_check": validate_matern_covariance_action(),
    }

    with ReactionSolver2D(config) as solver:
        coefficient_map = StructuredCoefficientGridMap(solver, grid)
        state_map = StructuredLagrangeStateGridMap(solver, grid)
        observations = FullFieldObservationOperator(state_map)

        # Smooth but nontrivial coefficient and physical direction.  The
        # coefficient stays well inside the required u >= 10 region.
        u_grid = (
            50.0
            + 7.0 * np.sin(2.0 * np.pi * X) * np.sin(np.pi * Y)
            + 3.0 * np.cos(np.pi * X) * np.cos(2.0 * np.pi * Y)
        )
        h_grid = (
            np.sin(np.pi * X) * np.sin(2.0 * np.pi * Y)
            + 0.35 * np.cos(3.0 * np.pi * X) * np.sin(np.pi * Y)
        )
        u_values = coefficient_map.grid_to_array(u_grid)
        h = coefficient_map.grid_to_function(
            h_grid,
            name="physical_direction_h",
        )

        # Exact coefficient permutation round trip.
        u_function = coefficient_map.grid_to_function(u_grid, name="u_test")
        coefficient_roundtrip = coefficient_map.function_to_grid(u_function)
        coefficient_roundtrip_error = float(
            np.max(np.abs(coefficient_roundtrip - u_grid))
        )
        if coefficient_roundtrip_error > 5.0e-14:
            raise AssertionError(
                "P1 coefficient/grid round trip is not exact: "
                f"{coefficient_roundtrip_error:.3e}."
            )
        metrics["coefficient_grid_roundtrip_linf"] = coefficient_roundtrip_error

        # A vertex grid determines all state DOFs only for P1.
        if args.state_degree == 1:
            probe_grid = X * (1.0 - X) * Y * (1.0 - Y)
            probe_state = state_map.grid_to_state(probe_grid, name="grid_probe")
            state_roundtrip_error = float(
                np.max(np.abs(state_map.state_to_grid(probe_state) - probe_grid))
            )
            if state_roundtrip_error > 5.0e-14:
                raise AssertionError(
                    "P1 state/grid round trip is not exact: "
                    f"{state_roundtrip_error:.3e}."
                )
            metrics["state_grid_roundtrip_linf"] = state_roundtrip_error
        else:
            metrics["state_grid_roundtrip_linf"] = None

        # The safeguard is an enforced invariant, not a warning or clip.
        unsafe = u_values.copy()
        unsafe[0] = solver.minimum_u - 1.0e-6
        try:
            solver.set_coefficient(unsafe)
        except ValueError:
            positivity_rejection = True
        else:
            positivity_rejection = False
        if not positivity_rejection:
            raise AssertionError("A coefficient below u=10 was accepted.")
        metrics["rejects_u_below_10"] = positivity_rejection

        solver.set_coefficient(u_values)
        state = solver.forward()
        pressure_grid = observations.apply_scalar(state, dtype=np.float64)

        boundary_values = np.concatenate(
            [
                pressure_grid[0, :],
                pressure_grid[-1, :],
                pressure_grid[:, 0],
                pressure_grid[:, -1],
            ]
        )
        boundary_linf = float(np.max(np.abs(boundary_values)))
        if boundary_linf > 1.0e-11:
            raise AssertionError(
                f"Dirichlet boundary values are not zero: {boundary_linf:.3e}."
            )
        if float(np.min(pressure_grid)) < -1.0e-11:
            raise AssertionError("The positive-source solution violates positivity.")
        metrics["pressure_boundary_linf"] = boundary_linf
        metrics["pressure_min"] = float(np.min(pressure_grid))
        metrics["pressure_max"] = float(np.max(pressure_grid))

        # The synthetic-data convention must never put an irreducible noise
        # floor on the exactly-known homogeneous Dirichlet boundary.
        observation_weights = trapezoid_weights(grid)
        observation_mask = interior_vertex_mask(grid)
        noise = add_scalar_heteroscedastic_noise(
            pressure_grid,
            weights=observation_weights,
            observation_mask=observation_mask,
            noise_percent=0.1,
            floor_fraction=0.05,
            seed=20260818,
        )
        boundary_noise = np.concatenate(
            [
                noise.noise[0, :],
                noise.noise[-1, :],
                noise.noise[:, 0],
                noise.noise[:, -1],
                noise.standard_deviation[0, :],
                noise.standard_deviation[-1, :],
                noise.standard_deviation[:, 0],
                noise.standard_deviation[:, -1],
            ]
        )
        boundary_noise_linf = float(np.max(np.abs(boundary_noise)))
        if boundary_noise_linf != 0.0:
            raise AssertionError(
                "Synthetic noise was added on the Dirichlet boundary: "
                f"{boundary_noise_linf:.3e}."
            )
        metrics["noise_boundary_linf"] = boundary_noise_linf
        metrics["noise_diagnostics"] = noise.diagnostics

        # One matrix/LU for the state followed by multiple right-hand sides.
        tangent = solver.tangent(state, h)
        _ = solver.tangent(state, -h.x.array)

        state_coords = solver.state_coordinates()
        q_values = (
            np.sin(np.pi * state_coords[:, 0])
            * np.sin(np.pi * state_coords[:, 1])
            + 0.2
            * np.sin(2.0 * np.pi * state_coords[:, 0])
            * np.sin(3.0 * np.pi * state_coords[:, 1])
        )
        q = solver.new_state_function(q_values, name="state_covector_q")
        adjoint = solver.adjoint(q)

        # Exact transpose of vertex observations under the trapezoidal grid
        # pairing.  For P2 the mass lift intentionally assigns a zero
        # coordinate load to edge-midpoint DOFs and solves the full P2 mass
        # system; simply interpolating the residual would be incorrect.
        residual_grid = (
            np.sin(np.pi * X) * np.sin(2.0 * np.pi * Y)
            + 0.3 * np.sin(3.0 * np.pi * X) * np.sin(np.pi * Y)
        )
        tangent_grid = observations.apply_scalar(tangent, dtype=np.float64)
        with ScalarVertexObservationAdjoint(solver, state_map) as vertex_adjoint:
            state_covector = vertex_adjoint.lift_weighted_residual(
                residual_grid,
                observation_weights,
                mask=observation_mask,
            )
            observation_adjoint = solver.adjoint(state_covector)
            coordinate_load_native = solver.parameter_coordinate_adjoint(
                state,
                observation_adjoint,
            )
            coordinate_load_grid = coefficient_map.array_to_grid(
                coordinate_load_native
            )
            vertex_adjoint_diagnostics = vertex_adjoint.diagnostics()

        observation_left = weighted_inner(
            residual_grid,
            tangent_grid,
            observation_weights,
            mask=observation_mask,
        )
        observation_right = float(
            np.sum(coordinate_load_grid * h_grid, dtype=np.float64)
        )
        observation_adjoint_relative_error = abs(
            observation_left - observation_right
        ) / max(abs(observation_left), abs(observation_right), 1.0e-14)
        if observation_adjoint_relative_error > args.adjoint_rtol:
            raise AssertionError(
                "Vertex-observation coordinate adjoint identity failed: "
                f"left={observation_left:.16e}, "
                f"right={observation_right:.16e}, "
                f"relative error={observation_adjoint_relative_error:.3e}."
            )
        metrics["vertex_observation_adjoint_identity"] = {
            "left": observation_left,
            "right": observation_right,
            "relative_error": observation_adjoint_relative_error,
            "lift": vertex_adjoint_diagnostics,
        }

        # The covariance-preconditioned coordinate direction must be positive
        # in its defining Euclidean pairing; no second quadrature factor is
        # inserted because coordinate_load_grid already is a nodal covector.
        with StationaryMaternCovariance2D(
            grid,
            nu=1.5,
            length_scale=0.15,
            variance=1.0,
        ) as covariance:
            matern_direction = covariance.apply_coordinate_covector(
                coordinate_load_grid
            )
        coordinate_direction_product = float(
            np.sum(coordinate_load_grid * matern_direction, dtype=np.float64)
        )
        if not np.isfinite(coordinate_direction_product) or (
            coordinate_direction_product <= 0.0
        ):
            raise AssertionError(
                "The Matérn-preconditioned coordinate product is not positive: "
                f"{coordinate_direction_product:.16e}."
            )
        metrics["coordinate_gradient_direction_product"] = (
            coordinate_direction_product
        )

        cache_before_coefficient_change = solver.cache_diagnostics()
        if cache_before_coefficient_change[
            "operator_assemblies_factorizations"
        ] != 1:
            raise AssertionError(
                "Forward/tangent/adjoint solves did not reuse one factorisation: "
                f"{cache_before_coefficient_change}."
            )
        metrics["cache_before_coefficient_change"] = cache_before_coefficient_change

        # Discrete L2 adjoint identity:
        #     <J h, q>_V = <h, -p lambda>_Q.
        gradient = solver.parameter_gradient(state, adjoint)
        left = solver.state_inner(tangent, q)
        right = solver.parameter_inner(h, gradient)
        adjoint_relative_error = abs(left - right) / max(
            abs(left),
            abs(right),
            1.0e-14,
        )
        if adjoint_relative_error > args.adjoint_rtol:
            raise AssertionError(
                "Discrete adjoint identity failed: "
                f"left={left:.16e}, right={right:.16e}, "
                f"relative error={adjoint_relative_error:.3e}."
            )
        metrics["adjoint_identity"] = {
            "left": left,
            "right": right,
            "relative_error": adjoint_relative_error,
        }

        # Second-order Taylor remainder for the physical-u tangent.
        epsilons = np.asarray([0.4, 0.2, 0.1, 0.05], dtype=np.float64)
        taylor_errors: list[float] = []
        for epsilon in epsilons:
            solver.set_coefficient(u_values + epsilon * h.x.array)
            perturbed = solver.forward()
            remainder = difference_function(
                solver,
                perturbed,
                state,
                tangent,
                float(epsilon),
            )
            taylor_errors.append(solver.state_norm(remainder))

        taylor_errors_array = np.asarray(taylor_errors)
        if np.any(taylor_errors_array <= 0.0):
            raise AssertionError(
                f"Non-positive Taylor errors encountered: {taylor_errors}."
            )
        rates = np.log(
            taylor_errors_array[:-1] / taylor_errors_array[1:]
        ) / np.log(2.0)
        minimum_rate = float(np.min(rates))
        if minimum_rate < args.taylor_rate_min:
            raise AssertionError(
                "Tangent Taylor remainder is not second order: "
                f"rates={rates.tolist()}."
            )
        metrics["taylor_test"] = {
            "epsilons": epsilons.tolist(),
            "remainder_norms": taylor_errors_array.tolist(),
            "rates": rates.tolist(),
            "minimum_rate": minimum_rate,
        }

        metrics["final_cache_diagnostics"] = solver.cache_diagnostics()
        metrics["state_dofs"] = solver.num_state_dofs()
        metrics["parameter_dofs"] = solver.num_parameter_dofs()

    print("Reaction core/grid checks passed.")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
