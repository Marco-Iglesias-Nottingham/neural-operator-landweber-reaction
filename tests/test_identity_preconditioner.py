#!/usr/bin/env python3
"""Unit tests for the fixed-grid identity-preconditioner ablation."""

from __future__ import annotations

import numpy as np
import unittest

from reaction_core.grid import RegularGrid2D
from reaction_core.inversion import (
    IdentityCovariance2D,
    StationaryMaternCovariance2D,
)
from scripts.inversion import landweber_surrogate


def production_grid() -> RegularGrid2D:
    return RegularGrid2D.from_domain(1.0, 1.0, 120, 120)


class IdentityPreconditionerTests(unittest.TestCase):
    def test_identity_returns_finite_float64_copy(self) -> None:
        grid = production_grid()
        original = np.arange(grid.size, dtype=np.float32).reshape(grid.shape)
        with IdentityCovariance2D(grid, ignored_keyword="allowed") as identity:
            result = identity.apply_coordinate_covector(original)
            self.assertEqual(result.shape, grid.shape)
            self.assertEqual(result.dtype, np.float64)
            np.testing.assert_array_equal(result, original)
            result[0, 0] = -1.0
            self.assertEqual(original[0, 0], 0.0)
            self.assertEqual(identity.approximate_workspace_megabytes, 0.0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            identity.apply_coordinate_covector(original)

    def test_identity_rejects_bad_coordinate_covectors(self) -> None:
        bad_values = (
            np.zeros((119, 120)),
            np.full((120, 120), np.nan),
            np.full((120, 120), np.inf),
        )
        with IdentityCovariance2D(production_grid()) as identity:
            for bad in bad_values:
                with self.subTest(shape=bad.shape, finite=np.all(np.isfinite(bad))):
                    with self.assertRaises(ValueError):
                        identity.apply_coordinate_covector(bad)

    def test_short_length_scale_matern_converges_to_scaled_identity(self) -> None:
        grid = production_grid()
        rng = np.random.default_rng(20260918)
        coordinate_covector = rng.standard_normal(grid.shape)
        variance = 2.5
        with StationaryMaternCovariance2D(
            grid,
            nu=1.5,
            length_scale=1.0e-4,
            variance=variance,
            fft_workers=1,
        ) as covariance:
            actual = covariance.apply_coordinate_covector(coordinate_covector)
        np.testing.assert_allclose(
            actual,
            variance * coordinate_covector,
            rtol=2.0e-12,
            atol=2.0e-12,
        )

    def test_surrogate_parser_defaults_to_matern_and_accepts_identity(self) -> None:
        required = [
            "--synthetic-file",
            "truth.npz",
            "--checkpoint",
            "model.pt",
        ]
        self.assertEqual(
            landweber_surrogate.parse_args(required).preconditioner, "matern"
        )
        self.assertEqual(
            landweber_surrogate.parse_args(
                [*required, "--preconditioner", "identity"]
            ).preconditioner,
            "identity",
        )
        with self.assertRaises(SystemExit):
            landweber_surrogate.parse_args(
                [*required, "--preconditioner", "not-a-preconditioner"]
            )

    def test_step_is_invariant_to_positive_preconditioner_scale(self) -> None:
        rng = np.random.default_rng(20260918)
        q = rng.standard_normal((12, 10))
        direction = rng.standard_normal((12, 10))
        q_dot_d = float(np.sum(q * direction))
        if q_dot_d <= 0.0:
            direction = -direction
        j_direction = rng.standard_normal((12, 10))
        damping = 0.95
        change_cap = 5.0

        def update(scale: float) -> np.ndarray:
            d = scale * direction
            jd = scale * j_direction
            formula = damping * float(np.sum(q * d)) / float(np.sum(jd * jd))
            cap = change_cap / float(np.max(np.abs(d)))
            return min(formula, cap) * d

        np.testing.assert_allclose(
            update(1.0), update(7.25), rtol=2.0e-15, atol=2.0e-15
        )


if __name__ == "__main__":
    unittest.main()
