from __future__ import annotations

import unittest

import numpy as np

from reaction_core.random_fields import matern_covariance


class MaternSmoothnessRevisionTests(unittest.TestCase):
    def test_half_integer_closed_forms(self) -> None:
        radius = np.linspace(0.0, 2.0, 1001)
        length_scale = 0.17
        variance = 1.7

        half = matern_covariance(radius, 0.5, length_scale, variance)
        half_expected = variance * np.exp(-radius / length_scale)
        np.testing.assert_allclose(half, half_expected, rtol=1.0e-12, atol=1.0e-14)

        three_half = matern_covariance(radius, 1.5, length_scale, variance)
        scaled = np.sqrt(3.0) * radius / length_scale
        three_half_expected = variance * (1.0 + scaled) * np.exp(-scaled)
        np.testing.assert_allclose(
            three_half, three_half_expected, rtol=1.0e-12, atol=1.0e-14
        )

    def test_nu_quarter_dense_subgrid_is_symmetric_positive_definite(self) -> None:
        rng = np.random.default_rng(20260924)
        full = np.stack(
            np.meshgrid(
                np.linspace(0.0, 1.0, 120),
                np.linspace(0.0, 1.0, 120),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 2)
        points = full[rng.choice(full.shape[0], size=400, replace=False)]
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        covariance = matern_covariance(
            distances, nu=0.25, length_scale=0.1, variance=1.0
        )

        np.testing.assert_allclose(covariance, covariance.T, rtol=0.0, atol=0.0)
        self.assertTrue(np.all(np.isfinite(covariance)))
        self.assertGreater(float(np.linalg.eigvalsh(covariance)[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
