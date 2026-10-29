#!/usr/bin/env python3
"""Guard the frozen paper distribution in the public data entry points."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from reaction_core.config import DistributionConfig


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "nu_values": (1.0, 1.5, 2.0, 2.5),
    "length_scales": (0.075, 0.10, 0.15, 0.20, 0.25),
    "mean_u_range": (95.0, 105.0),
    "std_u_range": (10.0, 30.0),
    "u_bounds": (0.0, 200.0),
}


def config_values(config: DistributionConfig) -> dict[str, tuple[float, ...]]:
    return {
        "nu_values": tuple(float(value) for value in config.nu_values),
        "length_scales": tuple(float(value) for value in config.length_scales),
        "mean_u_range": tuple(float(value) for value in config.mean_u_range),
        "std_u_range": tuple(float(value) for value in config.std_u_range),
        "u_bounds": (float(config.u_min), float(config.u_max)),
    }


def frozen_t1_distribution() -> dict[str, tuple[float, ...]]:
    path = ROOT / "scripts/data/generate_truth_T1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "PRODUCTION_DISTRIBUTION" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Call):
            break
        values = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in node.value.keywords
            if keyword.arg is not None
        }
        return {
            "nu_values": tuple(float(value) for value in values["nu_values"]),
            "length_scales": tuple(float(value) for value in values["length_scales"]),
            "mean_u_range": tuple(float(value) for value in values["mean_u_range"]),
            "std_u_range": tuple(float(value) for value in values["std_u_range"]),
            "u_bounds": (float(values["u_min"]), float(values["u_max"])),
        }
    raise AssertionError(f"Could not find PRODUCTION_DISTRIBUTION in {path}.")


class DataDistributionContractTests(unittest.TestCase):
    def test_shared_distribution_defaults(self) -> None:
        self.assertEqual(config_values(DistributionConfig()), EXPECTED)

    def test_frozen_t1_matches_distribution_defaults(self) -> None:
        self.assertEqual(frozen_t1_distribution(), EXPECTED)

    def test_data_builders_use_shared_defaults(self) -> None:
        for relative in ("scripts/data/generate_shard.py", "scripts/data/build_kl_basis.py"):
            text = (ROOT / relative).read_text(encoding="utf-8")
            with self.subTest(path=relative):
                self.assertIn("defaults = DistributionConfig()", text)
                for attribute in ("nu_values", "length_scales", "mean_u_range", "std_u_range"):
                    self.assertIn(f"default=list(defaults.{attribute})", text)
                self.assertIn("default=[defaults.u_min, defaults.u_max]", text)


if __name__ == "__main__":
    unittest.main()
