#!/usr/bin/env python3
"""Regenerate all five frozen Reaction-paper figures from Tier-A data."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(os.environ.get("REACTION_DATA_ROOT", ROOT / "data/reaction-paper-v1"))


def run(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/figures")
    args = parser.parse_args()
    data = args.data_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["REACTION_DATA_ROOT"] = str(data)
    env["REACTION_OUTPUT_DIR"] = str(output)
    run([sys.executable, "reproduce/_compact_convergence.py", "--overwrite"], env)
    run([sys.executable, "reproduce/_compact_derivative.py", "--overwrite"], env)
    run([sys.executable, "reproduce/make_field_figures.py", "--data-root", str(data), "--output-dir", str(output)], env)
    expected = {
        "noiseless_convergence_compact.pdf",
        "noiseless_oracle_reconstructions_rows.pdf",
        "derivative_geometry_compact.pdf",
        "preconditioner_identity_fields_T2.pdf",
        "noisy_convergence_T1_compact.pdf",
    }
    missing = sorted(name for name in expected if not (output / name).is_file())
    if missing:
        raise RuntimeError(f"Figure generation did not produce: {missing}")
    print(f"Wrote five paper figures to {output}")


if __name__ == "__main__":
    main()
