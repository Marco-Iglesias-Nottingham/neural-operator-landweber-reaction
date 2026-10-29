#!/usr/bin/env python3
"""One-command Tier-A reproduction of Reaction paper products."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    commands = (
        [sys.executable, "reproduce/download_data.py", "--verify-only"],
        [sys.executable, "reproduce/make_tables.py"],
        [sys.executable, "reproduce/make_figures.py"],
        [sys.executable, "reproduce/numbers_in_text.py"],
        [sys.executable, "reproduce/check_against_paper.py"],
    )
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
