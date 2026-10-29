#!/usr/bin/env python3
"""Compare regenerated Reaction products with the frozen paper references."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TABLES = (
    "headline_emulator_test.tex",
    "noiseless_headline_T1_T2.tex",
    "fno_capacity_semiconvergence.tex",
    "fno_training_size_semiconvergence.tex",
    "preconditioner_low_smoothness_errors.tex",
    "table_low_smoothness_diagnostics_single.tex",
    "noisy_tau_stopping_grouped.tex",
)
FIGURES = (
    "noiseless_convergence_compact.pdf",
    "noiseless_oracle_reconstructions_rows.pdf",
    "derivative_geometry_compact.pdf",
    "preconditioner_identity_fields_T2.pdf",
    "noisy_convergence_T1_compact.pdf",
)
AUTHOR_ACCEPTED_LAYOUT_PIXELS = {
    "noiseless_oracle_reconstructions_rows.pdf": 8252,
    "preconditioner_identity_fields_T2.pdf": 138045,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render(pdf: Path, output_stem: Path) -> Path:
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("pdftoppm is required for the 200-dpi pixel comparison")
    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", "200", str(pdf), str(output_stem)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return output_stem.with_suffix(".png")


def verify_bundle(data_root: Path) -> dict[str, object]:
    checksum_file = data_root / "SHA256SUMS"
    failures = []
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = data_root / relative
        checked += 1
        actual = sha256(path)
        if actual != expected:
            failures.append({"path": relative, "expected": expected, "actual": actual})
    return {"files_checked": checked, "failures": failures, "pass": not failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "reference/paper")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/reaction-paper-v1")
    parser.add_argument(
        "--strict-field-pixels",
        action="store_true",
        help="Treat the two documented field-figure layout differences as failures.",
    )
    args = parser.parse_args()
    report: dict[str, object] = {
        "tables": [],
        "figures": [],
        "tier_a_data": verify_bundle(args.data_root),
        "strict_field_pixels": bool(args.strict_field_pixels),
    }
    all_pass = bool(report["tier_a_data"]["pass"])

    for name in TABLES:
        generated = args.output_root / "tables" / name
        reference = args.reference_dir / name
        same = generated.read_bytes() == reference.read_bytes()
        report["tables"].append(
            {"file": name, "byte_identical": same, "generated_sha256": sha256(generated), "reference_sha256": sha256(reference)}
        )
        all_pass = all_pass and same

    with tempfile.TemporaryDirectory(prefix="reaction-release-pdf-check-") as temporary:
        temporary_root = Path(temporary)
        for name in FIGURES:
            generated = args.output_root / "figures" / name
            reference = args.reference_dir / name
            generated_png = render(generated, temporary_root / f"generated_{Path(name).stem}")
            reference_png = render(reference, temporary_root / f"reference_{Path(name).stem}")
            generated_pixels = np.asarray(Image.open(generated_png).convert("RGBA"), dtype=np.int16)
            reference_pixels = np.asarray(Image.open(reference_png).convert("RGBA"), dtype=np.int16)
            same_shape = generated_pixels.shape == reference_pixels.shape
            if same_shape:
                difference = np.abs(generated_pixels - reference_pixels)
                nonzero = int(np.count_nonzero(np.any(difference != 0, axis=2)))
                maximum = int(difference.max(initial=0))
            else:
                nonzero = -1
                maximum = -1
            same = same_shape and nonzero == 0
            accepted_layout_only = (
                same_shape
                and nonzero == AUTHOR_ACCEPTED_LAYOUT_PIXELS.get(name, -2)
            )
            passes_policy = same or (accepted_layout_only and not args.strict_field_pixels)
            report["figures"].append(
                {
                    "file": name,
                    "render_dpi": 200,
                    "same_pixel_shape": same_shape,
                    "nonzero_pixels": nonzero,
                    "maximum_channel_difference": maximum,
                    "pixel_identical": same,
                    "accepted_layout_only": accepted_layout_only,
                    "passes_selected_policy": passes_policy,
                    "generated_sha256": sha256(generated),
                    "reference_sha256": sha256(reference),
                }
            )
            all_pass = all_pass and passes_policy

    report["pass"] = all_pass
    destination = args.output_root / "verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
