#!/usr/bin/env python3
"""Install and verify the Reaction Tier-A bundle from Zenodo or local files."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from pathlib import PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
ZENODO_RECORD = "https://doi.org/10.5281/zenodo.23767592"
DEFAULT_URL = (
    "https://zenodo.org/records/23767592/files/"
    "reaction-paper-products-v1.zip?download=1"
)
ARCHIVE_PREFIX = PurePosixPath("paper_products/reaction-paper-v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root: Path) -> None:
    checksum_file = root / "SHA256SUMS"
    if not checksum_file.is_file():
        raise FileNotFoundError(f"Missing bundle checksum file: {checksum_file}")
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {relative}: {actual} != {expected}")


def install_zip(archive_path: Path, destination: Path) -> None:
    """Extract only the Tier-A subtree from the frozen Zenodo ZIP."""

    destination.mkdir(parents=True)
    extracted = 0
    with zipfile.ZipFile(archive_path, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename)
            try:
                relative = member_path.relative_to(ARCHIVE_PREFIX)
            except ValueError:
                continue
            if not relative.parts or member.is_dir():
                continue
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted += 1
    if extracted == 0:
        raise RuntimeError(
            f"Archive does not contain the expected {ARCHIVE_PREFIX.as_posix()} subtree"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Local unpacked reaction-paper-v1 directory")
    parser.add_argument("--archive", type=Path, help="Local reaction-paper-products-v1 ZIP")
    parser.add_argument(
        "--url",
        default=os.environ.get("REACTION_ZENODO_URL", DEFAULT_URL),
        help=f"Tier-A ZIP URL (default: reserved data record {ZENODO_RECORD})",
    )
    parser.add_argument("--destination", type=Path, default=ROOT / "data/reaction-paper-v1")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.destination)
        print(f"Verified {args.destination.resolve()}")
        return
    if args.destination.exists():
        raise FileExistsError(f"Refusing to replace existing data directory: {args.destination}")
    if args.source:
        shutil.copytree(args.source, args.destination)
    elif args.archive:
        install_zip(args.archive, args.destination)
    elif args.url:
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="reaction-download-") as temporary:
            archive = Path(temporary) / "reaction-paper-products-v1.zip"
            urllib.request.urlretrieve(args.url, archive)
            install_zip(archive, args.destination)
    else:
        raise SystemExit("Pass --source, --archive or --url.")
    verify(args.destination)
    print(f"Installed and verified {args.destination.resolve()}")


if __name__ == "__main__":
    main()
