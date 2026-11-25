"""Download and extract MovieLens datasets into ``data/raw/``.

Raw MovieLens files are gitignored and must never be committed or redistributed.
Dataset home: https://grouplens.org/datasets/movielens/

Usage:
    python scripts/download_movielens.py latest-small 1m
    python scripts/download_movielens.py all
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://files.grouplens.org/datasets/movielens"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO_ROOT / "data" / "raw"

# name -> (archive file name, extracted directory name, required files)
DATASETS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "latest-small": (
        "ml-latest-small.zip",
        "ml-latest-small",
        ("ratings.csv", "movies.csv", "tags.csv", "links.csv"),
    ),
    "1m": (
        "ml-1m.zip",
        "ml-1m",
        ("ratings.dat", "movies.dat", "users.dat"),
    ),
    "25m": (
        "ml-25m.zip",
        "ml-25m",
        (
            "ratings.csv",
            "movies.csv",
            "tags.csv",
            "links.csv",
            "genome-scores.csv",
            "genome-tags.csv",
        ),
    ),
}


def ssl_context() -> ssl.SSLContext:
    """Return a TLS context that trusts the certifi bundle when available.

    Windows Python installs often fail to build a complete default CA chain.
    certifi is a small, well-maintained bundle used by pip itself.
    """

    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def download(url: str, target: Path, force: bool = False) -> None:
    """Download ``url`` to ``target`` with a simple progress indicator."""

    if target.exists() and not force:
        print(f"  archive already present: {target.name}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")

    print(f"  downloading {url}")
    with (
        urllib.request.urlopen(url, context=ssl_context()) as response,
        partial.open("wb") as handle,
    ):
        total = int(response.headers.get("Content-Length") or 0)
        written = 0
        while chunk := response.read(256 * 1024):
            handle.write(chunk)
            written += len(chunk)
            if total:
                percent = written * 100 / total
                print(
                    f"\r  {percent:5.1f}%  {written / 1e6:7.1f} / {total / 1e6:.1f} MB",
                    end="",
                )
    print()
    partial.replace(target)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(archive: Path, expected_dir: Path, force: bool = False) -> None:
    """Extract ``archive`` next to itself and verify the expected files exist."""

    if expected_dir.exists() and not force:
        print(f"  already extracted: {expected_dir.name}")
        return

    if expected_dir.exists():
        shutil.rmtree(expected_dir)

    print(f"  extracting {archive.name}")
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(archive.parent)


def verify(dataset: str, expected_dir: Path) -> None:
    """Raise if any required file is missing."""

    required = DATASETS[dataset][2]
    missing = [name for name in required if not (expected_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{dataset}: missing required files: {', '.join(missing)}")


def fetch(dataset: str, dest: Path, force: bool = False) -> Path:
    """Download, extract and verify one dataset; return its directory."""

    archive_name, dir_name, _ = DATASETS[dataset]
    archive = dest / archive_name
    expected_dir = dest / dir_name

    print(f"[{dataset}]")
    download(f"{BASE_URL}/{archive_name}", archive, force=force)
    print(f"  sha256: {sha256(archive)}")
    extract(archive, expected_dir, force=force)
    verify(dataset, expected_dir)
    print(f"  ready: {expected_dir}")
    return expected_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=[*DATASETS, "all"],
        help="Datasets to fetch (or 'all').",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST}).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract.")
    return parser.parse_args()


def main() -> int:
    """Entry point."""

    args = parse_args()
    selected = list(DATASETS) if "all" in args.datasets else args.datasets

    print("MovieLens data is for academic use only. Do not redistribute the raw files.")
    for dataset in dict.fromkeys(selected):
        fetch(dataset, args.dest, force=args.force)

    print("\nDone. Raw files live under", args.dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
