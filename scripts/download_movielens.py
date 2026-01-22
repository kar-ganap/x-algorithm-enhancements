#!/usr/bin/env python3
"""Download MovieLens 100K dataset.

Usage:
    uv run python scripts/download_movielens.py
"""

import os
import urllib.request
import zipfile
from pathlib import Path


MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = Path(__file__).parent.parent / "data"


def download_movielens_100k(data_dir: Path = DATA_DIR) -> Path:
    """Download and extract MovieLens 100K dataset.

    Args:
        data_dir: Directory to store the data

    Returns:
        Path to extracted dataset directory
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "ml-100k.zip"
    extract_dir = data_dir / "ml-100k"

    # Check if already downloaded
    if extract_dir.exists() and (extract_dir / "u.data").exists():
        print(f"MovieLens 100K already exists at {extract_dir}")
        return extract_dir

    # Download
    print(f"Downloading MovieLens 100K from {MOVIELENS_100K_URL}...")
    urllib.request.urlretrieve(MOVIELENS_100K_URL, zip_path)
    print(f"Downloaded to {zip_path}")

    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    # Clean up zip
    os.remove(zip_path)

    print(f"MovieLens 100K extracted to {extract_dir}")
    print("\nDataset files:")
    for f in sorted(extract_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.name}: {size:,} bytes")

    return extract_dir


def main():
    extract_dir = download_movielens_100k()

    # Print basic stats
    data_file = extract_dir / "u.data"
    with open(data_file) as f:
        lines = f.readlines()

    print(f"\nDataset stats:")
    print(f"  Total ratings: {len(lines):,}")

    users = set()
    items = set()
    for line in lines:
        parts = line.strip().split('\t')
        users.add(parts[0])
        items.add(parts[1])

    print(f"  Unique users: {len(users):,}")
    print(f"  Unique items: {len(items):,}")
    print(f"  Density: {len(lines) / (len(users) * len(items)) * 100:.2f}%")


if __name__ == "__main__":
    main()
