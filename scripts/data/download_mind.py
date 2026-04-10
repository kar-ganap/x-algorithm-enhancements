#!/usr/bin/env python3
"""Download MIND (Microsoft News Dataset) small release.

MIND-small: 50K users, ~65K news articles, ~230K impressions.
Total size: ~100 MB compressed, ~450 MB extracted.

Usage:
    uv run python scripts/data/download_mind.py
"""

import os
import urllib.request
import zipfile
from pathlib import Path

MIND_TRAIN_URL = "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
MIND_DEV_URL = "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip"
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def download_and_extract(url: str, target_dir: Path, zip_name: str) -> None:
    zip_path = target_dir.parent / zip_name
    target_dir.mkdir(parents=True, exist_ok=True)

    if (target_dir / "news.tsv").exists() and (target_dir / "behaviors.tsv").exists():
        print(f"  Already exists at {target_dir}")
        return

    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"  Downloaded {zip_path.stat().st_size / 1e6:.1f} MB")

    print(f"  Extracting to {target_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    os.remove(zip_path)


def main():
    mind_dir = DATA_DIR / "mind-small"
    print("Downloading MIND-small ...")
    download_and_extract(MIND_TRAIN_URL, mind_dir / "train", "MINDsmall_train.zip")
    download_and_extract(MIND_DEV_URL, mind_dir / "dev", "MINDsmall_dev.zip")

    print(f"\nMIND-small ready at {mind_dir}")
    for subdir in ["train", "dev"]:
        sub = mind_dir / subdir
        if sub.exists():
            print(f"  {subdir}/")
            for f in sorted(sub.iterdir()):
                if f.is_file():
                    print(f"    {f.name}: {f.stat().st_size / 1e6:.1f} MB")

    # Quick stats
    news_train = mind_dir / "train" / "news.tsv"
    if news_train.exists():
        with open(news_train, encoding="utf-8") as f:
            train_count = sum(1 for _ in f)
        print(f"\nTrain news articles: {train_count:,}")


if __name__ == "__main__":
    main()
