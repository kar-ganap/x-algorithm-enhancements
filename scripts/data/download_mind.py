#!/usr/bin/env python3
"""Download MIND (Microsoft News Dataset) small release.

Source: Hugging Face mirror maintained by the Recommenders team.
The original Microsoft Azure blob storage (mind201910small.blob.core.windows.net)
is no longer publicly accessible (HTTP 409).

MIND-small: 50K users, ~65K news articles, ~230K impressions.
Total size: ~80 MB compressed, ~450 MB extracted.

Usage:
    uv run python scripts/data/download_mind.py
"""

import os
import zipfile
from pathlib import Path

import requests

MIND_TRAIN_URL = "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_train.zip"
MIND_DEV_URL = "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_dev.zip"
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def download_and_extract(url: str, target_dir: Path, zip_name: str) -> None:
    zip_path = target_dir.parent / zip_name
    target_dir.mkdir(parents=True, exist_ok=True)

    if (target_dir / "news.tsv").exists() and (target_dir / "behaviors.tsv").exists():
        print(f"  Already exists at {target_dir}")
        return

    print(f"  Downloading {url} ...")
    with requests.get(url, stream=True, allow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"    {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)",
                          end="\r", flush=True)
    print(f"\n  Downloaded {zip_path.stat().st_size / 1e6:.1f} MB")

    print(f"  Extracting to {target_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    os.remove(zip_path)


def main():
    mind_dir = DATA_DIR / "mind-small"
    print("Downloading MIND-small (HuggingFace mirror) ...")
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

    behaviors_train = mind_dir / "train" / "behaviors.tsv"
    if behaviors_train.exists():
        with open(behaviors_train, encoding="utf-8") as f:
            beh_count = sum(1 for _ in f)
        print(f"Train impressions:   {beh_count:,}")


if __name__ == "__main__":
    main()
