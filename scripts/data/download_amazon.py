#!/usr/bin/env python3
"""Download Amazon Reviews 2023 - Kindle Store subset.

Source: McAuley Lab's 2023 release (the 2018 release is no longer available).
- 5-core pre-filtered ratings CSV from HuggingFace (~932 MB)
- Item metadata (gzipped JSONL) from UCSD McAuley Lab (~2.27 GB)

After download, streams through files to build a subset:
  - Filter to top-20K items by rating count
  - Filter to users with ≥5 reviews on those items
  - Target: ~100K reviews, 10K users, 20K items

Schema differences from 2018 release:
  - reviews: 'rating' (not 'overall'), 'user_id' (not 'reviewerID'),
    'parent_asin' is the join key to metadata
  - metadata: includes 'price', hierarchical 'categories', 'average_rating',
    'rating_number' — richer than 2018

Usage:
    uv run python scripts/data/download_amazon.py
"""

import gzip
import json
from pathlib import Path

import requests

# McAuley Lab 2023 release. UCSD hosts gzipped metadata; HF hosts pre-filtered 5-core ratings.
RATINGS_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/benchmark/5core/rating_only/Kindle_Store.csv"
)
META_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/"
    "raw/meta_categories/meta_Kindle_Store.jsonl.gz"
)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
TARGET_DIR = DATA_DIR / "amazon-kindle"
TOP_K_ITEMS = 20_000
MIN_USER_REVIEWS = 5


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already exists: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"  Downloading {url} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, allow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"    {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB ({pct:.0f}%)",
                          end="\r", flush=True)
    print(f"\n  Downloaded {dest.stat().st_size / 1e6:.1f} MB")


def stream_ratings_csv(path: Path):
    """Yield (user_id, parent_asin, rating, timestamp) tuples from the 5-core CSV.
    Format: user_id,parent_asin,rating,timestamp (with header row)."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip()  # consume header
        del header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                yield parts[0], parts[1], float(parts[2]), int(parts[3])
            except ValueError:
                continue


def stream_meta_jsonl_gz(path: Path):
    """Yield dicts from gzipped JSONL metadata file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_subset(ratings_path: Path, meta_path: Path, out_dir: Path) -> None:
    subset_reviews_path = out_dir / "reviews_subset.jsonl"
    subset_meta_path = out_dir / "meta_subset.jsonl"

    if subset_reviews_path.exists() and subset_meta_path.exists():
        print(f"  Subset already exists at {out_dir}")
        return

    # Pass 1: count reviews per item
    print("  Pass 1: counting reviews per item ...")
    item_counts: dict[str, int] = {}
    for _, parent_asin, _, _ in stream_ratings_csv(ratings_path):
        item_counts[parent_asin] = item_counts.get(parent_asin, 0) + 1
    print(f"    Total unique items: {len(item_counts):,}")

    top_items = set(
        asin for asin, _ in sorted(item_counts.items(), key=lambda x: -x[1])[:TOP_K_ITEMS]
    )
    print(f"    Selected top-{len(top_items):,} items by review count")

    # Pass 2: count reviews per user among top items
    print("  Pass 2: counting reviews per user on top items ...")
    user_counts: dict[str, int] = {}
    for uid, asin, _, _ in stream_ratings_csv(ratings_path):
        if asin in top_items:
            user_counts[uid] = user_counts.get(uid, 0) + 1

    active_users = {uid for uid, n in user_counts.items() if n >= MIN_USER_REVIEWS}
    print(f"    Users with >= {MIN_USER_REVIEWS} reviews: {len(active_users):,}")

    # Pass 3: write filtered reviews
    print("  Pass 3: writing subset reviews ...")
    n_written = 0
    with open(subset_reviews_path, "w", encoding="utf-8") as f:
        for uid, asin, rating, ts in stream_ratings_csv(ratings_path):
            if asin in top_items and uid in active_users:
                record = {
                    "user_id": uid,
                    "parent_asin": asin,
                    "rating": rating,
                    "timestamp": ts,
                }
                f.write(json.dumps(record) + "\n")
                n_written += 1
    print(f"    Wrote {n_written:,} reviews to {subset_reviews_path}")

    # Pass 4: write metadata for top items (join key is parent_asin)
    print("  Pass 4: writing subset metadata ...")
    n_meta = 0
    with open(subset_meta_path, "w", encoding="utf-8") as f:
        for m in stream_meta_jsonl_gz(meta_path):
            if m.get("parent_asin") in top_items:
                minimal = {
                    "parent_asin": m.get("parent_asin"),
                    "title": m.get("title", ""),
                    "categories": m.get("categories", []),
                    "main_category": m.get("main_category", ""),
                    "price": m.get("price", ""),
                    "average_rating": m.get("average_rating"),
                    "rating_number": m.get("rating_number"),
                    "details": m.get("details", {}),
                }
                f.write(json.dumps(minimal) + "\n")
                n_meta += 1
    print(f"    Wrote {n_meta:,} metadata entries to {subset_meta_path}")


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    ratings_csv = TARGET_DIR / "Kindle_Store_5core_ratings.csv"
    meta_gz = TARGET_DIR / "meta_Kindle_Store.jsonl.gz"

    print("Downloading Amazon Kindle 5-core ratings + metadata ...")
    download(RATINGS_URL, ratings_csv)
    download(META_URL, meta_gz)

    print("\nBuilding subset ...")
    build_subset(ratings_csv, meta_gz, TARGET_DIR)

    print(f"\nAmazon Kindle subset ready at {TARGET_DIR}")
    for f in sorted(TARGET_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
