#!/usr/bin/env python3
"""Download Amazon Reviews 2018 - Kindle Store 5-core.

Kindle Store 5-core: ~2.2M reviews, ~140K items (5-core filtered).
Downloads reviews (~437 MB compressed) and metadata (~310 MB compressed).
After download, streams through the files to build a subset:
  - Filter to top-20K items by review count
  - Filter to users with ≥5 reviews on those items
  - Target: ~100K reviews, 10K users, 20K items

Usage:
    uv run python scripts/data/download_amazon.py
"""

import gzip
import json
import urllib.request
from pathlib import Path

# UCSD McAuley Lab mirror (2018 release)
REVIEWS_URL = "https://jmcauley.ucsd.edu/data/amazon/amazonReviews/Kindle_Store_5.json.gz"
META_URL = "https://jmcauley.ucsd.edu/data/amazon/metaFiles/meta_Kindle_Store.json.gz"

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
    urllib.request.urlretrieve(url, dest)
    print(f"  Downloaded {dest.stat().st_size / 1e6:.1f} MB")


def stream_jsonl_gz(path: Path):
    """Yield dicts from a gzipped JSON-lines file (Amazon's format uses
    Python dict literals via eval, not strict JSON). Handle both."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Amazon 2018 release sometimes uses Python-dict-like literals
                # with single quotes; convert using literal_eval via ast
                import ast
                try:
                    yield ast.literal_eval(line)
                except (SyntaxError, ValueError):
                    continue


def build_subset(reviews_path: Path, meta_path: Path, out_dir: Path) -> None:
    subset_reviews_path = out_dir / "reviews_subset.jsonl"
    subset_meta_path = out_dir / "meta_subset.jsonl"

    if subset_reviews_path.exists() and subset_meta_path.exists():
        print(f"  Subset already exists at {out_dir}")
        return

    # Pass 1: count reviews per item (streaming)
    print("  Pass 1: counting reviews per item ...")
    item_counts: dict[str, int] = {}
    for r in stream_jsonl_gz(reviews_path):
        asin = r.get("asin")
        if asin:
            item_counts[asin] = item_counts.get(asin, 0) + 1
    print(f"    Total unique items: {len(item_counts):,}")

    # Select top-K items
    top_items = set(
        asin for asin, _ in sorted(item_counts.items(), key=lambda x: -x[1])[:TOP_K_ITEMS]
    )
    print(f"    Selected top-{len(top_items):,} items by review count")

    # Pass 2: count reviews per user among top items
    print("  Pass 2: counting reviews per user on top items ...")
    user_counts: dict[str, int] = {}
    for r in stream_jsonl_gz(reviews_path):
        asin = r.get("asin")
        uid = r.get("reviewerID")
        if asin in top_items and uid:
            user_counts[uid] = user_counts.get(uid, 0) + 1

    active_users = {uid for uid, n in user_counts.items() if n >= MIN_USER_REVIEWS}
    print(f"    Users with >= {MIN_USER_REVIEWS} reviews: {len(active_users):,}")

    # Pass 3: write filtered reviews
    print("  Pass 3: writing subset reviews ...")
    n_written = 0
    with open(subset_reviews_path, "w", encoding="utf-8") as f:
        for r in stream_jsonl_gz(reviews_path):
            if r.get("asin") in top_items and r.get("reviewerID") in active_users:
                minimal = {
                    "reviewerID": r.get("reviewerID"),
                    "asin": r.get("asin"),
                    "overall": r.get("overall"),
                    "unixReviewTime": r.get("unixReviewTime"),
                }
                f.write(json.dumps(minimal) + "\n")
                n_written += 1
    print(f"    Wrote {n_written:,} reviews to {subset_reviews_path}")

    # Pass 4: write metadata for top items
    print("  Pass 4: writing subset metadata ...")
    n_meta = 0
    with open(subset_meta_path, "w", encoding="utf-8") as f:
        for m in stream_jsonl_gz(meta_path):
            if m.get("asin") in top_items:
                minimal = {
                    "asin": m.get("asin"),
                    "title": m.get("title", ""),
                    "category": m.get("category", []),
                    "price": m.get("price", ""),
                    "brand": m.get("brand", ""),
                    "description": m.get("description", []),
                    "details": m.get("details", {}),
                }
                f.write(json.dumps(minimal) + "\n")
                n_meta += 1
    print(f"    Wrote {n_meta:,} metadata entries to {subset_meta_path}")


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    reviews_gz = TARGET_DIR / "Kindle_Store_5.json.gz"
    meta_gz = TARGET_DIR / "meta_Kindle_Store.json.gz"

    print("Downloading Amazon Kindle Store 5-core (may take several minutes) ...")
    download(REVIEWS_URL, reviews_gz)
    download(META_URL, meta_gz)

    print("\nBuilding subset (streaming through raw files) ...")
    build_subset(reviews_gz, meta_gz, TARGET_DIR)

    print(f"\nAmazon Kindle subset ready at {TARGET_DIR}")
    for f in sorted(TARGET_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
