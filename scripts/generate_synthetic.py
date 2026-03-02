#!/usr/bin/env python3
"""Generate synthetic Twitter-like dataset.

Creates a synthetic dataset with planted patterns for verifying
that Phoenix can learn expected user behaviors.

Usage:
    uv run python scripts/generate_synthetic.py [--users N] [--posts N] [--engagements N]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "phoenix"))

from enhancements.data.ground_truth import (
    ARCHETYPE_DISTRIBUTION,
    TOPIC_DISTRIBUTION,
    ContentTopic,
    UserArchetype,
)
from enhancements.data.synthetic_twitter import (
    SyntheticTwitterGenerator,
    create_train_val_test_split,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Twitter-like dataset"
    )
    parser.add_argument(
        "--users", type=int, default=1000,
        help="Number of users to generate"
    )
    parser.add_argument(
        "--posts", type=int, default=50000,
        help="Number of posts to generate"
    )
    parser.add_argument(
        "--engagements", type=int, default=200000,
        help="Number of engagement events to generate"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output", type=str, default="data/synthetic_twitter",
        help="Output directory"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Synthetic Twitter Data Generation")
    print("=" * 70)
    print()

    print("Configuration:")
    print(f"  Users: {args.users}")
    print(f"  Posts: {args.posts}")
    print(f"  Engagements: {args.engagements}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {args.output}")
    print()

    # Show ground truth distributions
    print("User Archetype Distribution:")
    for arch, prob in ARCHETYPE_DISTRIBUTION.items():
        count = int(args.users * prob)
        print(f"  {arch.value:<15}: {prob*100:>5.1f}% (~{count} users)")
    print()

    print("Content Topic Distribution:")
    for topic, prob in TOPIC_DISTRIBUTION.items():
        count = int(args.posts * prob)
        print(f"  {topic.value:<15}: {prob*100:>5.1f}% (~{count} posts)")
    print()

    # Generate dataset
    print("Generating dataset...")
    generator = SyntheticTwitterGenerator(seed=args.seed)
    dataset = generator.generate(
        num_users=args.users,
        num_posts=args.posts,
        num_engagements=args.engagements,
    )

    print()
    print("Generated dataset:")
    print(f"  Users: {dataset.num_users}")
    print(f"  Authors: {dataset.num_authors}")
    print(f"  Posts: {dataset.num_posts}")
    print(f"  Engagements: {dataset.num_engagements}")
    print()

    # Show actual distributions
    print("Actual archetype distribution:")
    for arch in UserArchetype:
        count = len(dataset.get_users_by_archetype(arch))
        pct = 100.0 * count / dataset.num_users
        print(f"  {arch.value:<15}: {count:>5} ({pct:>5.1f}%)")
    print()

    print("Actual topic distribution:")
    for topic in ContentTopic:
        count = len(dataset.get_posts_by_topic(topic))
        pct = 100.0 * count / dataset.num_posts
        print(f"  {topic.value:<15}: {count:>5} ({pct:>5.1f}%)")
    print()

    # Create train/val/test splits
    print("Creating train/val/test splits...")
    train, val, test = create_train_val_test_split(
        dataset,
        train_ratio=0.8,
        val_ratio=0.1,
    )
    print(f"  Train: {len(train)} engagements")
    print(f"  Val: {len(val)} engagements")
    print(f"  Test: {len(test)} engagements")
    print()

    # Save dataset
    import os
    import pickle

    os.makedirs(args.output, exist_ok=True)

    dataset_path = os.path.join(args.output, "dataset.pkl")
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved dataset to: {dataset_path}")

    splits_path = os.path.join(args.output, "splits.pkl")
    with open(splits_path, "wb") as f:
        pickle.dump({"train": train, "val": val, "test": test}, f)
    print(f"Saved splits to: {splits_path}")

    # Show sample engagements
    print()
    print("Sample engagements (first 5):")
    for eng in dataset.engagements[:5]:
        user = dataset.get_user(eng.user_id)
        post = dataset.get_post(eng.post_id)
        author = dataset.get_author(post.author_id) if post else None

        actions_taken = [k.replace("_score", "") for k, v in eng.actions.items() if v > 0]
        print(f"  User {eng.user_id} ({user.archetype.value}) -> "
              f"Post {eng.post_id} ({post.topic.value if post else '?'}) by Author {post.author_id if post else '?'}")
        print(f"    Actions: {', '.join(actions_taken) or 'none'}")
    print()

    print("=" * 70)
    print("Generation complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
