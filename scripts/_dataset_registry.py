"""Dataset registry: single source of truth for dataset_name → modules.

Scripts that need to work on multiple datasets should use this instead
of hardcoded loader paths. The registry preserves the importlib.util
loading pattern (NOT package imports) to avoid the
enhancements/data/__init__.py Phoenix/grok import chain.

DO NOT convert this file to use ``from enhancements.data.movielens
import MovieLensDataset``. That triggers ``enhancements/data/__init__.py``
which eagerly imports modules depending on vendored Phoenix/grok and
breaks the entire script. See CLAUDE.md "Gotchas" for details.

Usage:
    from _dataset_registry import load_dataset, REGISTRY

    ds = load_dataset(args.dataset)
    configs, pool, topics = ds.configs, ds.pool, ds.topics
    base_probs = pool[np.newaxis, :, :]
    print(f"Loaded {ds.spec.name}: {ds.dataset.num_items} items, "
          f"{ds.spec.feature_dim} features")
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).parent.parent  # x-algorithm root


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata for one dataset family.

    Each refactored experiment script reads from a DatasetSpec rather
    than hardcoding stakeholder names, file paths, or item attribute
    conventions.
    """
    name: str
    data_dir: Path
    feature_dim: int
    stakeholder_family: str  # "movielens" | "mind" | "amazon"

    # Module loading
    loader_module: str
    loader_path: Path
    loader_class: str
    stakeholders_module: str
    stakeholders_path: Path

    # Function names (resolved via getattr at runtime)
    content_pool_fn: str
    preferences_fn: str

    # Stakeholder canonical ordering — load-bearing for byte-equivalence
    # of MovieLens regression tests. Scripts that iterate over
    # stakeholders for RNG-consuming operations MUST use this order,
    # not sorted(configs.keys()), or seed-state diverges.
    primary_stakeholder_order: tuple[str, ...]

    # Special-role stakeholders (named for cross-dataset clarity)
    scorer_stakeholder: str    # "platform"-equivalent (default scorer for LOSO experiments)
    diversity_stakeholder: str  # "diversity"-equivalent (used for {name}_utility field lookups)

    # Dataclass field name discrepancies between MovieLens and the
    # MIND/Amazon loaders. Adapter helpers in LoadedDataset hide these.
    item_attr_name: str   # "movies" (ML) | "items" (MIND/Amazon)
    rating_id_attr: str   # "movie_id" (ML) | "item_id" (MIND/Amazon)
    rating_type: str      # "ordinal_1_5" | "binary_click" | "ordinal_1_5_float"


REGISTRY: dict[str, DatasetSpec] = {
    "ml-100k": DatasetSpec(
        name="ml-100k",
        data_dir=ROOT / "data" / "ml-100k",
        feature_dim=19,
        stakeholder_family="movielens",
        loader_module="movielens",
        loader_path=ROOT / "enhancements" / "data" / "movielens.py",
        loader_class="MovieLensDataset",
        stakeholders_module="movielens_stakeholders",
        stakeholders_path=ROOT / "enhancements" / "data" / "movielens_stakeholders.py",
        content_pool_fn="generate_movielens_content_pool",
        preferences_fn="generate_movielens_preferences",
        primary_stakeholder_order=("user", "platform", "diversity"),
        scorer_stakeholder="platform",
        diversity_stakeholder="diversity",
        item_attr_name="movies",
        rating_id_attr="movie_id",
        rating_type="ordinal_1_5",
    ),
    "ml-1m": DatasetSpec(
        name="ml-1m",
        data_dir=ROOT / "data" / "ml-1m",
        feature_dim=19,
        stakeholder_family="movielens",
        loader_module="movielens",
        loader_path=ROOT / "enhancements" / "data" / "movielens.py",
        loader_class="MovieLensDataset",
        stakeholders_module="movielens_stakeholders",
        stakeholders_path=ROOT / "enhancements" / "data" / "movielens_stakeholders.py",
        content_pool_fn="generate_movielens_content_pool",
        preferences_fn="generate_movielens_preferences",
        primary_stakeholder_order=("user", "platform", "diversity"),
        scorer_stakeholder="platform",
        diversity_stakeholder="diversity",
        item_attr_name="movies",
        rating_id_attr="movie_id",
        rating_type="ordinal_1_5",
    ),
    "mind-small": DatasetSpec(
        name="mind-small",
        data_dir=ROOT / "data" / "mind-small",
        feature_dim=35,
        stakeholder_family="mind",
        loader_module="mind",
        loader_path=ROOT / "enhancements" / "data" / "mind.py",
        loader_class="MINDDataset",
        stakeholders_module="mind_stakeholders",
        stakeholders_path=ROOT / "enhancements" / "data" / "mind_stakeholders.py",
        content_pool_fn="generate_mind_content_pool",
        preferences_fn="generate_mind_preferences",
        # Domain-native ordering — see docs/f2/tier2_phase_a_retro.md.
        primary_stakeholder_order=(
            "reader", "publisher", "advertiser", "journalist", "civic_diversity",
        ),
        scorer_stakeholder="publisher",
        diversity_stakeholder="civic_diversity",
        item_attr_name="items",
        rating_id_attr="item_id",
        rating_type="binary_click",
    ),
    "amazon-kindle": DatasetSpec(
        name="amazon-kindle",
        data_dir=ROOT / "data" / "amazon-kindle",
        feature_dim=32,
        stakeholder_family="amazon",
        loader_module="amazon",
        loader_path=ROOT / "enhancements" / "data" / "amazon.py",
        loader_class="AmazonDataset",
        stakeholders_module="amazon_stakeholders",
        stakeholders_path=ROOT / "enhancements" / "data" / "amazon_stakeholders.py",
        content_pool_fn="generate_amazon_content_pool",
        preferences_fn="generate_amazon_preferences",
        primary_stakeholder_order=(
            "reader", "publisher", "indie_author", "premium_seller", "diversity",
        ),
        scorer_stakeholder="publisher",
        diversity_stakeholder="diversity",
        item_attr_name="items",
        rating_id_attr="item_id",
        rating_type="ordinal_1_5_float",
    ),
}


def _load(name: str, path: Path) -> ModuleType:
    """Load a Python module via importlib, registering in sys.modules first.

    Registration BEFORE exec_module is required for Python 3.11 dataclass
    type introspection (PEP-604 unions like ``int | None`` need
    ``cls.__module__`` resolvable in sys.modules during dataclass setup).
    Without this, dataclasses with union types fail with
    ``AttributeError: 'NoneType' object has no attribute '__dict__'``.

    See lesson 11 in tasks/lessons.md and the equivalent _load helpers
    in existing experiment scripts.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class LoadedDataset:
    """Bundle returned by ``load_dataset()`` — duck-types across families.

    Attributes:
        spec: Static metadata for this dataset.
        dataset: The instantiated loader (MovieLensDataset / MINDDataset / AmazonDataset).
        loader_mod: The loader module object (for accessing dataclasses, etc.).
        stakeholders_mod: The stakeholders module object (for `build_named_*`, etc.).
        configs: Dict of stakeholder name → weight vector.
        pool: Content pool features, shape ``[M, D]``.
        topics: Per-item primary topic index, shape ``[M]``.

    Adapter methods hide MovieLens-vs-MIND/Amazon attribute name discrepancies
    (e.g. ``.movies`` vs ``.items``, ``.movie_id`` vs ``.item_id``).
    """
    spec: DatasetSpec
    dataset: Any
    loader_mod: ModuleType
    stakeholders_mod: ModuleType
    configs: dict[str, np.ndarray] = field(default_factory=dict)
    pool: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    topics: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))

    @property
    def stakeholder_names(self) -> list[str]:
        """List of stakeholder names. NOT sorted — preserves canonical order
        from primary_stakeholder_order if configs were built that way."""
        return list(self.configs.keys())

    def rating_item_id(self, rating: Any) -> int:
        """Get the item id from a Rating dataclass.

        Hides the movie_id (MovieLens) vs item_id (MIND/Amazon) discrepancy.
        """
        return getattr(rating, self.spec.rating_id_attr)

    def item_dict(self) -> dict:
        """Get the item dict (movies for MovieLens, items for MIND/Amazon)."""
        return getattr(self.dataset, self.spec.item_attr_name)

    def generate_preferences(
        self,
        weights: np.ndarray,
        n_pairs: int,
        seed: int,
        noise_std: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate (preferred, rejected) pair tensors for one stakeholder.

        Resolves to the dataset-specific generate_*_preferences function
        via the spec.
        """
        fn = getattr(self.stakeholders_mod, self.spec.preferences_fn)
        return fn(self.pool, weights, n_pairs, seed, noise_std=noise_std)


def load_dataset(
    name: str,
    min_ratings: int = 5,
    seed: int = 42,
) -> LoadedDataset:
    """One-call dataset load: import modules, build dataset, content pool, configs.

    Args:
        name: Dataset name from REGISTRY.keys()
            (currently: "ml-100k", "ml-1m", "mind-small", "amazon-kindle")
        min_ratings: Minimum ratings/impressions per item for content pool inclusion.
        seed: Random seed for content pool ordering.

    Returns:
        LoadedDataset with spec, dataset, configs, pool, topics populated.

    Raises:
        ValueError: if name is not in REGISTRY.
        FileNotFoundError: if the dataset's data directory doesn't exist.
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {sorted(REGISTRY.keys())}"
        )
    spec = REGISTRY[name]

    if not spec.data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found for {name}: {spec.data_dir}. "
            f"Run scripts/data/download_{spec.stakeholder_family}.py first."
        )

    loader_mod = _load(spec.loader_module, spec.loader_path)
    stakeholders_mod = _load(spec.stakeholders_module, spec.stakeholders_path)

    DatasetClass = getattr(loader_mod, spec.loader_class)
    dataset = DatasetClass(str(spec.data_dir))

    raw_configs = stakeholders_mod.build_stakeholder_configs(dataset)
    # Re-order configs into canonical primary_stakeholder_order to ensure
    # downstream iterators see them in a deterministic order regardless
    # of how the stakeholder module returned them.
    configs: dict[str, np.ndarray] = {}
    for s in spec.primary_stakeholder_order:
        if s in raw_configs:
            configs[s] = raw_configs[s]
    # Append any extra stakeholders that aren't in the canonical order
    # (defensive — shouldn't happen unless a stakeholder module is updated
    # without updating the registry).
    for s, w in raw_configs.items():
        if s not in configs:
            configs[s] = w

    content_pool_fn = getattr(stakeholders_mod, spec.content_pool_fn)
    # Use positional args because the second-arg name varies by family:
    # MovieLens/Amazon use ``min_ratings``, MIND uses ``min_impressions``.
    # The semantics are identical (minimum interactions per item).
    pool, topics = content_pool_fn(dataset, min_ratings, seed)

    return LoadedDataset(
        spec=spec,
        dataset=dataset,
        loader_mod=loader_mod,
        stakeholders_mod=stakeholders_mod,
        configs=configs,
        pool=pool,
        topics=topics,
    )
