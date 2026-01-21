"""F2 Phase 1: JIT Optimization Utilities for Phoenix Inference.

This module provides static-shape JIT compilation wrappers for Phoenix,
enabling faster inference through ahead-of-time compilation with known shapes.

Key concepts:
- Static shapes: JAX JIT recompiles when shapes change; fixing shapes avoids this
- Padding: Input batches are padded to match compiled shapes
- Warmup: First call triggers compilation; subsequent calls use cached kernel
"""

import time
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from phoenix.recsys_model import RecsysBatch, RecsysEmbeddings
from phoenix.runners import RankingOutput, RecsysInferenceRunner


# Register RecsysEmbeddings as a JAX pytree (it's a dataclass, not NamedTuple)
def _recsys_embeddings_flatten(emb: RecsysEmbeddings):
    """Flatten RecsysEmbeddings for JAX pytree."""
    children = (
        emb.user_embeddings,
        emb.history_post_embeddings,
        emb.candidate_post_embeddings,
        emb.history_author_embeddings,
        emb.candidate_author_embeddings,
    )
    aux_data = None
    return children, aux_data


def _recsys_embeddings_unflatten(aux_data, children):
    """Unflatten RecsysEmbeddings from JAX pytree."""
    return RecsysEmbeddings(
        user_embeddings=children[0],
        history_post_embeddings=children[1],
        candidate_post_embeddings=children[2],
        history_author_embeddings=children[3],
        candidate_author_embeddings=children[4],
    )


# Register the pytree
jax.tree_util.register_pytree_node(
    RecsysEmbeddings,
    _recsys_embeddings_flatten,
    _recsys_embeddings_unflatten,
)


@dataclass
class StaticShapeConfig:
    """Configuration for static-shape JIT compilation."""

    batch_size: int = 4
    history_len: int = 32
    num_candidates: int = 8


class JITStats(NamedTuple):
    """Statistics from JIT compilation and execution."""

    compilation_time_ms: float
    first_run_time_ms: float
    warmup_avg_time_ms: float


def pad_to_shape(arr: np.ndarray, target_shape: tuple, pad_value: int = 0) -> np.ndarray:
    """Pad array to target shape along all dimensions.

    Args:
        arr: Input array
        target_shape: Desired shape
        pad_value: Value to use for padding

    Returns:
        Padded array with target_shape
    """
    if arr.shape == target_shape:
        return arr

    pad_widths = []
    for current, target in zip(arr.shape, target_shape):
        if current > target:
            raise ValueError(f"Cannot pad: current dim {current} > target {target}")
        pad_widths.append((0, target - current))

    return np.pad(arr, pad_widths, mode="constant", constant_values=pad_value)


def pad_batch_to_static(
    batch: RecsysBatch,
    config: StaticShapeConfig,
) -> RecsysBatch:
    """Pad a RecsysBatch to static shapes for JIT compilation.

    Args:
        batch: Input batch (may have variable shapes)
        config: Static shape configuration

    Returns:
        Padded batch with shapes matching config
    """
    bs = config.batch_size
    hist_len = config.history_len
    num_cand = config.num_candidates

    # Get hash dimensions from current batch
    num_user_hashes = batch.user_hashes.shape[-1]
    num_item_hashes = batch.history_post_hashes.shape[-1]
    num_author_hashes = batch.history_author_hashes.shape[-1]
    num_actions = batch.history_actions.shape[-1]

    return RecsysBatch(
        user_hashes=pad_to_shape(
            np.asarray(batch.user_hashes), (bs, num_user_hashes)
        ),
        history_post_hashes=pad_to_shape(
            np.asarray(batch.history_post_hashes), (bs, hist_len, num_item_hashes)
        ),
        history_author_hashes=pad_to_shape(
            np.asarray(batch.history_author_hashes), (bs, hist_len, num_author_hashes)
        ),
        history_actions=pad_to_shape(
            np.asarray(batch.history_actions), (bs, hist_len, num_actions)
        ),
        history_product_surface=pad_to_shape(
            np.asarray(batch.history_product_surface), (bs, hist_len)
        ),
        candidate_post_hashes=pad_to_shape(
            np.asarray(batch.candidate_post_hashes), (bs, num_cand, num_item_hashes)
        ),
        candidate_author_hashes=pad_to_shape(
            np.asarray(batch.candidate_author_hashes), (bs, num_cand, num_author_hashes)
        ),
        candidate_product_surface=pad_to_shape(
            np.asarray(batch.candidate_product_surface), (bs, num_cand)
        ),
    )


def pad_embeddings_to_static(
    embeddings: RecsysEmbeddings,
    config: StaticShapeConfig,
) -> RecsysEmbeddings:
    """Pad RecsysEmbeddings to static shapes for JIT compilation.

    Args:
        embeddings: Input embeddings (may have variable shapes)
        config: Static shape configuration

    Returns:
        Padded embeddings with shapes matching config
    """
    bs = config.batch_size
    hist_len = config.history_len
    num_cand = config.num_candidates

    # Get dimensions from current embeddings
    num_user_hashes = embeddings.user_embeddings.shape[1]
    num_item_hashes = embeddings.history_post_embeddings.shape[2]
    num_author_hashes = embeddings.history_author_embeddings.shape[2]
    emb_size = embeddings.user_embeddings.shape[-1]

    return RecsysEmbeddings(
        user_embeddings=pad_to_shape(
            np.asarray(embeddings.user_embeddings),
            (bs, num_user_hashes, emb_size),
        ),
        history_post_embeddings=pad_to_shape(
            np.asarray(embeddings.history_post_embeddings),
            (bs, hist_len, num_item_hashes, emb_size),
        ),
        candidate_post_embeddings=pad_to_shape(
            np.asarray(embeddings.candidate_post_embeddings),
            (bs, num_cand, num_item_hashes, emb_size),
        ),
        history_author_embeddings=pad_to_shape(
            np.asarray(embeddings.history_author_embeddings),
            (bs, hist_len, num_author_hashes, emb_size),
        ),
        candidate_author_embeddings=pad_to_shape(
            np.asarray(embeddings.candidate_author_embeddings),
            (bs, num_cand, num_author_hashes, emb_size),
        ),
    )


def create_static_rank_fn(
    runner: RecsysInferenceRunner,
    config: StaticShapeConfig,
    warmup_iterations: int = 3,
) -> tuple[Callable, JITStats]:
    """Create a JIT-compiled ranking function with static shapes.

    This function:
    1. Creates a JIT-compiled version of the rank_candidates function
    2. Warms up the compilation with a dummy batch
    3. Returns the compiled function and timing statistics

    Args:
        runner: Initialized RecsysInferenceRunner
        config: Static shape configuration
        warmup_iterations: Number of warmup iterations after compilation

    Returns:
        Tuple of (jit_rank_fn, JITStats)
    """
    # Create the JIT-compiled function
    # We unpack the RecsysEmbeddings dataclass to individual arrays for JAX
    @jax.jit
    def jit_rank_inner(
        params,
        batch: RecsysBatch,
        user_emb,
        hist_post_emb,
        cand_post_emb,
        hist_author_emb,
        cand_author_emb,
    ):
        embeddings = RecsysEmbeddings(
            user_embeddings=user_emb,
            history_post_embeddings=hist_post_emb,
            candidate_post_embeddings=cand_post_emb,
            history_author_embeddings=hist_author_emb,
            candidate_author_embeddings=cand_author_emb,
        )
        return runner.rank_candidates(params, batch, embeddings)

    def jit_rank(params, batch: RecsysBatch, embeddings: RecsysEmbeddings):
        """Wrapper that unpacks embeddings for the inner JIT function."""
        return jit_rank_inner(
            params,
            batch,
            embeddings.user_embeddings,
            embeddings.history_post_embeddings,
            embeddings.candidate_post_embeddings,
            embeddings.history_author_embeddings,
            embeddings.candidate_author_embeddings,
        )

    # Create dummy data with static shapes for warmup
    from phoenix.runners import create_example_batch

    model_config = runner.runner.model
    dummy_batch, dummy_embeddings = create_example_batch(
        batch_size=config.batch_size,
        emb_size=model_config.emb_size,
        history_len=config.history_len,
        num_candidates=config.num_candidates,
        num_actions=model_config.num_actions,
        num_user_hashes=model_config.hash_config.num_user_hashes,
        num_item_hashes=model_config.hash_config.num_item_hashes,
        num_author_hashes=model_config.hash_config.num_author_hashes,
        product_surface_vocab_size=model_config.product_surface_vocab_size,
    )

    # Clear caches to measure fresh compilation
    jax.clear_caches()

    # Measure compilation time (first call)
    compile_start = time.perf_counter()
    output = jit_rank(runner.params, dummy_batch, dummy_embeddings)
    jax.block_until_ready(output)
    compilation_time_ms = (time.perf_counter() - compile_start) * 1000

    # Measure first run after compilation
    first_run_start = time.perf_counter()
    output = jit_rank(runner.params, dummy_batch, dummy_embeddings)
    jax.block_until_ready(output)
    first_run_time_ms = (time.perf_counter() - first_run_start) * 1000

    # Warmup iterations
    warmup_times = []
    for _ in range(warmup_iterations):
        start = time.perf_counter()
        output = jit_rank(runner.params, dummy_batch, dummy_embeddings)
        jax.block_until_ready(output)
        warmup_times.append((time.perf_counter() - start) * 1000)

    warmup_avg_time_ms = float(np.mean(warmup_times)) if warmup_times else first_run_time_ms

    stats = JITStats(
        compilation_time_ms=compilation_time_ms,
        first_run_time_ms=first_run_time_ms,
        warmup_avg_time_ms=warmup_avg_time_ms,
    )

    return jit_rank, stats


class StaticJITRunner:
    """Phoenix runner wrapper with static-shape JIT optimization.

    This class wraps a RecsysInferenceRunner and provides JIT-compiled
    ranking with automatic padding to static shapes.

    Example:
        >>> runner = RecsysInferenceRunner(...)
        >>> runner.initialize()
        >>> jit_runner = StaticJITRunner(runner, StaticShapeConfig(batch_size=4))
        >>> output = jit_runner.rank(batch, embeddings)
    """

    def __init__(
        self,
        base_runner: RecsysInferenceRunner,
        config: Optional[StaticShapeConfig] = None,
        warmup_iterations: int = 3,
    ):
        """Initialize the static JIT runner.

        Args:
            base_runner: Initialized RecsysInferenceRunner
            config: Static shape config (uses defaults if None)
            warmup_iterations: Number of warmup iterations
        """
        self.base_runner = base_runner
        self.config = config or StaticShapeConfig()
        self._jit_fn: Optional[Callable] = None
        self._stats: Optional[JITStats] = None
        self._warmup_iterations = warmup_iterations

    def _ensure_compiled(self) -> None:
        """Ensure the JIT function is compiled."""
        if self._jit_fn is None:
            self._jit_fn, self._stats = create_static_rank_fn(
                self.base_runner,
                self.config,
                self._warmup_iterations,
            )

    @property
    def stats(self) -> Optional[JITStats]:
        """Get JIT compilation statistics."""
        return self._stats

    @property
    def params(self):
        """Get model parameters."""
        return self.base_runner.params

    def rank(
        self,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings,
    ) -> RankingOutput:
        """Rank candidates using JIT-compiled inference.

        The batch and embeddings will be padded to match the static shape
        configuration. Results are unpadded before returning.

        Args:
            batch: Input batch
            embeddings: Input embeddings

        Returns:
            RankingOutput with scores and rankings
        """
        self._ensure_compiled()

        # Get original batch size for unpadding
        original_batch_size = batch.user_hashes.shape[0]
        original_num_candidates = batch.candidate_post_hashes.shape[1]

        # Pad to static shapes
        padded_batch = pad_batch_to_static(batch, self.config)
        padded_embeddings = pad_embeddings_to_static(embeddings, self.config)

        # Run JIT-compiled inference
        output = self._jit_fn(self.base_runner.params, padded_batch, padded_embeddings)

        # Unpad results - slice back to original dimensions
        scores = output.scores[:original_batch_size, :original_num_candidates, :]
        ranked_indices = output.ranked_indices[:original_batch_size, :original_num_candidates]

        # Build unpadded RankingOutput
        return RankingOutput(
            scores=scores,
            ranked_indices=ranked_indices,
            p_favorite_score=scores[:, :, 0],
            p_reply_score=scores[:, :, 1],
            p_repost_score=scores[:, :, 2],
            p_photo_expand_score=scores[:, :, 3],
            p_click_score=scores[:, :, 4],
            p_profile_click_score=scores[:, :, 5],
            p_vqv_score=scores[:, :, 6],
            p_share_score=scores[:, :, 7],
            p_share_via_dm_score=scores[:, :, 8],
            p_share_via_copy_link_score=scores[:, :, 9],
            p_dwell_score=scores[:, :, 10],
            p_quote_score=scores[:, :, 11],
            p_quoted_click_score=scores[:, :, 12],
            p_follow_author_score=scores[:, :, 13],
            p_not_interested_score=scores[:, :, 14],
            p_block_author_score=scores[:, :, 15],
            p_mute_author_score=scores[:, :, 16],
            p_report_score=scores[:, :, 17],
            p_dwell_time=scores[:, :, 18],
        )

    def rank_no_unpad(
        self,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings,
    ) -> RankingOutput:
        """Rank with static shapes, no unpadding (for benchmarking).

        Use this when you want to measure pure JIT performance without
        the overhead of padding/unpadding.

        Args:
            batch: Input batch (must match static shape config)
            embeddings: Input embeddings (must match static shape config)

        Returns:
            RankingOutput (padded size)
        """
        self._ensure_compiled()
        return self._jit_fn(self.base_runner.params, batch, embeddings)
