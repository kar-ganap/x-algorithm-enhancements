# xAI Phoenix Enhancement Project - Design Document

## Overview

This document outlines the design and implementation plan for three features built on top of xAI's open-sourced Phoenix recommendation system. These features demonstrate skills relevant to xAI's ML engineering roles.

**Repository:** [xai-org/x-algorithm](https://github.com/xai-org/x-algorithm)

**Target Roles:**
- Member of Technical Staff, Search and Retrieval
- Member of Technical Staff, JAX & Compiler
- Member of Technical Staff, Inference / Applied Inference
- Member of Technical Staff, RL Infrastructure
- Member of Technical Staff, Model Evaluation

---

## Codebase Assessment

### What's Available (Fully Functional)

| Component | Location | Status |
|-----------|----------|--------|
| Grok Transformer | `phoenix/grok.py` | Complete - Ported from Grok-1 |
| Ranking Model | `phoenix/recsys_model.py` | Complete - Multi-action prediction |
| Retrieval Model | `phoenix/recsys_retrieval_model.py` | Complete - Two-tower architecture |
| Inference Runners | `phoenix/runners.py` | Complete - With synthetic data generation |
| Demo Scripts | `phoenix/run_*.py` | Runnable |
| Test Suite | `phoenix/test_*.py` | Comprehensive |

### What's NOT Available (Rust - Reference Only)

- Home Mixer orchestration (missing 15+ internal crates)
- Thunder post store (missing 8+ internal crates)
- Candidate Pipeline framework (no Cargo.toml)
- Scoring weights (excluded from release)

### Dependencies (All Public)

```toml
jax = "0.8.1"
dm-haiku = ">=0.0.13"
numpy = ">=1.26.4"
```

---

## Feature 1: JAX Optimization (F2)

### Objective

Optimize the Phoenix transformer for faster inference while maintaining accuracy. Target: **3-5x latency reduction**.

### Background

The current Phoenix model (`phoenix/grok.py`) uses standard JAX/Haiku patterns without inference-specific optimizations. Key bottlenecks:

1. **Attention computation** - O(n²) memory, no fusion
2. **No JIT with static shapes** - Dynamic recompilation overhead
3. **No quantization** - Full bfloat16 precision
4. **No KV-cache** - Recomputes user context every request

### Key Insight: Candidate Isolation Enables KV-Caching

The Phoenix attention mask has a unique property (from `grok.py:97-134`):

```
Attention Mask Structure:
                    User+History    Candidates
                    ┌───────────┬─────────────┐
User+History        │  Causal   │      0      │
                    ├───────────┼─────────────┤
Candidates          │  Can see  │  Only self  │
                    └───────────┴─────────────┘
```

**Implication:** Candidates only attend to user context + themselves, never to each other. This means:
- User context KV can be cached across requests for the same user
- Each candidate's computation is independent
- Massive parallelization opportunity

### Implementation Plan

#### Phase 1: Baseline & Profiling (Day 1)

**Goal:** Establish baseline metrics and identify bottlenecks.

```
phoenix/
├── optimization/
│   ├── __init__.py
│   ├── benchmark.py          # Benchmarking harness
│   ├── profiler.py           # JAX profiling utilities
│   └── baseline_metrics.json # Recorded baseline
```

**Tasks:**
1. Create synthetic batch generator with configurable sizes
2. Implement latency benchmark (p50, p95, p99)
3. Implement throughput benchmark (batches/sec)
4. Implement memory profiling
5. Run JAX profiler to identify hot paths
6. Document baseline metrics

**Deliverable:** Baseline report with:
- Latency distribution across batch sizes
- Memory usage breakdown
- Profiler flamegraph showing time distribution

#### Phase 2: JIT Compilation Strategy (Day 2)

**Goal:** Optimize compilation with static shapes.

**Current State:**
```python
# No explicit JIT - relies on Haiku's transform
def forward(batch, embeddings):
    return model(batch, embeddings)
```

**Target State:**
```python
@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def forward_optimized(params, batch, embeddings, seq_len, num_candidates, num_actions):
    """JIT-compiled forward with static shape hints."""
    ...
```

**Tasks:**
1. Identify all dynamic dimensions in the model
2. Create shape-specialized forward functions for common configurations
3. Implement batch padding for shape consistency
4. Benchmark compilation time vs runtime tradeoff
5. Document optimal batch size configurations

**Expected Improvement:** 1.5-2x from avoiding recompilation

#### Phase 3: KV-Cache Implementation (Days 3-4)

**Goal:** Cache user context KV-pairs across candidate batches.

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    KV-CACHE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Request 1: User A + [Candidates 1-32]                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Compute user context (seq 0 to history_len)       │   │
│  │ 2. Cache KV for user context                         │   │
│  │ 3. Compute candidate scores using cached KV          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Request 2: User A + [Candidates 33-64] (SAME USER)         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. Retrieve cached KV for user context ← SKIP COMPUTE│   │
│  │ 2. Compute candidate scores using cached KV          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Savings: ~50% compute for subsequent candidate batches     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Implementation:**

```python
# phoenix/optimization/kv_cache.py

from typing import NamedTuple
import jax.numpy as jnp

class KVCache(NamedTuple):
    """Cached key-value pairs for transformer layers."""
    keys: jax.Array    # [num_layers, batch, num_kv_heads, seq_len, head_dim]
    values: jax.Array  # [num_layers, batch, num_kv_heads, seq_len, head_dim]
    seq_len: int       # Current sequence length in cache


class CachedPhoenixModel:
    """Phoenix model with KV-caching for user context."""

    def __init__(self, config: PhoenixModelConfig):
        self.config = config
        self.cache = None

    def encode_user_context(
        self,
        params,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings
    ) -> KVCache:
        """Encode user + history and return KV cache."""
        # Run transformer up to candidate_start_offset
        # Extract and return KV pairs
        ...

    def score_candidates_with_cache(
        self,
        params,
        cache: KVCache,
        candidate_embeddings: jax.Array,
    ) -> jax.Array:
        """Score candidates using cached user context KV."""
        # Candidates attend to cached KV + self
        ...
```

**Tasks:**
1. Modify `Transformer.__call__` to optionally return KV pairs
2. Implement `KVCache` data structure
3. Implement cache-aware attention that uses pre-computed KV
4. Add cache invalidation logic (user context changes)
5. Benchmark cache hit rate and latency improvement

**Expected Improvement:** 1.5-2x for multi-batch candidate scoring

#### Phase 4: Attention Optimization (Days 5-6)

**Goal:** Implement memory-efficient attention.

**Current State** (`grok.py:336-354`):
```python
# Standard attention - O(n²) memory
attn_logits = jnp.einsum("...thHd,...Thd->...hHtT", query_heads, key_heads)
attn_logits = attn_logits / scale
attn_logits = jnp.where(mask, attn_logits, -1e30)
attn_weights = jax.nn.softmax(attn_logits)
attn = jnp.einsum("...hHtT,...Thd->...thHd", attn_weights, value_heads)
```

**Target:** Flash-style attention that:
1. Tiles computation to fit in SRAM
2. Never materializes full attention matrix
3. Computes softmax incrementally

**Implementation Options:**

| Option | Complexity | Expected Speedup | Memory Reduction |
|--------|------------|------------------|------------------|
| JAX `dot_product_attention` | Low | 1.5x | 2x |
| Custom tiled attention | High | 2-3x | 4x+ |
| Pallas kernel (if available) | Very High | 3-5x | 4x+ |

**Recommended Approach:** Start with JAX's built-in, then custom if needed.

```python
# Option 1: Use JAX's optimized attention (if available in jax 0.8.1)
from jax.nn import dot_product_attention

# Option 2: Custom implementation
def flash_attention_forward(q, k, v, mask, block_size=64):
    """Memory-efficient attention via tiling."""
    # Implementation based on Flash Attention paper
    ...
```

**Tasks:**
1. Research JAX 0.8.1 attention optimizations
2. Implement tiled attention if needed
3. Handle the custom candidate isolation mask
4. Benchmark memory usage and latency
5. Verify numerical equivalence with baseline

**Expected Improvement:** 2-3x latency, 4x memory reduction

#### Phase 5: Quantization (Day 7)

**Goal:** Reduce precision while maintaining accuracy.

**Current State:**
```python
fprop_dtype: Any = jnp.bfloat16  # Already using bfloat16
```

**Target:** int8 weights, bfloat16 activations

**Implementation:**
```python
# phoenix/optimization/quantization.py

def quantize_weights(params, bits=8):
    """Quantize model weights to int8."""
    def quantize_leaf(x):
        if x.dtype == jnp.bfloat16:
            scale = jnp.max(jnp.abs(x)) / 127.0
            return (x / scale).astype(jnp.int8), scale
        return x, None

    return jax.tree_util.tree_map(quantize_leaf, params)

def dequantize_matmul(x, w_quant, scale):
    """Dequantize-multiply in one fused operation."""
    return jnp.dot(x, w_quant.astype(x.dtype)) * scale
```

**Tasks:**
1. Implement weight quantization (int8)
2. Implement quantized matmul
3. Measure accuracy degradation on synthetic data
4. Benchmark memory and latency improvements
5. Find optimal quantization configuration

**Expected Improvement:** 2x memory, 1.3x latency (memory-bound ops)

### Success Metrics

| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| Latency (p50) | ~50ms | <20ms | <10ms |
| Latency (p99) | ~80ms | <40ms | <20ms |
| Throughput | ~20 batch/s | >60 batch/s | >100 batch/s |
| Memory | ~8GB | <4GB | <2GB |
| Accuracy loss | 0% | <0.1% | 0% |

### Measurement Plan

```python
# phoenix/optimization/benchmark.py

def run_full_benchmark(model, params, config):
    """Run comprehensive benchmark suite."""
    results = {
        'latency': benchmark_latency(model, params, config),
        'throughput': benchmark_throughput(model, params, config),
        'memory': benchmark_memory(model, params, config),
        'accuracy': benchmark_accuracy(model, params, config),
    }
    return results
```

### Deliverables

1. **Code:**
   - `phoenix/optimization/` module with all optimizations
   - Drop-in replacement for inference runner

2. **Benchmarks:**
   - `benchmark_results.json` with all metrics
   - Comparison charts (baseline vs optimized)

3. **Documentation:**
   - Technical writeup explaining each optimization
   - Profile analysis showing where time is saved

---

## Feature 2: RL Reward Modeling (F4)

### Objective

Reframe Phoenix engagement predictions as a reward model for RLHF-style training. Demonstrate preference learning and interpretable reward weights.

### Background

Phoenix predicts 19 engagement actions (`runners.py:202-222`):

```python
ACTIONS = [
    "favorite_score",      # Positive
    "reply_score",         # Positive
    "repost_score",        # Positive
    "photo_expand_score",  # Positive
    "click_score",         # Positive
    "profile_click_score", # Positive
    "vqv_score",           # Video quality view - Positive
    "share_score",         # Positive
    "share_via_dm_score",  # Positive
    "share_via_copy_link_score",  # Positive
    "dwell_score",         # Positive (time spent)
    "quote_score",         # Positive
    "quoted_click_score",  # Positive
    "follow_author_score", # Very Positive
    "not_interested_score",# Negative
    "block_author_score",  # Very Negative
    "mute_author_score",   # Negative
    "report_score",        # Very Negative
    "dwell_time",          # Continuous - Positive
]
```

The weighted scorer (Rust, not included) combines these into a final score:
```
Score = Σ(weight_i × P(action_i))
```

### Key Insight: This IS a Reward Model

```
┌─────────────────────────────────────────────────────────────┐
│           ENGAGEMENT PREDICTION → REWARD MODEL               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Standard Framing:                                          │
│  ────────────────                                           │
│  Phoenix predicts: P(like), P(repost), P(block), ...        │
│  Weighted scorer: Score = 0.5×P(like) + 0.3×P(repost) - ... │
│  Use: Sort posts by score, show top ones                    │
│                                                             │
│  RL Framing:                                                │
│  ───────────                                                │
│  State: User context (history, preferences)                 │
│  Action: Select a post to show                              │
│  Reward: R(s, a) = weighted engagement prediction           │
│                                                             │
│  Phoenix IS a reward model!                                 │
│  The weights define what "good" means.                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Plan

#### Phase 1: Reward Model Wrapper (Day 1)

**Goal:** Create a clean reward model interface around Phoenix.

```
phoenix/
├── reward_modeling/
│   ├── __init__.py
│   ├── reward_model.py       # Reward model wrapper
│   ├── learned_weights.py    # Learnable weight module
│   ├── preference_dataset.py # Preference data handling
│   └── training.py           # DPO/preference training
```

**Implementation:**

```python
# phoenix/reward_modeling/reward_model.py

from typing import NamedTuple
import jax
import jax.numpy as jnp
from runners import ACTIONS

class RewardWeights(NamedTuple):
    """Learnable or fixed weights for combining action probabilities."""
    weights: jax.Array  # [num_actions]

    @classmethod
    def default(cls):
        """Default weights (hand-tuned baseline)."""
        weights = jnp.array([
            1.0,   # favorite
            0.5,   # reply
            0.8,   # repost
            0.2,   # photo_expand
            0.3,   # click
            0.4,   # profile_click
            0.5,   # vqv
            0.6,   # share
            0.4,   # share_via_dm
            0.3,   # share_via_copy_link
            0.3,   # dwell
            0.7,   # quote
            0.2,   # quoted_click
            1.5,   # follow_author (high value)
            -0.5,  # not_interested
            -2.0,  # block_author (very negative)
            -1.0,  # mute_author
            -3.0,  # report (most negative)
            0.1,   # dwell_time
        ])
        return cls(weights=weights)


class PhoenixRewardModel:
    """Reward model built on Phoenix engagement predictions."""

    def __init__(self, phoenix_runner, weights: RewardWeights = None):
        self.phoenix = phoenix_runner
        self.weights = weights or RewardWeights.default()

    def get_action_probs(self, batch, embeddings) -> jax.Array:
        """Get engagement probabilities for all candidates.

        Returns:
            probs: [batch_size, num_candidates, num_actions]
        """
        output = self.phoenix.rank(batch, embeddings)
        return output.scores

    def compute_reward(self, batch, embeddings) -> jax.Array:
        """Compute scalar reward for each candidate.

        Returns:
            rewards: [batch_size, num_candidates]
        """
        probs = self.get_action_probs(batch, embeddings)
        rewards = jnp.einsum('bca,a->bc', probs, self.weights.weights)
        return rewards

    def rank_by_reward(self, batch, embeddings) -> jax.Array:
        """Rank candidates by reward (descending).

        Returns:
            indices: [batch_size, num_candidates] - sorted by reward
        """
        rewards = self.compute_reward(batch, embeddings)
        return jnp.argsort(-rewards, axis=-1)
```

**Tasks:**
1. Implement `RewardWeights` with sensible defaults
2. Implement `PhoenixRewardModel` wrapper
3. Add methods for reward computation and ranking
4. Test with synthetic data
5. Visualize reward breakdown by action type

#### Phase 2: Preference Learning (Days 2-3)

**Goal:** Learn reward weights from preference pairs.

**Preference Data Format:**
```python
PreferencePair = NamedTuple('PreferencePair', [
    ('user_batch', RecsysBatch),      # User context
    ('user_embeddings', RecsysEmbeddings),
    ('preferred_idx', int),            # Index of preferred candidate
    ('rejected_idx', int),             # Index of rejected candidate
])
```

**Bradley-Terry Model:**
```
P(preferred > rejected) = σ(R(preferred) - R(rejected))

Loss = -log(σ(R(preferred) - R(rejected)))
     = -log(σ(Σ w_i × (P_i(preferred) - P_i(rejected))))
```

**Implementation:**

```python
# phoenix/reward_modeling/training.py

import optax

def preference_loss(weights, phoenix_params, phoenix_model, batch):
    """Compute Bradley-Terry preference loss.

    Args:
        weights: RewardWeights to optimize
        phoenix_params: Frozen Phoenix model parameters
        phoenix_model: Phoenix model (frozen)
        batch: Batch of preference pairs

    Returns:
        loss: Scalar loss value
    """
    # Get action probabilities for all candidates
    probs = phoenix_model.apply(phoenix_params, batch.user_batch, batch.user_embeddings)

    # Extract preferred and rejected probs
    preferred_probs = probs[jnp.arange(len(batch)), batch.preferred_idx]  # [B, A]
    rejected_probs = probs[jnp.arange(len(batch)), batch.rejected_idx]    # [B, A]

    # Compute reward difference
    reward_diff = jnp.einsum('ba,a->b', preferred_probs - rejected_probs, weights.weights)

    # Bradley-Terry loss
    loss = -jnp.mean(jax.nn.log_sigmoid(reward_diff))

    return loss


def train_reward_weights(
    phoenix_runner,
    preference_data,
    num_epochs=100,
    learning_rate=0.01,
):
    """Train reward weights using preference data.

    Args:
        phoenix_runner: Initialized Phoenix inference runner
        preference_data: List of PreferencePair
        num_epochs: Training epochs
        learning_rate: Optimizer learning rate

    Returns:
        learned_weights: Optimized RewardWeights
        training_history: Loss history
    """
    # Initialize weights
    weights = RewardWeights.default()

    # Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(weights.weights)

    # Training loop
    history = []
    for epoch in range(num_epochs):
        for batch in preference_data:
            loss, grads = jax.value_and_grad(preference_loss)(
                weights, phoenix_runner.params, phoenix_runner, batch
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            weights = RewardWeights(optax.apply_updates(weights.weights, updates))
            history.append(loss)

    return weights, history
```

**Tasks:**
1. Implement preference loss function
2. Implement training loop with optax
3. Add logging and visualization
4. Test on synthetic preference data
5. Verify weight recovery (can we learn known weights?)

#### Phase 3: Synthetic Preference Generation (Day 4)

**Goal:** Create preference data for training and evaluation.

**Three Approaches:**

```python
# phoenix/reward_modeling/preference_dataset.py

def generate_synthetic_preferences_known_weights(
    phoenix_runner,
    true_weights: RewardWeights,
    num_pairs: int = 1000,
) -> List[PreferencePair]:
    """Generate preferences using known ground-truth weights.

    Used for: Validating that training can recover true weights.
    """
    pairs = []
    for _ in range(num_pairs):
        batch, embeddings = create_example_batch(...)

        # Compute rewards with true weights
        probs = phoenix_runner.rank(batch, embeddings).scores
        true_rewards = jnp.einsum('bca,a->bc', probs, true_weights.weights)

        # Sample preferred = higher reward, rejected = lower reward
        for b in range(batch_size):
            rewards = true_rewards[b]
            preferred = jnp.argmax(rewards)
            rejected = jnp.argmin(rewards)
            pairs.append(PreferencePair(batch, embeddings, preferred, rejected))

    return pairs


def generate_llm_preferences(
    phoenix_runner,
    num_pairs: int = 1000,
    model: str = "gpt-4",
) -> List[PreferencePair]:
    """Generate preferences using LLM-as-judge.

    Used for: Creating realistic preferences without real engagement data.
    """
    # Generate candidate descriptions
    # Ask LLM: "Which post would a typical user prefer?"
    # Parse response into preference pairs
    ...


def load_reddit_preferences(
    subreddit: str = "all",
    min_score_diff: int = 100,
) -> List[PreferencePair]:
    """Load preferences from Reddit upvotes.

    Used for: Real preference signal (higher score = preferred).
    """
    # Load Reddit data
    # Convert to Phoenix batch format
    # Create pairs where higher-scored post is preferred
    ...
```

**Tasks:**
1. Implement synthetic preference generation
2. Implement LLM-as-judge preference generation (optional)
3. Implement Reddit data loader (optional)
4. Create evaluation split (train/val/test)
5. Document data format and generation process

#### Phase 4: Evaluation & Analysis (Day 5)

**Goal:** Evaluate learned weights and analyze interpretability.

**Metrics:**

| Metric | Description | Target |
|--------|-------------|--------|
| Preference Accuracy | % of held-out pairs correctly ranked | >70% |
| Weight Recovery MSE | MSE between learned and true weights (synthetic) | <0.1 |
| Ranking Correlation | Spearman correlation with ground truth ranking | >0.8 |

**Interpretability Analysis:**

```python
def analyze_learned_weights(weights: RewardWeights):
    """Analyze and visualize learned reward weights."""

    # 1. Sign analysis
    positive_actions = [ACTIONS[i] for i, w in enumerate(weights.weights) if w > 0]
    negative_actions = [ACTIONS[i] for i, w in enumerate(weights.weights) if w < 0]

    # 2. Magnitude ranking
    sorted_actions = sorted(
        zip(ACTIONS, weights.weights),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    # 3. Comparison with defaults
    default = RewardWeights.default()
    weight_diff = weights.weights - default.weights

    return {
        'positive_actions': positive_actions,
        'negative_actions': negative_actions,
        'importance_ranking': sorted_actions,
        'deviation_from_default': weight_diff,
    }
```

**Tasks:**
1. Implement evaluation metrics
2. Run weight recovery experiment
3. Analyze learned weights for interpretability
4. Create visualization of weight importance
5. Write analysis report

### Success Metrics

| Metric | Target |
|--------|--------|
| Weight Recovery MSE (synthetic) | < 0.1 |
| Preference Accuracy (synthetic) | > 90% |
| Preference Accuracy (LLM/Reddit) | > 70% |
| Training convergence | < 100 epochs |
| Interpretable weights | Signs match intuition |

### Deliverables

1. **Code:**
   - `phoenix/reward_modeling/` module
   - Training scripts and notebooks

2. **Experiments:**
   - Weight recovery on synthetic data
   - Preference learning on proxy data

3. **Analysis:**
   - Weight interpretability report
   - Comparison with hand-tuned baselines

---

## Feature 3: Multimodal Retrieval (F1)

### Objective

Extend the two-tower retrieval model to support multimodal content (text + images). Enable cross-modal retrieval (text query → image results).

### Background

Current retrieval model (`recsys_retrieval_model.py`) uses hash-based embeddings:

```python
class PhoenixRetrievalModel:
    def build_user_representation(self, batch, embeddings):
        # User tower: transformer over user + history
        # Output: [B, D] user embedding

    def build_candidate_representation(self, batch, embeddings):
        # Candidate tower: MLP over post + author embeddings
        # Output: [B, C, D] candidate embeddings
```

**Limitation:** No image/video understanding - only hash-based IDs.

### Key Insight: Shared Embedding Space

```
┌─────────────────────────────────────────────────────────────┐
│              MULTIMODAL RETRIEVAL ARCHITECTURE               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    Shared Embedding Space                   │
│                    ──────────────────────                   │
│                                                             │
│    User Tower                         Candidate Tower       │
│    ──────────                         ───────────────       │
│  ┌─────────────┐                    ┌─────────────────┐    │
│  │ Text history│──┐                 │   Text embed    │──┐ │
│  └─────────────┘  │                 └─────────────────┘  │ │
│  ┌─────────────┐  │  Project to     ┌─────────────────┐  │ │
│  │Image history│──┼─► shared ──────►│  Image embed    │──┼─┤│
│  └─────────────┘  │  space D        │   (CLIP)        │  │ ││
│  ┌─────────────┐  │                 └─────────────────┘  │ ││
│  │Video history│──┘                 ┌─────────────────┐  │ ││
│  └─────────────┘                    │  Video embed    │──┘ ││
│        │                            └─────────────────┘    ││
│        ▼                                    │              ││
│   user_embed [D]                     candidate_embed [D]   ││
│        │                                    │              ││
│        └──────────── dot product ───────────┘              ││
│                          │                                 ││
│                          ▼                                 ││
│                   similarity score                         ││
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Plan

#### Phase 1: CLIP Integration (Day 1)

**Goal:** Add CLIP embeddings to candidate tower.

```
phoenix/
├── multimodal/
│   ├── __init__.py
│   ├── clip_encoder.py       # CLIP embedding extraction
│   ├── multimodal_tower.py   # Extended candidate tower
│   ├── multimodal_model.py   # Full retrieval model
│   └── data_utils.py         # Multimodal data handling
```

**Implementation:**

```python
# phoenix/multimodal/clip_encoder.py

import jax
import jax.numpy as jnp
from transformers import CLIPProcessor, FlaxCLIPModel

class CLIPEncoder:
    """CLIP encoder for image and text embeddings."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model = FlaxCLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.embed_dim = self.model.config.projection_dim  # 512 for base

    def encode_images(self, images: List[Image]) -> jax.Array:
        """Encode images to CLIP embeddings.

        Args:
            images: List of PIL images

        Returns:
            embeddings: [N, 512] normalized image embeddings
        """
        inputs = self.processor(images=images, return_tensors="np")
        outputs = self.model.get_image_features(**inputs)
        return outputs / jnp.linalg.norm(outputs, axis=-1, keepdims=True)

    def encode_texts(self, texts: List[str]) -> jax.Array:
        """Encode texts to CLIP embeddings.

        Args:
            texts: List of text strings

        Returns:
            embeddings: [N, 512] normalized text embeddings
        """
        inputs = self.processor(text=texts, return_tensors="np", padding=True)
        outputs = self.model.get_text_features(**inputs)
        return outputs / jnp.linalg.norm(outputs, axis=-1, keepdims=True)
```

**Tasks:**
1. Set up CLIP model loading (Hugging Face)
2. Implement image encoding
3. Implement text encoding
4. Test embedding quality
5. Benchmark encoding speed

#### Phase 2: Multimodal Candidate Tower (Day 2)

**Goal:** Extend candidate tower to fuse text + image embeddings.

**Current Candidate Tower** (`recsys_retrieval_model.py:18-42`):
```python
class CandidateTower(hk.Module):
    def __call__(self, post_author_embedding):
        # Mean pool over hash embeddings
        pooled = jnp.mean(post_author_embedding, axis=-2)
        # Normalize
        return pooled / jnp.linalg.norm(pooled, axis=-1, keepdims=True)
```

**Extended Multimodal Tower:**

```python
# phoenix/multimodal/multimodal_tower.py

class MultimodalCandidateTower(hk.Module):
    """Candidate tower with text + image fusion."""

    def __init__(
        self,
        emb_size: int,
        clip_dim: int = 512,
        fusion: str = "concat_project",  # or "add", "attention"
    ):
        super().__init__()
        self.emb_size = emb_size
        self.clip_dim = clip_dim
        self.fusion = fusion

    def __call__(
        self,
        hash_embedding: jax.Array,      # [B, C, num_hashes, D] - original
        image_embedding: jax.Array,     # [B, C, clip_dim] - CLIP image
        text_embedding: jax.Array = None,  # [B, C, clip_dim] - CLIP text (optional)
    ) -> jax.Array:
        """Fuse multimodal embeddings into single representation.

        Returns:
            candidate_embedding: [B, C, emb_size] - fused, normalized
        """
        # Pool hash embeddings (original method)
        hash_pooled = jnp.mean(hash_embedding, axis=-2)  # [B, C, D]

        # Project CLIP embeddings to model dimension
        clip_proj = hk.Linear(self.emb_size, name="clip_proj")
        image_proj = clip_proj(image_embedding)  # [B, C, D]

        if self.fusion == "add":
            # Simple addition
            fused = hash_pooled + image_proj

        elif self.fusion == "concat_project":
            # Concatenate and project
            concat = jnp.concatenate([hash_pooled, image_proj], axis=-1)
            fusion_proj = hk.Linear(self.emb_size, name="fusion_proj")
            fused = fusion_proj(concat)

        elif self.fusion == "attention":
            # Cross-attention fusion
            fused = self._attention_fusion(hash_pooled, image_proj)

        # Normalize
        return fused / jnp.linalg.norm(fused, axis=-1, keepdims=True)

    def _attention_fusion(self, text_emb, image_emb):
        """Fuse using cross-attention."""
        # Stack modalities
        modalities = jnp.stack([text_emb, image_emb], axis=-2)  # [B, C, 2, D]

        # Self-attention over modalities
        attn = hk.MultiHeadAttention(
            num_heads=4,
            key_size=self.emb_size // 4,
            w_init_scale=1.0,
        )
        attended = attn(modalities, modalities, modalities)  # [B, C, 2, D]

        # Pool over modalities
        return jnp.mean(attended, axis=-2)  # [B, C, D]
```

**Tasks:**
1. Implement `MultimodalCandidateTower`
2. Test different fusion strategies
3. Ensure backward compatibility (image_embedding=None → original behavior)
4. Benchmark latency overhead
5. Verify embedding quality

#### Phase 3: Multimodal User Tower (Day 3)

**Goal:** Encode user's multimodal engagement history.

**Challenge:** User history currently only contains hash-based post IDs. Need to:
1. Track which modality each history item is
2. Encode image/video history items with CLIP
3. Fuse into unified user representation

**Implementation:**

```python
# phoenix/multimodal/multimodal_model.py

class MultimodalRecsysBatch(NamedTuple):
    """Extended batch with multimodal embeddings."""
    # Original fields
    user_hashes: jax.Array
    history_post_hashes: jax.Array
    history_author_hashes: jax.Array
    history_actions: jax.Array
    history_product_surface: jax.Array
    candidate_post_hashes: jax.Array
    candidate_author_hashes: jax.Array
    candidate_product_surface: jax.Array

    # New multimodal fields
    history_image_embeddings: jax.Array  # [B, S, clip_dim] - CLIP embeds for image posts
    history_modality_mask: jax.Array     # [B, S] - 0=text, 1=image, 2=video
    candidate_image_embeddings: jax.Array  # [B, C, clip_dim]
    candidate_modality_mask: jax.Array     # [B, C]


class MultimodalUserTower(hk.Module):
    """User tower that encodes multimodal history."""

    def __init__(self, config, clip_dim: int = 512):
        super().__init__()
        self.config = config
        self.clip_dim = clip_dim

    def __call__(
        self,
        batch: MultimodalRecsysBatch,
        embeddings: RecsysEmbeddings,
    ) -> jax.Array:
        """Encode user with multimodal history.

        Returns:
            user_representation: [B, D] - normalized
        """
        # Original text-based history encoding
        text_history = self._encode_text_history(batch, embeddings)

        # Image history encoding
        image_history = self._encode_image_history(batch)

        # Combine with modality-aware attention
        combined = self._fuse_modalities(text_history, image_history, batch.history_modality_mask)

        # Run through transformer (same as original)
        user_rep = self._transformer_encode(combined)

        return user_rep / jnp.linalg.norm(user_rep, axis=-1, keepdims=True)
```

**Tasks:**
1. Define `MultimodalRecsysBatch` data structure
2. Implement multimodal user tower
3. Handle missing modalities gracefully
4. Test with synthetic multimodal data
5. Benchmark memory/latency

#### Phase 4: Cross-Modal Retrieval (Day 4)

**Goal:** Enable text query → image retrieval (and vice versa).

**Implementation:**

```python
# phoenix/multimodal/cross_modal.py

class CrossModalRetriever:
    """Retriever supporting cross-modal queries."""

    def __init__(
        self,
        multimodal_model: MultimodalPhoenixRetrieval,
        clip_encoder: CLIPEncoder,
    ):
        self.model = multimodal_model
        self.clip = clip_encoder

        # Corpus indices (built offline)
        self.text_corpus_embeddings = None
        self.image_corpus_embeddings = None

    def build_corpus(
        self,
        text_posts: List[str],
        image_posts: List[Image],
    ):
        """Build corpus embeddings for retrieval."""
        self.text_corpus_embeddings = self.clip.encode_texts(text_posts)
        self.image_corpus_embeddings = self.clip.encode_images(image_posts)

        # Project to shared space
        # ... (use candidate tower projection)

    def retrieve_text_to_image(
        self,
        text_query: str,
        user_context: MultimodalRecsysBatch,
        top_k: int = 10,
    ) -> List[int]:
        """Retrieve images given a text query.

        Args:
            text_query: Text description of desired content
            user_context: User history for personalization
            top_k: Number of results

        Returns:
            indices: Top-k image indices from corpus
        """
        # Encode query
        query_emb = self.clip.encode_texts([text_query])

        # Personalize with user context (optional)
        if user_context is not None:
            user_emb = self.model.encode_user(user_context)
            query_emb = (query_emb + user_emb) / 2  # Simple fusion

        # Retrieve from image corpus
        scores = jnp.dot(query_emb, self.image_corpus_embeddings.T)
        return jnp.argsort(-scores, axis=-1)[:, :top_k]
```

**Tasks:**
1. Implement cross-modal retriever
2. Support text→image and image→text
3. Add optional user personalization
4. Benchmark retrieval quality (Recall@k)
5. Create demo notebook

#### Phase 5: Evaluation on Standard Benchmarks (Day 5)

**Goal:** Evaluate on MS-COCO or Flickr30k for cross-modal retrieval.

**Benchmark Setup:**

```python
# phoenix/multimodal/evaluation.py

def evaluate_coco_retrieval(
    retriever: CrossModalRetriever,
    coco_data: COCODataset,
    split: str = "val",
) -> Dict[str, float]:
    """Evaluate cross-modal retrieval on MS-COCO.

    Returns:
        metrics: {
            'text_to_image_R@1': ...,
            'text_to_image_R@5': ...,
            'text_to_image_R@10': ...,
            'image_to_text_R@1': ...,
            'image_to_text_R@5': ...,
            'image_to_text_R@10': ...,
        }
    """
    results = {'text_to_image': [], 'image_to_text': []}

    for image, captions in coco_data:
        # Text → Image
        for caption in captions:
            retrieved = retriever.retrieve_text_to_image(caption, top_k=10)
            results['text_to_image'].append({
                'target': image_id,
                'retrieved': retrieved,
            })

        # Image → Text
        retrieved = retriever.retrieve_image_to_text(image, top_k=10)
        results['image_to_text'].append({
            'target': caption_ids,
            'retrieved': retrieved,
        })

    # Compute Recall@k
    metrics = {}
    for direction in ['text_to_image', 'image_to_text']:
        for k in [1, 5, 10]:
            recall = compute_recall_at_k(results[direction], k)
            metrics[f'{direction}_R@{k}'] = recall

    return metrics
```

**Tasks:**
1. Download MS-COCO or Flickr30k
2. Create data loader for evaluation
3. Run retrieval evaluation
4. Compare with CLIP baseline
5. Analyze failure cases

### Success Metrics

| Metric | CLIP Baseline | Target |
|--------|---------------|--------|
| Text→Image R@1 (COCO) | ~30% | >25% |
| Text→Image R@5 (COCO) | ~55% | >50% |
| Text→Image R@10 (COCO) | ~65% | >60% |
| Latency overhead | 0ms | <50ms |

Note: We may not beat CLIP directly, but demonstrating the integration and personalization capability is the goal.

### Deliverables

1. **Code:**
   - `phoenix/multimodal/` module
   - CLIP integration and fusion strategies

2. **Benchmarks:**
   - MS-COCO or Flickr30k evaluation
   - Comparison with baselines

3. **Demo:**
   - Interactive cross-modal retrieval demo
   - Visualization of retrieved results

---

## Timeline Summary

```
Week 1:
├── Days 1-2: F2 Phase 1-2 (Baseline + JIT)
├── Days 3-4: F2 Phase 3 (KV-Cache)
├── Days 5-6: F2 Phase 4 (Attention)
└── Day 7: F2 Phase 5 (Quantization)

Week 2:
├── Days 1-2: F4 Phase 1-2 (Reward Model + Preference Learning)
├── Days 3-4: F4 Phase 3-4 (Data + Evaluation)
└── Day 5: F4 Writeup

Week 3:
├── Days 1-2: F1 Phase 1-2 (CLIP + Candidate Tower)
├── Days 3-4: F1 Phase 3-4 (User Tower + Cross-Modal)
└── Day 5: F1 Evaluation + Writeup
```

---

## Repository Structure

```
x-algorithm/
├── phoenix/
│   ├── grok.py                    # Original - Transformer
│   ├── recsys_model.py            # Original - Ranking model
│   ├── recsys_retrieval_model.py  # Original - Retrieval model
│   ├── runners.py                 # Original - Inference runners
│   ├── run_ranker.py              # Original - Demo
│   ├── run_retrieval.py           # Original - Demo
│   │
│   ├── optimization/              # F2: JAX Optimization
│   │   ├── __init__.py
│   │   ├── benchmark.py
│   │   ├── profiler.py
│   │   ├── kv_cache.py
│   │   ├── flash_attention.py
│   │   ├── quantization.py
│   │   └── optimized_runner.py
│   │
│   ├── reward_modeling/           # F4: RL Reward Modeling
│   │   ├── __init__.py
│   │   ├── reward_model.py
│   │   ├── learned_weights.py
│   │   ├── preference_dataset.py
│   │   ├── training.py
│   │   └── evaluation.py
│   │
│   └── multimodal/                # F1: Multimodal Retrieval
│       ├── __init__.py
│       ├── clip_encoder.py
│       ├── multimodal_tower.py
│       ├── multimodal_model.py
│       ├── cross_modal.py
│       └── evaluation.py
│
├── experiments/                   # Experiment notebooks
│   ├── f2_optimization.ipynb
│   ├── f4_reward_learning.ipynb
│   └── f1_multimodal.ipynb
│
├── results/                       # Benchmark results
│   ├── f2_baseline.json
│   ├── f2_optimized.json
│   ├── f4_training.json
│   └── f1_retrieval.json
│
└── design_doc.md                  # This document
```

---

## Appendix: Quick Reference

### Running Phoenix Baseline

```bash
cd x-algorithm/phoenix
uv run run_ranker.py      # Ranking demo
uv run run_retrieval.py   # Retrieval demo
uv run pytest             # Tests
```

### Key Files to Understand

| File | Purpose |
|------|---------|
| `grok.py:97-134` | `make_recsys_attn_mask` - Candidate isolation |
| `grok.py:336-354` | Attention computation (optimization target) |
| `recsys_model.py:50-100` | `PhoenixModel.build_inputs` - Input construction |
| `recsys_retrieval_model.py:18-42` | `CandidateTower` - Embedding fusion |
| `runners.py:202-222` | `ACTIONS` - All 19 engagement types |
| `runners.py:340-370` | `RankingOutput` - Model outputs |

### Dependencies to Add

```toml
# For F1 (Multimodal)
transformers = ">=4.30.0"
pillow = ">=10.0.0"

# For F4 (RL)
optax = ">=0.1.5"

# For benchmarking
matplotlib = ">=3.7.0"
pandas = ">=2.0.0"
```
