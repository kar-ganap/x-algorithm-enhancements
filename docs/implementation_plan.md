# Implementation Plan: Testable Phases & Go/No-Go Gates

## Project Structure (Option 2)

```
x-algorithm-enhancements/
│
├── phoenix/                          # ORIGINAL xAI CODE (untouched)
│   ├── grok.py
│   ├── recsys_model.py
│   ├── recsys_retrieval_model.py
│   ├── runners.py
│   ├── run_ranker.py
│   ├── run_retrieval.py
│   └── test_*.py
│
├── enhancements/                     # OUR CODE
│   ├── __init__.py
│   ├── common/                       # Shared utilities
│   │   ├── __init__.py
│   │   ├── config.py                 # Shared configurations
│   │   ├── metrics.py                # Benchmark/eval utilities
│   │   └── viz.py                    # Visualization helpers
│   │
│   ├── optimization/                 # F2: JAX Optimization
│   │   ├── __init__.py
│   │   ├── benchmark.py
│   │   ├── profiler.py
│   │   ├── jit_utils.py
│   │   ├── kv_cache.py
│   │   ├── attention.py
│   │   ├── quantization.py
│   │   └── optimized_runner.py
│   │
│   ├── reward_modeling/              # F4: RL Reward Modeling
│   │   ├── __init__.py
│   │   ├── reward_model.py
│   │   ├── weights.py
│   │   ├── preference_data.py
│   │   ├── training.py
│   │   └── evaluation.py
│   │
│   └── multimodal/                   # F1: Multimodal Retrieval
│       ├── __init__.py
│       ├── clip_encoder.py
│       ├── multimodal_batch.py
│       ├── candidate_tower.py
│       ├── user_tower.py
│       ├── retrieval.py
│       └── evaluation.py
│
├── tests/                            # All tests for our code
│   ├── __init__.py
│   ├── test_optimization/
│   │   ├── test_benchmark.py
│   │   ├── test_kv_cache.py
│   │   ├── test_attention.py
│   │   └── test_quantization.py
│   ├── test_reward_modeling/
│   │   ├── test_reward_model.py
│   │   ├── test_training.py
│   │   └── test_evaluation.py
│   └── test_multimodal/
│       ├── test_clip_encoder.py
│       ├── test_towers.py
│       └── test_retrieval.py
│
├── experiments/                      # Notebooks & scripts
│   ├── f2_optimization_experiments.ipynb
│   ├── f4_reward_experiments.ipynb
│   └── f1_multimodal_experiments.ipynb
│
├── results/                          # Benchmark outputs
│   ├── f2/
│   ├── f4/
│   └── f1/
│
├── design_doc.md                     # High-level design
├── implementation_plan.md            # This file
├── pyproject.toml                    # Project dependencies
└── README.md                         # Updated with our work
```

---

# F2: JAX Optimization - Testable Phases

## Phase 0: Setup & Baseline (Gate: Can we run and measure?)

### Objective
Verify we can run Phoenix model and establish baseline metrics.

### Deliverables
```
enhancements/
├── __init__.py
├── common/
│   ├── __init__.py
│   └── metrics.py
└── optimization/
    ├── __init__.py
    └── benchmark.py

tests/
└── test_optimization/
    └── test_benchmark.py
```

### Implementation
```python
# enhancements/optimization/benchmark.py

@dataclass
class BenchmarkConfig:
    batch_sizes: List[int] = (1, 2, 4, 8)
    history_lens: List[int] = (32, 64, 128)
    num_candidates: List[int] = (8, 16, 32)
    num_warmup: int = 5
    num_runs: int = 50

@dataclass
class BenchmarkResult:
    config: dict
    latency_ms: dict  # {p50, p95, p99, mean, std}
    throughput: float  # batches/sec
    memory_mb: float  # peak GPU memory
    timestamp: str

def benchmark_phoenix_baseline(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Run comprehensive baseline benchmarks."""
    ...

def save_results(results: List[BenchmarkResult], path: str):
    """Save results to JSON."""
    ...
```

### Tests
```python
# tests/test_optimization/test_benchmark.py

def test_phoenix_model_loads():
    """Verify we can load and initialize Phoenix model."""
    runner = RecsysInferenceRunner(...)
    runner.initialize()
    assert runner.params is not None

def test_synthetic_batch_creation():
    """Verify we can create valid synthetic batches."""
    batch, embeddings = create_example_batch(batch_size=2, ...)
    assert batch.user_hashes.shape[0] == 2

def test_forward_pass_runs():
    """Verify forward pass completes without error."""
    output = runner.rank(batch, embeddings)
    assert output.scores.shape == (batch_size, num_candidates, num_actions)

def test_benchmark_produces_results():
    """Verify benchmark harness works."""
    results = benchmark_phoenix_baseline(BenchmarkConfig(num_runs=3))
    assert len(results) > 0
    assert all(r.latency_ms['p50'] > 0 for r in results)
```

### Go/No-Go Gate
| Criterion | Required | How to Test |
|-----------|----------|-------------|
| Phoenix model loads | ✅ | `test_phoenix_model_loads` passes |
| Forward pass works | ✅ | `test_forward_pass_runs` passes |
| Benchmark harness works | ✅ | `test_benchmark_produces_results` passes |
| Baseline metrics recorded | ✅ | `results/f2/baseline.json` exists with valid data |

**Gate Decision:** All 4 criteria must pass. If any fail, debug before proceeding.

---

## Phase 1: JIT Optimization (Gate: Measurable speedup from JIT?)

### Objective
Implement static-shape JIT compilation and measure improvement.

### Deliverables
```
enhancements/optimization/
├── jit_utils.py          # NEW
└── benchmark.py          # UPDATED with JIT benchmarks
```

### Implementation
```python
# enhancements/optimization/jit_utils.py

def create_static_forward(
    runner: RecsysInferenceRunner,
    batch_size: int,
    history_len: int,
    num_candidates: int,
) -> Callable:
    """Create JIT-compiled forward with static shapes."""

    @functools.partial(jax.jit, static_argnums=())
    def forward_static(params, batch, embeddings):
        # Ensure shapes match expected
        return runner.rank_candidates(params, batch, embeddings)

    # Warmup compilation
    dummy_batch, dummy_emb = create_padded_batch(batch_size, history_len, num_candidates)
    _ = forward_static(runner.params, dummy_batch, dummy_emb)

    return forward_static

def pad_batch_to_static(batch, target_batch_size, target_history_len, target_candidates):
    """Pad batch to static shapes for JIT."""
    ...
```

### Tests
```python
# tests/test_optimization/test_jit_utils.py

def test_static_forward_compiles():
    """Verify static forward function compiles."""
    forward_fn = create_static_forward(runner, batch_size=4, history_len=32, num_candidates=8)
    assert callable(forward_fn)

def test_static_forward_output_matches_baseline():
    """Verify JIT output matches non-JIT output."""
    baseline_output = runner.rank(batch, embeddings)
    jit_output = forward_static(runner.params, batch, embeddings)
    np.testing.assert_allclose(baseline_output.scores, jit_output.scores, rtol=1e-5)

def test_jit_faster_than_baseline():
    """Verify JIT version is faster after warmup."""
    baseline_latency = benchmark_latency(runner.rank, num_runs=20)
    jit_latency = benchmark_latency(forward_static, num_runs=20)
    assert jit_latency['p50'] < baseline_latency['p50']
```

### Go/No-Go Gate
| Criterion | Required | How to Test |
|-----------|----------|-------------|
| JIT compiles without error | ✅ | `test_static_forward_compiles` passes |
| Output matches baseline | ✅ | `test_static_forward_output_matches_baseline` passes (rtol=1e-5) |
| Speedup > 1.2x | ✅ | `test_jit_faster_than_baseline` passes |
| No accuracy degradation | ✅ | Scores within rtol=1e-5 of baseline |

**Gate Decision:**
- If speedup < 1.2x: Document findings, proceed anyway (JIT is foundation for later phases)
- If output doesn't match: Debug until it does, do not proceed

---

## Phase 2: KV-Cache (Gate: Cache hit provides speedup?)

### Objective
Implement KV-caching for user context, enabling faster multi-batch candidate scoring.

### Deliverables
```
enhancements/optimization/
├── kv_cache.py           # NEW
└── benchmark.py          # UPDATED with cache benchmarks
```

### Implementation
```python
# enhancements/optimization/kv_cache.py

class KVCache(NamedTuple):
    keys: jax.Array      # [num_layers, batch, num_kv_heads, seq_len, head_dim]
    values: jax.Array    # [num_layers, batch, num_kv_heads, seq_len, head_dim]
    user_hash: int       # For cache invalidation

class CachedPhoenixRunner:
    """Phoenix runner with KV-caching for user context."""

    def __init__(self, base_runner: RecsysInferenceRunner):
        self.base = base_runner
        self._cache: Optional[KVCache] = None
        self._cache_user_hash: Optional[int] = None

    def encode_user_context(self, batch, embeddings) -> KVCache:
        """Encode user context and return KV cache."""
        ...

    def score_with_cache(self, cache: KVCache, candidate_batch, candidate_embeddings) -> jax.Array:
        """Score candidates using cached user context."""
        ...

    def rank(self, batch, embeddings, use_cache: bool = True) -> RankingOutput:
        """Rank with optional caching."""
        user_hash = hash(batch.user_hashes.tobytes())

        if use_cache and self._cache_user_hash == user_hash:
            # Cache hit
            return self.score_with_cache(self._cache, batch, embeddings)
        else:
            # Cache miss - encode and cache
            self._cache = self.encode_user_context(batch, embeddings)
            self._cache_user_hash = user_hash
            return self.score_with_cache(self._cache, batch, embeddings)
```

### Tests
```python
# tests/test_optimization/test_kv_cache.py

def test_kv_cache_creation():
    """Verify KV cache can be created."""
    cache = runner.encode_user_context(batch, embeddings)
    assert cache.keys.shape[0] == num_layers
    assert cache.values.shape[0] == num_layers

def test_cached_output_matches_baseline():
    """Verify cached scoring matches non-cached."""
    baseline = runner.rank(batch, embeddings, use_cache=False)
    cached = runner.rank(batch, embeddings, use_cache=True)
    np.testing.assert_allclose(baseline.scores, cached.scores, rtol=1e-5)

def test_cache_hit_faster_than_miss():
    """Verify cache hit is faster than cache miss."""
    # First call - cache miss
    t1 = time_call(lambda: runner.rank(batch, embeddings, use_cache=True))
    # Second call - cache hit (same user)
    t2 = time_call(lambda: runner.rank(batch, embeddings, use_cache=True))
    assert t2 < t1 * 0.8  # At least 20% faster

def test_cache_invalidation_on_user_change():
    """Verify cache invalidates when user changes."""
    runner.rank(batch1, embeddings1, use_cache=True)
    old_hash = runner._cache_user_hash
    runner.rank(batch2, embeddings2, use_cache=True)  # Different user
    assert runner._cache_user_hash != old_hash
```

### Go/No-Go Gate
| Criterion | Required | How to Test |
|-----------|----------|-------------|
| KV cache creates successfully | ✅ | `test_kv_cache_creation` passes |
| Cached output matches baseline | ✅ | `test_cached_output_matches_baseline` passes |
| Cache hit speedup > 1.3x | ⚠️ | `test_cache_hit_faster_than_miss` passes |
| Cache invalidates correctly | ✅ | `test_cache_invalidation_on_user_change` passes |

**Gate Decision:**
- If cache hit speedup < 1.3x: Investigate why, document, but can proceed
- If output doesn't match baseline: Do not proceed, debug first

---

## Phase 3: Attention Optimization (Gate: Memory or latency improvement?)

### Objective
Implement memory-efficient attention (flash-style or JAX built-in).

### Deliverables
```
enhancements/optimization/
├── attention.py          # NEW
└── benchmark.py          # UPDATED with attention benchmarks
```

### Implementation
```python
# enhancements/optimization/attention.py

def efficient_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    mask: jax.Array,
    scale: float,
) -> jax.Array:
    """Memory-efficient attention implementation.

    Tries JAX built-in first, falls back to custom tiled implementation.
    """
    # Option 1: Try JAX's dot_product_attention if available
    try:
        from jax.nn import dot_product_attention
        return dot_product_attention(query, key, value, mask=mask, scale=scale)
    except ImportError:
        pass

    # Option 2: Custom tiled implementation
    return _tiled_attention(query, key, value, mask, scale)

def _tiled_attention(q, k, v, mask, scale, block_size=64):
    """Tiled attention that doesn't materialize full attention matrix."""
    ...

def replace_attention_in_transformer(transformer_config):
    """Monkey-patch transformer to use efficient attention."""
    ...
```

### Tests
```python
# tests/test_optimization/test_attention.py

def test_efficient_attention_output_matches():
    """Verify efficient attention matches standard."""
    q, k, v = create_random_qkv(batch=2, heads=4, seq=128, dim=64)
    mask = create_causal_mask(128)

    standard = standard_attention(q, k, v, mask)
    efficient = efficient_attention(q, k, v, mask)

    np.testing.assert_allclose(standard, efficient, rtol=1e-4)

def test_efficient_attention_memory_usage():
    """Verify memory usage is lower than standard."""
    # Use larger sequence to see memory difference
    q, k, v = create_random_qkv(batch=4, heads=8, seq=512, dim=64)

    standard_mem = measure_memory(lambda: standard_attention(q, k, v, mask))
    efficient_mem = measure_memory(lambda: efficient_attention(q, k, v, mask))

    assert efficient_mem < standard_mem * 0.7  # At least 30% reduction

def test_efficient_attention_with_candidate_isolation_mask():
    """Verify efficient attention works with Phoenix's special mask."""
    mask = make_recsys_attn_mask(seq_len=64, candidate_start_offset=32)
    output = efficient_attention(q, k, v, mask)
    assert output.shape == expected_shape
```

### Go/No-Go Gate
| Criterion | Required | How to Test |
|-----------|----------|-------------|
| Output matches standard attention | ✅ | `test_efficient_attention_output_matches` (rtol=1e-4) |
| Memory reduction > 30% | ⚠️ | `test_efficient_attention_memory_usage` |
| Works with candidate isolation mask | ✅ | `test_efficient_attention_with_candidate_isolation_mask` |
| Latency improvement > 1.2x | ⚠️ | Benchmark comparison |

**Gate Decision:**
- Memory OR latency improvement required (at least one)
- If neither improves: Document why, consider skipping this optimization

---

## Phase 4: Quantization (Gate: Acceptable accuracy/speed tradeoff?)

### Objective
Implement int8 quantization with minimal accuracy loss.

### Deliverables
```
enhancements/optimization/
├── quantization.py       # NEW
└── benchmark.py          # UPDATED with quantization benchmarks
```

### Implementation
```python
# enhancements/optimization/quantization.py

@dataclass
class QuantizationConfig:
    weight_bits: int = 8
    activation_bits: int = 16  # Keep activations higher precision
    per_channel: bool = True   # Per-channel vs per-tensor quantization

def quantize_params(params: hk.Params, config: QuantizationConfig) -> Tuple[hk.Params, dict]:
    """Quantize model parameters.

    Returns:
        quantized_params: Quantized parameters
        scales: Scale factors for dequantization
    """
    ...

def create_quantized_forward(runner, config: QuantizationConfig):
    """Create forward function using quantized weights."""
    quantized_params, scales = quantize_params(runner.params, config)

    def forward(batch, embeddings):
        # Dequantize on-the-fly during matmuls
        ...

    return forward
```

### Tests
```python
# tests/test_optimization/test_quantization.py

def test_quantize_dequantize_roundtrip():
    """Verify quantize->dequantize preserves values approximately."""
    original = jnp.array([0.1, -0.5, 0.9, -0.2])
    quantized, scale = quantize_tensor(original, bits=8)
    recovered = dequantize_tensor(quantized, scale)
    np.testing.assert_allclose(original, recovered, atol=0.01)

def test_quantized_forward_accuracy():
    """Verify quantized model accuracy is within tolerance."""
    baseline_output = runner.rank(batch, embeddings)
    quantized_output = quantized_runner.rank(batch, embeddings)

    # Check ranking preservation (more important than exact scores)
    baseline_ranks = jnp.argsort(-baseline_output.scores, axis=-1)
    quantized_ranks = jnp.argsort(-quantized_output.scores, axis=-1)

    # Top-3 should match at least 90% of the time
    top3_match = (baseline_ranks[:, :, :3] == quantized_ranks[:, :, :3]).mean()
    assert top3_match > 0.9

def test_quantized_memory_reduction():
    """Verify memory usage is reduced."""
    baseline_mem = measure_param_memory(runner.params)
    quantized_mem = measure_param_memory(quantized_runner.params)
    assert quantized_mem < baseline_mem * 0.5  # At least 50% reduction

def test_quantized_latency():
    """Verify latency is not significantly worse."""
    baseline_latency = benchmark_latency(runner.rank)
    quantized_latency = benchmark_latency(quantized_runner.rank)
    assert quantized_latency < baseline_latency * 1.2  # No more than 20% slower
```

### Go/No-Go Gate
| Criterion | Required | How to Test |
|-----------|----------|-------------|
| Top-3 ranking preserved > 90% | ✅ | `test_quantized_forward_accuracy` |
| Memory reduction > 40% | ✅ | `test_quantized_memory_reduction` |
| Latency not worse than 1.2x | ✅ | `test_quantized_latency` |

**Gate Decision:**
- If ranking accuracy < 90%: Try different quantization config, or skip
- If memory reduction < 40%: Not worth the complexity, skip

---

## Phase 5: Integration & Final Benchmark (Gate: Overall improvement?)

### Objective
Combine all optimizations and produce final benchmark report.

### Deliverables
```
enhancements/optimization/
└── optimized_runner.py   # NEW - combines all optimizations

results/f2/
├── baseline.json
├── phase1_jit.json
├── phase2_kv_cache.json
├── phase3_attention.json
├── phase4_quantization.json
└── final_combined.json
```

### Implementation
```python
# enhancements/optimization/optimized_runner.py

class OptimizedPhoenixRunner:
    """Phoenix runner with all optimizations applied."""

    def __init__(
        self,
        base_runner: RecsysInferenceRunner,
        use_jit: bool = True,
        use_kv_cache: bool = True,
        use_efficient_attention: bool = True,
        quantization_config: Optional[QuantizationConfig] = None,
    ):
        self.config = {...}
        self._setup_optimizations()

    def rank(self, batch, embeddings) -> RankingOutput:
        """Optimized ranking."""
        ...

    def benchmark_report(self) -> dict:
        """Generate full benchmark report comparing to baseline."""
        ...
```

### Tests
```python
# tests/test_optimization/test_optimized_runner.py

def test_optimized_runner_output_matches_baseline():
    """Verify optimized runner produces same rankings as baseline."""
    baseline = baseline_runner.rank(batch, embeddings)
    optimized = optimized_runner.rank(batch, embeddings)

    baseline_ranks = jnp.argsort(-baseline.scores[..., 0], axis=-1)
    optimized_ranks = jnp.argsort(-optimized.scores[..., 0], axis=-1)

    assert (baseline_ranks == optimized_ranks).mean() > 0.95

def test_optimized_runner_speedup():
    """Verify combined optimizations achieve target speedup."""
    baseline_latency = benchmark_latency(baseline_runner.rank)
    optimized_latency = benchmark_latency(optimized_runner.rank)

    speedup = baseline_latency['p50'] / optimized_latency['p50']
    assert speedup >= 2.0  # Target: at least 2x

def test_optimized_runner_memory():
    """Verify memory usage is reduced."""
    baseline_mem = measure_inference_memory(baseline_runner)
    optimized_mem = measure_inference_memory(optimized_runner)

    assert optimized_mem < baseline_mem * 0.7  # At least 30% reduction
```

### Final Go/No-Go Gate (F2 Complete)
| Criterion | Required | Target |
|-----------|----------|--------|
| Ranking accuracy > 95% | ✅ | Matches baseline |
| Latency speedup | ✅ | >= 2x |
| Memory reduction | ⚠️ | >= 30% (nice to have) |
| All tests pass | ✅ | 100% |
| Benchmark report generated | ✅ | `results/f2/final_combined.json` |

---

## Phase 6: Learned Weights & Analysis Validation (Gate: Real dynamics demonstrated?)

### Problem Statement
Phases 1-5 validated optimizations on **random weights**:
- "100% accuracy" = preserving random rankings
- Analysis tools show uniform scores, no dynamics
- Speedups are real, but "so what?" is unclear

**We need learned weights to demonstrate genuine value.**

### Objective
Train Phoenix on MovieLens to:
1. Validate optimizations work on non-trivial weights
2. Demonstrate analysis tools reveal real dynamics
3. Show genuine filter bubble emergence

### Why MovieLens?
| Criterion | MovieLens | Twitter Data | Synthetic |
|-----------|-----------|--------------|-----------|
| Real human patterns | ✅ | ✅ | ❌ Designed |
| Accessible | ✅ Free | ❌ API costs | ✅ Free |
| Credible | ✅ Academic benchmark | ✅ Most relevant | ⚠️ Circular |

### Data Mapping
```
MovieLens → Phoenix:
- User ID → User hash
- Movie ID → Item hash (post_hash)
- Rating history → Engagement history
- Movie genres → Item embeddings
- Ratings: 5★→"like", 4★→"positive", 3★→"neutral"
```

### Deliverables
```
enhancements/data/
├── __init__.py
├── movielens.py              # Dataset loader
├── movielens_adapter.py      # Phoenix format adapter
└── embeddings.py             # Embedding utilities

enhancements/training/
├── __init__.py
├── train_phoenix.py          # Training loop
├── losses.py                 # Loss functions
└── metrics.py                # Evaluation metrics

scripts/
├── download_movielens.py     # Data download
├── train_movielens.py        # Training entry point
└── run_analysis.py           # Run analysis tools

models/movielens_phoenix/
└── best_model.pkl            # Trained weights
```

### Implementation Steps
1. **Data Infrastructure**: Download MovieLens, create dataset class
2. **Phoenix Adapter**: Map MovieLens to RecsysBatch/RecsysEmbeddings
3. **Embeddings**: Generate movie embeddings from genres
4. **Training Loop**: Loss function, optimizer, checkpointing
5. **Validation**: NDCG@10, Hit Rate metrics
6. **Re-run Analysis**: Trajectory, diversity, counterfactual
7. **Re-run Benchmarks**: Verify optimizations preserve learned rankings

### Tests
```python
# tests/test_data/test_movielens.py

def test_movielens_loads():
    """Verify MovieLens dataset loads correctly."""
    dataset = MovieLensDataset("100k")
    assert len(dataset.train) > 0
    assert len(dataset.test) > 0

def test_adapter_produces_valid_batch():
    """Verify adapter creates valid Phoenix batch."""
    adapter = MovieLensToPhoenixAdapter(dataset, model_config)
    batch, embeddings = adapter.get_batch(user_id=1)

    assert batch.user_hashes.shape[0] == 1
    assert embeddings.user_embeddings.shape[-1] == model_config.emb_size

def test_model_learns():
    """Verify loss decreases during training."""
    initial_loss = trainer.evaluate(val_data)
    trainer.train(epochs=5)
    final_loss = trainer.evaluate(val_data)

    assert final_loss < initial_loss * 0.8  # At least 20% reduction
```

### Go/No-Go Gates
| Gate | Criterion | Measurement |
|------|-----------|-------------|
| Training works | Loss decreases | Training curves |
| Model learns | Val NDCG@10 > random | Evaluation metrics |
| Analysis shows dynamics | Non-uniform scores | Trajectory plots |
| Filter bubbles emerge | Diversity decreases | Coverage/Gini metrics |
| Counterfactuals meaningful | Some τ < 0.9 | Ablation analysis |
| Optimizations preserve accuracy | <1% degradation | Before/after comparison |

### Success Criteria
Phase 6 is successful when we can demonstrate:
1. Filter bubbles emerge naturally when following recommendations
2. Early engagement history influences later recommendations
3. Optimizations preserve ranking quality on learned weights

---

## Phase 7: Synthetic Twitter Data & Verification (Gate: Model learns planted patterns?)

### Objective
Create synthetic Twitter-like data with known ground truth patterns, train Phoenix, and verify the model recovers those patterns. This validates Phoenix on the full 19-action space.

### Motivation
MovieLens only tests 1 action type. Synthetic data enables:
- Testing all 19 action types (like, retweet, reply, block, etc.)
- Verifying author preference learning
- Testing negative signal effectiveness (block/mute)
- Ground truth comparison (we know the planted patterns)

### Deliverables
```
enhancements/
├── data/
│   ├── synthetic_twitter.py      # Data generator
│   ├── synthetic_adapter.py      # Phoenix format adapter
│   └── ground_truth.py           # Planted pattern definitions
│
├── verification/                  # NEW
│   ├── __init__.py
│   ├── embedding_probes.py       # Cluster analysis
│   ├── behavioral_tests.py       # Topic/author preference tests
│   ├── action_tests.py           # Action differentiation tests
│   ├── counterfactual_tests.py   # Intervention tests
│   └── suite.py                  # Orchestrator

scripts/
├── generate_synthetic.py         # Generate dataset
├── train_synthetic.py            # Train model
└── verify_synthetic.py           # Run verification

tests/test_verification/
├── test_embedding_probes.py
├── test_behavioral_tests.py
└── test_action_tests.py
```

### Data Model

**User Archetypes (1,000 users)**
| Archetype | % | Behavior |
|-----------|---|----------|
| `sports_fan` | 15% | Like/RT sports, ignore politics |
| `political_L` | 15% | Engage left, block right |
| `political_R` | 15% | Engage right, block left |
| `tech_bro` | 15% | Like/RT tech content |
| `lurker` | 20% | Like only, never RT/reply |
| `power_user` | 20% | High RT/reply ratio |

**Content Topics (50,000 posts)**
| Topic | % | Authors |
|-------|---|---------|
| `sports` | 25% | 20 sports accounts |
| `politics_L` | 12.5% | 15 left-leaning accounts |
| `politics_R` | 12.5% | 15 right-leaning accounts |
| `tech` | 20% | 20 tech accounts |
| `entertainment` | 20% | 20 entertainment accounts |
| `news` | 10% | 10 mixed accounts |

**Engagement Rules (Ground Truth)**
```python
ENGAGEMENT_RULES = {
    ('sports_fan', 'sports'): {
        'favorite': 0.70, 'repost': 0.30, 'reply': 0.10,
    },
    ('sports_fan', 'politics'): {
        'favorite': 0.05, 'not_interested': 0.10,
    },
    ('political_L', 'politics_L'): {
        'favorite': 0.65, 'repost': 0.45, 'follow_author': 0.20,
    },
    ('political_L', 'politics_R'): {
        'block_author': 0.25, 'not_interested': 0.30,
    },
    ('lurker', '*'): {
        'favorite': 0.20, 'repost': 0.01, 'reply': 0.00,
    },
    ('power_user', '*'): {
        'favorite': 0.45, 'repost': 0.35, 'reply': 0.25,
    },
}
```

### Verification Tests

**1. Embedding Probes**
```python
def test_user_archetype_clustering():
    """Users of same archetype should cluster in embedding space."""
    user_embs = get_user_embeddings(all_users)
    labels = [get_archetype(u) for u in all_users]
    silhouette = silhouette_score(user_embs, labels)
    assert silhouette > 0.2  # Meaningful clustering

def test_topic_clustering():
    """Posts of same topic should cluster."""
    post_embs = get_post_embeddings(all_posts)
    labels = [get_topic(p) for p in all_posts]
    silhouette = silhouette_score(post_embs, labels)
    assert silhouette > 0.2
```

**2. Behavioral Tests**
```python
def test_topic_preference(archetype, topic, expected):
    """Verify archetype prefers expected topic."""
    users = get_users_by_archetype(archetype)
    posts = get_posts_by_topic(topic)

    scores = []
    for user in sample(users, 50):
        for post in sample(posts, 20):
            score = model.predict_engagement(user, post)
            scores.append(score)

    actual = np.mean(scores)
    assert abs(actual - expected) < 0.15  # Within 15%
```

**3. Action Differentiation Tests**
```python
def test_lurker_action_distribution():
    """Lurkers: high like, low retweet/reply."""
    lurkers = get_users_by_archetype('lurker')
    action_dist = predict_action_distribution(lurkers)

    assert action_dist['favorite'] > action_dist['repost'] * 10
    assert action_dist['reply'] < 0.05

def test_power_user_vs_lurker():
    """Power users have higher retweet ratio than lurkers."""
    lurker_dist = predict_action_distribution(get_users_by_archetype('lurker'))
    power_dist = predict_action_distribution(get_users_by_archetype('power_user'))

    lurker_rt_ratio = lurker_dist['repost'] / lurker_dist['favorite']
    power_rt_ratio = power_dist['repost'] / power_dist['favorite']

    assert power_rt_ratio > lurker_rt_ratio * 3  # 3x higher
```

**4. Counterfactual Tests**
```python
def test_block_reduces_ranking():
    """Blocking author should reduce their posts' ranking."""
    user = sample_user()
    author = sample_author()

    baseline_rank = get_rank(user, author_posts)

    # Add block to history
    add_block_to_history(user, author)
    counterfactual_rank = get_rank(user, author_posts)

    assert counterfactual_rank > baseline_rank  # Worse rank

def test_archetype_flip():
    """Replacing history with different archetype's should flip predictions."""
    sports_fan = sample_user_by_archetype('sports_fan')
    tech_bro_history = get_typical_history('tech_bro')

    # Give sports fan a tech bro's history
    predictions = predict_with_custom_history(sports_fan, tech_bro_history)

    # Should now prefer tech over sports
    assert predictions['tech'] > predictions['sports']
```

### Go/No-Go Gates

| Gate | Criterion | Threshold |
|------|-----------|-----------|
| G1: Data generates | 50K posts, 200K engagements | Complete |
| G2: Training works | Loss decreases | NDCG > random |
| G3: Embedding probes | Archetype clustering | Silhouette > 0.2 |
| G4: Behavioral tests | Topic preference accuracy | > 70% |
| G5: Action differentiation | Lurker vs power user | Ratio > 2x |
| G6: Counterfactual tests | Block reduces ranking | > 50% of cases |

### Failure Interpretation

| If This Fails | It Suggests |
|---------------|-------------|
| G2 (training) | Bug in adapter or training loop |
| G3 (embeddings) | Model not learning user/item structure |
| G4 (behavioral) | Topic preferences not captured |
| G5 (actions) | Phoenix treats all actions same |
| G6 (counterfactual) | Negative signals not learned |

### Implementation Order

| Step | Component | Depends On |
|------|-----------|------------|
| 7.1 | Ground truth definitions | - |
| 7.2 | Data generator | 7.1 |
| 7.3 | Phoenix adapter | 7.2 |
| 7.4 | Training script | 7.3 |
| 7.5 | Embedding probes | 7.4 |
| 7.6 | Behavioral tests | 7.4 |
| 7.7 | Action tests | 7.4 |
| 7.8 | Counterfactual tests | 7.4 |
| 7.9 | Verification suite | 7.5-7.8 |

---

# F4: RL Reward Modeling - Testable Phases

## Phase 0: Reward Model Wrapper (Gate: Can compute rewards?)

### Objective
Create clean reward model interface around Phoenix predictions.

### Deliverables
```
enhancements/reward_modeling/
├── __init__.py
├── reward_model.py
└── weights.py

tests/test_reward_modeling/
└── test_reward_model.py
```

### Tests
```python
def test_reward_model_initialization():
    """Verify reward model initializes with Phoenix runner."""
    reward_model = PhoenixRewardModel(phoenix_runner)
    assert reward_model.weights is not None

def test_get_action_probs_shape():
    """Verify action probabilities have correct shape."""
    probs = reward_model.get_action_probs(batch, embeddings)
    assert probs.shape == (batch_size, num_candidates, num_actions)

def test_compute_reward_shape():
    """Verify rewards have correct shape."""
    rewards = reward_model.compute_reward(batch, embeddings)
    assert rewards.shape == (batch_size, num_candidates)

def test_reward_ordering_matches_weights():
    """Verify higher weighted actions increase reward."""
    # Create batch where one candidate has high P(like), another has high P(block)
    # Verify the liker gets higher reward
    ...
```

### Go/No-Go Gate
| Criterion | Required |
|-----------|----------|
| Reward model initializes | ✅ |
| Action probs correct shape | ✅ |
| Rewards correct shape | ✅ |
| Reward ordering sensible | ✅ |

---

## Phase 1: Preference Data & Loss (Gate: Loss computes and decreases?)

### Objective
Implement preference data format and Bradley-Terry loss.

### Deliverables
```
enhancements/reward_modeling/
├── preference_data.py    # NEW
└── training.py           # NEW (partial)
```

### Tests
```python
def test_preference_pair_creation():
    """Verify preference pairs can be created."""
    pairs = generate_synthetic_preferences(phoenix_runner, num_pairs=10)
    assert len(pairs) == 10
    assert all(p.preferred_idx != p.rejected_idx for p in pairs)

def test_preference_loss_computes():
    """Verify loss function runs without error."""
    loss = preference_loss(weights, phoenix_runner, batch_of_pairs)
    assert jnp.isfinite(loss)
    assert loss > 0  # Cross-entropy loss should be positive

def test_preference_loss_gradient():
    """Verify loss has valid gradients."""
    loss, grads = jax.value_and_grad(preference_loss)(weights, ...)
    assert all(jnp.isfinite(g).all() for g in jax.tree_util.tree_leaves(grads))
```

### Go/No-Go Gate
| Criterion | Required |
|-----------|----------|
| Preference pairs generate | ✅ |
| Loss computes (finite) | ✅ |
| Gradients are finite | ✅ |

---

## Phase 2: Training Loop (Gate: Can recover known weights?)

### Objective
Implement training loop and validate with weight recovery experiment.

### Deliverables
```
enhancements/reward_modeling/
└── training.py           # COMPLETE
```

### Tests
```python
def test_training_loop_runs():
    """Verify training loop completes without error."""
    learned_weights, history = train_reward_weights(
        phoenix_runner,
        synthetic_preferences,
        num_epochs=10
    )
    assert len(history) > 0

def test_loss_decreases():
    """Verify loss decreases during training."""
    _, history = train_reward_weights(..., num_epochs=50)
    assert history[-1] < history[0]  # Final loss < initial loss

def test_weight_recovery():
    """Verify we can recover known ground-truth weights."""
    # Generate preferences using known weights
    true_weights = RewardWeights.default()
    preferences = generate_synthetic_preferences(phoenix_runner, true_weights, num_pairs=1000)

    # Train from random initialization
    init_weights = RewardWeights(jnp.zeros(num_actions))
    learned_weights, _ = train_reward_weights(..., init_weights=init_weights)

    # Check recovery (correlation, not exact match)
    correlation = jnp.corrcoef(true_weights.weights, learned_weights.weights)[0, 1]
    assert correlation > 0.8  # Strong correlation
```

### Go/No-Go Gate
| Criterion | Required |
|-----------|----------|
| Training completes | ✅ |
| Loss decreases | ✅ |
| Weight recovery correlation > 0.8 | ✅ |

---

## Phase 3: Evaluation (Gate: Preference accuracy > 70%?)

### Objective
Implement evaluation metrics and test on held-out data.

### Deliverables
```
enhancements/reward_modeling/
└── evaluation.py         # NEW

results/f4/
├── weight_recovery.json
└── preference_accuracy.json
```

### Tests
```python
def test_preference_accuracy_computation():
    """Verify accuracy metric computes correctly."""
    # Create pairs where we know the answer
    acc = compute_preference_accuracy(reward_model, test_pairs)
    assert 0 <= acc <= 1

def test_learned_model_beats_random():
    """Verify learned model beats random baseline."""
    random_acc = 0.5  # Random guessing
    learned_acc = compute_preference_accuracy(learned_reward_model, test_pairs)
    assert learned_acc > random_acc + 0.1  # At least 10% better than random

def test_interpretability_analysis():
    """Verify weight analysis produces sensible results."""
    analysis = analyze_learned_weights(learned_weights)

    # Negative actions should have negative weights
    assert 'block_author_score' in analysis['negative_actions']
    assert 'report_score' in analysis['negative_actions']

    # Positive actions should have positive weights
    assert 'favorite_score' in analysis['positive_actions']
```

### Final Go/No-Go Gate (F4 Complete)
| Criterion | Required | Target |
|-----------|----------|--------|
| Preference accuracy (synthetic) | ✅ | > 85% |
| Weight recovery correlation | ✅ | > 0.8 |
| Interpretable weights | ✅ | Signs match intuition |
| All tests pass | ✅ | 100% |

---

# F1: Multimodal Retrieval - Testable Phases

## Phase 0: CLIP Integration (Gate: Can encode images/text?)

### Objective
Set up CLIP encoder and verify it works.

### Deliverables
```
enhancements/multimodal/
├── __init__.py
└── clip_encoder.py

tests/test_multimodal/
└── test_clip_encoder.py
```

### Tests
```python
def test_clip_model_loads():
    """Verify CLIP model loads successfully."""
    encoder = CLIPEncoder()
    assert encoder.model is not None
    assert encoder.embed_dim == 512  # or 768 for large

def test_encode_images():
    """Verify image encoding works."""
    images = [create_random_image() for _ in range(4)]
    embeddings = encoder.encode_images(images)
    assert embeddings.shape == (4, 512)

    # Verify normalized
    norms = jnp.linalg.norm(embeddings, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

def test_encode_texts():
    """Verify text encoding works."""
    texts = ["a cat", "a dog", "a bird", "a fish"]
    embeddings = encoder.encode_texts(texts)
    assert embeddings.shape == (4, 512)

def test_cross_modal_similarity():
    """Verify cross-modal similarity is sensible."""
    image_emb = encoder.encode_images([cat_image])
    text_embs = encoder.encode_texts(["a photo of a cat", "a photo of a car"])

    similarities = jnp.dot(image_emb, text_embs.T)
    assert similarities[0, 0] > similarities[0, 1]  # "cat" more similar than "car"
```

### Go/No-Go Gate
| Criterion | Required |
|-----------|----------|
| CLIP loads | ✅ |
| Image encoding works | ✅ |
| Text encoding works | ✅ |
| Cross-modal similarity sensible | ✅ |

---

## Phase 1: Multimodal Candidate Tower (Gate: Fusion improves retrieval?)

### Objective
Extend candidate tower to fuse hash + CLIP embeddings.

### Deliverables
```
enhancements/multimodal/
├── multimodal_batch.py   # NEW
└── candidate_tower.py    # NEW
```

### Tests
```python
def test_multimodal_batch_creation():
    """Verify multimodal batch can be created."""
    batch = create_multimodal_batch(...)
    assert batch.candidate_image_embeddings is not None

def test_multimodal_tower_output_shape():
    """Verify tower produces correct output shape."""
    tower = MultimodalCandidateTower(emb_size=128)
    output = tower(hash_emb, image_emb)
    assert output.shape == (batch_size, num_candidates, 128)

def test_multimodal_tower_normalized():
    """Verify output is normalized."""
    output = tower(hash_emb, image_emb)
    norms = jnp.linalg.norm(output, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

def test_backward_compatibility():
    """Verify tower works without image embeddings (original behavior)."""
    output = tower(hash_emb, image_emb=None)
    assert output.shape == (batch_size, num_candidates, 128)
```

### Go/No-Go Gate
| Criterion | Required |
|-----------|----------|
| Multimodal batch creates | ✅ |
| Tower output correct shape | ✅ |
| Output normalized | ✅ |
| Backward compatible | ✅ |

---

## Phase 2: Cross-Modal Retrieval (Gate: Retrieval works?)

### Objective
Implement text→image and image→text retrieval.

### Deliverables
```
enhancements/multimodal/
└── retrieval.py          # NEW
```

### Tests
```python
def test_build_corpus():
    """Verify corpus can be built."""
    retriever = CrossModalRetriever(model, clip_encoder)
    retriever.build_corpus(text_posts, image_posts)
    assert retriever.image_corpus_embeddings is not None

def test_text_to_image_retrieval():
    """Verify text→image retrieval returns results."""
    results = retriever.retrieve_text_to_image("a sunset", top_k=5)
    assert len(results) == 5
    assert all(0 <= idx < corpus_size for idx in results)

def test_image_to_text_retrieval():
    """Verify image→text retrieval returns results."""
    results = retriever.retrieve_image_to_text(sunset_image, top_k=5)
    assert len(results) == 5

def test_retrieval_relevance():
    """Verify retrieved results are relevant."""
    # Insert known image with caption
    corpus_images = [cat_image, dog_image, car_image]
    corpus_texts = ["a cat", "a dog", "a car"]

    retriever.build_corpus(corpus_texts, corpus_images)

    results = retriever.retrieve_text_to_image("feline", top_k=1)
    assert results[0] == 0  # Should retrieve cat image
```

### Go/No-Go Gate
| Criterion | Required |
|-----------|----------|
| Corpus builds | ✅ |
| Text→image returns results | ✅ |
| Image→text returns results | ✅ |
| Results are relevant | ✅ |

---

## Phase 3: Benchmark Evaluation (Gate: Reasonable Recall@k?)

### Objective
Evaluate on standard benchmark (COCO or Flickr30k).

### Deliverables
```
enhancements/multimodal/
└── evaluation.py         # NEW

results/f1/
├── coco_retrieval.json
└── comparison_with_clip.json
```

### Tests
```python
def test_coco_evaluation_runs():
    """Verify evaluation completes on small subset."""
    metrics = evaluate_coco_retrieval(retriever, coco_subset, split='val')
    assert 'text_to_image_R@1' in metrics
    assert 'image_to_text_R@1' in metrics

def test_recall_above_baseline():
    """Verify recall is above random baseline."""
    metrics = evaluate_coco_retrieval(retriever, coco_val)

    # Random baseline for 1000 images: R@10 ≈ 1%
    assert metrics['text_to_image_R@10'] > 0.05  # At least 5%

def test_comparison_with_clip():
    """Compare with pure CLIP baseline."""
    our_metrics = evaluate_coco_retrieval(our_retriever, coco_val)
    clip_metrics = evaluate_coco_retrieval(clip_baseline, coco_val)

    # We may not beat CLIP, but document the comparison
    print(f"Ours: {our_metrics}")
    print(f"CLIP: {clip_metrics}")
```

### Final Go/No-Go Gate (F1 Complete)
| Criterion | Required | Target |
|-----------|----------|--------|
| Text→Image R@10 | ✅ | > 30% |
| Image→Text R@10 | ✅ | > 30% |
| All tests pass | ✅ | 100% |
| Benchmark report generated | ✅ | `results/f1/coco_retrieval.json` |

---

# Summary: All Gates

## F2: JAX Optimization
| Phase | Gate Criterion | Required |
|-------|---------------|----------|
| 0 | Can run and measure baseline | ✅ |
| 1 | JIT speedup > 1.2x, output matches | ✅ |
| 2 | KV-cache hit faster, output matches | ✅ |
| 3 | Memory OR latency improvement | ⚠️ |
| 4 | Ranking preserved > 90%, memory reduced | ✅ |
| 5 | Overall speedup >= 2x | ✅ |
| 6 | MovieLens training, NDCG > random | ✅ |
| 7 | Synthetic verification: patterns recovered | ⬜ |

## F4: RL Reward Modeling
| Phase | Gate Criterion | Required |
|-------|---------------|----------|
| 0 | Can compute rewards | ✅ |
| 1 | Loss computes, gradients finite | ✅ |
| 2 | Weight recovery correlation > 0.8 | ✅ |
| 3 | Preference accuracy > 85% | ✅ |

## F1: Multimodal Retrieval
| Phase | Gate Criterion | Required |
|-------|---------------|----------|
| 0 | CLIP encodes images/text | ✅ |
| 1 | Multimodal tower works | ✅ |
| 2 | Cross-modal retrieval works | ✅ |
| 3 | R@10 > 30% on COCO | ✅ |
