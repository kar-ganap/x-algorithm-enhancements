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
│   ├── optimization/                 # F1: JAX Optimization
│   │   ├── __init__.py
│   │   ├── benchmark.py
│   │   ├── profiler.py
│   │   ├── jit_utils.py
│   │   ├── kv_cache.py
│   │   ├── attention.py
│   │   ├── quantization.py
│   │   └── optimized_runner.py
│   │
│   ├── reward_modeling/              # F2: RL Reward Modeling (Pluralistic + Multi-Stakeholder)
│   │   ├── __init__.py
│   │   ├── reward_model.py           # Basic + Contextual reward models
│   │   ├── pluralistic.py            # Pluralistic (mixture) reward model
│   │   ├── weights.py                # RewardWeights dataclass
│   │   ├── preference_data.py        # Preference pair handling
│   │   ├── training.py               # Training loops
│   │   ├── structural_recovery.py    # Ground truth verification
│   │   ├── causal_verification.py    # Intervention tests
│   │   ├── objectives.py             # Multi-stakeholder objectives
│   │   ├── pareto.py                 # Pareto frontier computation
│   │   ├── stakeholders.py           # Stakeholder definitions
│   │   ├── policy_analysis.py        # Policy impact analysis
│   │   ├── game_theory.py            # (Optional) Game-theoretic models
│   │   └── visualization.py          # Tradeoff visualizations
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
│   │   ├── test_reward_model.py      # Basic + contextual reward tests
│   │   ├── test_pluralistic.py       # Pluralistic model tests
│   │   ├── test_training.py          # Training loop tests
│   │   ├── test_structural.py        # Structural recovery tests
│   │   ├── test_causal.py            # Causal verification tests
│   │   ├── test_multi_objective.py   # Pareto frontier tests
│   │   └── test_stakeholders.py      # Stakeholder analysis tests
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

# F1: JAX Optimization - Testable Phases

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

# F2: RL Reward Modeling - Testable Phases

## Overview: Risk-Tiered Approach

F2 builds progressively from basic reward learning to novel multi-stakeholder analysis:

```
Day 13+:   Phase 6 (Game Theory) ─────────────── Tier 3: Novel (optional)
Day 11-12: Phase 5 (Stakeholder Utilities) ──── Tier 2: Multi-Stakeholder
Day 9-10:  Phase 4 (Multi-Objective) ─────────── Tier 2: Multi-Stakeholder
Day 7-8:   Phase 3 (Causal Verification) ─────── Tier 1: Core
Day 5-6:   Phase 2 (Pluralistic Rewards) ─────── Tier 1: Core
Day 3-4:   Phase 1 (Context-Dependent) ───────── Foundation: Ambitious
Day 1-2:   Phase 0 (Basic Reward) ────────────── Foundation: Table Stakes
```

**Key Principle:** Each phase is a valid exit point with publishable deliverables.

### F2 Integration

F2 leverages F2 components throughout:

| F2 Component | F2 Phase | How Used |
|--------------|----------|----------|
| Trained Phoenix | All | Base model for action probabilities |
| Synthetic Data | Phase 2+ | Ground truth archetypes |
| Ground Truth | Phase 2+ | True reward functions for verification |
| Verification Suite | Phase 3 | Extended for causal reward tests |
| KV-Cache | All | 3-4x speedup in reward training |
| JIT | All | Fast reward computation |

---

## Phase 0: Basic Reward Model (Gate: Can compute rewards?)

### Objective
Create reward model that computes scalar rewards from Phoenix action probabilities.

### Deliverables
```
enhancements/reward_modeling/
├── __init__.py
├── reward_model.py          # PhoenixRewardModel
├── weights.py               # RewardWeights dataclass
└── preference_data.py       # Preference pair handling

tests/test_reward_modeling/
└── test_reward_model.py
```

### Implementation
```python
# enhancements/reward_modeling/reward_model.py

from enhancements.optimization import OptimizedPhoenixRunner  # F2 integration

@dataclass
class RewardWeights:
    """Learnable reward weights for 19 actions."""
    weights: jnp.ndarray  # [19]

    @classmethod
    def default(cls) -> 'RewardWeights':
        """Default weights: positive for engagement, negative for block/mute."""
        w = jnp.array([
            1.0,   # favorite
            0.5,   # reply
            0.8,   # repost
            0.1,   # photo_expand
            0.2,   # click
            0.3,   # profile_click
            0.1,   # vqv
            0.6,   # share
            0.4,   # share_via_dm
            0.3,   # share_via_copy_link
            0.1,   # dwell
            0.7,   # quote
            0.2,   # quoted_click
            1.2,   # follow_author
            -0.5,  # not_interested
            -1.5,  # block_author
            -1.0,  # mute_author
            -2.0,  # report
        ])
        return cls(weights=w)

class PhoenixRewardModel:
    """Reward model wrapping Phoenix for RL training."""

    def __init__(
        self,
        phoenix_runner: OptimizedPhoenixRunner,  # Use F2's optimized runner
        weights: Optional[RewardWeights] = None,
    ):
        self.runner = phoenix_runner
        self.weights = weights or RewardWeights.default()

    def get_action_probs(self, batch, embeddings) -> jnp.ndarray:
        """Get action probabilities from Phoenix. Shape: [B, C, 19]"""
        output = self.runner.rank(batch, embeddings)
        return output.scores  # Already probabilities from sigmoid

    def compute_reward(self, batch, embeddings) -> jnp.ndarray:
        """Compute scalar reward per candidate. Shape: [B, C]"""
        probs = self.get_action_probs(batch, embeddings)
        # Weighted sum: R = w · P(actions)
        rewards = jnp.einsum('bca,a->bc', probs, self.weights.weights)
        return rewards
```

### Tests
```python
# tests/test_reward_modeling/test_reward_model.py

def test_reward_model_initialization():
    """Verify reward model initializes with optimized Phoenix runner."""
    from enhancements.optimization import OptimizedPhoenixRunner

    opt_runner = OptimizedPhoenixRunner(phoenix_runner, use_kv_cache=True)
    reward_model = PhoenixRewardModel(opt_runner)
    assert reward_model.weights is not None
    assert reward_model.weights.weights.shape == (19,)  # Changed from 18 to 19

def test_get_action_probs_shape():
    """Verify action probabilities have correct shape."""
    probs = reward_model.get_action_probs(batch, embeddings)
    assert probs.shape == (batch_size, num_candidates, 19)
    assert jnp.all((probs >= 0) & (probs <= 1))  # Valid probabilities

def test_compute_reward_shape():
    """Verify rewards have correct shape."""
    rewards = reward_model.compute_reward(batch, embeddings)
    assert rewards.shape == (batch_size, num_candidates)

def test_reward_reflects_weights():
    """Verify reward ordering matches weight signs."""
    # High P(favorite) should give higher reward than high P(block)
    probs_like = jnp.zeros((1, 2, 19)).at[0, 0, 0].set(0.9)   # High favorite
    probs_block = jnp.zeros((1, 2, 19)).at[0, 1, 15].set(0.9)  # High block

    rewards = reward_model._compute_reward_from_probs(
        jnp.concatenate([probs_like, probs_block], axis=1)
    )
    assert rewards[0, 0] > rewards[0, 1]  # Like > Block

def test_kv_cache_speedup():
    """Verify F2's KV-cache provides speedup in reward computation."""
    # Same user, different candidates - should hit cache
    t1 = time_fn(lambda: reward_model.compute_reward(batch1, emb1))  # Cache miss
    t2 = time_fn(lambda: reward_model.compute_reward(batch2, emb2))  # Cache hit
    assert t2 < t1 * 0.5  # At least 2x faster
```

### Go/No-Go Gate
| Criterion | Required | How to Test |
|-----------|----------|-------------|
| Reward model initializes | ✅ | `test_reward_model_initialization` |
| Action probs correct shape | ✅ | `test_get_action_probs_shape` |
| Rewards correct shape | ✅ | `test_compute_reward_shape` |
| Reward ordering sensible | ✅ | `test_reward_reflects_weights` |
| KV-cache integration works | ✅ | `test_kv_cache_speedup` |

---

## Phase 1: Context-Dependent Rewards (Gate: Per-archetype weights learned?)

### Objective
Extend reward model to learn archetype-specific weights using F2's synthetic data.

### Deliverables
```
enhancements/reward_modeling/
├── reward_model.py          # UPDATED: ContextualRewardModel
├── training.py              # NEW: Training loop
└── preference_data.py       # UPDATED: Load from synthetic data

tests/test_reward_modeling/
├── test_reward_model.py     # UPDATED
└── test_training.py         # NEW
```

### Implementation
```python
# enhancements/reward_modeling/reward_model.py

class ContextualRewardModel:
    """Reward model with archetype-specific weights."""

    def __init__(
        self,
        phoenix_runner: OptimizedPhoenixRunner,
        num_archetypes: int = 6,
        num_actions: int = 19,
    ):
        self.runner = phoenix_runner
        self.num_archetypes = num_archetypes
        # Weight matrix: [K, 19] - one weight vector per archetype
        self.weights = jnp.zeros((num_archetypes, num_actions))

    def compute_reward(
        self,
        batch,
        embeddings,
        archetype_ids: jnp.ndarray,  # [B] archetype index per user
    ) -> jnp.ndarray:
        """Compute reward using archetype-specific weights."""
        probs = self.get_action_probs(batch, embeddings)  # [B, C, 19]

        # Select weights for each user's archetype
        user_weights = self.weights[archetype_ids]  # [B, 19]

        # Compute reward: R = w[archetype] · P(actions)
        rewards = jnp.einsum('bca,ba->bc', probs, user_weights)
        return rewards


# enhancements/reward_modeling/training.py

from enhancements.data import SyntheticTwitterAdapter  # F2 integration

def train_contextual_rewards(
    reward_model: ContextualRewardModel,
    adapter: SyntheticTwitterAdapter,
    num_epochs: int = 50,
    lr: float = 0.01,
) -> Tuple[jnp.ndarray, List[float]]:
    """Train archetype-specific reward weights."""

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(reward_model.weights)

    history = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_data in adapter.get_preference_batches():
            batch, embeddings, archetype_ids, preferred, rejected = batch_data

            loss, grads = jax.value_and_grad(preference_loss)(
                reward_model.weights,
                reward_model,
                batch,
                embeddings,
                archetype_ids,
                preferred,
                rejected,
            )

            updates, opt_state = optimizer.update(grads, opt_state)
            reward_model.weights = optax.apply_updates(reward_model.weights, updates)
            epoch_loss += loss

        history.append(epoch_loss)

    return reward_model.weights, history
```

### Tests
```python
# tests/test_reward_modeling/test_training.py

def test_contextual_reward_shape():
    """Verify contextual rewards have correct shape."""
    archetype_ids = jnp.array([0, 1, 2, 3])  # 4 users, different archetypes
    rewards = contextual_model.compute_reward(batch, embeddings, archetype_ids)
    assert rewards.shape == (4, num_candidates)

def test_different_archetypes_different_rewards():
    """Verify different archetypes produce different rewards for same content."""
    sports_post_batch = create_sports_post_batch()

    # Sports fan should reward sports content higher than political user
    rewards_sports_fan = contextual_model.compute_reward(
        sports_post_batch, emb, archetype_ids=jnp.array([0])  # Sports fan
    )
    rewards_political = contextual_model.compute_reward(
        sports_post_batch, emb, archetype_ids=jnp.array([1])  # Political L
    )

    assert rewards_sports_fan[0, 0] > rewards_political[0, 0]

def test_training_loss_decreases():
    """Verify loss decreases during training."""
    _, history = train_contextual_rewards(model, adapter, num_epochs=20)
    assert history[-1] < history[0] * 0.5  # At least 50% reduction

def test_archetype_weight_differentiation():
    """Verify learned weights differ across archetypes."""
    weights, _ = train_contextual_rewards(model, adapter, num_epochs=50)

    # Weights should not be identical across archetypes
    for i in range(6):
        for j in range(i+1, 6):
            assert not jnp.allclose(weights[i], weights[j])
```

### Go/No-Go Gate
| Criterion | Required | How to Test |
|-----------|----------|-------------|
| Contextual rewards compute | ✅ | `test_contextual_reward_shape` |
| Archetypes produce different rewards | ✅ | `test_different_archetypes_different_rewards` |
| Training loss decreases | ✅ | `test_training_loss_decreases` |
| Weights differentiate | ✅ | `test_archetype_weight_differentiation` |

---

## Phase 2: Pluralistic Rewards (Gate: Structural recovery > 0.8?)

### Objective
Model rewards as mixture over K value systems, verify structural recovery against F2's ground truth.

### Deliverables
```
enhancements/reward_modeling/
├── pluralistic.py           # NEW: Mixture model
├── structural_recovery.py   # NEW: Verification against ground truth
└── training.py              # UPDATED

tests/test_reward_modeling/
├── test_pluralistic.py      # NEW
└── test_structural.py       # NEW
```

### Implementation
```python
# enhancements/reward_modeling/pluralistic.py

class PluralRewardModel:
    """Pluralistic reward model with K latent value systems."""

    def __init__(
        self,
        phoenix_runner: OptimizedPhoenixRunner,
        num_value_systems: int = 6,
        num_actions: int = 19,
        embedding_dim: int = 64,
    ):
        self.runner = phoenix_runner
        self.K = num_value_systems

        # K reward functions (value systems)
        self.reward_weights = jnp.zeros((num_value_systems, num_actions))

        # Mixture weights predictor: user_embedding -> π(k)
        self.mixture_mlp = hk.Sequential([
            hk.Linear(embedding_dim),
            jax.nn.relu,
            hk.Linear(num_value_systems),
            jax.nn.softmax,
        ])

    def compute_mixture_weights(self, user_embeddings: jnp.ndarray) -> jnp.ndarray:
        """Predict mixture weights π(user) over K value systems. Shape: [B, K]"""
        return self.mixture_mlp(user_embeddings)

    def compute_reward(self, batch, embeddings) -> jnp.ndarray:
        """Compute pluralistic reward as mixture over value systems."""
        probs = self.get_action_probs(batch, embeddings)  # [B, C, 19]

        # Get user embeddings and predict mixture
        user_emb = embeddings.user_embeddings  # [B, D]
        mixture = self.compute_mixture_weights(user_emb)  # [B, K]

        # Reward for each value system
        system_rewards = jnp.einsum('bca,ka->bck', probs, self.reward_weights)  # [B, C, K]

        # Final reward: mixture over systems
        rewards = jnp.einsum('bck,bk->bc', system_rewards, mixture)  # [B, C]
        return rewards

    def get_dominant_system(self, user_embeddings: jnp.ndarray) -> jnp.ndarray:
        """Get dominant value system for each user. Shape: [B]"""
        mixture = self.compute_mixture_weights(user_embeddings)
        return jnp.argmax(mixture, axis=-1)


# enhancements/reward_modeling/structural_recovery.py

from enhancements.data.ground_truth import (
    UserArchetype, get_engagement_probs, ENGAGEMENT_RULES
)

def compute_ground_truth_weights(archetype: UserArchetype) -> jnp.ndarray:
    """Derive ground truth reward weights from F2's engagement rules."""
    weights = jnp.zeros(19)

    # Average over all topics for this archetype
    for topic in ContentTopic:
        probs = get_engagement_probs(archetype, topic)
        action_array = jnp.array(probs.to_array())

        # Weight by engagement value (positive) or disengagement (negative)
        # Positive: favorite, repost, reply, follow, share, etc.
        # Negative: block, mute, report, not_interested
        weights = weights + action_array

    return weights / len(ContentTopic)  # Average

def measure_structural_recovery(
    learned_model: PluralRewardModel,
    adapter: SyntheticTwitterAdapter,
) -> dict:
    """Measure how well learned value systems match ground truth archetypes."""

    results = {
        'system_to_archetype': {},
        'correlations': [],
        'assignment_accuracy': 0.0,
    }

    # Get ground truth weights for each archetype
    gt_weights = {
        arch: compute_ground_truth_weights(arch)
        for arch in UserArchetype
    }

    # Match learned systems to archetypes (Hungarian algorithm)
    correlation_matrix = jnp.zeros((learned_model.K, len(UserArchetype)))
    for k in range(learned_model.K):
        for i, arch in enumerate(UserArchetype):
            corr = jnp.corrcoef(
                learned_model.reward_weights[k],
                gt_weights[arch]
            )[0, 1]
            correlation_matrix = correlation_matrix.at[k, i].set(corr)

    # Find best matching (greedy for simplicity)
    matched = []
    for k in range(learned_model.K):
        best_arch_idx = jnp.argmax(correlation_matrix[k])
        matched.append((k, list(UserArchetype)[best_arch_idx], correlation_matrix[k, best_arch_idx]))

    results['correlations'] = [m[2] for m in matched]
    results['mean_correlation'] = jnp.mean(jnp.array(results['correlations']))

    # Measure user assignment accuracy
    correct = 0
    total = 0
    for user in adapter.get_all_users():
        true_archetype = user.archetype
        predicted_system = learned_model.get_dominant_system(user.embedding)
        # Check if predicted system maps to correct archetype
        if matched[predicted_system][1] == true_archetype:
            correct += 1
        total += 1

    results['assignment_accuracy'] = correct / total

    return results
```

### Tests
```python
# tests/test_reward_modeling/test_pluralistic.py

def test_mixture_weights_sum_to_one():
    """Verify mixture weights form valid distribution."""
    mixture = plural_model.compute_mixture_weights(user_embeddings)
    assert mixture.shape == (batch_size, 6)
    np.testing.assert_allclose(mixture.sum(axis=-1), 1.0, atol=1e-5)

def test_pluralistic_reward_shape():
    """Verify pluralistic rewards have correct shape."""
    rewards = plural_model.compute_reward(batch, embeddings)
    assert rewards.shape == (batch_size, num_candidates)

def test_structural_recovery():
    """Verify learned value systems recover ground truth archetypes."""
    # Train model
    trained_model = train_pluralistic_rewards(model, adapter, epochs=100)

    # Measure recovery
    results = measure_structural_recovery(trained_model, adapter)

    # Key gate: mean correlation > 0.8
    assert results['mean_correlation'] > 0.8

    # Secondary: assignment accuracy > 70%
    assert results['assignment_accuracy'] > 0.7

def test_value_systems_interpretable():
    """Verify each value system has interpretable weights."""
    for k in range(plural_model.K):
        weights = plural_model.reward_weights[k]

        # Block (index 15) should be negative
        assert weights[15] < 0

        # Favorite (index 0) should be positive
        assert weights[0] > 0
```

### Go/No-Go Gate
| Criterion | Required | How to Test |
|-----------|----------|-------------|
| Mixture weights valid | ✅ | `test_mixture_weights_sum_to_one` |
| Rewards compute correctly | ✅ | `test_pluralistic_reward_shape` |
| Structural recovery > 0.8 | ✅ | `test_structural_recovery` |
| Value systems interpretable | ✅ | `test_value_systems_interpretable` |

---

## Phase 3: Causal Verification (Gate: Intervention tests pass?)

### Objective
Verify reward model captures causal relationships, not just correlations, using F2's verification methodology.

### Deliverables
```
enhancements/reward_modeling/
├── causal_verification.py   # NEW: Intervention tests for rewards
└── training.py              # UPDATED: Contrastive objectives

tests/test_reward_modeling/
└── test_causal.py           # NEW
```

### Implementation
```python
# enhancements/reward_modeling/causal_verification.py

from enhancements.verification import CounterfactualTests  # Extend F2's suite

class RewardCausalVerification:
    """Causal verification tests for reward models."""

    def __init__(
        self,
        reward_model: PluralRewardModel,
        adapter: SyntheticTwitterAdapter,
    ):
        self.model = reward_model
        self.adapter = adapter

    def test_block_reduces_reward(self, num_tests: int = 100) -> dict:
        """Test: blocking an author should reduce their posts' reward."""
        passed = 0

        for _ in range(num_tests):
            # Get random user and author
            user, author = self.adapter.get_random_user_author_pair()
            author_post = self.adapter.get_post_by_author(author)

            # Compute reward before block
            batch_before, emb_before = self.adapter.create_batch(user, [author_post])
            reward_before = self.model.compute_reward(batch_before, emb_before)[0, 0]

            # Inject synthetic block (not in training data)
            user_with_block = self.adapter.inject_block(user, author)
            batch_after, emb_after = self.adapter.create_batch(user_with_block, [author_post])
            reward_after = self.model.compute_reward(batch_after, emb_after)[0, 0]

            # Causal test: reward should decrease
            if reward_after < reward_before:
                passed += 1

        return {
            'test': 'block_reduces_reward',
            'passed': passed,
            'total': num_tests,
            'rate': passed / num_tests,
        }

    def test_history_affects_reward(self, num_tests: int = 100) -> dict:
        """Test: matching history should give higher reward than mismatched."""
        passed = 0

        for _ in range(num_tests):
            # Get post with known topic
            post, topic = self.adapter.get_post_with_topic()

            # Create matching history (same topic preference)
            matching_history = self.adapter.create_history_for_topic(topic)

            # Create mismatched history (different topic)
            other_topic = self.adapter.get_different_topic(topic)
            mismatched_history = self.adapter.create_history_for_topic(other_topic)

            # Compute rewards
            batch_match, emb_match = self.adapter.create_batch_with_history(
                matching_history, [post]
            )
            batch_mismatch, emb_mismatch = self.adapter.create_batch_with_history(
                mismatched_history, [post]
            )

            reward_match = self.model.compute_reward(batch_match, emb_match)[0, 0]
            reward_mismatch = self.model.compute_reward(batch_mismatch, emb_mismatch)[0, 0]

            # Causal test: matching history should give higher reward
            if reward_match > reward_mismatch:
                passed += 1

        return {
            'test': 'history_affects_reward',
            'passed': passed,
            'total': num_tests,
            'rate': passed / num_tests,
        }

    def run_all_tests(self) -> dict:
        """Run all causal verification tests."""
        results = {}
        results['block'] = self.test_block_reduces_reward()
        results['history'] = self.test_history_affects_reward()
        results['all_passed'] = (
            results['block']['rate'] > 0.5 and
            results['history']['rate'] > 0.5
        )
        return results
```

### Tests
```python
# tests/test_reward_modeling/test_causal.py

def test_block_intervention():
    """Verify blocking causally reduces reward."""
    verification = RewardCausalVerification(trained_model, adapter)
    result = verification.test_block_reduces_reward(num_tests=50)

    # Gate: >50% of interventions should work
    assert result['rate'] > 0.5

def test_history_intervention():
    """Verify history causally affects reward."""
    verification = RewardCausalVerification(trained_model, adapter)
    result = verification.test_history_affects_reward(num_tests=50)

    # Gate: >50% of interventions should work
    assert result['rate'] > 0.5

def test_causal_vs_correlational():
    """Compare causal model vs correlational baseline."""
    # Train without contrastive loss (correlational)
    corr_model = train_pluralistic_rewards(model, adapter, use_contrastive=False)

    # Train with contrastive loss (causal)
    causal_model = train_pluralistic_rewards(model, adapter, use_contrastive=True)

    # Verify causal model passes more intervention tests
    corr_results = RewardCausalVerification(corr_model, adapter).run_all_tests()
    causal_results = RewardCausalVerification(causal_model, adapter).run_all_tests()

    assert causal_results['block']['rate'] > corr_results['block']['rate']
```

### Go/No-Go Gate
| Criterion | Required | How to Test |
|-----------|----------|-------------|
| Block intervention > 50% | ✅ | `test_block_intervention` |
| History intervention > 50% | ✅ | `test_history_intervention` |
| Causal > Correlational | ✅ | `test_causal_vs_correlational` |

---

## Phase 4: Multi-Objective Analysis (Gate: Pareto frontier computed?)

### Objective
Define multiple objectives (user, platform, society) and compute Pareto frontier.

### Deliverables
```
enhancements/reward_modeling/
├── objectives.py            # NEW: Objective function definitions
├── pareto.py                # NEW: Pareto frontier computation
└── visualization.py         # NEW: Tradeoff visualization

results/f4/
├── pareto_frontier.json
└── tradeoff_analysis.json
```

### Implementation
```python
# enhancements/reward_modeling/objectives.py

from enhancements.data.ground_truth import get_engagement_probs

class ObjectiveFunctions:
    """Multi-stakeholder objective functions."""

    def __init__(self, adapter: SyntheticTwitterAdapter):
        self.adapter = adapter

    def user_engagement(self, user, recommendations) -> float:
        """User wants content they engage with positively."""
        total = 0
        for post in recommendations:
            probs = get_engagement_probs(user.archetype, post.topic)
            # Positive engagement
            total += probs.favorite + probs.repost + probs.reply
        return total / len(recommendations)

    def user_satisfaction(self, user, recommendations) -> float:
        """User engagement minus discomfort."""
        engagement = self.user_engagement(user, recommendations)

        discomfort = 0
        for post in recommendations:
            probs = get_engagement_probs(user.archetype, post.topic)
            discomfort += probs.block_author + probs.mute_author + probs.not_interested

        return engagement - discomfort / len(recommendations)

    def platform_utility(self, all_users, all_recommendations) -> float:
        """Platform wants total engagement across all users."""
        total = sum(
            self.user_engagement(u, recs)
            for u, recs in zip(all_users, all_recommendations)
        )
        return total

    def society_utility(self, all_users, all_recommendations) -> float:
        """Society wants low polarization (cross-group hostility)."""
        cross_blocks = 0

        for user, recs in zip(all_users, all_recommendations):
            for post in recs:
                # Count cross-group blocks (L blocking R content or vice versa)
                if self._is_cross_group(user.archetype, post.topic):
                    probs = get_engagement_probs(user.archetype, post.topic)
                    cross_blocks += probs.block_author

        # Negative because lower is better
        return -cross_blocks

    def _is_cross_group(self, archetype, topic) -> bool:
        """Check if this is a cross-group interaction (e.g., political L seeing R)."""
        cross_pairs = [
            (UserArchetype.POLITICAL_L, ContentTopic.POLITICS_R),
            (UserArchetype.POLITICAL_R, ContentTopic.POLITICS_L),
        ]
        return (archetype, topic) in cross_pairs


# enhancements/reward_modeling/pareto.py

def compute_pareto_frontier(
    reward_model: PluralRewardModel,
    adapter: SyntheticTwitterAdapter,
    objectives: ObjectiveFunctions,
    num_weightings: int = 20,
) -> List[dict]:
    """Compute Pareto frontier by varying objective weights."""

    frontier = []

    for alpha in np.linspace(0, 1, num_weightings):
        # Train model with weighted objective
        # R = alpha * user_engagement + (1-alpha) * society_utility
        model = train_with_weighted_objective(
            reward_model, adapter,
            user_weight=alpha,
            society_weight=(1 - alpha),
        )

        # Evaluate on all objectives
        recs = generate_recommendations(model, adapter)

        point = {
            'alpha': alpha,
            'user_engagement': objectives.user_engagement_total(recs),
            'user_satisfaction': objectives.user_satisfaction_total(recs),
            'platform_utility': objectives.platform_utility(recs),
            'society_utility': objectives.society_utility(recs),
        }
        frontier.append(point)

    return frontier
```

### Tests
```python
# tests/test_reward_modeling/test_multi_objective.py

def test_objectives_compute():
    """Verify all objective functions compute correctly."""
    objs = ObjectiveFunctions(adapter)

    user = adapter.get_random_user()
    recs = adapter.get_random_posts(10)

    eng = objs.user_engagement(user, recs)
    sat = objs.user_satisfaction(user, recs)

    assert 0 <= eng <= 1
    assert sat <= eng  # Satisfaction <= engagement (discomfort subtracted)

def test_pareto_frontier_shape():
    """Verify Pareto frontier has correct structure."""
    frontier = compute_pareto_frontier(model, adapter, objectives, num_weightings=10)

    assert len(frontier) == 10
    assert all('user_engagement' in p for p in frontier)
    assert all('society_utility' in p for p in frontier)

def test_pareto_tradeoff_exists():
    """Verify there's a tradeoff between engagement and society."""
    frontier = compute_pareto_frontier(model, adapter, objectives, num_weightings=20)

    # Extract engagement and society values
    eng = [p['user_engagement'] for p in frontier]
    soc = [p['society_utility'] for p in frontier]

    # Should see tradeoff: max engagement != max society
    max_eng_idx = np.argmax(eng)
    max_soc_idx = np.argmax(soc)

    assert max_eng_idx != max_soc_idx  # Different optimal points

def test_quantified_tradeoff():
    """Compute specific tradeoff ratio."""
    frontier = compute_pareto_frontier(model, adapter, objectives, num_weightings=50)

    # Find balanced point
    balanced = frontier[len(frontier) // 2]
    max_eng_point = max(frontier, key=lambda p: p['user_engagement'])

    eng_cost = (max_eng_point['user_engagement'] - balanced['user_engagement']) / max_eng_point['user_engagement']
    soc_gain = (balanced['society_utility'] - max_eng_point['society_utility']) / abs(max_eng_point['society_utility'])

    print(f"Trading {eng_cost:.1%} engagement for {soc_gain:.1%} society improvement")
```

### Go/No-Go Gate
| Criterion | Required | How to Test |
|-----------|----------|-------------|
| Objectives compute | ✅ | `test_objectives_compute` |
| Pareto frontier generates | ✅ | `test_pareto_frontier_shape` |
| Tradeoff exists | ✅ | `test_pareto_tradeoff_exists` |
| Tradeoff quantified | ✅ | `test_quantified_tradeoff` |

---

## Phase 5: Stakeholder Utility Analysis (Gate: Impact table generated?)

### Objective
Attribute objectives to stakeholders, analyze who wins/loses under different policies.

### Deliverables
```
enhancements/reward_modeling/
├── stakeholders.py          # NEW: Stakeholder definitions
├── policy_analysis.py       # NEW: Policy impact analysis
└── visualization.py         # UPDATED: Impact tables

results/f4/
├── stakeholder_impact.json
└── policy_recommendations.json
```

### Implementation
```python
# enhancements/reward_modeling/stakeholders.py

@dataclass
class Stakeholder:
    """A stakeholder with utility function."""
    name: str
    utility_fn: Callable
    description: str

class StakeholderRegistry:
    """Registry of all stakeholders."""

    def __init__(self, adapter: SyntheticTwitterAdapter):
        self.adapter = adapter
        self.objectives = ObjectiveFunctions(adapter)

        self.stakeholders = {
            'user_L': Stakeholder(
                name='Political L Users',
                utility_fn=lambda recs: self._user_group_utility(recs, UserArchetype.POLITICAL_L),
                description='Left-leaning users want engaging left content',
            ),
            'user_R': Stakeholder(
                name='Political R Users',
                utility_fn=lambda recs: self._user_group_utility(recs, UserArchetype.POLITICAL_R),
                description='Right-leaning users want engaging right content',
            ),
            'platform': Stakeholder(
                name='Platform',
                utility_fn=self.objectives.platform_utility,
                description='Platform wants total engagement + retention',
            ),
            'society': Stakeholder(
                name='Society',
                utility_fn=self.objectives.society_utility,
                description='Society wants low polarization',
            ),
        }


# enhancements/reward_modeling/policy_analysis.py

def analyze_policy_impact(
    policies: Dict[str, PluralRewardModel],
    stakeholders: StakeholderRegistry,
    adapter: SyntheticTwitterAdapter,
) -> pd.DataFrame:
    """Analyze impact of different policies on each stakeholder."""

    results = []

    for policy_name, model in policies.items():
        # Generate recommendations under this policy
        recs = generate_recommendations(model, adapter)

        row = {'policy': policy_name}
        for stake_name, stakeholder in stakeholders.stakeholders.items():
            utility = stakeholder.utility_fn(recs)
            row[stake_name] = utility

        results.append(row)

    df = pd.DataFrame(results)

    # Normalize to show % change from baseline
    baseline = df[df['policy'] == 'max_engagement'].iloc[0]
    for col in df.columns[1:]:
        df[f'{col}_pct'] = (df[col] - baseline[col]) / abs(baseline[col]) * 100

    return df
```

### Tests
```python
# tests/test_reward_modeling/test_stakeholders.py

def test_stakeholder_utilities_compute():
    """Verify all stakeholder utilities compute."""
    registry = StakeholderRegistry(adapter)
    recs = generate_recommendations(model, adapter)

    for name, stakeholder in registry.stakeholders.items():
        utility = stakeholder.utility_fn(recs)
        assert np.isfinite(utility)

def test_policy_impact_analysis():
    """Verify policy impact table generates correctly."""
    policies = {
        'max_engagement': train_with_objective(model, 'engagement'),
        'max_society': train_with_objective(model, 'society'),
        'balanced': train_with_objective(model, 'balanced'),
    }

    impact = analyze_policy_impact(policies, stakeholders, adapter)

    assert len(impact) == 3
    assert 'user_L' in impact.columns
    assert 'society' in impact.columns

def test_stakeholder_conflicts():
    """Verify stakeholder conflicts are captured."""
    impact = analyze_policy_impact(policies, stakeholders, adapter)

    # Max engagement should hurt society
    max_eng = impact[impact['policy'] == 'max_engagement'].iloc[0]
    assert max_eng['society_pct'] < 0  # Society worse off

    # Max society should hurt platform
    max_soc = impact[impact['policy'] == 'max_society'].iloc[0]
    assert max_soc['platform_pct'] < 0  # Platform worse off

def test_balanced_policy_exists():
    """Verify balanced policy improves all stakeholders vs worst case."""
    impact = analyze_policy_impact(policies, stakeholders, adapter)

    balanced = impact[impact['policy'] == 'balanced'].iloc[0]

    # Balanced should be positive for society (vs max_engagement baseline)
    # and not terrible for platform
    assert balanced['society_pct'] > 0
    assert balanced['platform_pct'] > -20  # Not more than 20% worse
```

### Go/No-Go Gate
| Criterion | Required | How to Test |
|-----------|----------|-------------|
| Stakeholder utilities compute | ✅ | `test_stakeholder_utilities_compute` |
| Impact table generates | ✅ | `test_policy_impact_analysis` |
| Conflicts captured | ✅ | `test_stakeholder_conflicts` |
| Balanced policy exists | ✅ | `test_balanced_policy_exists` |

---

## Phase 6: Game-Theoretic Analysis (Optional) (Gate: Equilibrium characterized?)

### Objective
Model stakeholders as strategic agents, analyze equilibria.

### Deliverables
```
enhancements/reward_modeling/
├── game_theory.py           # NEW: Game-theoretic models
├── equilibrium.py           # NEW: Equilibrium computation
└── mechanism_design.py      # NEW: Mechanism analysis

results/f4/
├── equilibrium_analysis.json
└── mechanism_insights.json
```

### Implementation (Sketch)
```python
# enhancements/reward_modeling/game_theory.py

class RecommendationGame:
    """Game-theoretic model of recommendation ecosystem."""

    def __init__(self, stakeholders: StakeholderRegistry):
        self.stakeholders = stakeholders

    def user_best_response(self, platform_policy, user_type):
        """Given platform policy, what's user's best response?"""
        # User can: engage, reduce engagement, or leave
        options = ['full_engage', 'selective_engage', 'leave']
        utilities = [
            self._user_utility(platform_policy, user_type, opt)
            for opt in options
        ]
        return options[np.argmax(utilities)]

    def find_nash_equilibrium(self):
        """Find Nash equilibrium of the game."""
        # Iterative best response
        platform_policy = 'max_engagement'  # Start
        user_responses = {}

        for _ in range(100):  # Max iterations
            # Users best respond to platform
            for user_type in UserArchetype:
                user_responses[user_type] = self.user_best_response(
                    platform_policy, user_type
                )

            # Platform best responds to users
            new_policy = self.platform_best_response(user_responses)

            if new_policy == platform_policy:
                break  # Equilibrium found
            platform_policy = new_policy

        return {
            'platform_policy': platform_policy,
            'user_responses': user_responses,
            'is_equilibrium': new_policy == platform_policy,
        }
```

### Go/No-Go Gate (Optional Phase)
| Criterion | Required | How to Test |
|-----------|----------|-------------|
| Game model defined | ⚠️ | Code compiles |
| Best responses compute | ⚠️ | Unit tests |
| Equilibrium found | ⚠️ | `find_nash_equilibrium` terminates |
| Insights documented | ⚠️ | Results JSON generated |

---

## Summary: F2 Gates

| Phase | Gate Criterion | Required | Target |
|-------|---------------|----------|--------|
| 0 | Can compute rewards | ✅ | Basic reward model works |
| 1 | Per-archetype weights learned | ✅ | Weights differentiate |
| 2 | Structural recovery > 0.8 | ✅ | Pluralistic structure recovered |
| 3 | Intervention tests > 50% | ✅ | Causal understanding verified |
| 4 | Pareto frontier computed | ✅ | Tradeoffs quantified |
| 5 | Impact table generated | ✅ | Stakeholder analysis complete |
| 6 | Equilibrium characterized | ⚠️ | (Optional) Game theory insights |

## F2 Exit Points

| After Phase | Deliverable | Value |
|-------------|-------------|-------|
| 0 | Basic reward model | Table stakes |
| 1 | Context-dependent rewards | Ambitious |
| 2 | Pluralistic rewards + verification | Novel (structural) |
| 3 | + Causal verification | Novel (causal) |
| 4 | + Pareto frontier | Multi-stakeholder |
| 5 | + Stakeholder impact | Full framework |
| 6 | + Game theory | Research contribution |

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

## F1: JAX Optimization
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

## F2: RL Reward Modeling (Pluralistic + Multi-Stakeholder)
| Phase | Gate Criterion | Required |
|-------|---------------|----------|
| 0 | Basic rewards compute, KV-cache integrated | ✅ |
| 1 | Context-dependent (per-archetype) weights | ✅ |
| 2 | Pluralistic structural recovery > 0.8 | ✅ |
| 3 | Causal intervention tests > 50% | ✅ |
| 4 | Pareto frontier computed | ✅ |
| 5 | Stakeholder impact table generated | ✅ |
| 6 | Game-theoretic equilibrium (optional) | ⚠️ |

## F1: Multimodal Retrieval
| Phase | Gate Criterion | Required |
|-------|---------------|----------|
| 0 | CLIP encodes images/text | ✅ |
| 1 | Multimodal tower works | ✅ |
| 2 | Cross-modal retrieval works | ✅ |
| 3 | R@10 > 30% on COCO | ✅ |
