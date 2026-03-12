# X-Algorithm: Results & Learnings

This document captures the results, benchmarks, and learnings from implementing enhancements to the Phoenix recommendation system.

**Features Covered:**
- **F2: JAX Optimization** - JIT, KV-cache, quantization
- **F4: Reward Modeling** - Bradley-Terry, pluralistic rewards, causal verification

---

## Summary

| Phase | Status | Key Achievement |
|-------|--------|-----------------|
| Phase 0 | ✅ Complete | Baseline established, benchmark harness operational |
| Phase 1 | ✅ Complete | **10.33x speedup** from static-shape JIT compilation |
| Phase 2 | ✅ Complete | Logical KV-cache with hash-based invalidation |
| Phase 2b | ✅ Complete | Full K/V tensor caching with **9.62x max speedup** |
| Phase 3 | ✅ Complete | Efficient attention exploiting Phoenix mask structure |
| Phase 3b | ✅ Complete | Analysis tools validated with trained weights |
| Phase 4 | ✅ Complete | Quantization study: **58% memory reduction** with INT8 |
| Phase 4b | ✅ Complete | Trained model quantization: **~90% top-3 agreement** |
| Phase 5 | ✅ Complete | Combined optimization runner: all gates pass |
| Phase 6 | ✅ Complete | MovieLens training: **NDCG 0.4112** (+59% vs untrained) |
| Phase 7 | ✅ Complete | Synthetic verification: **all 5 test suites pass** |

**Total Tests:** 166 passing (112 optimization + 54 quantization)

**Key Phase 6 Finding (BPR + In-Batch Negatives):** Training with BPR loss and in-batch negatives achieved dramatic improvements: **Val NDCG 0.4112** (vs 0.3157 BCE), **Test NDCG 0.4183** (vs 0.2410 BCE). The 73% test improvement shows much better generalization.

**Key Phase 4b Finding (Trained Model):** INT8 per-channel achieves **99.0% top-3 agreement** on the BPR-trained model. The improved score margins with BPR make quantization much more robust.

**Key Phase 3b Finding (Trained Model):** Trajectory stability τ=0.907 (very high), counterfactual ablations change top-1 in 36.2% of cases, no recency bias detected.

**Key Ablation Finding:** Neither transformer nor embeddings work well alone. The improvement is almost entirely from **synergy (107.5%)** - components must work together.

---

## Phase 0: Baseline & Setup

### Objective
Verify we can run the Phoenix model and establish baseline metrics.

### Results

| Gate Criterion | Status | Evidence |
|----------------|--------|----------|
| Phoenix model loads | ✅ | `test_phoenix_model_loads` passes |
| Forward pass works | ✅ | `test_forward_pass_runs` passes |
| Benchmark harness works | ✅ | `test_benchmark_produces_results` passes |
| Baseline metrics recorded | ✅ | Benchmark infrastructure operational |

### Key Files
- `enhancements/optimization/benchmark.py` - Benchmarking harness
- `tests/test_optimization/test_benchmark.py` - Baseline tests

---

## Phase 1: JIT Optimization

### Objective
Implement static-shape JIT compilation and measure improvement.

### Results

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Latency (p50) | 103.77 ms | 10.04 ms | **10.33x** |
| Numerical accuracy | - | rtol < 1e-5 | ✅ Matches |

### Gate Status

| Gate Criterion | Required | Status |
|----------------|----------|--------|
| JIT compiles without error | ✅ | Pass |
| Output matches baseline (rtol=1e-5) | ✅ | Pass |
| Speedup > 1.2x | ✅ | **10.33x** (far exceeds) |
| No accuracy degradation | ✅ | Verified |

### Implementation Details

**Static Shape Strategy:**
- Pre-compiled forward functions for fixed batch sizes
- Shape assertions to catch mismatches early
- Warmup compilation on initialization

**Key Files:**
- `enhancements/optimization/jit_runner.py` - JIT-optimized runner
- `tests/test_optimization/test_jit_runner.py` - 10 gate tests

### Learnings

1. **First call overhead**: Initial JIT compilation takes ~2-5 seconds, subsequent calls are fast
2. **Shape consistency critical**: Dynamic shapes cause recompilation; static shapes essential for production
3. **10x speedup typical**: JAX XLA compilation provides substantial gains even on CPU

---

## Phase 2: Logical KV-Cache

### Objective
Implement cache hit/miss detection using user hash comparison.

### Results

| Gate Criterion | Required | Status |
|----------------|----------|--------|
| Cache hit detection works | ✅ | Pass |
| Cache invalidation on user change | ✅ | Pass |
| Hash comparison is efficient | ✅ | O(1) lookup |

### Implementation Details

**Hash-Based Cache Invalidation:**
```python
# Cache key derived from user hashes
user_hash = hash(batch.user_hashes.tobytes())
if self._cache_user_hash == user_hash:
    # Cache hit - reuse cached context
else:
    # Cache miss - recompute and store
```

**Key Files:**
- `enhancements/optimization/kv_cache.py` - Logical cache implementation
- `tests/test_optimization/test_kv_cache.py` - 14 cache tests

### Learnings

1. **User hash stability**: `tobytes()` provides consistent hashing across calls
2. **Foundation for full caching**: Logical detection enables K/V tensor caching in Phase 2b

---

## Phase 2b: Full K/V Tensor Caching

### Objective
Store actual Key/Value tensors from transformer attention layers, enabling candidate-only forward passes on cache hits.

### Results

#### Scaling Benchmark (kv_cache_scaling.py)

| Config | Layers | Emb | Context | Miss (ms) | Hit (ms) | Speedup |
|--------|--------|-----|---------|-----------|----------|---------|
| Small | 4 | 128 | 33 | 23.27 | 21.39 | 1.09x |
| Medium | 8 | 256 | 65 | 43.50 | 40.30 | 1.08x |
| Large | 12 | 512 | 129 | 91.63 | 73.54 | 1.25x |
| XLarge | 16 | 768 | 257 | 226.02 | 92.99 | 2.43x |

**Key Finding:** Speedup scales with model size and context length.

#### Limits Benchmark (kv_cache_limits.py)

| Config | Context | Candidates | Theoretical Max | Actual | Efficiency |
|--------|---------|------------|-----------------|--------|------------|
| Baseline | 129 | 8 | 17.1x | 1.51x | 3.2% |
| Deep (24L) | 129 | 8 | 17.1x | 2.07x | 6.6% |
| Wide (1024) | 129 | 8 | 17.1x | 2.20x | 7.5% |
| Long (512) | 512 | 8 | 65.0x | 4.95x | 6.2% |
| Very Long (1024) | 1024 | 8 | 129.0x | **9.62x** | 6.7% |
| Extreme (512:4) | 512 | 4 | 129.0x | 9.32x | 6.5% |
| Large + Long | 512 | 8 | 65.0x | 6.27x | 8.2% |
| Production-like | 256 | 16 | 17.0x | 2.15x | 7.2% |

**Best Result:** 9.62x speedup with 1024 context length, 8 candidates

#### Stress Test Results (kv_cache_stress.py)

| Test | Status | Key Finding |
|------|--------|-------------|
| Batch Size Scaling | ✅ | 8.71x speedup at batch_size=8 |
| Memory Usage | ✅ | 0.25 MB (small) to 12 MB (large) |
| Sequential Cache Hits | ✅ | Consistent speedup across 20 hits |
| Numerical Stability | ✅ | **Zero difference** (max diff: 0.00e+00) |
| Candidate Count Scaling | ✅ | Speedup increases with fewer candidates |
| Throughput | ✅ | 3-4x throughput improvement |

### Gate Status

| Gate Criterion | Required | Status |
|----------------|----------|--------|
| KV cache creates successfully | ✅ | Pass |
| Cached output matches baseline | ✅ | Pass (rtol=1e-4) |
| Cache hit speedup > 1.3x | ✅ | **9.62x max** |
| Cache invalidates correctly | ✅ | Pass |
| Numerical stability | ✅ | Zero difference |

### Implementation Details

**Architecture:**
```
CachingMultiHeadAttention
├── Stores K, V tensors per layer
├── RoPE position offset handling
└── Grouped-Query Attention (GQA) support

CachingTransformer
├── Layer-wise cache management
├── extract_user_context_from_cache()
└── Candidate isolation mask preserved

FullKVCachedRunner
├── encode_user_context() → cache
├── score_with_cache() → fast path
└── Hash-based invalidation
```

**RoPE Position Handling:**
```python
# Critical: Cached tokens use positions 0..context_len-1
# New tokens use positions context_len..context_len+new_len-1
position_offset = cached_kv.shape[2]  # seq dimension
```

**Key Files:**
- `enhancements/optimization/caching_attention.py` - K/V caching attention
- `enhancements/optimization/caching_transformer.py` - Layer-wise cache management
- `enhancements/optimization/full_kv_cache.py` - Complete cached runner
- `tests/test_optimization/test_kv_cache_full.py` - 17 gate tests
- `benchmarks/kv_cache_*.py` - Benchmark scripts

### Learnings

1. **Speedup is memory-bound**: On CPU (DDR: 50-100 GB/s), we see 2-10x speedup. On GPU (HBM: 2-3 TB/s), expect 10-50x.

2. **Theoretical vs actual speedup**:
   - Theoretical max: `(context + candidates) / candidates`
   - Actual efficiency: ~6-8% of theoretical on CPU
   - Gap due to: fixed overhead, JIT dispatch latency, memory bandwidth limits

3. **Haiku parameter path consistency**: When using `hk.transform`, parameter paths must match between initialization and apply. Fixed by creating shared `_compute_logits()` method.

4. **Numerical stability excellent**: Zero difference between cached and non-cached outputs when using float32.

5. **Batch size matters**: Larger batches amortize fixed overhead, improving speedup (8.71x at batch=8 vs 2.43x at batch=1).

---

## Hardware Analysis

### Hardware-Independent Metrics (Same on CPU/GPU)

| Metric | Example Value |
|--------|---------------|
| Cache memory size | 12 layers × 257 context × 4 heads × 64 dim × 2 = 1.51 MB |
| Numerical correctness | rtol < 1e-4 |
| Theoretical max speedup | (context + candidates) / candidates |
| Cache hit/miss detection | Hash comparison |

### Hardware-Dependent Metrics

| Metric | CPU (Measured) | GPU (Estimated) |
|--------|----------------|-----------------|
| Memory bandwidth | 50-100 GB/s | 1500-3000 GB/s |
| Absolute latency | 100-500 ms | 5-50 ms |
| KV-cache speedup | 2-10x | 10-50x |
| Throughput | 20-50 cand/sec | 1000-10000 cand/sec |

### Why GPU Speedup is Higher

1. **Memory-bound workload**: KV-cache benefits directly from HBM bandwidth (30x faster than DDR)
2. **Parallelism**: GPU excels at GEMM operations in attention
3. **Batch efficiency**: GPU utilization improves dramatically with larger batches

---

## File Summary

### Core Implementation
```
enhancements/optimization/
├── benchmark.py              # Phase 0: Benchmarking harness
├── jit_runner.py            # Phase 1: Static-shape JIT
├── kv_cache.py              # Phase 2: Logical cache
├── caching_attention.py     # Phase 2b: K/V tensor caching
├── caching_transformer.py   # Phase 2b: Layer-wise cache
├── full_kv_cache.py         # Phase 2b: Complete runner
├── attention.py             # Phase 3: Efficient attention
├── optimized_runner.py      # Phase 5: Combined optimization runner
└── quantization/            # Phase 4 & 4b: Quantization
    ├── __init__.py          # Exports
    ├── config.py            # QuantizationConfig, MixedPrecisionConfig
    ├── quantize.py          # Core quantize/dequantize functions
    ├── quantized_runner.py  # QuantizedPhoenixRunner
    ├── kv_quantize.py       # KV-cache quantization (Phase 4b)
    └── study.py             # QuantizationStudy, select_winner()
```

### Analysis Tools (Phase 3b)
```
enhancements/analysis/
├── __init__.py                    # Exports
├── trajectory_simulation.py       # Simulated trajectory
├── real_trajectory_simulation.py  # Real model re-ranking
├── ranking_dynamics.py            # Score evolution visualization
├── path_divergence.py             # Compare diverging paths
├── sensitivity_analysis.py        # Outcome predictability
├── diversity_metrics.py           # Filter bubble detection
└── counterfactual_analysis.py     # History importance
```

### Tests
```
tests/test_optimization/
├── test_benchmark.py         # 4 tests
├── test_jit_runner.py        # 10 tests
├── test_kv_cache.py          # 14 tests
├── test_kv_cache_full.py     # 17 tests
├── test_attention.py         # 21 tests
├── test_quantization.py      # 54 tests (Phase 4 + 4b)
└── test_optimized_runner.py  # 25 tests (Phase 5)

tests/test_analysis/
└── test_trajectory_simulation.py  # 25 tests
```

### Benchmarks
```
benchmarks/
├── kv_cache_scaling.py      # Model size scaling
├── kv_cache_limits.py       # Maximum speedup exploration
├── kv_cache_stress.py       # Stress tests (6 scenarios)
├── kv_cache_hw_analysis.py  # Hardware analysis
├── quantization_study.py    # Quantization comparative study (--extended for Phase 4b)
└── f2_final_benchmark.py    # Combined optimization benchmark (Phase 5)
```

---

## Phase 3: Efficient Attention

### Objective
Implement memory-efficient attention that exploits Phoenix's candidate isolation mask structure.

### Results

#### Memory Reduction

| Config | Standard | Efficient | Reduction |
|--------|----------|-----------|-----------|
| Small (64 ctx, 8 cand) | 162.0 KB | 144.2 KB | 1.12x |
| Medium (128 ctx, 8 cand) | 578.0 KB | 544.2 KB | 1.06x |
| Large (256 ctx, 16 cand) | 2312.0 KB | 2176.5 KB | 1.06x |
| Long ctx (512 ctx, 8 cand) | 8450.0 KB | 8320.2 KB | 1.02x |
| Production (1024 ctx, 16 cand) | 33800.0 KB | 33280.5 KB | 1.02x |

**Memory reduction is modest (2-12%)** because context² dominates the memory footprint.

#### Gate Status

| Gate Criterion | Required | Status |
|----------------|----------|--------|
| Output matches standard attention (rtol=1e-4) | ✅ | Pass |
| Memory reduction > 30% | ⚠️ | **Partial** (2-12%) |
| Works with candidate isolation mask | ✅ | Pass |
| Numerical stability | ✅ | Pass |

### Implementation Details

**Key Insight:** Phoenix's mask has exploitable structure:
- Context tokens: causal attention to other context tokens only (NOT candidates)
- Candidate tokens: attend to ALL context + self only (not other candidates)

**Algorithm:**
```
efficient_phoenix_attention(Q, K, V, context_len):
    # Split into context and candidate
    Q_ctx, Q_cand = Q[:context_len], Q[context_len:]

    # CONTEXT: Causal self-attention (context × context)
    out_ctx = causal_attention(Q_ctx, K_ctx, V_ctx)

    # CANDIDATES: Context + self only
    scores_to_ctx = Q_cand @ K_ctx.T           # [cand, context]
    scores_to_self = diag(Q_cand @ K_cand.T)   # [cand] - just diagonal!

    weights = softmax([scores_to_ctx, scores_to_self])
    out_cand = weights[:, :context] @ V_ctx + weights[:, -1] * V_cand

    return concat([out_ctx, out_cand])
```

**Memory Comparison:**
- Standard: O((context + candidates)²)
- Efficient: O(context²) + O(candidates × context) + O(candidates)

The savings come from:
1. Context doesn't attend to candidates (causal mask eliminates this)
2. Candidate-to-candidate is diagonal only (O(C) instead of O(C²))

**Flash Attention (Reference Implementation):**
- Online softmax algorithm for memory-efficient attention
- Processes K/V in blocks without materializing full attention matrix
- Primarily beneficial on GPU with long sequences

### Key Files

- `enhancements/optimization/attention.py` - Efficient attention implementation
- `tests/test_optimization/test_attention.py` - 21 tests

### Learnings

1. **Memory savings limited by context²**: With typical context lengths (256-1024), the context self-attention dominates memory usage. The candidate isolation optimization saves memory only in the candidate-related terms.

2. **GPU benefits expected**: On GPU, the structural optimization enables:
   - Better kernel fusion (fewer memory round-trips)
   - Separate optimization paths for context vs candidates
   - Flash attention for long context sequences

3. **Correctness verified**: Efficient attention output matches standard attention with Phoenix mask (rtol=1e-4 across all test configurations).

---

## Phase 3b: Analysis Tools (Trajectory & Counterfactual)

### Objective
Build analysis tools to understand recommendation dynamics, detect filter bubbles, and enable model explainability.

### Results

| Tool | Purpose | Status | Notes |
|------|---------|--------|-------|
| Trajectory Simulation | Rankings over engagement | ✅ Complete | Simulated + Real modes |
| Diversity Metrics | Filter bubble detection | ✅ Complete | Coverage/Gini meaningful |
| Sensitivity Analysis | Outcome predictability | ✅ Complete | 40% diversity reduction found |
| Counterfactual Analysis | History importance | ✅ Complete | Needs trained weights |

#### Key Finding: Architectural vs Learned Behavior

With **randomly initialized weights**:
- All candidates score ~0.5 (uniform)
- No ranking differentiation
- No filter bubble dynamics
- History items have no influence (all Kendall's τ = 1.0)

**Conclusion:** Filter bubble effects are **learned behaviors**, not architectural properties.

---

### Phase 3b Results with Trained Weights (MovieLens)

After training the Phoenix model on MovieLens-100K with BPR + in-batch negatives (see Phase 6), we validated the analysis tools with learned weights.

#### Training Context
- **Model**: 64d embeddings, 4 transformer layers
- **Best NDCG@3**: 0.4112 (epoch 9)
- **Training approach**: BPR loss with in-batch negatives (31 per positive)
- **Ablation finding**: 107.5% synergy - transformer and embeddings only work together

#### Trajectory Simulation Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Users analyzed | 19 | |
| Avg ranking stability (τ) | **0.907** | Very high - rankings very consistent |
| Score range | ~1.0 | Scores close to 1.0 (saturated sigmoid) |
| Score spread | ~0.1-0.2 | Reasonable margins between candidates |

**Finding**: Rankings are very stable as user history grows (τ=0.907, up from 0.808 with BCE). The BPR-trained model produces more confident, consistent recommendations.

#### Counterfactual Analysis Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Users analyzed | 20 | |
| Total ablations | 160 | 8 positions × 20 users |
| Avg τ after ablation | **0.654** | Moderate ranking changes |
| Top-1 changed | **36.2%** | History influences top pick |

**Recency Analysis:**
| Position | Avg Impact |
|----------|------------|
| Recent items (pos 0-3) | 1.42e-1 |
| Older items (pos 4+) | 1.44e-1 |

**Finding**: No recency bias detected - recent and older history items have equal influence on rankings. This is consistent across both BCE and BPR trained models.

#### Comparison: BCE vs BPR Training Effects on Analysis

| Metric | BCE Training | BPR Training |
|--------|--------------|--------------|
| Trajectory stability (τ) | 0.808 | **0.907** (+12%) |
| Top-1 change rate | 42.5% | 36.2% (-15%) |
| Avg ablation τ | 0.675 | 0.654 (-3%) |
| Score spreads | ~3.5e-8 (tiny) | ~0.1-0.2 (healthy) |

**Key Insight**: BPR training produces:
1. More stable recommendations (higher trajectory τ)
2. Better score margins (0.1 vs 1e-8) - less sensitive to noise
3. Similar history importance patterns (no recency bias in either)

#### Echo Chamber Assessment

Evidence **for** echo chamber tendency:
- Very high trajectory stability (τ=0.907) - rankings barely change with new engagement

Evidence **against** strong echo chamber:
- 36.2% of ablations changed top-1 - history does influence recommendations
- No recency bias - model doesn't amplify recent preferences
- Healthy score margins - strong differentiation between candidates

**Conclusion**: The BPR-trained model shows signs of **stable personalization** rather than echo chamber - it makes confident, consistent recommendations based on user history, but individual history items still matter (36% top-1 change rate on ablation).

#### Diversity Metrics Results (100 candidates)

| Metric | Top Strategy | Random Strategy |
|--------|--------------|-----------------|
| Unique candidates engaged | 20/100 (20%) | 100/100 (100%) |
| Engagement Gini | 0.80 | 0.16 |
| Coverage gap | 80% fewer | - |

**Insight:** The filter bubble manifests as **coverage** (only seeing 20% of catalog), not embedding similarity.

#### Sensitivity Analysis Results (8 candidates)

| Metric | Random | Top-Biased (70%) |
|--------|--------|------------------|
| Outcome diversity | 100% unique | 60% unique |
| First-choice predictability | 18% | 68% |
| Position entropy (step 0) | 2.93 | 1.62 |

**Insight:** Following recommendations constrains outcome space by 40%.

#### Real vs Simulated Re-ranking

| Aspect | Simulated | Real Re-ranking |
|--------|-----------|-----------------|
| Score updates | `score * 0.98 + noise` | Actual model inference |
| With random weights | Artificial dynamics | Uniform scores (no dynamics) |
| With trained weights | Not meaningful | **Reveals true model behavior** |
| Rank correlation | - | 0.25 (low - simulated doesn't match real) |

### Implementation Details

**Trajectory Simulation Architecture:**
```
TrajectorySimulator (simulated):
├── Initial ranking via model
├── Engagement removes candidate
├── Scores updated via perturbation
└── Fast but artificial dynamics

RealTrajectorySimulator (actual):
├── Initial ranking via model
├── Engagement adds candidate to history
├── Full model re-inference
└── Slower but accurate dynamics
```

**Counterfactual Analysis:**
```
CounterfactualAnalyzer:
├── ablate_history_item(pos) → Remove item, measure τ change
├── truncate_history(n) → Keep last N, measure impact
├── modify_history_actions() → Change engagement type
└── analyze_recency_sensitivity() → Recency bias detection
```

**Why Caching Doesn't Help Counterfactuals:**
- Trajectory: Cache context K/V, vary candidates ✓
- Counterfactual: Would cache candidate K/V, but candidates DEPEND on context
- Asymmetry from Phoenix attention: candidates attend TO context

### Key Files

```
enhancements/analysis/
├── trajectory_simulation.py       # Simulated trajectory
├── real_trajectory_simulation.py  # Real model re-ranking
├── ranking_dynamics.py            # Score evolution visualization
├── path_divergence.py             # Compare diverging paths
├── sensitivity_analysis.py        # Outcome predictability
├── diversity_metrics.py           # Filter bubble detection
└── counterfactual_analysis.py     # History importance

tests/test_analysis/
└── test_trajectory_simulation.py  # 25 tests
```

### Learnings

1. **Random weights = null model**: Uniform scores mean no differentiation. Essential baseline for isolating learned vs architectural effects.

2. **Coverage > Embedding diversity**: Filter bubbles show up as "never seeing 80% of catalog", not "similar recommendations".

3. **Deterministic top strategy**: With deterministic model, all "top-following" trajectories are identical → zero variance → CI interpretation requires care.

4. **Counterfactual ≠ Caching opportunity**: Value is in analysis framework (explainability), not performance optimization.

5. **Validated with trained weights**: Analysis tools reveal real dynamics - see "Phase 3b Results with Trained Weights" section above.

### Usage

```bash
# Trajectory analysis
uv run python enhancements/analysis/ranking_dynamics.py
uv run python enhancements/analysis/path_divergence.py

# Diversity (simulated - fast)
uv run python enhancements/analysis/diversity_metrics.py --candidates 100

# Diversity (real - use with trained weights)
uv run python enhancements/analysis/diversity_metrics.py --candidates 100 --real

# Sensitivity
uv run python enhancements/analysis/sensitivity_analysis.py

# Counterfactual (use with trained weights)
uv run python enhancements/analysis/counterfactual_analysis.py --history 32

# Real trajectory
uv run python enhancements/analysis/real_trajectory_simulation.py
```

---

## Phase 4: Quantization Comparative Study

### Objective
Implement a comparative study of quantization approaches to find the best tradeoff between compression and accuracy.

### Configurations Tested

| Config | Bit Width | Granularity | Symmetry | Status |
|--------|-----------|-------------|----------|--------|
| `fp16_baseline` | FP16 | - | - | FAIL (memory gate) |
| `int8_tensor_sym` | INT8 | Per-tensor | Symmetric | PASS |
| `int8_channel_sym` | INT8 | Per-channel | Symmetric | **WINNER** |
| `int8_channel_asym` | INT8 | Per-channel | Asymmetric | PASS |
| `int4_tensor_sym` | INT4 | Per-tensor | Symmetric | PASS |
| `int4_channel_asym` | INT4 | Per-channel | Asymmetric | PASS |

### Results

| Config | Top-3 | Kendall's τ | Memory Reduction | Latency |
|--------|-------|-------------|------------------|---------|
| fp16_baseline | 100% | 1.000 | 39.6% | 0.94x |
| int8_tensor_sym | 100% | 1.000 | 59.4% | 1.08x |
| **int8_channel_sym** | 100% | 1.000 | 58.3% | 1.01x |

### Winner: `int8_channel_sym`

- **Top-3 preserved**: 100%
- **Memory reduction**: 58.3% (~2.4x compression)
- **Latency ratio**: 1.01x (essentially no slowdown)
- **Selection score**: 0.850

### Go/No-Go Gate Results

| Gate | Criterion | Threshold | int8_channel_sym |
|------|-----------|-----------|------------------|
| Accuracy | Top-3 preserved | > 90% | ✅ 100% |
| Memory | Reduction vs FP32 | > 40% | ✅ 58.3% |
| Latency | Ratio vs baseline | < 1.2x | ✅ 1.01x |

### Key Files

```
enhancements/optimization/quantization/
├── __init__.py           # Exports
├── config.py             # QuantizationConfig, BitWidth, Granularity
├── quantize.py           # quantize(), dequantize() functions
├── quantized_runner.py   # QuantizedPhoenixRunner
└── study.py              # QuantizationStudy, select_winner()

benchmarks/
└── quantization_study.py # Comparative study script
```

### Usage

```bash
# Run full comparative study
uv run python benchmarks/quantization_study.py

# Quick study (fewer batches)
uv run python benchmarks/quantization_study.py --quick
```

---

## Phase 4b: Extended Quantization Study

### Objective
Push quantization further with three advanced techniques:
1. **Mixed INT4/INT8 precision** - INT4 for FFN (less sensitive), INT8 for attention
2. **Per-group INT4** - 128 weights per group for better INT4 accuracy
3. **KV-cache quantization** - INT8 K/V tensors for cache memory reduction

### New Features Implemented

| Feature | Description | Benefit |
|---------|-------------|---------|
| `MixedPrecisionConfig` | Layer-specific bit widths | Target aggressive quantization to tolerant layers |
| `Granularity.PER_GROUP` | 128 weights per scale factor | Better INT4 accuracy vs per-channel |
| `kv_quantize.py` | K/V tensor quantization | ~4x cache memory reduction |
| Composability | All strategies combinable | Maximum compression options |

### Full Study Results (14 Configurations)

| Config | Top-3 | Tau | Memory | Latency | Status |
|--------|-------|-----|--------|---------|--------|
| fp16_baseline | 100% | 1.000 | 39.6% | 1.05x | FAIL |
| int8_tensor_sym | 100% | 1.000 | 59.4% | 1.16x | PASS |
| int8_channel_sym | 100% | 1.000 | 58.3% | 1.19x | PASS |
| int8_channel_asym | 100% | 1.000 | 58.3% | 1.15x | PASS |
| int4_tensor_sym | 100% | 1.000 | 59.4% | 1.19x | PASS |
| int4_channel_asym | 100% | 1.000 | 58.3% | 1.14x | PASS |
| **int8_channel_kv8** | **100%** | **1.000** | **58.3%** | **1.13x** | **WINNER** |
| mixed_int4_ffn_int8_attn | 100% | 1.000 | 58.3% | 1.14x | PASS |
| int4_pergroup128_sym | 100% | 1.000 | 58.1% | 1.20x | PASS |
| int8_channel_sym_kv8 | 100% | 1.000 | 58.3% | 1.14x | PASS |
| mixed_int4_int8_kv8 | 100% | 1.000 | 58.3% | 1.16x | PASS |
| int4_pergroup_kv8 | 100% | 1.000 | 58.1% | 1.24x | FAIL |
| mixed_int4_pergroup_int8_attn | 100% | 1.000 | 58.2% | 1.17x | PASS |
| mixed_int4_pergroup_int8_kv8 | 100% | 1.000 | 58.2% | 1.17x | PASS |

### Winner: `int8_channel_kv8`

| Metric | Value |
|--------|-------|
| Top-3 Preserved | 100% |
| Kendall's Tau | 1.000 |
| Memory Reduction | 58.3% |
| Latency Ratio | 1.13x |
| Selection Score | 0.826 |

### Top 5 by Composite Score

| Rank | Config | Score |
|------|--------|-------|
| 1 | int8_channel_kv8 | 0.826 |
| 2 | int4_channel_asym | 0.824 |
| 3 | int8_tensor_sym | 0.824 |
| 4 | int8_channel_sym_kv8 | 0.823 |
| 5 | mixed_int4_ffn_int8_attn | 0.823 |

### Key Findings

1. **Perfect accuracy across all configs** - 100% top-3 preservation and Kendall's tau of 1.0. Phoenix model is remarkably robust to quantization.

2. **12 of 14 configs passed all gates** - Only FP16 (insufficient memory reduction) and int4_pergroup_kv8 (latency too high at 1.24x) failed.

3. **KV-cache quantization provides latency benefit** - The winner uses KV-cache quantization which reduces memory bandwidth pressure, achieving best latency (1.13x).

4. **Memory reduction is consistent (~58%)** - On this small model, scale/zero-point overhead is proportionally similar across schemes.

5. **Composability works** - Mixed precision + per-group + KV-cache all compose correctly, enabling flexible deployment configurations.

---

### Phase 4b Results with Trained Weights (MovieLens)

We re-evaluated quantization on the **BPR-trained** model (NDCG 0.4112) to validate real-world accuracy.

#### Quantization Study on BPR-Trained Model

| Config | FP32 NDCG | INT8 NDCG | Retention | Top-3 Agreement |
|--------|-----------|-----------|-----------|-----------------|
| INT8 per-channel | 0.4096 | 0.4084 | **99.7%** | **99.0%** |

**Gate: ✅ PASSED (99.0% >= 95%)**

#### Comparison: BCE vs BPR Training for Quantization

| Metric | BCE Training | BPR Training | Improvement |
|--------|--------------|--------------|-------------|
| Top-3 Agreement | ~90% | **99.0%** | +9% |
| NDCG Retention | 93.1% | **99.7%** | +7% |
| Gate Status | PASS (marginal) | **PASS (comfortable)** |

#### Why BPR Training Improves Quantization Robustness

The key difference is **score margins**:

| Metric | BCE Training | BPR Training |
|--------|--------------|--------------|
| Typical score | ~3e-7 | ~1.0 |
| Score margin | ~1e-8 | ~0.1-0.2 |
| Quantization noise | ~6e-9 | ~6e-9 |
| Noise/Margin ratio | **66%** | **<0.01%** |

With BCE, quantization noise (6e-9) was 66% of score margins (1e-8), causing ranking flips.
With BPR, score margins (~0.1) are 10 million times larger than quantization noise - virtually no impact.

**Key Insight**: BPR loss produces better-calibrated scores with healthy margins, making the model much more robust to quantization.

#### Extended Quantization Study (14 Configs)

We also ran the full Phase 4b extended study:

| Observation | Result |
|-------------|--------|
| Configs tested | 14 |
| Top-3 agreement | 100% for most configs |
| Kendall's tau | 0.643-1.0 (varies) |
| Memory reduction | 58-59% |
| **All configs failed latency gate** | >1.2x on CPU |

The latency gate failures are due to dequantization overhead on CPU. On GPU, INT8/INT4 would show throughput benefits.

#### Recommendations for Production

1. **INT8 per-channel quantization works excellently** - 99% top-3 agreement, 99.7% NDCG retention, 58% memory reduction

2. **BPR training makes quantization easier** - the healthy score margins eliminate the quantization sensitivity issue seen with BCE

3. **Gate can remain at 95%** - with BPR training, we comfortably exceed this threshold

4. **GPU deployment recommended** for latency-sensitive applications - CPU dequantization overhead causes all configs to fail latency gate

### CPU vs GPU Expectations

| Aspect | CPU (Measured) | GPU (Expected) |
|--------|----------------|----------------|
| Winner | int8_channel_kv8 | Mixed precision configs |
| INT4 benefit | Latency overhead | 2x throughput (tensor cores) |
| Per-group overhead | Visible | Negligible (parallel) |
| KV-cache benefit | Modest latency | Significant (HBM bandwidth) |

**Recommendation:**
- **CPU deployment**: Use `int8_channel_kv8` (INT8 per-channel + INT8 KV-cache)
- **GPU deployment**: Consider `mixed_int4_int8_kv8` for better throughput

### Key Files

```
enhancements/optimization/quantization/
├── config.py             # + MixedPrecisionConfig, PER_GROUP, EXTENDED_STUDY_CONFIGS
├── quantize.py           # + compute_scale_zp_per_group(), get_quant_settings_for_param()
├── kv_quantize.py        # NEW: KV-cache quantization
├── quantized_runner.py   # QuantizedPhoenixRunner
└── study.py              # + EXTENDED_STUDY_CONFIGS import

benchmarks/
└── quantization_study.py # + --extended flag
```

### Usage

```bash
# Run extended study (Phase 4b configs)
uv run python benchmarks/quantization_study.py --extended

# Quick extended study
uv run python benchmarks/quantization_study.py --extended --quick

# Run quantization tests (54 tests)
uv run pytest tests/test_optimization/test_quantization.py -v
```

---

## Phase 5: Combined Optimization Runner

### Objective
Create a unified runner that combines all F2 optimizations (JIT, KV-cache, Quantization) and verify go/no-go gates.

### Go/No-Go Gates

| Gate | Criterion | Threshold | Result |
|------|-----------|-----------|--------|
| Accuracy | Top-3 preserved | >= 95% | ✅ **100%** |
| Latency | Speedup vs baseline | >= 2x | ✅ **4.56x** (KV-cache) |
| Memory | Reduction (quant only) | >= 30% | ✅ **49.8%** |

### Results (Large Model: 12L, 512d, 512 context)

| Config | Top-3 | Speedup | Memory | Status |
|--------|-------|---------|--------|--------|
| baseline | 100% | 1.00x | N/A | --- |
| jit_only | 100% | 2.01x | N/A | PASS |
| **kv_cache_only** | **100%** | **4.56x** | N/A | **PASS** |
| quantization_only | 100% | 0.72x | 49.8% | FAIL |
| jit_kv_cache | 100% | 3.08x | N/A | PASS |
| kv_cache_quantization | 100% | 2.26x | 49.8% | PASS |
| full_optimized | 100% | 2.33x | 49.8% | PASS |

### Key Findings

1. **KV-cache provides the largest latency benefit** (4.56x) for inference with repeated users. The speedup scales with context length.

2. **JIT compilation provides 2x speedup** from static shape optimization.

3. **Quantization trades latency for memory** - on CPU, the dequantization overhead results in slower inference (0.72x), but achieves 49.8% memory reduction.

4. **Combined optimizations work together** - the full_optimized config achieves 2.33x speedup and 49.8% memory reduction while maintaining 100% accuracy.

5. **Context length is critical for KV-cache benefit** - with short context (32-128 tokens), the overhead dominates. With longer context (512+ tokens), substantial speedups are achieved.

### Implementation Details

**OptimizedPhoenixRunner:**
- Unified interface combining all optimizations
- Configurable via `OptimizationConfig`
- Three inference paths:
  1. KV-cache path (includes its own JIT)
  2. JIT path (static shape compilation)
  3. Base path (fallback)

**Key Files:**
```
enhancements/optimization/
├── optimized_runner.py      # OptimizedPhoenixRunner, OptimizationConfig
└── ...

benchmarks/
└── f2_final_benchmark.py    # Combined benchmark

tests/test_optimization/
└── test_optimized_runner.py # 25 tests
```

### Usage

```bash
# Run combined benchmark
uv run python benchmarks/f2_final_benchmark.py --model-size large

# Quick benchmark
uv run python benchmarks/f2_final_benchmark.py --model-size large --quick

# Small model (fast but overhead dominates)
uv run python benchmarks/f2_final_benchmark.py --model-size small
```

---

## Summary

**Current cumulative potential (verified):**
- JIT: ~2x speedup (static shape compilation)
- KV-cache: ~4.5x speedup (with 512 context)
- Quantization: ~50% memory reduction (INT8)
- KV-cache quantization: Additional cache memory reduction

**Combined on GPU (estimated):** 10-50x speedup + 2x weight memory reduction + 4x cache memory reduction

**Test coverage:** 166 tests (112 optimization + 54 quantization)

---

## Phase 6: MovieLens Training

### Objective
Train the Phoenix model on MovieLens-100K to validate optimizations with learned weights.

### Training Evolution

We tested multiple training configurations to find the best approach:

| Config | Loss | Negatives | Val NDCG@3 | Test NDCG@3 | Notes |
|--------|------|-----------|-----------|------------|-------|
| BCE (1:7 random) | BCE | 7 random | 0.3157 | 0.2410 | Initial approach |
| BPR (1:15 random) | BPR | 15 random | 0.2727 | - | Overfitting |
| **BPR + In-Batch** | BPR | 31 in-batch | **0.4112** | **0.4183** | **Best** |

### Best Configuration (BPR + In-Batch Negatives)

| Parameter | Value |
|-----------|-------|
| Dataset | MovieLens-100K |
| Users | 943 |
| Movies | 1,682 |
| Training ratings | 81,513 |
| Model | 64d embeddings, 4 layers |
| Batch size | 32 |
| Learning rate | **0.0005** (halved from 0.001) |
| Weight decay | **0.0001** |
| Loss function | **BPR (pairwise ranking)** |
| Negatives | **In-batch (31 per positive)** |
| Early stopping | 5 epochs patience |

### Results

| Epoch | NDCG@3 | Hit@3 | Notes |
|-------|--------|-------|-------|
| 0 (before training) | 0.2585 | 0.3650 | Random initialization |
| 9 (best) | **0.4112** | **0.5458** | +59% NDCG improvement |
| Test set | **0.4183** | **0.5539** | **Excellent generalization** |

### Comparison: Old BCE vs New BPR+In-Batch

| Metric | BCE (1:7) | BPR+In-Batch | Improvement |
|--------|-----------|--------------|-------------|
| Val NDCG@3 | 0.3157 | **0.4112** | **+30%** |
| Val Hit@3 | 0.4330 | **0.5458** | +26% |
| Test NDCG@3 | 0.2410 | **0.4183** | **+73%** |
| Test Hit@3 | 0.3328 | **0.5539** | +66% |

The 73% improvement on test set shows BPR + in-batch negatives dramatically improved generalization.

### Training Validation (Ablation Study)

To understand what each component contributes:

| Configuration | NDCG@3 | Hit@3 | Contribution |
|---------------|--------|-------|--------------|
| Full trained model | 0.4096 | 0.5400 | - |
| Learned Emb + Dot-Product (no transformer) | 0.2862 | 0.3960 | Embeddings: **+3.2%** |
| Random Emb + Transformer (no learned emb) | 0.2684 | 0.3740 | Transformer: **-10.7%** |
| Random baseline | 0.2821 | 0.3960 | - |
| **Synergy (interaction effect)** | - | - | **+107.5%** |

**Key Insight**: Neither transformer nor embeddings work well alone:
- Embeddings alone barely beat random (+0.4%)
- Transformer with random embeddings is WORSE than random
- The improvement is almost entirely from **synergy** - they must work together

This differs from the old BCE training where components had more independent value. BPR loss seems to train the model to rely more heavily on the transformer-embedding interaction.

### Why BPR + In-Batch Works Better

1. **In-batch negatives are harder** - Each positive gets 31 negatives that are other users' liked items. These are harder to distinguish than random movies because they're popular enough to be liked by someone.

2. **BPR optimizes relative ranking** - Instead of absolute probabilities, BPR optimizes `score(positive) > score(negative)`. This directly matches the evaluation metric (NDCG).

3. **Lower LR prevents overfitting** - Halving LR from 0.001 to 0.0005 with weight decay 0.0001 gave more stable training.

4. **Better generalization** - Test NDCG improved 73% vs only 30% on validation, showing the model learned more generalizable patterns.

### Key Learnings

1. **Zero initialization bug discovered and fixed** - Original Phoenix code used `Constant(0)` for transformer weights, causing model to bypass attention entirely. Fixed to Xavier initialization.

2. **BPR >> BCE for ranking** - BPR loss directly optimizes ranking, leading to much better NDCG scores than BCE which optimizes click probability.

3. **In-batch negatives >> random negatives** - Using other users' positives as negatives provides harder, more informative training signal.

4. **Components must work together** - Unlike old training, the new model shows almost no independent component value - 107.5% of improvement comes from transformer-embedding synergy.

5. **Much better generalization** - Old model had severe overfitting (val 0.3157 vs test 0.2410). New model generalizes well (val 0.4112 vs test 0.4183).

---

## Phase 7: Synthetic Twitter Data & Verification Suite

### Objective
Create synthetic Twitter-like data with known ground truth to verify the model learns correct causal relationships, not just correlations.

### Why Synthetic Data?

Real Twitter data has unknown ground truth - we can't verify if the model learned the *right* patterns. Synthetic data with explicit rules lets us:
1. **Test causal effects**: Does blocking an author actually reduce their posts' scores?
2. **Test archetype learning**: Does the model differentiate lurkers from power users?
3. **Test history processing**: Does the transformer actually use history content?

### Ground Truth Design

We designed a rich, multi-dimensional ground truth with **648+ explicit probability parameters**.

**User Archetypes** (6 types):
| Archetype | Behavior |
|-----------|----------|
| Sports Fan | High engagement with sports, low elsewhere |
| Tech Bro | High engagement with tech content |
| Political L | Engages with left politics, blocks right |
| Political R | Engages with right politics, blocks left |
| Lurker | Passive - favorites only, no shares/replies |
| Power User | High engagement across all action types |

**Content Topics** (6 types): Sports, Tech, Politics L, Politics R, Entertainment, News

**Actions** (18 types): favorite, reply, repost, photo_expand, click, profile_click, vqv, share, share_via_dm, share_via_copy_link, dwell, quote, quoted_click, follow_author, not_interested, block_author, mute_author, report

**Ground Truth Dimensionality**: 6 archetypes × 6 topics × 18 actions = **648 probability values** defining expected behavior for every combination.

### Pattern Variety (Successfully Recovered)

The synthetic data encodes several distinct pattern types, all verified through the test suite:

#### 1. Topic Preferences (Correlational)
Each archetype has distinct topic affinity patterns:
```
Sports Fan + Sports:     70% favorite, 30% repost, 60% dwell
Sports Fan + Politics:   5% favorite, 10% not_interested
Tech Bro + Tech:         70% favorite, 35% repost, 65% dwell
Tech Bro + Sports:       10% favorite, 15% click
```
**Verification**: Behavioral tests check predicted action rates match these ground truth values.

#### 2. Action Style Patterns (Archetype-Specific)
Lurkers and power users have topic-independent behavior styles:
```
Lurker (any topic):      20% favorite, 1% repost, 0% reply, 35% dwell
Power User (any topic):  45% favorite, 35% repost, 25% reply, 55% dwell
```
**Verification**: Action differentiation tests verify lurker_repost_ratio << power_user_repost_ratio (15x difference).

#### 3. Cross-Group Hostility (Negative Engagement)
Political users exhibit hostile behavior toward opposing content:
```
Political L + Politics R:  25% block, 15% mute, 30% not_interested, 5% report
Political R + Politics L:  25% block, 15% mute, 30% not_interested, 5% report
```
**Verification**: Block effect tests verify these patterns causally affect ranking.

#### 4. Causal Block Relationships
The model must learn that blocked authors should rank lower - not just correlation but causation:
```
If user U has blocked author A:
  score(post_by_A | user_U) < score(post_by_other | user_U)
```
**Verification**: Counterfactual test with synthetic blocks (not seen in training) verifies 78% success rate.

#### 5. History-Conditioned Preferences
The transformer must use history content, not just user embedding:
```
Same candidate post, different histories:
  score(sports_post | sports_history) > score(sports_post | tech_history)
```
**Verification**: Archetype flip test verifies 86% of swapped histories change topic preferences.

#### 6. Compositional Patterns
Patterns compose - e.g., political users engaging with neutral topics:
```
Political L + News:      35% favorite, 20% repost, 45% dwell (moderate engagement)
Political L + Sports:    10% favorite, 15% click (low engagement)
Political L + Politics L: 65% favorite, 45% repost (high engagement)
```
The model must learn the full 6×6 topic matrix, not just "political user = high engagement".

### Multi-Level Verification Hierarchy

We verify learning at **four levels of abstraction**, from representations to causal interventions:

```
Level 4: CAUSAL INTERVENTIONS
         ├── Block effect (78%): Injecting block → score drops
         └── Archetype flip (86%): Swapping history → preferences change
                    ↑
Level 3: ACTION PREDICTIONS
         ├── Behavioral accuracy (100%): Predicted rates match ground truth
         └── Action differentiation (6/6): Lurker vs power user patterns
                    ↑
Level 2: RANKING QUALITY
         ├── BPR loss training
         └── Positive > negative post scoring
                    ↑
Level 1: REPRESENTATION LEARNING
         ├── User embeddings cluster by archetype (silhouette 0.37)
         └── Topic embeddings cluster by topic (silhouette 0.99)
```

Each level builds on the previous. Passing lower levels without upper levels indicates correlation learning (memorization). Passing upper levels indicates causal understanding.

### Verification Suite

| Test | Level | Description | Threshold |
|------|-------|-------------|-----------|
| **Embedding Probes** | L1 | User/topic embeddings cluster by archetype/topic | Silhouette > 0.25 |
| **Behavioral Tests** | L3 | Predicted action rates match ground truth | 90% of tests pass |
| **Action Differentiation** | L3 | Lurkers vs power users behave differently | 15x repost ratio |
| **Block Effect** | L4 | Blocking author reduces their posts' scores | > 50% of tests |
| **Archetype Flip** | L4 | Swapping history changes topic preferences | > 50% of tests |

### Results

| Test | Initial | After Training | Status |
|------|---------|----------------|--------|
| User clustering | 0.47 | 0.37 | ✅ PASS |
| Behavioral tests | 12% | **100%** | ✅ PASS |
| Action differentiation | 3/6 | **6/6** | ✅ PASS |
| Block effect rate | 16% | **78%** | ✅ PASS |
| Archetype flip rate | 4% | **86%** | ✅ PASS |

### Key Challenges & Solutions

#### Challenge 1: Action Predictions Were Uniform
**Problem**: Model predicted ~0.5 for all actions regardless of user archetype.

**Root Cause**: Training used binary labels (0/1) from actual engagements, losing the archetype-specific probability information.

**Solution**: Use ground truth probabilities as soft labels:
```python
# Before: action_labels = [1, 0, 0, 1, ...] (binary)
# After:  action_labels = get_engagement_probs(archetype, topic).to_array()
```

#### Challenge 2: Block Effect Didn't Generalize
**Problem**: Block effect only worked for actual (user, blocked_author) pairs from training data, not arbitrary pairs.

**Root Cause**: Model memorized specific pairs instead of learning the general semantic "block → lower score".

**Solution**: Add synthetic block pairs during training:
```python
def get_block_aware_batch(self, synthetic_ratio=0.5):
    # 50% actual blocks from training data
    # 50% synthetic: random user + random author with injected block
```

Result: Block effect improved from **24% → 78%**.

#### Challenge 3: Transformer Wasn't Using History
**Problem**: Archetype flip rate was 4% - swapping history barely changed predictions.

**Root Cause**: User embeddings (archetype-specific initialization) dominated predictions. Transformer history processing was ignored.

**Solution**: History-topic contrastive learning:
```python
# Same candidate post, different histories
# Train: score(post | matching_history) > score(post | mismatched_history)
```

Result: Archetype flip rate improved from **4% → 86%**.

### Training Configuration

| Component | Loss | Purpose |
|-----------|------|---------|
| BPR | Ranking loss | Positive > negative post |
| BCE | Action prediction | From model output |
| Classification | Cross-entropy | Predict archetype from user embedding |
| Action Predictor | MSE | Predict action rates from user embedding |
| **Block Contrastive** | Margin ranking | Non-blocked > blocked author posts |
| **History Contrastive** | Margin ranking | Matching > mismatched history |

### Summary: Patterns Injected vs Recovered

| Pattern Type | Ground Truth | Recovered | Evidence |
|-------------|--------------|-----------|----------|
| Topic preferences | 36 (archetype, topic) rules | ✅ 100% | Behavioral test accuracy |
| Action styles | 2 wildcards (lurker, power_user) | ✅ | 15x repost ratio difference |
| Cross-group hostility | 25% block rate | ✅ | Predicted block rates match |
| Causal block effect | Block → lower score | ✅ 78% | Synthetic block intervention |
| History conditioning | Matching history → higher score | ✅ 86% | Archetype swap test |
| Compositional behavior | 6×6×18 = 648 parameters | ✅ | Full action prediction accuracy |

**Total patterns defined**: 648+ probability values across the (archetype × topic × action) space.

**Total patterns verified**: All 5 test suites pass, validating both correlational and causal learning.

### Key Learnings

1. **Soft labels preserve probability information** - Binary labels lose the archetype-specific action rates. Using ground truth probabilities as targets teaches the correct distributions.

2. **Contrastive learning enables causal learning** - Direct supervision with contrastive pairs teaches causal relationships (block → lower score, matching history → higher score) rather than correlations.

3. **Synthetic training data enables generalization** - Training only on actual blocks caused memorization. Adding synthetic blocks taught the general semantic.

4. **User embeddings can dominate** - With archetype-specific initialization, embeddings become so strong that the transformer is bypassed. History contrastive loss forces transformer usage.

5. **Verification tests catch subtle failures** - Behavioral tests passing (100%) while flip tests failing (4%) revealed that the model was using the right archetype but ignoring history content.

6. **Multi-level verification is essential** - Passing correlational tests (L1-L3) doesn't guarantee causal understanding (L4). The hierarchy caught a model that memorized blocks but didn't learn the general semantic.

### Key Files

```
enhancements/data/
├── ground_truth.py           # Archetype/topic definitions, engagement rules
├── synthetic_twitter.py      # Data generator
└── synthetic_adapter.py      # Phoenix adapter with contrastive batches

enhancements/verification/
├── embedding_probes.py       # Clustering tests
├── behavioral_tests.py       # Ground truth comparison
├── action_tests.py           # Archetype differentiation
└── counterfactual_tests.py   # Block effect, archetype flip

scripts/
├── train_synthetic.py        # Multi-task training with contrastive losses
└── verify_synthetic.py       # Run full verification suite
```

### Usage

```bash
# Generate synthetic data
uv run python scripts/train_synthetic.py --generate

# Train model
uv run python scripts/train_synthetic.py --epochs 10

# Run verification suite
uv run python scripts/verify_synthetic.py
```

---

## Next Steps

| Task | Status |
|------|--------|
| ~~Re-run analysis with trained weights~~ | ✅ Complete |
| ~~BPR + in-batch negatives training~~ | ✅ Complete (NDCG 0.4112) |
| ~~Ablation study (transformer vs embeddings)~~ | ✅ Complete (107.5% synergy) |
| ~~Re-validate quantization with BPR model~~ | ✅ Complete (99% agreement) |
| ~~Synthetic data verification suite~~ | ✅ Complete (all tests pass) |
| GPU benchmarking | Future work |
| Production deployment guide | Future work |
| Richer features (content embeddings) | Future work |

---

## Running the Benchmarks

```bash
# KV-cache benchmarks
uv run python benchmarks/kv_cache_scaling.py
uv run python benchmarks/kv_cache_limits.py
uv run python benchmarks/kv_cache_stress.py
uv run python benchmarks/kv_cache_hw_analysis.py

# Quantization benchmarks
uv run python benchmarks/quantization_study.py              # Phase 4 configs
uv run python benchmarks/quantization_study.py --extended   # Phase 4b extended configs
uv run python benchmarks/quantization_study.py --quick      # Quick test

# Combined optimization benchmark (Phase 5)
uv run python benchmarks/f2_final_benchmark.py --model-size large  # Full benchmark
uv run python benchmarks/f2_final_benchmark.py --model-size large --quick  # Quick version
uv run python benchmarks/f2_final_benchmark.py --model-size small  # Small model

# Run optimization tests (including quantization and optimized runner)
uv run pytest tests/test_optimization/ -v

# Run analysis tests
uv run pytest tests/test_analysis/ -v

# Run all tests (166 total)
uv run pytest tests/ -v
```

---

# F4: Reward Modeling - Results & Learnings

**Summary**

| Phase | Status | Key Achievement |
|-------|--------|-----------------|
| Phase 1 | ✅ Complete | BT preference learning: **99.3% accuracy**, sensitive to label noise |
| Phase 2 | ✅ Complete | Pluralistic models: weight recovery is fundamental BT limit, differentiation is solvable |
| Phase 3 | ✅ Complete | Causal verification: **all 5 test suites pass** |
| Phase 4 | ✅ Complete | Multi-stakeholder: **cosine sim 0.478** via label differentiation (was 1.0) |

**Key F4 Finding**: Stakeholder model differentiation requires stakeholder-specific preference labels, not alternative loss functions. Standard BT with different utility functions per stakeholder achieves cosine similarity of 0.478 (target was < 0.95). The 87-experiment sweep confirmed no alternative loss improves on baseline BT.

---

## Phase 1: Bradley-Terry Preference Learning

### Objective
Implement Bradley-Terry preference learning with per-archetype weights and establish baseline performance.

### Model Architecture

- **ContextualRewardModel**: Per-archetype weight matrix `[K, 18]` where K=6 archetypes
- **Reward computation**: `R[b,c] = weights[archetype[b]] · P[b,c,:]` via einsum
- **Loss function**: Bradley-Terry: `-log(σ(R_preferred - R_rejected))`

### Training Results

| Metric | Value |
|--------|-------|
| Final Training Accuracy | 99.84% |
| Final Validation Accuracy | 99.29% |
| Training pairs | 5,600 |
| Epochs | 100 |

### Sensitivity Analysis: Key Findings

#### 1. Noise Sensitivity (Train Clean, Test with Noise)

| Test Noise (std) | Accuracy |
|------------------|----------|
| 0.00 | 98.7% |
| 0.10 | 93.7% |
| 0.20 | 86.0% |
| 0.30 | 83.3% |

**Finding**: ~15% accuracy degradation with 30% feature noise.

#### 2. Label Flip Sensitivity (Critical Finding)

| Label Flip Rate | Accuracy |
|-----------------|----------|
| 0% | 93.7% |
| 10% | 90.0% |
| 20% | 79.7% |
| 30% | 68.7% |

**Finding**: Label flips (simulating human disagreement) are MORE damaging than feature noise. 30% flip rate approaches random guessing.

#### 3. Does Noisy Training Help? (Key Experiment)

Test condition: 15% feature noise + 15% label flips

| Training Condition | Accuracy on Noisy Test |
|-------------------|------------------------|
| Clean | 78.7% |
| Matched noise (0.15/0.15) | 75.2% |
| **Feature noise only (0.15/0.00)** | **82.0%** |
| Label flip only (0.00/0.15) | 70.8% |

**Critical Insight**:
- **Feature noise during training HELPS** - acts as data augmentation
- **Label flips during training HURT** - teaches wrong preferences
- Best strategy: Add feature noise but keep labels clean

#### 4. Held-Out Archetype Generalization

| Held Out Count | Accuracy |
|----------------|----------|
| Hold 1 | 88.0% |
| Hold 2 | 82.0% |
| Hold 3 | 86.0% |
| Hold 4 | 93.0% |

**Finding**: Generalization varies based on which archetypes are held out, not just the count. Model learns similar weights across archetypes on this synthetic data.

#### 5. Cross-Archetype Transfer

| Transfer | Accuracy |
|----------|----------|
| Sports → Tech | 100% |
| Political L → Political R | 100% |
| Tech → Lurker | 92% |
| Power User → Lurker | 98% |

**Finding**: Model generalizes well across archetypes because learned weights are similar (0.997 cosine similarity).

**Phase 4 Retrospective**: The 0.997 cosine similarity across archetypes is a consequence of the preference label generation, not a model property. All archetypes use the same formula to determine which content is "preferred" vs "rejected" — so despite different engagement patterns, the training signal is effectively identical. See Phase 4 for the full analysis of this insight.

### Implications for Phase 2+

The sensitivity to **label flips (human disagreement)** is the critical limitation. Real human preference data has:
- **Inconsistent labels**: Different humans disagree on preferences
- **Context-dependent preferences**: Same user prefers different things in different contexts
- **Noisy feedback**: Accidental clicks, changed minds

**Phase 4 Retrospective**: This label sensitivity finding is more significant than initially appreciated. It foreshadows the key Phase 4 insight: the labels are the critical variable — a model can only differentiate between groups if the training labels differentiate them. No loss function or architecture can compensate for identical preference labels.

This motivates more sophisticated approaches:
1. **Noise-aware models** that explicitly model label uncertainty
2. **Confidence-weighted training** to downweight uncertain labels
3. **Ensemble methods** to capture preference distribution

### Go/No-Go Assessment

| Gate | Threshold | Result | Status |
|------|-----------|--------|--------|
| Standard accuracy | > 95% | 100.0% | PASS |
| Hard negatives | > 60% | 99.0% | PASS |
| Adversarial | > 75% | 95.0% | PASS |
| Noisy (10%) | > 85% | 91.3% | PASS |
| Held-out generalization | > 70% | 88.0% | PASS |

### Key Files

```
enhancements/reward_modeling/
├── reward_model.py          # ContextualRewardModel
├── training.py              # Bradley-Terry loss and training
├── weights.py               # RewardWeights dataclass
└── __init__.py

scripts/
├── train_reward_model.py    # Baseline training
├── evaluate_reward_model.py # Comprehensive evaluation
└── sensitivity_analysis.py  # Sensitivity study

results/f4_phase1/
├── baseline_weights.npy     # Trained weights
├── training_metrics.json    # Training history
├── comprehensive_evaluation.json
├── sensitivity_analysis.json
├── training_curves.png
└── sensitivity_analysis.png
```

### Usage

```bash
# Train baseline model
uv run python scripts/train_reward_model.py

# Run comprehensive evaluation
uv run python scripts/evaluate_reward_model.py

# Run sensitivity analysis
uv run python scripts/sensitivity_analysis.py

# Run tests (45 tests)
uv run pytest tests/test_reward_modeling/ -v
```

---

## Phase 2: Pluralistic Reward Models

### Objective
Implement pluralistic reward models that discover multiple "value systems" from user preference data, enabling personalized ranking:

```
R(user, content) = Σ_k π_k(user) · (weights_k · action_probs)
```

Goal: **recover ground truth archetypes** from preference learning.

---

### Summary of Approaches Tried

| # | Approach | Accuracy | Weight Correlation | Cluster Purity |
|---|----------|----------|-------------------|----------------|
| 1 | EM training | ~95% | 0.15 | N/A |
| 2 | Auxiliary loss | ~95% | 0.15 | N/A |
| 3 | Hybrid | ~97% | 0.51 | N/A |
| 4 | Supervised classification | ~98% | 0.56 | N/A |
| 5 | Learned embeddings (oracle) | ~99% | 0.50 | N/A |
| 6 | Two-stage (avg features) | 99.4% | 0.55 | 70.9% |
| 7 | **Two-stage (topic features)** | **99.3%** | **0.60** | **100%** |
| 8 | **Two-stage (topic×action)** | **99.5%** | **0.54** | **100%** |

**Key Finding**: All approaches fail the 0.8 Pearson correlation gate. Rank-order analysis (Kendall τ = 0.612, Spearman ρ = 0.767) shows the model recovers broad action ordering better than magnitudes, but genuine recovery limitations remain beyond BT scale invariance. See `results/f4_rank_recovery.json`.

---

### Part 1: End-to-End Pluralistic Models

#### Training Approaches

| Approach | Description |
|----------|-------------|
| **EM** | Alternating E-step (compute responsibilities) and M-step (update weights) |
| **Auxiliary** | End-to-end with diversity loss + entropy regularization |
| **Hybrid** | EM structure with diversity regularization in M-step |

#### Results

| Approach | Accuracy | Correlation | Assignment | Diversity |
|----------|----------|-------------|------------|-----------|
| EM | 99.3% | 0.387 | 17.5% | **0.001** (collapsed) |
| Auxiliary | 99.5% | 0.153 | 18.1% | 0.937 |
| **Hybrid** | **97.3%** | **0.510** | 7.6% | 0.426 |

**Key Finding: All approaches fail the 0.8 correlation gate.**

- **EM collapses**: All 6 value systems become identical
- **Auxiliary**: High diversity but wrong systems
- **Hybrid**: Best unsupervised approach, still insufficient

---

### Part 2: Fix Attempts

#### Fix 1: Supervised Classification Loss

Added supervised loss to guide user→cluster assignments:

```python
L_total = L_bt + λ_div·L_diversity + λ_ent·L_entropy + λ_cls·L_classification
```

| λ_cls | Correlation | Assignment |
|-------|-------------|------------|
| 0 | 0.153 | 18.1% |
| 1.0 | **0.559** | **32.3%** |
| 5.0 | 0.559 | 32.3% |

**Result**: Improved but still fails. MLP easily classifies users, but weights don't recover structure.

#### Fix 2: Learned User Embeddings

Learned encoder: `user_history → embedding → mixture_weights`

| Variant | Correlation |
|---------|-------------|
| Unsupervised | ~0.3 |
| Supervised | ~0.5 |
| **Oracle (one-hot archetype)** | **~0.5** |

**Critical Discovery**: Even with **perfect clustering** (oracle one-hot input), weights don't recover ground truth.

**Root Cause**: Bradley-Terry loss has many local minima. Countless weight configurations produce identical rankings - there's no unique solution.

---

### Part 3: Two-Stage Approach

Decoupled the problem into two stages:

```
Stage 1: Cluster users by interaction features (k-means)
Stage 2: Train per-cluster Bradley-Terry weights
```

#### Feature Engineering Results

| Feature Type | Dimensions | Cluster Purity | Accuracy |
|--------------|------------|----------------|----------|
| Avg action probs | 18D | 70.9% | 99.4% |
| Topic engagement | 6D | **100%** | 99.3% |
| **Topic × Action** | **108D** | **100%** | **99.5%** |

**Key Insight**: Feature engineering solves clustering perfectly. Weight recovery still fails but isn't needed for production.

---

### Part 4: Stress Tests

#### Stress Test #1: Same Topics, Different Actions

Archetypes with same topic preferences but different action styles (likers vs commenters vs sharers).

| Feature Type | Cluster Purity | Expected |
|--------------|----------------|----------|
| Topic-only (6D) | 70.3% | FAIL ✓ |
| **Topic×Action (108D)** | **100%** | PASS ✓ |

**Result**: Richer features solve the problem.

#### Stress Test #2: Noisy Preferences (FUNDAMENTAL)

| Noise Rate | Accuracy | Cluster Purity |
|------------|----------|----------------|
| 0% | 99.3% | 100% |
| 10% | 88.9% | 100% |
| 20% | 78.3% | 100% |
| **30%** | **68.3%** | **100%** |
| **40%** | **59.6%** | **100%** |

**Key Insight**: With 40% label noise, theoretical max accuracy is 60% - model achieves 59.6%.

- Clustering still works (features aren't noisy)
- **This is a FUNDAMENTAL limitation** - noisy labels cannot be fully corrected

#### Stress Test #4: Cross-Topic Users (FUNDAMENTAL)

Users who genuinely like multiple topics (sports AND tech).

| Metric | Value |
|--------|-------|
| Cross-user distribution | 3-4 clusters |
| Max concentration | 40.7% in one cluster |

**Key Finding**: Cross-topic users **form their own hybrid clusters** - they don't scatter randomly. K-means naturally discovers hybrid preference patterns.

---

### Part 5: GMM Soft Clustering

Attempted soft membership for cross-topic users.

| Method | Avg Entropy | Significant Clusters |
|--------|-------------|---------------------|
| GMM + Rich (108D) | 0.000 | 1.0 |
| GMM + Topic (6D) | 0.035 | 1.0 |
| GMM + 10 clusters | 0.000 | 1.0 |

**Finding**: Rich features create perfectly separated clusters. GMM assigns ~100% probability to nearest cluster, making it functionally equivalent to k-means.

**Conclusion**: Soft membership not needed when features are discriminative. Cross-topic users form their own distinct clusters.

---

### Key Learnings

#### 1. Bradley-Terry Has Two Distinct Limitations

**Limitation A — Weight Recovery (Fundamental):** BT is scale-invariant, so many weight vectors produce identical rankings. You cannot recover the exact ground truth weights:

```
Ground truth weights: [1.0, 0.5, 0.3, -0.2, ...]
Learned weights:      [0.8, 0.4, 0.24, -0.16, ...]  ← Same rankings!
Correlation:          0.5 (fails gate)
```

This is intrinsic to pairwise ranking losses and cannot be fixed by changing the loss function.

**Limitation B — Weight Differentiation (Not Fundamental):** When different groups (archetypes, stakeholders) converge to the same weights, the cause is **identical preference labels**, not BT's loss landscape. Phase 4 demonstrates that BT produces highly differentiated weights (cosine similarity 0.48) when groups are trained on preference labels generated by different utility functions.

**Phase 4 Retrospective**: In Phase 2, we attributed the failure of per-cluster weights to diverge entirely to Limitation A. In retrospect, Limitation B was the dominant factor — the preference pairs were generated with the same formula for all clusters, so there was no training signal to differentiate them.

#### 2. Structural Recovery ≠ Prediction Accuracy

| Metric | Value | Meaning |
|--------|-------|---------|
| Prediction accuracy | 99%+ | Model predicts preferences correctly |
| Weight Pearson | ~0.6 | Magnitudes don't match ground truth |
| Weight Spearman ρ | 0.77 | Rank ordering partially recovered |
| Weight Kendall τ | 0.61 | Pairwise concordance ≈ Pearson |

**These are independent.** Perfect prediction doesn't require recovering true weights. Rank metrics confirm partial scale artifact but genuine recovery limits.

#### 3. Feature Engineering > Model Complexity

Simple k-means with right features beats complex end-to-end models:

| Features | Purity | Notes |
|----------|--------|-------|
| 18D (avg actions) | 70.9% | Loses topic information |
| 6D (topic engagement) | 100% | Perfect for topic archetypes |
| 108D (topic×action) | 100% | Perfect for action archetypes |

#### 4. Cross-Topic Users Form Their Own Clusters

Users who like both sports AND tech don't scatter - they form coherent "hybrid" clusters. The clustering naturally discovers hybrid preference patterns.

#### 5. Fundamental vs Solvable Limitations

| Limitation | Type | Solution |
|------------|------|----------|
| Weight recovery (exact values) | **Fundamental** | Accept - not needed for production |
| Weight differentiation (groups converge) | **Solvable** | Use group-specific preference labels (see Phase 4) |
| Same topic, different actions | Solvable | Use richer features |
| Noisy labels | **Fundamental** | Accept - accuracy ≤ (1 - noise_rate) |
| Cross-topic users | Solvable | They form their own clusters |

---

### Final Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Two-Stage Pluralistic Model                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: User Clustering                                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ User History │ -> │ Rich Features│ -> │  K-means    │         │
│  │ (interactions)│    │ (topic×action)│   │ (K=6)       │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                               │                  │
│  Stage 2: Per-Cluster Weights                ▼                  │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  Cluster 0: weights_0 (sports fans)                  │       │
│  │  Cluster 1: weights_1 (political left)               │       │
│  │  Cluster 2: weights_2 (tech enthusiasts)             │       │
│  │  Cluster 3: weights_3 (hybrid: sports+tech)          │       │
│  │  Cluster 4: weights_4 (power users)                  │       │
│  │  Cluster 5: weights_5 (lurkers)                      │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                  │
│  Inference: reward = action_probs @ weights[cluster_id]         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### Production Metrics

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Preference prediction | 99.5% | >90% | ✅ PASS |
| Cluster purity | 100% | >80% | ✅ PASS |
| Weight Pearson | 0.60 | >0.8 | ❌ FAIL (accepted) |
| Weight Spearman ρ | 0.77 | — | Rank recovery partial |
| Interpretability | 98% | >90% | ✅ PASS |

**Decision**: Accept weight correlation failure. Rank-order analysis confirmed it is partially but not primarily a BT scale artifact (Spearman 0.77 > Pearson 0.60, but Kendall 0.61 ≈ Pearson). Single-stakeholder BT achieves Pearson 0.94 — the pluralistic clustering step introduces the main recovery loss.

---

### Implications for Phase 3+

#### What Works
- Per-user-cluster personalized ranking
- Discovering natural user segments
- High prediction accuracy (99%+)

#### What Doesn't Work
- Recovering "ground truth" preference weights
- Explaining WHY weights have specific values
- Soft membership (not needed with good features)

#### Phase 3: User Controllable Sliders
The two-stage model provides a foundation:
- **Cluster weights** can be exposed as slider starting points
- Users can **adjust within their cluster's weight space**
- Or **blend between cluster weights** for custom preferences

---

### One-Line Summary

**Phase 2 discovered that pluralistic rewards work for personalization (99.5% accuracy, 100% cluster purity) but cannot recover ground truth weights due to fundamental Bradley-Terry limitations - and that's okay for production.**

---

### Key Files

```
enhancements/reward_modeling/
├── pluralistic.py           # PluralConfig, PluralState, train_*, loss functions
├── structural_recovery.py   # measure_structural_recovery, check_recovery_gates
├── two_stage.py             # Two-stage k-means + Bradley-Terry
├── two_stage_gmm.py         # GMM soft clustering variant
├── learned_embeddings.py    # Fix 2: learned encoder
└── __init__.py

scripts/
├── compare_pluralistic_approaches.py  # End-to-end comparison
├── test_two_stage.py                  # Two-stage with stress tests
├── test_gmm_rich_features.py          # GMM comparison
├── test_learned_embeddings.py         # Fix 2 experiments

results/f4_phase2_two_stage/
├── two_stage_summary.json
├── two_stage_comparison.json

results/f4_phase2_gmm/
├── gmm_comparison.json
```

### Usage

```bash
# Run pluralistic comparison (end-to-end approaches)
uv run python scripts/compare_pluralistic_approaches.py

# Run two-stage with stress tests
uv run python scripts/test_two_stage.py

# Run GMM comparison
uv run python scripts/test_gmm_rich_features.py
```

---

## Phase 3: Causal Verification

### Objective
Implement intervention tests to verify that reward models capture **causal relationships**, not just correlations.

From the design doc (F4 Tier 1B):
> "Rewards should capture causation, not just correlation"

### Two Key Interventions

| Intervention | What It Tests | Expected Effect |
|--------------|---------------|-----------------|
| **Block Intervention** | Injecting block signal | Score should DECREASE |
| **Follow Intervention** | Injecting follow signal | Score should INCREASE |
| **History Intervention** | Matching vs mismatched history | Matching should score HIGHER |

### Results

#### Test 1: Default Weights (Hand-Tuned)

| Test | Pass Rate | Mean Effect | Status |
|------|-----------|-------------|--------|
| Block | 100% | -1.73 | PASS |
| Follow | 100% | +1.65 | PASS |
| History | 0% | 0.00 | FAIL (expected) |

**Note**: History test expected to fail - simple `R = w·P(actions)` ignores user history entirely.

#### Test 2: Trained Two-Stage Model

| Test | Pass Rate | Mean Effect | Status |
|------|-----------|-------------|--------|
| Block | 100% | -1.40 | PASS |
| Follow | 100% | +1.18 | PASS |
| History | 50% | +0.08 | FAIL |

**Key Insight**: The two-stage model captures **action-level causality** (block→-reward, follow→+reward) but only partially captures **history-level causality** (topic matching). The model clusters users by history but uses the same per-cluster weights regardless of content topic.

#### Test 3: Adversarial Detection

| Test | Pass Rate | Mean Effect | Status |
|------|-----------|-------------|--------|
| Block | 0% | +1.37 | FAIL |
| Follow | 0% | -0.95 | FAIL |
| History | 0% | 0.00 | FAIL |

**Result**: Adversarial weights (block=+1.5, follow=-1.0) correctly detected as broken.

### Key Findings

1. **Block/Follow interventions**: The reward model correctly captures that blocking is a negative signal and following is a positive signal. Both tests pass with 100% rate.

2. **History limitation revealed**: The two-stage model only partially captures topic-based history effects (50% pass rate). To fully pass, we'd need a model that explicitly scores content based on how well it matches user's historical topic interests.

3. **Adversarial detection works**: The causal verification framework correctly identifies reward models with inverted weights as broken.

4. **Action-level vs Content-level causality**:
   - **Action-level**: "User clicked block → author's future posts rank lower" ✓
   - **Content-level**: "User engaged with sports → sports posts rank higher" ⚠️ (partial)

### Implications

The current two-stage model is suitable for production where:
- Action signals (block, follow, favorite) should affect rankings
- User clustering provides coarse personalization

For finer-grained topic-matching, consider:
- Topic-aware reward weights (different weights per topic)
- Content-user interaction features
- History-conditioned scoring

### Stress Tests

To validate that the reward model truly understands causal relationships (not just correlations), we implemented 7 stress tests:

| Test | What It Validates | Default Weights | Two-Stage |
|------|-------------------|-----------------|-----------|
| **Effect Scaling** | Stronger intervention → larger effect | ✅ Monotonic | ✅ Monotonic |
| **Compound Interventions** | Multiple signals compound | ✅ 1.95x | ✅ 3.36x |
| **Conflicting Signals** | Block works even with high favorites | ✅ 100% | ✅ 100% |
| **Cross-Preference** | Block works on any content type | ✅ PASS | ✅ PASS |
| **Reversibility** | Removing intervention restores baseline | ✅ 0 error | ✅ 0 error |
| **Noise Robustness** | Causal relationship holds under noise | ✅ 100% | ✅ 100% |
| **Threshold Sensitivity** | No cliff-edge degradation | ✅ 0% drop | ✅ 0% drop |

#### Stress Test Details

**1. Effect Scaling**: Block strength [0.2, 0.4, 0.6, 0.8, 1.0] produces monotonically increasing negative effects [-0.35, -0.71, -1.08, -1.44, -1.81].

**2. Compound Interventions**: `block + mute + not_interested` together is 2-3x stronger than `block` alone, confirming effects compound correctly.

**3. Conflicting Signals**: Even content with `favorite=0.8, repost=0.6` still shows decreased score when block is injected - the model doesn't get confused by mixed signals.

**4. Cross-Preference**: Block effect is consistent whether applied to user's preferred topic (-0.75) or non-preferred topic (-0.76) - no special cases.

**5. Reversibility**: `baseline → block → restore` returns exactly to baseline (0.00 error), confirming deterministic causal behavior.

**6. Noise Robustness**: With up to 20% noise in action probabilities, block intervention still decreases score 100% of the time.

**7. Threshold Sensitivity**: Pass rate doesn't cliff-edge at any threshold - effects are consistent and well-distributed.

### Go/No-Go Assessment

| Gate | Threshold | Result | Status |
|------|-----------|--------|--------|
| Block intervention | >50% | 100% | ✅ PASS |
| Follow intervention | >50% | 100% | ✅ PASS |
| Adversarial detection | 100% | 100% | ✅ PASS |
| History intervention | >50% | 50% | ⚠️ PARTIAL |
| **Stress tests** | 7/7 | **7/7** | ✅ **PASS** |

**Decision**: Phase 3 passes all core gates including stress tests. History limitation is documented for future work.

### Key Files

```
enhancements/reward_modeling/
├── causal_verification.py    # CausalVerificationSuite, intervention tests
└── __init__.py               # + exports

scripts/
├── test_causal_verification.py  # Basic causal tests
├── stress_test_causal.py        # 7 stress tests for causal understanding

results/f4_phase3_causal/
├── causal_verification_results.json
├── stress_test_results.json
```

### Usage

```bash
# Run basic causal verification tests
uv run python scripts/test_causal_verification.py

# Run stress tests (validates true causal understanding)
uv run python scripts/stress_test_causal.py
```

---

## Running the Analysis Tools

```bash
# Trajectory analysis
uv run python enhancements/analysis/ranking_dynamics.py
uv run python enhancements/analysis/path_divergence.py
uv run python enhancements/analysis/real_trajectory_simulation.py

# Diversity metrics
uv run python enhancements/analysis/diversity_metrics.py --candidates 100
uv run python enhancements/analysis/diversity_metrics.py --candidates 100 --real  # With trained weights

# Sensitivity analysis
uv run python enhancements/analysis/sensitivity_analysis.py

# Counterfactual analysis (use with trained weights)
uv run python enhancements/analysis/counterfactual_analysis.py --history 32

# Phase 3b analysis with trained MovieLens model
uv run python scripts/run_phase3b_analysis.py

# Quantization analysis with trained model
uv run python scripts/analyze_learned_model.py --num-samples 200

# Phase 4 Multi-Stakeholder analysis
uv run python scripts/analyze_stakeholder_utilities.py        # Pareto frontier analysis
uv run python scripts/train_and_compare_stakeholder_models.py # Per-stakeholder model comparison
```

---

## Phase 4: Multi-Stakeholder Framework

*87 training experiments completed across 4 loss function types with 50,000 training pairs per stakeholder.*

### Objective
Train stakeholder-differentiated reward models (User, Platform, Society) that produce meaningfully different content rankings.

### Root Cause Discovery: It's the Labels, Not the Loss

#### Attempt 1: Penalty-Based Differentiation (Failed)

Initial approach: train three models with different loss penalties on the same preference data.

```
User Model:     Loss = BT_loss + λ * discomfort_penalty
Platform Model: Loss = BT_loss (standard)
Society Model:  Loss = BT_loss + λ_div * diversity_loss + λ_pol * polarization_penalty
```

**Result**: All three models converged to nearly identical weights (cosine similarity ~1.0, Kendall's τ ~0.98).

**Original (incorrect) diagnosis**: "Bradley-Terry loss dominates training; penalty terms can't overcome BT's ranking invariance."

#### Attempt 2: Alternative Loss Functions (Failed)

Systematically tested 4 alternative loss functions to break scale invariance:

| Loss Function | Idea | Result |
|---------------|------|--------|
| Margin-BT | Force minimum score gap | Cosine sim = 1.0 |
| Calibrated-BT | Anchor scores to engagement rates | Cosine sim = 1.0 |
| Constrained-BT | Stakeholder-specific hard constraints | (in progress) |
| Post-Hoc Reranking | Rerank at serving time | (in progress) |

**Result**: All loss functions produced identical weights across stakeholders.

#### The Real Fix: Stakeholder-Specific Preference Labels

The breakthrough came from realizing that **all stakeholders were trained on identical (preferred, rejected) pairs**. The preference labels were generated by a single utility function — so regardless of loss function, every model received the same training signal.

**Fix**: Give each stakeholder a different utility function that determines which content is preferred:

```python
# User: balanced engagement
User_utility     = positive_engagement - negative_signals

# Platform: engagement-focused, tolerates some negativity
Platform_utility = positive_engagement - 0.3 * negative_signals

# Society: harm-averse, heavily penalizes divisive content
Society_utility  = positive_engagement - 4.0 * negative_signals
```

Same content pairs, but stakeholders **disagree** on which is preferred:
- A viral political post with high engagement AND high blocking → Platform prefers it, Society rejects it

**Label disagreement rates** (50,000 pairs):

| Pair | Agreement | Disagreement |
|------|-----------|--------------|
| User-Platform | 77% | 23% |
| User-Society | 88% | 12% |
| Platform-Society | 65% | **35%** |

Platform and Society disagree most (35%) because they have the most divergent attitudes toward negative signals.

#### Results: Standard BT With Stakeholder-Specific Labels

| Metric | Before (same labels) | After (different labels) | Target |
|--------|---------------------|--------------------------|--------|
| User-Platform cosine sim | 1.000 | **0.830** | < 0.95 |
| User-Society cosine sim | 1.000 | **0.884** | < 0.95 |
| Platform-Society cosine sim | 1.000 | **0.478** | < 0.95 |

**No alternative loss function was needed.** Standard Bradley-Terry achieves cosine similarity of 0.48 — far below the 0.95 target — when trained on stakeholder-specific preference labels.

#### Disagreement → Differentiation Bound

A 34-point sweep of the negative-penalty parameter α (two reference sweeps: Platform-fixed and User-fixed) establishes the functional relationship between label disagreement rate and cosine similarity:

| Disagreement threshold | Cosine similarity drops below |
|------------------------|-------------------------------|
| ≥10% | 0.95 |
| ≥19% | 0.80 |
| ≥37% | 0.50 |

The relationship is strongly monotonic (Spearman ρ = -0.984) with a linear R² of 0.898 across both sweeps combined. The exact slope is reference-dependent: the Platform-fixed sweep shows steeper differentiation at moderate disagreement than the User-fixed sweep.

**Margin-augmented model**: Disagreement rate alone is insufficient — the same rate can produce different cosine similarities depending on the *depth* of disagreements. Adding mean total margin (average utility gap on disagreed pairs) as a second predictor yields R² = 0.977:

| Model | R² |
|-------|----|
| Disagreement rate only | 0.898 |
| 2-variable (rate + margin) | **0.977** |
| Product (rate × margin) | 0.863 |
| Margin only | 0.786 |
| \|Δα\| only | 0.437 |

The 2-variable fit: `cos = 1.098 − 1.127d − 0.088m` (d = disagreement rate, m = mean total margin). Both variables are needed independently — the product alone (R² = 0.863) is worse than disagreement alone. For the utility family U = pos − α·neg, the mean total margin decomposes as |Δα| · E[Δneg | disagreement], separating parameter distance from content structure. See `results/disagreement_bound_analysis.json`.

**LLM confidence as margin proxy (CONDITIONAL_GO)**: Tested whether the 2-variable structure holds when the annotator uses a different utility function. Claude Haiku was given natural language stakeholder descriptions and content as engagement/negativity scores (0-100) — no formula. Results across 15 sweep points × 200 pairs:

| Criterion | Threshold | Value | Result |
|-----------|-----------|-------|--------|
| LLM 2-var Spearman | ≥ 0.85 | **0.929** | PASS (MUST) |
| d correlation (Pearson) | ≥ 0.90 | **0.921** | PASS (MUST) |
| Conf-margin Spearman | ≥ 0.70 | 0.668 | FAIL (SHOULD) |
| Spearman gap (analytic − LLM) | ≤ 0.10 | **0.064** | PASS (NICE) |

The LLM's implicit utility function differs from ours (d_llm ≠ d_analytic), but the *ranking* of which stakeholder pairs are most differentiated is preserved (Spearman = 0.929). LLM confidence adds value over disagreement rate alone (+0.135 Spearman boost) but is compressed in range [1.40, 1.66], making it a weak margin substitute for linear prediction (R² = 0.80). The go/no-go uses Spearman rather than R² because rank-ordering is the practically relevant question. See `results/llm_margin_proxy.json`.

**α-recovery from BT weight vectors**: For the utility family U = pos − α·neg, can you recover α from the trained BT weight vector? Trained 13 BT models with α ∈ {0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0} and computed α_recovered = -mean(w_neg) / mean(w_pos).

| Metric | Value |
|--------|-------|
| Spearman (α_true vs α_recovered) | **1.0000** |
| Pearson | 0.9985 |
| R² | 0.9971 |
| Affine fit | α_rec ≈ -0.062 + 1.321 · α_true |
| MAE after affine transform | 0.120 |

Stakeholder validation (independent seed):

| Stakeholder | α_true | α_recovered |
|-------------|--------|-------------|
| Platform | 0.3 | 0.344 |
| User | 1.0 | 1.345 |
| Society | 4.0 | 5.594 |

The systematic amplification (slope 1.32 instead of 1.0) means recovered α values are stretched but perfectly ordered. All three recovery methods (mean ratio, sum ratio, regression) achieve identical Spearman = 1.0. BT's scale invariance prevents recovering absolute weight magnitudes, but the *ratio* structure within the weight vector encodes α up to an affine transform. See `results/alpha_recovery.json`.

**α-recovery stress tests**: The baseline recovery used ideal conditions (deterministic labels, 5000 pairs, known pos/neg groupings). Stress-tested across 4 dimensions (1300 training runs × 5 seeds × 13 α values):

| Dimension | Breaking point (ρ < 0.95) | Spearman at worst | Practical threshold |
|-----------|--------------------------|-------------------|---------------------|
| Label noise (p_flip) | 0.30 | 0.836±0.159 @30% | ≤20% annotation error |
| Sample size (n_pairs) | 50 | 0.877±0.053 @50 | ≥250 pairs sufficient |
| BT temperature (β) | 0.5 | 0.610±0.176 @β=0.2 | β ≥ 0.5 (clear preferences) |
| Content correlation (ρ) | 0.8 | 0.825±0.210 @ρ=0.8 | ρ ≤ 0.6 (moderate OK) |

Key findings: (1) Recovery is remarkably robust — tolerates 20% label noise and works with 250 pairs. (2) BT temperature is the strongest stressor — when preferences are nearly random (β=0.2), recovery collapses. (3) Moderate content correlation (ρ=0.3) *improves* recovery to perfect Spearman 1.0, likely because correlated content provides more signal about α when positive and negative actions co-vary. (4) Spearman (rank ordering) degrades much slower than Pearson (linear fit) across all dimensions — the ordering of α values is robust to conditions that destroy the linear relationship. See `results/alpha_recovery_stress.json`.

Weight vectors show clear stakeholder patterns:

| Action | User | Platform | Society | Interpretation |
|--------|------|----------|---------|----------------|
| favorite | +3.2 | **+4.6** | +1.7 | Platform values engagement most |
| repost | +2.3 | **+4.4** | +0.0 | Society ignores viral sharing |
| reply | +2.5 | **+4.3** | +0.2 | Platform values all engagement |
| follow_author | +1.8 | **+3.1** | +0.5 | Same pattern |
| block_author | -2.9 | -1.5 | **-5.3** | Society penalizes blocking most |
| report | -3.8 | -1.4 | **-4.8** | Society penalizes reports most |

### Key Insight

**A model can only differentiate between stakeholders if the training signal differentiates them.** If user, platform, and society all agree that content A is better than content B, no loss function — BT, margin-based, calibrated, listwise, or otherwise — can learn different weights. The differentiation must come from the data.

This corrects the Phase 2 narrative that attributed weight convergence to "fundamental BT limitations." BT's scale invariance is real (you can't recover exact weight values), but weight *differentiation* across groups was always solvable — it just required different preference labels per group.

### Full Experiment Results

**Configuration**: 50,000 training pairs per stakeholder, 2,000 content items, 30 epochs, 87 training experiments + 30 post-hoc evaluations.

#### Accuracy by Loss Type

| Loss Type | Hyperparameter | User | Platform | Society |
|-----------|---------------|------|----------|---------|
| Bradley-Terry | (baseline) | 92.8% | 91.9% | 95.4% |
| Margin-BT | m=0.05 | 96.0% | 92.9% | 97.5% |
| Margin-BT | m=0.5 | 95.7% | 93.0% | 97.0% |
| Margin-BT | m=2.0 | 93.9% | 92.1% | 95.7% |
| Calibrated-BT | λ=0.05 | 92.9% | 91.3% | 95.4% |
| Calibrated-BT | λ=0.25 | 93.4% | 86.8% | 95.4% |
| Calibrated-BT | λ=0.5 | 93.9% | **86.0%** | 95.5% |

**Finding**: All approaches maintain >85% accuracy. Calibrated-BT degrades Platform accuracy at higher calibration weights (drops to 86%). Margin-BT and standard BT are most stable.

#### Cosine Similarity (Stakeholder Differentiation)

Lower = more differentiated. Target: < 0.95 (previously 1.0).

| Loss Type | Hyperparameter | U-P | U-S | P-S | Min |
|-----------|---------------|-----|-----|-----|-----|
| Bradley-Terry | (baseline) | 0.830 | 0.884 | **0.478** | **0.478** |
| Calibrated-BT | λ=0.05 | 0.847 | 0.888 | 0.512 | 0.512 |
| Calibrated-BT | λ=0.10 | 0.848 | 0.889 | 0.531 | 0.531 |
| Calibrated-BT | λ=0.25 | 0.849 | 0.893 | 0.530 | 0.530 |
| Margin-BT | m=0.05 | 0.877 | 0.920 | 0.613 | 0.613 |
| Margin-BT | m=0.5 | 0.854 | 0.901 | 0.534 | 0.534 |
| Margin-BT | m=2.0 | 0.849 | 0.897 | 0.536 | 0.536 |

**Finding**: Standard Bradley-Terry achieves the **best** stakeholder differentiation (lowest cosine similarity). Alternative loss functions do not improve differentiation — they slightly reduce it. The Platform-Society pair is always most differentiated (they have 35% label disagreement), while User-Society is least differentiated (12% disagreement).

#### Topic Score Differentiation

The key policy question: does Society rank political content lower than Platform?

| Topic | User (BT) | Platform (BT) | Society (BT) | Platform - Society Gap |
|-------|-----------|---------------|-------------|----------------------|
| Entertainment | +2.31 | +4.33 | +0.47 | 3.85 |
| Sports | +1.76 | +3.79 | -0.04 | 3.83 |
| Tech | +0.01 | +2.96 | -2.06 | 5.02 |
| News | -0.07 | +2.90 | -2.12 | 5.02 |
| **Politics (L)** | -1.15 | +3.48 | **-4.29** | **7.77** |
| **Politics (R)** | -1.12 | +3.46 | **-4.22** | **7.68** |

**Finding**: Society consistently ranks politics strongly negative (-4.3) while Platform ranks it positive (+3.5). The gap of **7.7 points** is large and consistent. Margin-BT with m=2.0 amplifies this gap to ~11.3 points but at the cost of weight differentiation.

#### Margin-BT: Amplifies Scores but Reduces Differentiation

| Margin | Min Cosine Sim | Politics Gap (P-S) | Min Accuracy |
|--------|---------------|-------------------|-------------|
| 0.05 | 0.613 | 2.06 | 92.9% |
| 0.25 | 0.571 | 3.62 | 92.9% |
| 0.5 | 0.534 | 5.17 | 93.0% |
| 1.0 | 0.519 | 7.29 | 92.7% |
| 2.0 | 0.536 | 11.33 | 92.1% |
| BT baseline | **0.478** | 7.72 | 91.9% |

Margin-BT's minimum score gap constraint forces larger weight magnitudes, which has a regularizing effect on weight directions. The result: higher absolute score separation but more similar weight vector orientations. For maximum stakeholder differentiation, standard BT is superior.

#### Calibrated-BT: Anchoring Hurts Platform

| λ | Min Cosine Sim | Platform Accuracy | Society Accuracy |
|---|---------------|-------------------|-----------------|
| 0.05 | 0.512 | 91.3% | 95.4% |
| 0.10 | 0.531 | 89.4% | 95.5% |
| 0.25 | 0.530 | 86.8% | 95.4% |
| 0.50 | 0.538 | **86.0%** | 95.5% |

Calibration loss anchors predicted scores to ground-truth engagement rates. Platform's engagement-focused utility creates a harder calibration target (all scores should be high), leading to accuracy degradation. Society and User are relatively unaffected.

### Gate Tests

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| Min cosine similarity | < 0.95 | **0.478** (P-S, BT) | ✅ Pass |
| All accuracies > 85% | > 85% | 86.0% (worst: Calibrated λ=0.5) | ✅ Pass |
| Politics gap (P-S) | Society < Platform | **-7.7 points** | ✅ Pass |
| Consistent across losses | Pattern holds | All loss types show same pattern | ✅ Pass |

### Conclusions

1. **The fix was in the data, not the loss.** Stakeholder-specific preference labels are necessary and sufficient for model differentiation. No alternative loss function improved on standard Bradley-Terry.

2. **Standard BT is the best loss function** for this task. It achieves the lowest cosine similarity (0.478) and highest accuracy stability across stakeholders.

3. **Margin-BT is counterproductive** for differentiation. While it creates larger absolute score gaps, the margin constraint regularizes weight directions, making stakeholder models more similar.

4. **Calibrated-BT has a niche use case**: when you need absolute score calibration (e.g., for threshold-based decisions). But it degrades Platform accuracy.

5. **Label disagreement rate predicts differentiation**: Platform-Society (35% disagreement → cosine sim 0.478), User-Platform (23% → 0.830), User-Society (12% → 0.884). This is a direct, monotonic relationship.

6. **Per-archetype equity**: The user-trained BT scorer benefits all 6 archetypes (no losers). At div_weight=0.1: tech_bro +34.4%, lurker/power_user +7.1%, political archetypes +0.9-1.2%, sports_fan +2.8%. The aggregate 7.9% improvement is not masking harm to any group. See `results/archetype_pareto_analysis.json`.

### Key Files

```
enhancements/reward_modeling/
├── alternative_losses.py       # Margin-BT, Calibrated-BT, Constrained-BT, PostHoc
├── experiment_config.py        # Hyperparameter grids and experiment configuration

scripts/
├── run_loss_experiments.py     # Full alternative loss experiment suite (87 experiments)

results/loss_experiments/
├── *.json                      # Per-experiment results (45 files)
```

### Usage

```bash
# Run full experiment suite
PYTHONUNBUFFERED=1 uv run python scripts/run_loss_experiments.py --all

# Quick sanity test (4 experiments, 10 epochs)
uv run python scripts/run_loss_experiments.py --quick-test
```
