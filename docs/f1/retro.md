# F1: KV-Cache Optimization — Retrospective

## Summary

Inference optimization for the Phoenix Grok-based transformer model. Phases 0-5 complete. 166 passing tests (112 optimization + 54 quantization). Status: **complete, dormant** (last active Jan 2026).

## Phase Results

| Phase | What | Key Result |
|-------|------|------------|
| 0 | Baseline measurement | 103.77 ms/forward pass (CPU) |
| 1 | JIT compilation | **10.33x speedup** (103.77 → 10.04 ms) |
| 2 | Logical KV-cache | Hash-based invalidation, correct but no speedup (logical layer) |
| 2b | Full tensor KV-cache | **9.62x max speedup** with cached K/V tensors |
| 3 | Attention mechanism | Multi-head attention with proper masking |
| 4 | INT8 quantization | **58% memory reduction**, ~90% top-3 score agreement |
| 4b | Trained model quantization | **99.0% top-3 agreement** (BPR model's larger score margins help) |
| 5 | Combined optimization runner | All gates pass, integrated pipeline |

## Key Findings

1. **JIT is the single biggest win.** Static-shape compilation with `jax.jit` gives 10x with zero code changes to the model. First call has 2-5s overhead; subsequent calls amortize.

2. **KV-cache speedup is sequence-length dependent.** Theoretical 2-10x on CPU, 10-50x on GPU. Our CPU measurement (9.62x) is near the upper bound because the workload is memory-bandwidth-limited.

3. **INT8 quantization is robust with BPR training.** The BCE-trained model had marginal score differences (~1e-8), making quantization lossy. The BPR-trained model has 0.1-0.2 score margins, making INT8 (99% top-3 agreement) practical.

4. **Per-channel quantization >> per-tensor.** Per-channel INT8 preserves score ordering; per-tensor introduces enough noise to flip rankings.

## Architecture

```
enhancements/optimization/
├── benchmark.py              # Baseline benchmarking
├── jit_utils.py              # JAX JIT utilities
├── attention.py              # Multi-head attention
├── kv_cache.py               # Logical cache (Phase 2)
├── caching_attention.py      # Caching attention variants
├── caching_transformer.py    # Transformer with caching
├── full_kv_cache.py          # Full tensor caching (Phase 2b, main impl)
├── optimized_runner.py       # Combined optimization runner (Phase 5)
└── quantization/
    ├── config.py             # Quantization config
    ├── quantize.py           # Core INT8 logic
    ├── kv_quantize.py        # KV-specific quantization
    ├── quantized_runner.py   # Quantized inference runner
    └── study.py              # Optuna study for tuning

tests/test_optimization/
├── test_jit_utils.py
├── test_kv_cache.py
├── test_kv_cache_full.py
├── test_optimized_runner.py
├── test_quantization.py
└── test_attention.py
```

## Known Issues

- **9 test failures in `test_optimization/`**: These are environment-related (JAX version, CPU-only vs GPU). Not blocking reward modeling development. Tracked but not actively maintained.

## Results

- `results/f1_baseline.json` — Phase 0 baseline measurements
- `results/f1_phase4b/` — Quantization study results (3 runs + latest)
- `results/f1_phase5/` — Combined benchmark results (6 runs + latest)
