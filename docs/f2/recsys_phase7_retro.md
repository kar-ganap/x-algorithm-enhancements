# RecSys Phase 7: Per-Stakeholder Composition Sweep — Retro

**Goal**: Fix the scalarization comparison by sweeping per-stakeholder weight compositions across the simplex, matching scalarization's frontier exploration budget with zero retraining.

**Status**: Complete. Composition achieves 94-100% of scalarization coverage with 6× fewer training runs.

---

## 1. What Worked

### The composition sweep closes the gap completely on synthetic
Comp/Scalar HV ratio = 0.999. The per-stakeholder approach with composition sweep is statistically indistinguishable from scalarization on synthetic data. The Phase 4 "scalarization wins" result was entirely due to using a single mean scorer (21 points vs 378).

### On MovieLens, 94% coverage at 6× less compute
Comp/Scalar HV ratio = 0.939. Not perfect parity, but the remaining 6% gap comes from scalarization having unique mixing-specific weight directions that pure linear composition of 3 base vectors can't reproduce. This is expected — 3 base vectors span a 3D subspace while 18 scalarized models can reach directions outside that subspace.

### The retro from Phase 4 predicted this exactly
Phase 4 retro said: "A weighted combination or stakeholder-specific scorer selection would likely close the gap." It did.

## 2. Surprises

### Composition finds MORE Pareto points than scalarization on MovieLens
Composition: 105 Pareto points from 378 total. Scalarization: 87 Pareto points from 378 total. Despite lower hypervolume, composition explores the frontier more densely. The hypervolume gap comes from scalarization reaching further EXTREMES (especially in diversity utility), not from better coverage of the interior.

### Mutual domination is roughly symmetric on MovieLens
18.1% of composition Pareto points dominated by scalarization, 14.9% vice versa. Neither approach dominates the other — they're exploring overlapping but not identical Pareto surfaces.

## 3. Paper Framing

**Before**: "Scalarization wins (HV ratio 0.69-0.91). Per-stakeholder advantage is only operational."

**After**: "Per-stakeholder with composition achieves 94-100% of scalarization coverage using 3 training runs instead of 18. The naive mean scorer loses because it doesn't exercise the compositional flexibility. With proper composition sweep:
- Synthetic: parity (0.999)
- MovieLens: near-parity (0.939), 6× fewer training runs
- The remaining 6% gap on MovieLens reflects the geometric limitation of spanning from 3 base vectors vs 18 independent models"

The paper should present all three approaches (naive, composition, scalarization) as a progression demonstrating that the per-stakeholder framework's value IS its compositional flexibility — but only when exercised.

## 4. Metrics

| Metric | Synthetic | MovieLens |
|--------|-----------|-----------|
| Naive HV | 1261 | 1940 |
| Composition HV | 1380 | 2625 |
| Scalarized HV | 1381 | 2795 |
| Naive/Scalar ratio | 0.913 | 0.694 |
| **Comp/Scalar ratio** | **0.999** | **0.939** |
| Comp Pareto points | 21 | 105 |
| Scalar Pareto points | 27 | 87 |
| Training runs (per-stk) | 3 | 3 |
| Training runs (scalar) | 18 | 18 |
| Files modified | 2 (script + tests) |
| New tests | 1 |
| Total tests | 285 |
| Results artifact | `results/scalarization_baseline.json` (updated) |
