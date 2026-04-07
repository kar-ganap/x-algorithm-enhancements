# RecSys Phase 4: Scalarization Baseline — Retro

**Goal**: Compare per-stakeholder BT training against scalarization (mixed-preference training) as a baseline. Run on both synthetic and MovieLens.

**Status**: Complete. Scalarization wins on hypervolume. Honest finding — challenges the assumed superiority of per-stakeholder training.

---

## 1. What Worked

### Fair comparison design
Giving scalarization 21 models × 21 diversity weights = 378 points (vs per-stakeholder's 21) was the right call — it's generous to the baseline, so when scalarization wins, the result is robust. No one can say we sandbagged the baseline.

### Both datasets tell the same story
Synthetic: scalarization HV 1381 vs per-stakeholder 1261 (ratio 0.91). MovieLens: 2683 vs 1838 (ratio 0.69). The baseline is competitive to dominant on both, which means this is a structural finding, not a dataset artifact.

### Honest reporting catches a real limitation
If we had assumed per-stakeholder wins and not run the baseline, we'd have submitted a paper with an implicit claim ("our multi-model approach is better") that's falsifiable.

## 2. Surprises

### Scalarization wins — the paper's framing needs adjustment
The paper implicitly assumes per-stakeholder training is superior to naive alternatives. This experiment shows it isn't, at least on hypervolume. Scalarization with diversity sweep is a strong baseline.

**Why it wins**: Scalarization trains 18 different models (one per mixing point minus 3 vertex points), each representing a different stakeholder tradeoff. When each is swept across diversity weights, the union of 378 points covers the Pareto surface more thoroughly than 21 points from a single composite scorer.

### Per-stakeholder approach has a negative diversity utility on MovieLens
Max diversity utility from per-stakeholder: -3.15. From scalarization: +4.52. The per-stakeholder composite scorer (mean of user+platform+diversity weights) is dominated by user+platform direction (cos 0.96 between them), which actively harms diversity. Scalarization with high diversity mixing weight produces scorers that genuinely serve the diversity stakeholder.

### The problem is the composite scorer, not the separate training
Per-stakeholder training produces 3 good weight vectors. The limitation is in how they're COMBINED at serving time — a simple mean favors the aligned user+platform pair over the opposed diversity stakeholder. A weighted combination or stakeholder-specific scorer selection would likely close the gap.

## 3. Deviations from Plan

### MC hypervolume instead of exact
Used Monte Carlo (50K samples) for 3D hypervolume. Exact algorithms exist but add complexity. MC is sufficient for comparison — both approaches are measured with the same method.

### Skipped vertex points in scalarization
Points with mixing weight > 0.99 on one stakeholder are excluded (they're equivalent to per-stakeholder training). This leaves 18 of 21 mixing points, producing 378 frontier evaluations.

## 4. Implicit Assumptions Made Explicit

- **Per-stakeholder ≠ better by default**: The advantage of separate models is compositional flexibility at serving time. But if composition is done naively (mean), that flexibility isn't realized. Scalarization explores the space more thoroughly by construction.
- **Hypervolume favors thorough exploration**: With 378 vs 21 points, scalarization has a structural advantage. A fairer comparison would use k-stakeholder-specific scorers (use each of the 3 learned weight vectors separately, generating 63 points). This would test the compositional advantage directly.
- **The diversity knob is doing most of the work**: Both approaches rely on the same diversity weight sweep for tradeoffs. The difference is only in the engagement scorer. The diversity mechanism is the primary frontier-shaping mechanism, not the scorer.

## 5. Implications for Paper

This finding strengthens the paper if framed correctly:

**Wrong framing**: "Per-stakeholder training produces better frontiers than scalarization."
**Right framing**: "Scalarization is a competitive baseline (HV ratio 0.69-0.91), but requires retraining for each operating point. Per-stakeholder training produces reusable weight vectors that can be composed at serving time without retraining — an operational advantage even if the Pareto coverage is narrower with naive composition."

The operational cost argument: scalarization needs K models for K operating points. Per-stakeholder needs K models total for unlimited operating points (via composition). The paper should frame this as a cost-coverage tradeoff, not a dominance claim.

## 6. Metrics

| Metric | Synthetic | MovieLens |
|--------|-----------|-----------|
| Per-stakeholder HV | 1260.6 | 1838.0 |
| Scalarized HV | 1381.4 | 2682.6 |
| HV ratio (per-stk/scalar) | 0.91 | 0.69 |
| Dominated fraction | 0% | 3.4% |
| Per-stk Pareto points | 4 | 13 |
| Scalarized Pareto points | 27 | 87 |
| Scalarized total points | 378 | 378 |
| Runtime | 345s total | (included) |
| New tests | 8 | |
| Total tests | 276 | |
| Files created | 3 (script, tests, retro) | |
| Results artifact | `results/scalarization_baseline.json` | |
