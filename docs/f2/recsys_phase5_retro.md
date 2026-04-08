# RecSys Phase 5/5b: Goodhart Effect on MovieLens — Retro

**Goal**: Replicate the Goodhart effect (more data amplifies misspecification) on MovieLens.

**Status**: Complete after two iterations. Phase 5 was confounded (learned scorer changed content selection). Phase 5b fixed with fixed scorer + weight normalization. Result: **Goodhart inconclusive on MovieLens** — weak signal within noise, Strategy 1 non-monotonic.

---

## 1. What Worked

### Goodhart signal exists but is weak
Strategy 2 shows Hausdorff minimum at N=50 (4.689) rising to N=2000 (5.743) — a 22% increase. The pattern is real but much flatter than synthetic (which showed a 3× increase). The Goodhart mechanism exists on real data, but the effect size is smaller.

### Experiment ran cleanly
325 BT training runs + frontier computations in 30 min. No crashes, all conditions converged. The infrastructure from Phases 1-3 held up.

## 2. Surprises

### Strategy 1 is NOT monotonic
Hausdorff INCREASES as σ → 0: from 5.785 (σ=0.5) to 6.040 (σ=0.0). Better specification makes things *worse* by this metric. This is the opposite of synthetic, where Strategy 1 is perfectly monotonic.

**Root cause**: The baseline frontier uses a composite scorer (mean of user+platform+diversity). The individual BT-learned weights (even with σ=0, trained on true user preferences) produce a different scorer than the composite. The Hausdorff distance includes scorer mismatch, not just specification error.

On synthetic data this isn't a problem because all 3 stakeholder weights are more distinct (cos 0.48-0.88), so the composite scorer is a reasonable middle ground. On MovieLens, user and platform are nearly identical (cos 0.96), and diversity is opposed (cos -0.4). The composite is dominated by user+platform, so a BT model trained on true user weights produces weights close to the composite — but the diversity component in the composite shifts it enough to create measurable Hausdorff.

### Binary genre features limit the Goodhart effect
With continuous action probabilities (synthetic), misspecification creates fine-grained mislabeling — many pairs get flipped by small utility differences. With binary genre vectors, most pairs have the same genre-level ranking regardless of small weight perturbation. The misspecification needs to flip genre importance ordering (e.g., "Drama better than Action" → "Action better than Drama") to change labels, which requires larger σ.

## 3. Deviations from Plan

### Longer runtime than estimated
Plan estimated ~12 min; actual was ~30 min. The `compute_scorer_eval_frontier` on 1305 movies with greedy top-K is slower than on 100-500 synthetic items. 650 frontier computations × ~2.5s each ≈ 27 min.

## 4. Implicit Assumptions Made Explicit

- **Hausdorff distance conflates scorer mismatch with specification error**: On MovieLens, even σ=0 produces Hausdorff ~6.0 because the learned scorer differs from the composite baseline scorer. The Goodhart signal (the increase from N=50 to N=2000) sits on top of this baseline. The paper should report the DELTA (N=2000 Hausdorff - N=50 Hausdorff = 1.05) rather than absolute values.
- **Binary features dampen the Goodhart effect**: The mechanism requires misspecification to produce learnable-but-wrong signals. Binary genre features are less susceptible than continuous action probabilities.
- **The Goodhart effect is feature-space dependent**: Synthetic (D=18, continuous) shows 3× amplification; MovieLens (D=19, binary) shows 1.2×. This is itself a finding — the effect magnitude depends on feature richness.

## 5. Implications for Paper

The Goodhart finding partially replicates: the direction (more data eventually hurts under misspecification) is confirmed, but the magnitude is much weaker on real data. 

**Honest framing**: "The Goodhart curve replicates on MovieLens (Hausdorff minimum at N=50, rising 22% by N=2000), but the effect is weaker than on synthetic data (3× vs 1.2×). Binary genre features dampen the effect because misspecification needs to reverse genre importance ordering, not just shift continuous probabilities."

## Phase 5b: Fixed Scorer + Weight Normalization

Phase 5 was confounded: learned weights drove content SELECTION, not just evaluation. Fixed by:
1. **Fixed scorer** (platform weights) for content selection in both baseline and learned conditions
2. **Weight normalization** — normalize learned weights to same L2 norm as true weights (BT is scale-invariant)

### Phase 5b Results (normalized)

**Strategy 1** still not monotonic: σ=0.5 → 3.25, σ=0.0 → 4.44. Better specification makes things WORSE. This means BT-learned weight DIRECTION systematically differs from true direction, even under correct specification, because BT optimizes for preference ordering accuracy not weight recovery.

**Strategy 2**: min at N=200 (2.82), rises to N=2000 (3.42). 21% increase, but error bars overlap (std ~1.3). Signal is within noise.

**Conclusion**: Goodhart effect on MovieLens is **inconclusive**. The mechanism (BT over-learning misspecified labels) may exist but cannot be cleanly separated from BT's inherent weight direction bias on sparse binary features.

### Key insight: Why synthetic works but MovieLens doesn't

The synthetic Goodhart experiment succeeds because:
1. Continuous [0,1] action probabilities → fine-grained mislabeling
2. Shared fixed scorer (hardcoded engagement formula) → zero confound
3. Well-separated stakeholders (cos 0.48-0.88) → large misspecification signal
4. BT weight scale naturally bounded by [0,1] features

MovieLens breaks these conditions:
1. Binary genre features → coarse mislabeling
2. BT weight direction bias on sparse features
3. User-platform near-identical → small misspecification window
4. BT scale invariance unbounded → needed normalization hack

## 6. Metrics

| Metric | Phase 5 (confounded) | Phase 5b (clean, normalized) |
|--------|---------------------|------------------------------|
| σ=0.0 Hausdorff | 6.04 | 4.44 |
| Strategy 1 monotonic | No | No |
| Strategy 2 min | N=50, 4.69 | N=200, 2.82 |
| Strategy 2 peak | N=2000, 5.74 | N=2000, 3.42 |
| Goodhart amplification | 1.22× | 1.21× |
| Goodhart detected | Yes (weak) | Yes (weak, within noise) |
| Files created | 3 (script, tests, retro) | Modified script |
| Total tests | 282 | 282 |
| Results artifact | `results/movielens_goodhart.json` (overwritten) |
