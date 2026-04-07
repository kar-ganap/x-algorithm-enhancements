# RecSys Phase 5: Goodhart Effect on MovieLens — Retro

**Goal**: Replicate the Goodhart effect (more data amplifies misspecification) on MovieLens.

**Status**: Complete. Weak Goodhart signal detected (min at N=50, gradual rise). Strategy 1 non-monotonicity reveals a baseline scorer confound.

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

## 6. Metrics

| Metric | Value |
|--------|-------|
| Files created | 3 (script, tests, retro) |
| New tests | 6 |
| Total tests | 282 |
| BT training runs | ~325 |
| Runtime | 1830s (~30 min) |
| **Strategy 1 (better spec)** | |
| σ=0.5 Hausdorff | 5.785 |
| σ=0.0 Hausdorff | 6.040 (NOT monotonic) |
| **Strategy 2 (more data, σ=0.3)** | |
| N=50 (minimum) | 4.689 |
| N=2000 (peak) | 5.743 |
| Goodhart amplification | 1.22× (vs synthetic 3.0×) |
| Goodhart detected | Yes (weak) |
| Results artifact | `results/movielens_goodhart.json` |
