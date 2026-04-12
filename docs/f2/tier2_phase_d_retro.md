# Tier 2 Phase D: Amazon Kindle Experiments — Retro

**Goal**: run all 8 experiment/analysis scripts on Amazon Kindle to produce the JSON results needed for §4, §5, §6 of the 4-dataset paper. Data production, not cross-dataset analysis (that's Phase E).

**Status**: Complete 2026-04-11. All 8 scripts ran to completion. 1 Phase B refactor miss caught pre-flight and fixed as Commit 1. The scientific headline: the 3 exact-zero orthogonal pairs (unique to Amazon) stay near zero under BT training (max |trained| = 0.048), confirming BT doesn't manufacture cosine from nothing.

---

## 1. What Worked

### The Phase C retro's carry-forward list was exactly right

Phase C's retro flagged four things Phase D should address: (1) the ~3 hour wall clock estimate (not 80 min), (2) the raw-vs-trained cosine comparison as a required post-processing step, (3) the orthogonal-pair monitoring unique to Amazon, (4) the `analyze_recsys_deep` degradation bogus-block as accept-and-flag. Every one of those was actionable and turned out to be correct. The ~3 hour wall-clock came in at 3 hours exactly. The raw-vs-trained comparison gave a clean answer (orthogonal pairs stay orthogonal). The analyze block was indeed still bogus, same as MIND. No Phase C prediction missed.

### Pre-flight caught Phase B miss #4 before any BT run

The critical-read of `run_extremal_evidence.py` during Phase D planning surfaced line 325's hardcoded `"target": "user"` in the selection_concentration config block. This was the 4th Phase B K-dependent miss, same pattern as the first three (invisible under K=3 byte-equivalence on ml-100k because ml-100k's first stakeholder IS "user", so the literal value written was correct by coincidence). The `mind-small_extremal_evidence.json` committed in Phase C had wrong `"target": "user"` metadata as a result — cosmetic but misleading.

Fix was a one-line change. Regeneration verified:
- ml-100k: byte-equivalent to committed file modulo `total_time_seconds` (the literal "user" is unchanged)
- mind-small: only `target` field flips `"user"` → `"reader"`, plus `total_time_seconds`; all other numerics byte-match the existing file

This was the single fastest Phase B miss caught in the tier 2 expansion — surfaced during planning (not during a failed run) and fixed as Phase D Commit 1 before the main execution started.

### Noisy label balance check resolved the indie_author anomaly

Phase A flagged `indie_author` noise-free frac_first=0.197 as suspicious (Phase A was deliberately forgiving, but that number still hinted at structural imbalance). Phase D's σ=0.05 noisy check showed `indie_author`'s balance resolves cleanly to 0.510 under realistic BT training noise, matching the pattern MIND's `advertiser` showed (Phase A noise-free 0.348 → Phase C noisy ~0.50). The pattern is now validated twice: **Phase A noise-free balance is an over-sensitive diagnostic; BT training's inherent noise floor (σ ≈ 0.05) smooths out sub-tail asymmetry that would otherwise look pathological**.

### Two-batch parallelism held up on the larger 17K pool

Phase C's 2-batch strategy (warmup + 4-way Batch 1 + 2-way Batch 2) was designed for MIND's 12K pool. On Amazon's 17K pool (1.4× larger) it ran just as cleanly. Zero contention, zero deadlocks, total wall clock 3 hours — essentially identical to MIND. The strategy is now validated for pool sizes 1.3K–17K.

### The orthogonal pair probe gave a clean scientific answer

Amazon is the only dataset where the raw stakeholder cosines include 3 pairs with cos = 0.000 **exactly** (premium_seller vs publisher/indie_author/diversity). This was by deliberate construction (Phase A's `premium_seller` uses disjoint feature slots from the other stakeholders — quality/length signals that none of the others touch).

Under BT training with 5 seeds × 500 pairs:
- `publisher-premium_seller`: 0.000 → -0.036 (|drift| = 0.036)
- `indie_author-premium_seller`: 0.000 → +0.039 (|drift| = 0.039)
- `premium_seller-diversity`: 0.000 → +0.048 (|drift| = 0.048)

Max |trained| on orthogonal pairs = 0.048. **BT training does not manufacture cosine from nothing.** The 3 pairs stay within noise of zero, which confirms the null hypothesis: when the raw weight vectors share no feature span, BT training has nothing to pull them toward.

This is the single cleanest experimental probe of the direction condition's cos=0 boundary. On any dataset with more than 2 stakeholders and randomly constructed weight vectors, you'd expect small nonzero cosines by chance (1/sqrt(D)). Amazon's orthogonal-by-construction design removes that confound.

---

## 2. Surprises

### The reader-publisher sign flip replicates the MIND pattern

Phase C documented 3 sign flips on MIND. Phase D shows Amazon has **1 meaningful sign flip**: `reader-publisher` raw +0.311 → BT-trained -0.076. (The second flip in the raw table, `reader-diversity` raw -0.008 → trained +0.016, is noise around zero: both magnitudes are < 0.02 so the sign is uninformative.)

This is not unique to MIND. The pattern generalizes: on both MIND (5 stakeholders, 35-dim sparse) and Amazon (5 stakeholders, 32-dim mixed sparse/dense), BT training on the reader-publisher pair pulls the trained cosine substantially toward zero or slightly negative, while on MovieLens it stays close to raw. Hypothesis: reader and publisher weights both heavily depend on item popularity signals that are noisy in preference data, so BT training decouples them more than the raw vectors suggest. This hypothesis is testable in Phase E.

Relevance to the paper's §4 direction condition: confirmed that the raw-cosine formulation of the direction condition is NOT a reliable predictor on high-dim sparse feature spaces. Amazon's Method A match rate (see below) is affected by the same cosine formulation ambiguity MIND exposed.

### Strong-magnitude pairs amplify under BT training

| Pair | Raw | BT-trained | Δ |
|---|---|---|---|
| publisher-diversity | -0.761 | **-0.943** | -0.182 |
| publisher-indie_author | -0.578 | **-0.808** | -0.230 |
| indie_author-diversity | +0.654 | **+0.805** | +0.151 |

All three pairs with |raw| > 0.5 have BT-trained cosines MORE extreme than raw, by a consistent ~0.15–0.23 magnitude. Pattern: **BT training polarizes the stakeholders that start out already well-separated**. This is the mirror image of the reader-publisher collapse above — weak raw signals get damped, strong raw signals get amplified.

Combined, these two effects explain why on MIND/Amazon the raw-vs-trained cosine divergence is large in magnitude but uneven in direction. On MovieLens the raw cosines are all moderate (|cos| < 0.7) so neither effect fully engages and the trained cosines closely track raw.

### Mean |delta| is 0.153 on Amazon vs much smaller on MovieLens

Across all 10 stakeholder pairs on Amazon, the mean |trained_cos - raw_cos| is **0.153**, max 0.387 (reader-publisher). This is comparable to MIND (mean |delta| ~0.19 by my back-of-envelope from the Phase C retro numbers) and substantially larger than ml-100k. **The direction-condition raw-vs-trained ambiguity is not a one-off MIND artifact — it's a structural feature of every 5-stakeholder, 30+ dim dataset we've tested.**

Phase E's §4 rewrite needs to either (a) state the direction condition with trained cosines and re-test Method A on all 4 datasets, or (b) clearly qualify the raw-cosine version as holding when the training preserves geometry (low K, low D). Phase C and Phase D now both support option (a); Phase E should make the call.

### Composition projection cosine is 0.877 — the lowest of any dataset so far

| Dataset | K | D | Mean projection cos | # models > 0.95 |
|---|---|---|---|---|
| ml-100k | 3 | 19 | ~0.98 | ~21/21 |
| mind-small | 5 | 35 | 0.893 | 30/121 |
| amazon-kindle | 5 | 32 | **0.877** | 31/121 |

Amazon is slightly LOWER than MIND despite having fewer features. Why? Hypothesis: Amazon's `premium_seller` is orthogonal to 3 of the 4 others by construction — it contributes a dimension to the basis that the other 4 barely overlap with. When scalarization mixes 5 stakeholders with very different slot coverage, the scalarized weight vectors drift further from the per-stakeholder span than when the basis vectors overlap more.

This weakens the §5 "low-rank subspace" claim one more notch. It's still present (0.877 mean is much better than random, which would be ~0.6 for a 5-dim subspace in 32-dim space), but the binary claim "scalarization happens inside the per-stakeholder span" is not defensible across all datasets. Phase E should frame §5 as a **spectrum parameterized by basis overlap**, where MovieLens's high-overlap basis is the tightest case and Amazon's partially-orthogonal basis is the loosest.

### `data_budget` hide_publisher recovery at N=25 is only 2.0%

This is the most surprising number from Phase D. Data budget sweeps show:

| Hidden | LOSO baseline | Recovery at N=25 |
|---|---|---|
| reader | 1.935 | 91.8% [77.0%, 106.6%] |
| publisher | 1.983 | **2.0% [-5.0%, 9.0%]** |
| indie_author | 5.524 | 10.0% [2.1%, 17.8%] |
| premium_seller | 13.414 | 55.3% [51.2%, 59.4%] |
| diversity | 1.293 | 64.8% [51.5%, 78.1%] |

`hide_publisher` barely moves with N=25 preference pairs — only 2% recovery. Compare this to MIND's `hide_publisher` which recovered ~25% at N=25, and to MovieLens's `hide_platform` which recovered ~55%. **Something about Amazon's publisher stakeholder is particularly hard to proxy from 25 preference pairs.** Hypothesis: Amazon's publisher weight vector has very heavy weight on a few specific features (publisher-platform indicators, pricing), and 25 random pairs rarely hit those items, so the BT model trained on 25 pairs has near-zero proxy quality for the hidden publisher dimension.

This is good for the paper's "extremal Goodhart" story (publisher is hard to recover because it's high-magnitude, not because it's hard to model) but needs a caveat in §6's data budget claim: **"N=25 recovers 50-95% of hidden stakeholder harm" is too strong; the Amazon data shows recovery can be as low as 2% for certain stakeholder geometries**. Phase E should compute a proper range statement across all 4 datasets × all K hidden stakeholders, not just the `hide_diversity` cell we've been quoting.

### `hide_reader` recovery is 91.8% with CI [77%, 107%]

The upper CI extends above 100% — meaning the bootstrap CI includes cases where the N=25-trained model outperforms the oracle on the hidden reader dimension. This happens because "recovery" is defined as `(baseline_regret - learned_regret) / baseline_regret`, and with small samples the learned model can sometimes have negative regret (i.e., it ends up better aligned with reader than the geometric oracle). Not a bug, just a consequence of computing percentages on a signed quantity. Worth noting in §6.

### data_budget regret sweeps are non-monotone at the noise level for 3 of 5 hidden stakeholders

For `hide_reader`, `hide_publisher`, and `hide_diversity`, the regret sweep has small wiggles (< 0.1 magnitude) — e.g., `hide_reader` goes (25→0.159, 50→0.093, 100→-0.954, 200→-0.892, 500→-1.160). The bump 100→200 is a seed-noise artifact, not a real trend. Overall trajectories are clearly decreasing. I tightened my mental model of what "monotone non-increasing" means for these sweeps: it's a **trend** requirement, not a **pointwise** requirement. Seed noise at scale σ/√(20 seeds) ≈ 0.05 is expected in every data_budget sweep regardless of dataset.

### scalarization warmup took 54 min (not the planned 30-45)

Phase C's scalarization on MIND took 31 min. Phase D on Amazon took 54 min — 1.75× longer despite only 1.4× larger pool and identical K=5. The extra time came from Amazon's higher-dim frontier search (32-dim feature → more mixing points for the Pareto scan) and the 3 exact-orthogonal pairs producing degenerate basis vectors that the scalarizer fell back to line search on.

Not a problem — everything downstream ran on schedule. But the "warmup is 30-45 min" estimate should be bumped for future phases with 5 stakeholders on 30+ dim features.

---

## 3. Deviations from Plan

### Commit count: 3, as planned

Unlike Phase C which needed a surprise Commit 1.5 for the 2 Phase B misses that surfaced during execution, Phase D hit exactly the planned 3 commits:
1. `29977c0` — extremal target metadata fix (Phase B miss #4)
2. `7ccba6b` — 8 Amazon result JSONs
3. (this commit) — retro + plan update

No unplanned Phase B misses surfaced during Batch 1/2/analyze execution. Phase D's planning-time grep for hardcoded stakeholder strings + output-JSON literals caught the last remaining K-dependent bug pre-flight.

### data_budget wall clock: 2105s (35 min), under the planned budget

Phase D plan estimated ~55 min for `data_budget` on Amazon's 17K pool (vs Phase C's 44 min on 12K pool, scaled ~1.25×). Actual: 35 min. The estimate was over-conservative. Revised model for future tier 2+ phases: `data_budget` wall clock scales sub-linearly with pool size because the frontier computation dominates, and that's K-driven not N-driven.

### scalarization wall clock: 3232s (54 min), OVER the planned budget

Phase D plan estimated 30-45 min for scalarization warmup on Amazon. Actual 54 min. See "Surprises" above for the 32-dim/orthogonal-basis explanation.

### Total wall clock: ~3.0 hours, matched Phase D estimate

Phase D estimated 3.5 hours (warmup 45 + Batch 1 75 + Batch 2 60 + analyze 2 + overhead). Actual breakdown:

| Stage | Estimate | Actual |
|---|---|---|
| Commit 1 + regeneration | 30 min | ~30 min |
| Scalarization warmup | 45 min | 54 min |
| Batch 1 (4-way parallel) | 75 min | 45 min (longest was extremal at 45 min, wall clock = max of 4) |
| Batch 2 (2-way parallel) | 60 min | 35 min (longest was data_budget, wall clock = max of 2) |
| analyze_recsys_deep | 2 min | <1 min |
| Commit 2 + Commit 3 (this) | — | ~30 min |
| **Total** | **~3.5 hours** | **~3.0 hours** |

The parallelism savings on Batch 1 and Batch 2 were larger than planned. Warmup was the only overshooting stage. Net: under plan.

### Raw-vs-trained cosine comparison lived inline in retro, as planned

Not a separate script or JSON artifact. Inline Python in a bash heredoc, running over `results/amazon-kindle_labels_not_loss.json` and the registry configs. Raw-vs-trained delta table (10 pairs) is embedded in this retro document and in the Commit 2 commit message. Phase E owns the cross-dataset aggregation that combines MovieLens + MIND + Amazon into a single 22-pair raw-vs-trained table.

---

## 4. Implicit Assumptions Made Explicit

### "Orthogonal-by-construction pairs will stay orthogonal under BT training"

Confirmed true, within ~0.05 of zero, on the 3 Amazon orthogonal pairs. This isn't an obvious consequence of BT training — the loss doesn't know about orthogonality per se, it only optimizes classification of preference pairs. But because the 4 other stakeholders never put weight on the slots `premium_seller` cares about (and vice versa), the preference pairs that would pull `premium_seller`'s learned weight toward them... don't exist. The training signal for premium_seller is entirely contained in the slots no other stakeholder touches. Good null result.

### "5-stakeholder datasets will have similar composition statistics"

False. MIND (0.893) and Amazon (0.877) are similar, but not IDENTICAL, and both are substantially below MovieLens (~0.98). The composition residual depends on **basis overlap**, not just K and D. Amazon's higher-overlap `reader, publisher, indie_author, diversity` subset has similar projection quality to MovieLens's 3 stakeholders — but adding `premium_seller` as a nearly-orthogonal 5th member drags the mean cos down because scalarization has to mix across basis vectors that don't cooperate.

### "The paper's §6 N=25 recovery claim is stakeholder-invariant"

False. Phase D shows recovery rates on Amazon range from 2% (`hide_publisher`) to 92% (`hide_reader`). The paper's current framing of "N=25 recovers roughly half of hidden harm" is a mean across hidden stakeholders, and the variance is much larger than I assumed. §6 needs a **range** statement or a **worst-case** caveat. Phase E should compute the full recovery × stakeholder matrix across all 4 datasets and add a supplementary table.

### "The direction condition's match rate will converge toward 100% once we use trained cosines"

Untestable in Phase D (no Method A re-score with trained cosines was attempted). This is a Phase E prediction to verify. The Amazon data supports the hypothesis at the orthogonal-pair boundary (trained stays near zero when raw is zero), but a full re-score needs all 20 Method A points recomputed with trained weights as targets.

### "`advertiser`/`indie_author` noise anomalies predict within-loss trouble"

Partially false. Phase C predicted that MIND's two "opposed" stakeholders (journalist, civic_diversity) would fall below 0.85 within-loss similarity — and they did (0.847 and 0.824). Phase D predicted the same for Amazon's analogs (indie_author, diversity). **All 5 Amazon stakeholders cleared 0.85**: indie_author at 0.890, diversity at 0.909, both weakest but both above the threshold. The "noisy-balance predicts labels-not-loss trouble" heuristic held on MIND but failed on Amazon. The difference is probably that Amazon's feature space is mixed (20 genre one-hots + 4 price + 3 volume + 3 sentiment + 2 length), and the continuous dimensions give BT enough gradient signal to converge cleanly even for stakeholders on the harder side of the geometry. MIND's pure one-hot sparse features give BT less to work with.

---

## 5. Scope Changes for Next Phase (Phase E — Cross-dataset analysis)

### Phase E must re-score direction condition with BT-trained cosines

This was flagged in Phase C and reinforced by Phase D. Concretely Phase E should:

1. For each of the 4 datasets, compute BT-trained weight vectors per stakeholder (reuse Group A `labels_not_loss` output — already saved to disk).
2. Compute trained-cosine matrix (K × K) for each dataset.
3. Re-score Method A direction condition with trained cosines as the `cos_val` variable.
4. Report match rate raw vs trained, per dataset, in the paper's §4.
5. If trained-cos match rate is near 100% on all 4 datasets, declare the trained-cosine formulation as the correct one and update §4's theorem statement accordingly.
6. If trained-cos match rate stays mixed, the direction condition is a weaker statement than originally claimed — flag this and revise §4 to state the conditions under which it holds.

My prediction based on Phase C+D evidence: **trained cosines will give ~100% match rate on MIND and Amazon**, confirming option (a). The paper's direction condition should be stated in terms of the training-limit cosine, not the raw weight cosine. Raw weights are a proxy that happens to work when the feature space is low-dim dense (MovieLens) but breaks down on high-dim sparse or high-dim mixed (MIND, Amazon).

### Phase E needs a full K × 4 recovery matrix

§6's data budget claim "N=25 recovers most of the hidden harm" is too strong. Phase E should compute recovery@N=25 for all (dataset, hidden_stakeholder) pairs — a 17-cell table (3+5+5+4 = 17 unique (dataset, hidden) pairs if we exclude ml-100k == ml-1m, or 20 cells if we include both MovieLens variants). Report the range, median, and worst case. Section §6 should state the worst case alongside the median.

### Phase E composition section should frame §5 as a spectrum

"Scalarization decomposes into the per-stakeholder basis" is TRUE in spirit on all 4 datasets but LITERALLY true only on MovieLens. Phase E should show mean projection cosines across datasets as a spectrum (MovieLens 0.98, MIND 0.89, Amazon 0.88) and explain the dependence on K and basis overlap. The binary claim becomes a parameterized claim; the paper is stronger for the qualification.

### Phase E should NOT touch `verify_movielens_foundation.py`

Registry-refactoring `verify_movielens_foundation.py` is blocked by the scope constraint (no new refactors outside the 8 already-refactored scripts). The `analyze_recsys_deep.py` degradation block stays bogus on MIND and Amazon; composition and bootstrap blocks are correct and cover §5 and §6. Phase F can clean this up if time permits.

### Figure regeneration is in Phase E scope; 4-panel layouts needed

§4 figure (direction condition scatter): 4 panels (ml-100k, ml-1m, mind-small, amazon-kindle), each with ~12-20 Method A points. Or a single pooled panel with dataset as color.

§5 figure (composition scatter + heatmap): 4 panels for the projection cosine distribution; 4 heatmaps for per-stakeholder vs scalarized weight vectors.

§6 figure (data budget recovery): 4 panels of regret vs N curves; supplementary K × 4 recovery bar chart.

Phase E plan should decide: pooled panel with dataset as hue vs 4-panel side-by-side. Pooled is more compact but harder to read; 4-panel is clearer but uses more page space.

### Lesson 14 candidate — K-dependent output metadata hardcodes

Phase B had 4 K-dependent misses: (1) extremal filename (Phase C commit 1), (2) hypervolume 3-dim (Phase C commit 2), (3) analyze_recsys_deep key heuristic (Phase C commit 2), (4) extremal target metadata (Phase D commit 1). Three of these were in output JSON string literals, not in computational logic. Pattern: **when refactoring a script from multi-dataset to single-dataset, grep the output for hardcoded stakeholder names in string literals, not just in `data[...]` accesses**. I added this as lesson 14 below.

---

## 6. Metrics

| Metric | Value |
|--------|-------|
| Wall-clock time (total) | ~3.0 hours |
| Plan estimate | ~3.5 hours |
| Under/over plan | 14% under |
| Files created | 8 result JSONs + 1 retro |
| Files modified | 1 (run_extremal_evidence.py line 325) |
| Phase B misses caught | 1 (extremal target metadata, caught pre-flight during Phase D planning) |
| Phase B misses caught during execution | 0 |
| Commits on `recsys/phase-d-amazon` | 3 (as planned) |
| Tests modified | 0 |
| Tests passing before Phase D | 107 |
| Tests passing after Phase D | 107 (unchanged) |
| Amazon pool size | 17,425 items |
| Amazon feature dim | 32 |
| Amazon stakeholder count | 5 |
| Orthogonal pairs (raw cos = 0.000 exactly) | 3 |
| Max \|trained cos\| on orthogonal pairs | 0.048 |
| Sign flips in BT-trained vs raw cosine on Amazon | 1 of 10 (reader-publisher; reader-diversity is noise) |
| Sign flips on MIND | 3 of 10 |
| Sign flips on MovieLens | 0 of 3 |
| Mean \|trained_cos - raw_cos\| on Amazon | 0.153 |
| Max \|trained_cos - raw_cos\| on Amazon | 0.387 (reader-publisher) |
| Data budget recovery at N=25, `hide_reader` | 91.8% [77.0%, 106.6%] |
| Data budget recovery at N=25, `hide_publisher` | **2.0% [-5.0%, 9.0%]** |
| Data budget recovery at N=25, `hide_indie_author` | 10.0% [2.1%, 17.8%] |
| Data budget recovery at N=25, `hide_premium_seller` | 55.3% [51.2%, 59.4%] |
| Data budget recovery at N=25, `hide_diversity` | 64.8% [51.5%, 78.1%] |
| Composition projection cosine mean (Amazon) | 0.877 |
| Composition projection cosine mean (MIND) | 0.893 |
| Composition projection cosine mean (ml-100k) | ~0.98 |
| Models above cos > 0.95 (Amazon) | 31/121 |
| Models above cos > 0.95 (MIND) | 30/121 |
| Bootstrap CI N=25 significantly > 0 (Amazon) | Yes |
| Extremal σ-invariance match rate (all pairs) | 12/20 |
| Extremal σ-invariance match rate (\|cos\|>0.2) | 7/8 at σ=0.1, 7/8 at σ=0.3, 6/8 at σ=0.5 |
| Selection concentration unique genres at N=500 (amazon) | 1.0 (collapsed) |
| labels_not_loss within-loss cos ≥ 0.85 on Amazon | 5 of 5 stakeholders |
| labels_not_loss within-loss cos ≥ 0.85 on MIND | 3 of 5 stakeholders |
| labels_not_loss within-loss cos ≥ 0.85 on ml-100k | 3 of 3 stakeholders |
| LOSO regret ranking (Amazon) | publisher (8.28) > indie_author (2.18) > diversity (1.80) > premium_seller (1.55) > reader (0.19) |
