# Tier 2 Phase C: MIND-small Experiments — Retro

**Goal**: run all 8 experiment/analysis scripts on MIND-small to produce the JSON results needed for §4, §5, §6 of the paper. Data production, not analysis or figure regeneration (those are Phase E).

**Status**: Complete 2026-04-11. All 8 scripts ran to completion. 3 Phase B refactor misses surfaced and were fixed as Phase C commits 1-2. The most scientifically important finding is that BT-trained weight cosines diverge from raw stakeholder cosines on MIND's high-dim sparse features, affecting direction-condition predictive power.

---

## 1. What Worked

### Pre-flight smoke tests caught zero drift vs Phase A

The registry smoke test reproduced Phase A's cosine matrix to within 1e-3 at every entry. The noisy label-balance check (with σ=0.05 matching BT's noise) showed all 5 stakeholders at frac_first ≈ 0.5, not the misleading 0.348 that the noise-free metric had flagged for advertiser. This was a cheap pre-flight that would have caught any mid-Phase-B drift in the registry or stakeholder modules. It flagged nothing — confirming the Phase B refactor preserved the Phase A geometry exactly.

### Two-batch parallelism proved safer than 4- or 8-way

The first Phase B baseline attempt deadlocked 7 parallel `uv run` invocations on cold JAX cache. Phase C's strategy — warmup (scalarization) + Batch 1 (4-way) + Batch 2 (2-way) — ran to completion with zero contention issues. Wall clock: ~3 hours total (warmup 31 min + Batch 1 ~70 min + Batch 2 ~44 min + analyze + fix-up). The warmup cost was higher than estimated (31 min vs ~3 min on ml-100k) because the 5-simplex has 126 mixing points vs the 3-simplex's 21 — but it still acted as the JAX cache warmer for Batch 1.

### Regression tests on ml-100k after every code change

Both Phase B-miss fixes (extremal rename, hypervolume K-dim) required re-verification that ml-100k byte-equivalence still held. Both passed. The byte-equivalence guarantee from Phase B continued to hold across 2 additional Phase C code changes without me having to manually reason about whether the changes could break the K=3 path.

### Cross-script geometry sanity cross-checks

`goodhart`, `extremal`, and `expanded_direction` all report cosines on the same (target, hidden) pairs. On MIND the raw stakeholder cosines match Phase A's geometry check to within rounding. This caught nothing because there was nothing to catch — but if the registry or stakeholder module had silently drifted, any of the 3 scripts would have surfaced it independently.

---

## 2. Surprises

### THE SCIENTIFIC FINDING: BT-trained cosines flip sign vs raw cosines on MIND

This is the single most important result from Phase C and needs to be in the paper.

On MovieLens, the BT-trained weight cosines preserve the raw stakeholder cosines to within ~0.05. On MIND they do NOT. Three specific pairs flip sign between raw and BT-trained:

| Pair | Raw cosine | BT-trained cosine | Sign flip? |
|---|---|---|---|
| reader ↔ journalist | -0.152 | **+0.371** | YES |
| reader ↔ advertiser | +0.396 | **-0.161** | YES |
| publisher ↔ journalist | -0.038 | **+0.385** | YES |
| publisher ↔ civic_diversity | -0.839 | -0.958 | No (stronger) |
| publisher ↔ advertiser | -0.057 | **-0.654** | No (much stronger) |
| reader ↔ publisher | +0.482 | +0.081 | No (much weaker) |
| advertiser ↔ civic_diversity | +0.401 | +0.686 | No (stronger) |

Two observations:
- **Substantial divergence is not symmetric**. Some pairs get amplified (publisher-advertiser from -0.057 to -0.654), some get weakened (reader-publisher from +0.482 to +0.081), some flip sign outright.
- **The divergence is much larger than on MovieLens**, where most trained-vs-raw cosines matched within ±0.05.

**Why this matters for the paper's §4 direction condition**: the direction condition's predictive power is stated as "cos(w_target_hidden_true, w_target_hidden_model) < 0 iff hidden stakeholder degrades as N grows." The critical question is: which cosine — raw weights or BT-trained weights?

- If raw: MovieLens evidence is valid (they agree), MIND evidence contradicts (match rate 65%).
- If trained: need to re-score Method A using trained cosines and check.

My Phase C Method A implementation uses raw weights for the cosine (via `configs[stakeholder]` from the registry). This is the "intent" cosine — what an analyst would compute before running any training. The match rate of 65% on MIND tells us: **the direction condition with raw cosines is not reliable on high-dim sparse feature spaces where BT training substantially remaps the geometry.**

**The paper needs to either (a) state the condition with trained cosines and re-test, or (b) add a qualifier that the raw-cosine version only holds when the BT training roughly preserves the raw geometry (which is empirically true for low-dim dense MovieLens features but NOT for high-dim sparse MIND features).**

I suspect the correct formulation is (a) — trained cosine is the "mechanism" cosine, raw cosine is a proxy that happens to work when geometry is preserved. Phase E should explicitly compute trained cosines on MIND and re-score Method A. If the match rate rises to ~100%, we have the correct formulation.

### 3 Phase B misses surfaced at Phase C launch

This was the lesson-generating surprise of Phase C. All three were K-dependent bugs invisible under ml-100k regression testing:

1. **`run_extremal_evidence.py` line 360**: `out_path = ROOT / "results" / "extremal_evidence.json"` — the Phase B refactor changed the script from multi-dataset (loops over ml-100k + ml-1m) to single-dataset (`--dataset` flag), but missed updating the output filename to include the dataset prefix. Running Phase C on mind-small would have silently overwritten the ml-100k Phase B baseline. Caught by the Phase C plan's critical-read of the script.

2. **`run_scalarization_baseline.py` line 155**: `samples = rng.uniform(mins, maxs, size=(n_samples, 3))` — the function was named `compute_hypervolume_3d` (giveaway sign), and the hardcoded `3` was a K-dependent literal that only worked for MovieLens's 3 stakeholders. On MIND's 5-stakeholder simplex, the `mins` array has shape (5,) and the broadcast fails. Caught by running the warmup on mind-small — it crashed immediately.

3. **`scripts/analysis/analyze_recsys_deep.py` line 86**: `ds_key = "movielens" if "movielens" in data else list(data.keys())[0]` — the fallback `list(data.keys())[0]` picks whatever key is first in the top-level dict. On ml-100k this happens to be "config" but then the `if "movielens" in data` branch short-circuits. On mind-small "movielens" isn't a key and the fallback picks "config" — not the dataset subtree. Caught by running analyze_recsys_deep on mind-small — it returned an error about missing `weight_vectors`.

All three bugs had the SAME root cause: **byte-equivalence on a fixed-K dataset cannot catch K-dependent bugs**. This is a fundamental limitation of the Phase B regression test design. See Lesson 13 below.

### Data budget recovery is WAY higher on MIND than MovieLens

N=25 recovery for `hide_civic_diversity` (the diversity-equivalent on MIND) via bootstrap analysis is **84.2% [72.3%, 94.5%]**. On ml-100k it was 56.3% [48.2%, 63.9%] for `hide_diversity`. Why?

Hypothesis: MIND's pool is 12K items (9× MovieLens's 1.3K) which gives each stakeholder much more signal per preference pair. The `hide_civic_diversity` regret is absolute 3.38 on MIND vs 3.26 on ml-100k (similar magnitude), but N=25 training recovers much more of it on MIND.

Alternative hypothesis: MIND's 5 stakeholders form a richer geometric structure where training a single "hidden" direction is more informative than MovieLens's 3-stakeholder structure. The hidden stakeholder is maximally distinguishable from the observed 4 in MIND, vs the observed 2 in MovieLens.

This finding strengthens the §6 data budget claim: 25 preference pairs recover 42-85% of hidden stakeholder harm depending on the dataset, and the MIND numbers are actually more favorable for the "N=25 is enough" story.

### Composition projection cosine is lower on MIND than MovieLens

Mean projection cosine on MIND: **0.893** (vs MovieLens ~0.97+). Min: 0.564 (vs MovieLens ~0.95). 30/121 models above 0.95 (vs nearly all on MovieLens).

The K-dimensional subspace claim in §5 says: scalarized weight vectors should project onto the span of per-stakeholder vectors with cosine > 0.97. On MIND this is only true for 30/121 of the scalarized models. The low-rank structure is weaker, but still present (mean ~0.89 is far from random).

**Why**: the 5-simplex has 126 mixing points (vs 3-simplex's 21), so there are 6× more scalarized models to project onto a 5-dim subspace. The subspace is also higher-dim (5 vs 3) which may be easier to fit in theory, but the extra mixing points expose more corner cases where scalarized training finds weights outside the span.

The paper's §5 claim needs a weaker version on MIND. Phase E should discuss the subspace claim as a spectrum, not a binary, and show MIND as the intermediate case.

### Direction consistency on MIND is 85% — slightly better than MovieLens

MIND: 17/20 pairs have σ-consistent direction (rate 85%). MovieLens: 10/12 = 83%. The 3 inconsistencies on MIND are all in the transition zone |cos| < 0.2 — matching the MovieLens pattern exactly. Direction consistency holds uniformly across σ values for pairs in the strong-cosine regions.

This is a clean positive result: the Goodhart "direction depends only on geometry, not noise level" claim holds on MIND as well.

### labels_not_loss partial success on MIND

Within-loss similarity (core labels-not-loss claim: different losses produce the same weights when given the same labels):
- **3 of 5 stakeholders ABOVE 0.85 threshold**: advertiser (0.965), publisher (0.944), reader (0.911)
- **2 of 5 stakeholders BELOW threshold**: journalist (0.847), civic_diversity (0.824)

MovieLens had all 3/3 clearly above 0.95. MIND's 2 borderline cases are exactly the 2 stakeholders with the strongest negative relationships to the cluster: `journalist` (which opposes everything popular) and `civic_diversity` (anti-concentration). Hypothesis: when a stakeholder's weight vector has extreme cosines with the others, BT training is more sensitive to the loss function's regularization. Needs Phase D to confirm (does Amazon's `diversity` and `indie_author` show the same borderline pattern?).

### `run_data_budget_all_hidden.py` serial runs take ~45 min per run on MIND

The plan estimated 16-25 min. Actual: 44 min. The larger pool (12K vs 1.3K) and 5 hidden stakeholders (vs 3) made this much slower than anticipated. Total Phase C wall clock: ~3 hours instead of the planned ~80 min. Not a blocker, but worth updating the Phase D estimate.

---

## 3. Deviations from Plan

### 2 additional code-change commits beyond the 1 planned

The plan's Commit 1 was the extremal filename fix. I added a Commit 1.5 (`9b8d27e`) with TWO more Phase B-miss fixes (hypervolume 3→nd, analyze_recsys_deep key heuristic). Both had to be fixed for Phase C to complete, so this was necessary — but it violates the planned "3 commits, 1 code fix + 1 results + 1 retro" structure. New structure:

1. `49e504a` — extremal_evidence filename fix (planned)
2. `9b8d27e` — hypervolume + analyze_recsys_deep key fixes (NEW, Phase B miss)
3. `2685d36` — MIND result JSONs (planned, Commit 2)
4. This commit (retro + plan update + lessons — Commit 3)

Total 4 commits on `recsys/phase-c-mind`. The plan said 3; actual is 4 because of the 2 surfaced Phase B misses. Documented.

### scalarization ran THREE times during Phase C

1. First attempt (pre-fix): CRASHED on hypervolume shape mismatch
2. ml-100k re-verify (post-fix): PASSED byte-equivalent
3. mind-small (post-fix): ran to completion in 31 min

The extra two runs added ~35 min wall clock. Acceptable.

### `analyze_recsys_deep.py` ran TWICE on mind-small

1. First attempt (pre-fix): composition block errored with "weight_vectors not in results"
2. Second attempt (post-fix): composition block populated

Extra ~1 min wall clock. Negligible.

### Total wall clock: ~3 hours instead of planned ~80 min

The plan's ~80 min estimate was based on MovieLens timings. On MIND, every BT-based script takes 1.5-2× longer due to the larger pool and 5-stakeholder simplex. Revised Phase D estimate: expect ~3 hours for Amazon Kindle experiments (similar pool scale, 5 stakeholders).

---

## 4. Implicit Assumptions Made Explicit

### "The refactored scripts will just work on a new dataset"

False — the Phase B regression proved byte-equivalence on ml-100k, but the guarantee does not extend to non-3-stakeholder datasets. Three bugs invisible under K=3 surfaced immediately under K=5. The right frame: **Phase B's regression test is a NECESSARY condition (refactor preserves K=3 behavior) but not SUFFICIENT (refactor works on K≠3)**. A complete regression test would need to also run on a K≠3 dataset — which is exactly what Phase C became, unintentionally.

### "The raw stakeholder cosine is a good proxy for the BT-trained cosine"

False on high-dim sparse features. On MovieLens's 19-dim (mostly binary genre features but each movie has multiple genres set) the raw and trained cosines agree to ~0.05. On MIND's 35-dim (one-hot categories + one-hot subcategories = mostly zero vectors), BT training substantially remaps the geometry. Three pairs even flip sign. The paper's direction condition needs to be stated carefully about WHICH cosine matters.

### "labels_not_loss will hold universally"

False on borderline stakeholders. The Within-loss similarity > 0.85 threshold is a simplification. On MIND, 2 of 5 stakeholders (journalist, civic_diversity) are below it. The labels-not-loss claim is a SPECTRUM parameterized by dataset geometry, not a binary. MovieLens is the easy case where it's cleanly above threshold for all stakeholders.

### "analyze_recsys_deep is read-only"

True (it only reads JSON and computes), but that didn't stop it from having a bug — the dataset-key heuristic silently picks the wrong subtree. The "read-only = safe" assumption led to minimal testing of this script in Phase B.

### "Per-script sanity checks are sufficient"

Partially false. The inline `jq` checks after each script caught structural issues (wrong number of points, missing keys). But they couldn't catch the SCIENTIFIC surprise — the trained-vs-raw cosine divergence would have passed all sanity checks (numbers are finite, fields are populated, shapes are right) while silently being a major finding. Phase C's "scientific interpretation" pass — comparing BT-trained across-stakeholder cosines against raw cosines — was an ad-hoc addition I did after labels_not_loss ran, not a planned sanity check.

---

## 5. Scope Changes for Next Phase (Phase D — Amazon Kindle)

### Add raw-vs-trained cosine comparison to the Phase D run report

Phase D should explicitly compute and report `abs(raw_cos - trained_cos)` for every pair on Amazon. If Amazon also shows substantial divergence (likely, given premium_seller's exact-zero orthogonals), the finding generalizes. If Amazon shows agreement (like MovieLens), the finding is MIND-specific and suggests a feature-density explanation.

### Expect ~3 hours wall clock on Amazon, not the planned ~80 min

Revised Phase D estimate based on Phase C actuals. Amazon Kindle has 17K items, 5 stakeholders, 32 features — similar scale to MIND. Budget ~3 hours for the BT runs + retro + commits.

### Amazon orthogonal pairs need a focused analysis

The Phase A retro flagged `premium_seller` vs `publisher`/`indie_author`/`diversity` all having cos = 0.000 exactly. Phase D should report what the direction condition predicts at cos = 0 and whether the observed behavior is random (+/- 50%) or biased. This is a unique test case not available on MovieLens or MIND.

### Phase E will need more than "just regenerate figures with 4 datasets"

The raw-vs-trained cosine finding is a paper-level structural change. Phase E needs to:
1. Compute trained cosines on all 4 datasets (requires running labels_not_loss through an extended analysis, not just figure-rendering)
2. Re-score Method A of the direction condition using trained cosines
3. Compare trained-cos match rate to raw-cos match rate on each dataset
4. Update §4 of the paper to clearly distinguish the two cosine formulations
5. If trained-cos is the right formulation, the paper's direction condition becomes a post-hoc prediction (you need to train to know), which has different implications than the raw-cos version (analysts can check before training)

This is a larger-than-planned scope bump for Phase E, but it's the honest finding.

### The `verify_movielens_foundation.py` refactor is blocking clean degradation analysis

The degradation block in analyze_recsys_deep is still bogus on non-MovieLens datasets because `verify_movielens_foundation.py` was not registry-refactored in Phase B. Phase F (or a dedicated cleanup) should fix this. Low priority — composition and bootstrap blocks cover the key §5 and §6 claims.

---

## 6. Metrics

| Metric | Value |
|--------|-------|
| Wall-clock time (total) | ~3 hours |
| Plan estimate | ~80 min |
| Overrun factor | 2.25× |
| Files created | 8 result JSONs + 1 retro |
| Files modified | 3 (analyze_recsys_deep.py, run_scalarization_baseline.py, ml-100k_deep_analysis.json) |
| Files modified (Phase B misses) | 3 (extremal, hypervolume, analyze_recsys_deep) |
| Commits on `recsys/phase-c-mind` | 4 (extremal fix, hypervolume+analyze fix, MIND results, retro) |
| Planned commits | 3 |
| Phase B misses caught + fixed | 3 |
| Phase B misses that would have shipped silently without Phase C | 3 (extremal would have overwritten ml-100k; hypervolume would crash; analyze_recsys_deep would silently mis-read) |
| Scripts refactored | 0 (Phase C is run-only except for the 3 Phase B fixes) |
| Tests modified | 0 |
| Tests passing before Phase C | 107 |
| Tests passing after Phase C | 107 (unchanged) |
| MIND pool size | 12,261 items (vs MovieLens 1,305 — ~9.4× larger) |
| MIND feature dim | 35 |
| MIND stakeholder count | 5 |
| Direction condition match rate on MIND (|cos|>0.2) | 65% (raw cos) |
| Direction condition match rate on ml-100k (|cos|>0.2) | ~100% |
| Data budget recovery at N=25, civic_diversity hidden | 84.2% [72.3%, 94.5%] |
| Data budget recovery at N=25, diversity hidden on ml-100k | 56.3% [48.2%, 63.9%] |
| Composition projection cosine mean (MIND) | 0.893 |
| Composition projection cosine mean (ml-100k) | 0.979 |
| labels_not_loss within-loss cos ≥ 0.85 on MIND | 3 of 5 stakeholders |
| labels_not_loss within-loss cos ≥ 0.85 on ml-100k | 3 of 3 stakeholders |
| Sign flips in BT-trained vs raw cosine on MIND | 3 of 10 pairs |
| Sign flips on MovieLens | 0 of 3 pairs |
| Direction consistency across σ on MIND | 17/20 (85%) |
| Direction consistency across σ on ml-100k | 10/12 (83%) |
