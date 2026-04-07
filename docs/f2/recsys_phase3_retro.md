# RecSys Phase 3: LOSO + Data Budget on MovieLens — Retro

**Goal**: Replicate partial observation findings on MovieLens: which stakeholder is costliest to hide, and how many preference pairs to recover.

**Status**: Complete. LOSO regret ranking matches expected structure. Data budget shows 53% recovery at N=25.

---

## 1. What Worked

### LOSO regret ranking replicates
Diversity (5.469) > platform (5.108) > user (3.533). The stakeholder most opposed to the engagement scorer (diversity, cos -0.42 with platform) incurs the most regret. Same structural finding as synthetic (society > platform > user).

### Data budget shows rapid recovery
N=25 pairs reduce regret by 53% (3.257 → 1.535). Stronger than synthetic's 42%, likely because 19-dim genre features are a cleaner signal than 18-dim synthetic actions.

### `compute_scorer_eval_frontier` was the right abstraction
Separating the selection scorer from the evaluation weights avoids the BT scale-invariance problem that caused the smoke test failure. Learned weights drive content selection; true weights evaluate utility.

## 2. Surprises

### Negative regret at N≥200
At N=200+, diversity regret goes negative (-0.437 to -6.181). The learned diversity scorer shifts content selection toward diverse content so effectively that the frontier EXCEEDS the full baseline on diversity utility. This is correct: the baseline uses platform weights for selection (which favor popular content), while the diversity-augmented scorer includes diversity knowledge. The "improvement" beyond the baseline is real — having diversity info in the scorer, not just in evaluation, is strictly better.

This is actually a finding worth reporting: **adding stakeholder knowledge to the content selection scorer (not just the evaluation) can improve beyond the full-information baseline** when that baseline uses a stakeholder-agnostic scorer.

### Single-user frontier is sufficient
Using `base_probs[1, M, 19]` (no user dimension) worked cleanly. The frontier is over content diversity, not user personalization. This is honest and avoids fabricating per-user variation.

### `num_topics = len(np.unique())` bug
The index-out-of-bounds on genre 18 (with `np.unique` returning 18 values but max index being 18) required fixing to `max(content_topics) + 1`. This was a latent bug in `compute_k_frontier` that only manifested with sparse genre indices.

## 3. Deviations from Plan

### Geometric LOSO only (no training-based)
Plan mentioned adding training-based LOSO if time permitted. Went with geometric LOSO only, which is the cleaner comparison. Training-based LOSO would re-select content using only 2 observed stakeholder weights, but the separation of scorer/evaluator in `compute_scorer_eval_frontier` achieves this more cleanly.

### Spearman correlation as secondary metric
Added weight-space Spearman correlation between learned and true diversity weights as a secondary data-budget metric. This validates that BT is recovering the right genre-level signal (ρ = 0.47 at N=25, 0.76 at N=2000).

## 4. Implicit Assumptions Made Explicit

- **BT scale invariance affects regret computation**: Learned weights can have arbitrary scale, so comparing `w_learned @ features` against `w_true @ features` directly produces meaningless regret. The fix: use learned weights for SELECTION only, evaluate with true weights. This is the same issue the synthetic pipeline sidesteps by using a fixed engagement scorer.
- **"Diversity" is the hardest stakeholder to observe on MovieLens**: This parallels "society" on synthetic. Both are anti-correlated with the default engagement scorer. The finding is structural: stakeholders opposed to the platform objective are costliest to miss.
- **Negative regret is informative, not broken**: When adding hidden-stakeholder info to the scorer improves beyond the baseline, it demonstrates the value of stakeholder-aware content selection — a practical finding for recommendation system design.

## 5. Scope Changes for Next Phase

Phase 4 (Scalarization Baseline) should:
- Use the same `compute_scorer_eval_frontier` pattern for fair comparison
- Compare the per-stakeholder frontier against scalarization (mixed-preference training)
- Consider reporting the "exceeds baseline" finding from negative regret

## 6. Metrics

| Metric | Value |
|--------|-------|
| Files created | 2 (`run_movielens_loso.py`, `test_movielens_loso.py`) |
| Files modified | 1 (`k_stakeholder_frontier.py` — lazy import, index fix) |
| New tests | 5 (smoke tests) |
| Total tests | 268 passed |
| Runtime | 98s |
| **LOSO Regret** | |
| Hide diversity | 5.469 |
| Hide platform | 5.108 |
| Hide user | 3.533 |
| Regret ranking | diversity > platform > user ✓ |
| **Data Budget** | |
| LOSO baseline (N=0) | 3.257 |
| N=25 regret | 1.535 (53% recovery) |
| N=200 regret | -0.437 (exceeds baseline) |
| N=2000 regret | -6.181 |
| Weight Spearman at N=25 | 0.470 |
| Weight Spearman at N=2000 | 0.760 |
| Results artifact | `results/movielens_loso.json` |
