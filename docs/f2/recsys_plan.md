# RecSys 2026 Submission Plan

## Context

The multi-stakeholder reward modeling preprint (21 pages, single-column) needs two things for RecSys 2026 main track (abstract Apr 14, paper Apr 21):

1. **New experiments** to address the two biggest reviewer concerns:
   - All multi-stakeholder results are synthetic-only (MovieLens only validates BT training)
   - No comparison baselines (no scalarization or other multi-stakeholder methods)
2. **Paper compression** from 21 pages to 8 pages (ACM 2-column) + anonymization

## Key Decision: Genre Features (D=19), Not Action Remapping (D=18)

The existing MovieLens adapter maps ratings to 3 of 18 Phoenix actions via a hand-designed lookup table. Using this as "real data" is indefensible — a reviewer would immediately see the signal comes from the lookup table, not from real users.

Instead: use the **19-dim binary genre vector** as the movie feature representation. BT learns weights over genres (w ∈ R^19). Stakeholder utilities are defined over genres using real MovieLens data:

- **User**: genre weights derived from the user's actual rating history (mean rating per genre)
- **Platform**: genre weights = popularity × avg_rating (engagement/retention proxy)
- **Diversity**: genre weights = negative genre popularity fraction (penalizes overrepresented genres)

This is honest: "Preference pairs derived from real user-movie interactions using genre features." The dimension change (19 vs 18) is a *feature* — it shows the methodology generalizes beyond Phoenix's action space.

## What NOT to Replicate on MovieLens

- **α-recovery**: The pos-neg structure (U = pos − α·neg) has no natural counterpart in genre space.
- **Nonlinear robustness**: Already adequately covered on synthetic data.
- **Stress testing**: Not worth the extra complexity.

## Phases

### Phase 1: MovieLens Multi-Stakeholder Foundation
Create `movielens_stakeholders.py` — stakeholder utility definitions, preference pair generation, content pool construction. Generalize `train_with_loss()` for arbitrary D. Tests.

### Phase 2: Labels-Not-Loss on MovieLens
Run 4 BT loss variants × 3 stakeholders on MovieLens genre preferences. Verify convergence (cos > 0.90).

### Phase 3: LOSO + Data Budget on MovieLens
Hide each stakeholder, measure regret. Sweep hidden stakeholder sample size N ∈ {0..2000}.

### Phase 4: Scalarization Baseline
Implement scalarization (mixed-preference training). Run on both synthetic and MovieLens. Compare frontier coverage against per-stakeholder approach.

### Phase 5: Goodhart on MovieLens (if time permits)
Misspecify genre weights, show data amplifies error. Lowest priority — drop if behind.

### Phase 6: Paper Compression
ACM 2-column reformat, 21→8 pages, integrate new results, anonymize, submit.

## Experiments → Paper Claims

| Paper Claim | Synthetic Result | MovieLens Expected | If Doesn't Replicate |
|-------------|-----------------|-------------------|---------------------|
| Labels-not-loss (cos > 0.92) | 79/87 converge | cos > 0.90 across 4 losses | Report as "domain-dependent" |
| Society costs 10× to hide | Regret 1.08 vs 0.11 | Diversity-hidden > user-hidden | Still informative |
| 25 pairs = -42% regret | 20 seeds, p < 0.01 | Similar recovery curve | Report actual numbers |
| Goodhart (data amplifies misspec) | Hausdorff ↑ after N=100 | Same pattern expected | Drop from MovieLens section |
| Scalarization < per-stakeholder | Not yet tested | Per-stakeholder dominates | Report honestly |

## Risk Mitigation

1. **BT fails on D=19 genres**: Test Day 1. Fallback: augment with avg_rating, num_ratings (D=22).
2. **Stakeholders don't disagree enough**: Check label disagreement Day 1. If < 5%, increase diversity weight magnitude.
3. **Claims don't replicate**: Report honestly as a nuanced finding.
4. **Time overrun**: Priority: Phase 4 (scalarization) > Phase 2 (labels-not-loss) > Phase 3 (LOSO) > Phase 5 (Goodhart). Drop Phase 5 first.

## Timeline

| Day | Phase | Deliverable |
|-----|-------|-------------|
| 1 (Mon Apr 7) | Phase 1 | `movielens_stakeholders.py` + tests passing |
| 2 (Tue Apr 8) | Phase 2 | `results/movielens_labels_not_loss.json` |
| 3 (Wed Apr 9) | Phase 3 | `results/movielens_loso.json`, `results/movielens_data_budget.json` |
| 4 (Thu Apr 10) | Phase 4 | `results/scalarization_*.json` |
| 5 (Fri Apr 11) | Phase 5 + viz | Figures + analysis |
| 6 (Sat Apr 12) | Results writing | Draft expanded §7 |
| 7 (Sun Apr 13) | Abstract submit | Abstract by Apr 14 AoE |
| 8-15 (Apr 14-21) | Phase 6 | 8-page paper submitted |

## Output Files

```
results/movielens_labels_not_loss.json
results/movielens_loso.json
results/movielens_data_budget.json
results/movielens_goodhart.json
results/scalarization_synthetic.json
results/scalarization_movielens.json
docs/f2/paper/figures/fig7_movielens.pdf
```

## Critical Files

| File | Action | Why |
|------|--------|-----|
| `enhancements/data/movielens_stakeholders.py` | CREATE | MovieLens → multi-stakeholder bridge |
| `enhancements/reward_modeling/scalarization_baseline.py` | CREATE | Scalarization comparison |
| `scripts/experiments/run_movielens_multistakeholder.py` | CREATE | Experiment runner |
| `tests/test_reward_modeling/test_movielens_stakeholders.py` | CREATE | Test suite |
| `scripts/visualization/visualize_movielens_multistakeholder.py` | CREATE | Figure generation |
| `enhancements/reward_modeling/alternative_losses.py` | MODIFY | Generalize for D≠18 |
| `enhancements/reward_modeling/k_stakeholder_frontier.py` | MODIFY | Custom scorer support |
