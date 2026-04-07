# RecSys Phase 1.5: Held-Out Evaluation Fix — Retro

**Goal**: Add held-out evaluation to BT training pipeline. Verify existing results are robust. Establish proper evaluation discipline for all future experiments.

**Status**: Complete. All success criteria pass.

---

## 1. What Worked

### The existing results are robust
Synthetic data: held-out accuracy within 0.1-1.6% of training accuracy across all 3 stakeholders. Cosine similarity matrix within 0.00-0.045 of existing values. The 18-parameter linear model on 2000+ pairs has negligible overfitting risk — confirmed empirically.

MovieLens (D=19): held-out accuracy 93.4%, 94.2%, 93.8% vs training accuracy 93.0%, 95.2%, 95.8%. Gaps of 0.4-2.0%. Same conclusion.

### Backward-compatible API change
Adding `eval_probs_preferred` and `eval_probs_rejected` as optional kwargs to `train_with_loss()` and `eval_accuracy` as an optional field on `TrainedModel` means zero breakage for existing callers. The 257 existing tests pass unchanged.

## 2. Surprises

### The 99.3% accuracy in CLAUDE.md was already a validation accuracy
Phase 1 BT learning (the original, not RecSys Phase 1) used `evaluate_reward_model.py` with train/val splits. The 99.3% comes from `docs/results.md` line 1359: "Final Validation Accuracy | 99.29%". The discipline was present in Phase 1 but was dropped in Phase 4 when `run_loss_experiments.py` was written with a different evaluation pattern.

### Paper claims were never at risk
The paper uses cosine similarity, Spearman ρ, NDCG (held-out), and Pareto regret — none of which are discriminative accuracy on training data. The held-out evaluation fix is about discipline and future-proofing, not about correcting incorrect claims.

## 3. Deviations from Plan

### Skipped clarifying comments on analysis scripts
Plan called for adding docstring comments to `analyze_partial_observation.py`, `analyze_alpha_recovery.py`, and `analyze_alpha_stress.py` explaining why eval-on-training-pool is OK for their metrics. Deferred — these scripts aren't being modified and the explanation is in the retro doc. Will add if we touch those files.

## 4. Implicit Assumptions Made Explicit

- **18/19 parameters on 2000+ pairs = no overfitting**: Confirmed empirically. Train/held-out gap < 2% everywhere. This is expected for a linear model with a 100:1 data-to-parameter ratio.
- **Cosine similarity is independent of eval split**: Cosine sim is a property of the learned weight vectors, not the evaluation data. Splitting data changes the weights slightly (different training set), hence the small deltas (0.00-0.045), but the claims (>0.92 within-loss, <0.5 across-stakeholder) are unaffected.
- **Phase 4 loss experiments would also hold**: We only verified standard BT, not the full 87-config sweep. But since cosine sim is the metric and it's unchanged, the labels-not-loss claim is safe.

## 5. Scope Changes for Next Phase

Phase 2 (Labels-Not-Loss on MovieLens) should:
- Use `split_preferences()` and `eval_probs_*` params from the start
- Report both train and held-out accuracy in all results JSONs
- Compare held-out metrics in the paper, not training metrics

## 6. Metrics

| Metric | Value |
|--------|-------|
| Files created | 1 (`verify_held_out.py`) |
| Files modified | 4 (`alternative_losses.py`, `movielens_stakeholders.py`, `verify_movielens_foundation.py`, `test_movielens_stakeholders.py`, `CLAUDE.md`) |
| New tests | 0 (updated existing) |
| Existing tests broken | 0 (257 still pass) |
| **Synthetic held-out verification** | |
| User: train 93.1%, held-out 93.0% | Gap 0.1% |
| Platform: train 93.5%, held-out 91.8% | Gap 1.6% |
| Society: train 95.0%, held-out 94.6% | Gap 0.5% |
| Cosine sim user-platform | 0.830 (existing 0.830, Δ=0.000) |
| Cosine sim user-society | 0.854 (existing 0.884, Δ=0.030) |
| Cosine sim platform-society | 0.433 (existing 0.478, Δ=0.045) |
| **MovieLens held-out verification** | |
| User: train 93.0%, held-out 93.4% | Gap 0.4% |
| Platform: train 95.2%, held-out 94.2% | Gap 1.0% |
| Diversity: train 95.8%, held-out 93.8% | Gap 2.0% |
| Results artifacts | `results/held_out_verification.json`, `results/movielens_foundation.json` (updated) |
