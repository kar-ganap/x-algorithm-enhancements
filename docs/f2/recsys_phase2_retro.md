# RecSys Phase 2: Labels-Not-Loss on MovieLens — Retro

**Goal**: Replicate labels-not-loss on real MovieLens data, plus 4 MovieLens-specific extensions (temporal, downstream, user-groups, genre correlation).

**Status**: Complete. All success criteria pass. 85 training runs + analysis in 19 minutes.

---

## 1. What Worked

### Labels-not-loss replicates convincingly on real data
Within-loss cosine similarity: user 0.957, platform 0.987, diversity 0.958. All exceed the 0.85 threshold AND the synthetic benchmark's 0.92. This is stronger than expected — BT convergence is even more reliable on real genre features than on synthetic action probabilities.

### Temporal generalization is the standout finding
Training on early MovieLens ratings and evaluating on late ratings: 93-95% held-out accuracy with within-loss convergence of 0.962-0.988. This is something synthetic data fundamentally cannot provide. It shows the learned weight vectors generalize across time, not just across random held-out samples.

### User-group heterogeneity shows universal convergence
All 7 genre-preference groups (Film-Noir fans, Documentary fans, etc.) show BT-margin cosine sim > 0.989. Labels-not-loss isn't just an aggregate effect — it holds per real user subpopulation with messy, overlapping boundaries.

### Genre correlation doesn't break convergence
Despite condition number 921.2 and genre correlations up to 0.465 (Animation-Children's), the BT convergence result is robust. Multicollinearity in features doesn't prevent labels-not-loss from holding.

## 2. Surprises

### Within-loss similarity is HIGHER on MovieLens than synthetic
MovieLens: 0.957-0.987. Synthetic: 0.92+. This is counterintuitive — real data should be noisier. Likely explanation: the 19-dim genre features are sparser (1.8 genres/movie) and more structured than the 18-dim synthetic action probabilities (all non-zero). Less feature noise → tighter convergence.

### User-platform alignment is even stronger than Phase 1 showed
Across-stakeholder cos (BT): user-platform 0.962. In Phase 1 (single seed), it was 0.958. With 5 seeds, it's 0.962 ± 0.005. These stakeholders are nearly identical on MovieLens — both reward highly-rated popular content. The differentiation story is entirely about diversity vs. the user+platform bloc.

### Downstream prediction is very weak
User BT Spearman = 0.058. Popularity baseline = 0.369. Genre-level utility weights capture aggregate stakeholder preferences but are far too coarse to predict individual ratings. This is an important finding: the BT weights are valid for *which direction* each stakeholder pulls, not for *how much* any specific user will enjoy a specific movie.

### Diversity weights anti-correlate with ratings
Diversity BT Spearman = -0.085 with actual ratings. This is correct by construction — diversity penalizes popular, highly-rated genres — but it's the first empirical confirmation that the diversity stakeholder genuinely opposes user preferences at the individual rating level.

### User groups cluster on niche genres
The top genre groups are Film-Noir (136), Documentary (83), War (76), etc. — these are niche genres. The majority genre groups (Drama, Comedy, Action) presumably don't form distinct clusters because most users rate movies in those genres. This is a MovieLens-specific artifact worth noting.

## 3. Deviations from Plan

### Temporal pool uses full pool for late eval, not late-only
Plan said to filter to late-only movies. In practice, we generate preference pairs from the full pool (1440 movies) for temporal evaluation, since many movies span both periods. The stakeholder utilities are computed from early ratings only, which is the key constraint.

### Calibrated-BT engagement targets use utility as proxy
Plan specified `np.clip(features @ stakeholder_weights, 0, 1)` as engagement target. This works but means the "calibration" signal is the same information the BT loss uses for ordering. Calibrated-BT still converges (eval acc > 94%) but the calibration doesn't add independent information.

## 4. Implicit Assumptions Made Explicit

- **Genre features are sufficient for stakeholder differentiation**: 19 binary genres produce meaningful user-diversity disagreement (66%) but near-zero user-platform disagreement (8%). The differentiation capacity depends entirely on which stakeholders you're comparing.
- **"Labels-not-loss" is a BT convergence property, not a feature-space property**: It holds on both D=18 synthetic actions and D=19 MovieLens genres, across temporal splits, across user subgroups, and despite multicollinearity. This is about the loss landscape, not the data.
- **Downstream prediction requires per-user features, not per-stakeholder features**: Genre-level weights predict stakeholder direction but not individual rating magnitudes. Adding user-specific genre affinities would likely close the gap but is out of scope.

## 5. Scope Changes for Next Phase

Phase 3 (LOSO + data budget on MovieLens) should:
- Use the same content pool and stakeholder configs
- Focus on user-diversity and platform-diversity regret (user-platform are too aligned for interesting LOSO)
- Consider whether temporal LOSO (hide diversity in early data, reveal in late) adds value

## 6. Metrics

| Metric | Value |
|--------|-------|
| Files created | 2 (`run_movielens_labels_not_loss.py`, `test_movielens_labels_not_loss.py`) |
| Files modified | 1 (`movielens_stakeholders.py` — added temporal pool, user-group functions) |
| New tests | 6 (smoke tests) |
| Total tests | 263 passed |
| Training runs | ~85 |
| Total runtime | 19 min |
| **Group A: Within-loss cos (mean)** | user 0.957, platform 0.987, diversity 0.958 |
| **Group A: Across-stakeholder cos (BT)** | user-plat 0.962, user-div -0.338, plat-div -0.423 |
| **Group B: Hyperparameter cos** | margin-BT 0.998, calibrated-BT 0.930-0.989 |
| **Group C: Temporal eval accuracy** | user 93.0%, platform 95.8%, diversity 94.2% |
| **Group C: Temporal within-loss cos** | 0.962-0.988 |
| **Group D: Downstream Spearman** | user 0.058, platform 0.030, diversity -0.085, popularity 0.369 |
| **Group E: Per-group within-loss cos** | All 7 groups > 0.989 |
| **Group F: Condition number** | 921.2 |
| Results artifact | `results/movielens_labels_not_loss.json` |
