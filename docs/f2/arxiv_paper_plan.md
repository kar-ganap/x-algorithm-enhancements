# arXiv Preprint Plan

## The One-Sentence Paper

In multi-stakeholder BT preference learning, the cosine similarity between the optimization target and a hidden stakeholder's utility direction predicts whether more training data helps (cos > 0) or hurts (cos < 0) that stakeholder — validated 6/6 across 3 datasets.

## Importance Ranking of Findings

### Tier 1: The paper exists because of this

**The Goodhart direction condition.** cos(w_target, w_hidden) < 0 → hidden stakeholder degrades with more data. cos > 0 → improves. 6/6 validated. Nobody has stated or tested this before in the multi-stakeholder literature.

### Tier 2: This makes the paper more than a one-trick observation

**The Hausdorff metric correction.** The widely-used frontier distance metric produces false positive Goodhart when stakeholders are aligned. Utility-based metrics reveal no Goodhart on synthetic (all cos > 0). This corrects the original preprint's central claim.

### Tier 3: These explain WHY the condition holds

**Labels-not-loss / low-rank weight space.** BT convergence is identical regardless of loss variant → weight space is K-dimensional → convergence is movement in a K-dim space → direction is everything → cos predicts direction. This is the theoretical spine.

**Composition ≈ scalarization.** Projection cosine > 0.97 confirms the low-rank structure empirically. 3 models = 99% of 18 models' coverage. Consequence of low-rank.

### Tier 4: These make the findings believable on real data

**MovieLens validation.** Labels-not-loss replicates (cos 0.96-0.99). Temporal generalization holds (93-96%). User-group universality (7/7 groups). Scale validation (ML-1M replicates ML-100K).

**Data budget with bootstrap CIs.** 25 pairs = 50% recovery [34%, 63%]. Statistically significant.

### Tier 5: These make the findings actionable

**Audit toolkit.** Applied to X's 18 prediction targets. The threshold: does the platform weight negatives positively? One observable bit predicts societal risk. 5-step audit procedure for DSA regulators.

**Diversity weight as independent control.** Selection mechanism independence protects against individual misspecification. Design principle for recommendation systems.

### Tier 6: Interesting but appendix material

- Temporal generalization details (per-seed, per-group)
- Genre correlation robustness (condition number 921)
- Hyperparameter robustness (margin/calibrated BT)
- Downstream rating prediction (weak signal, directional)
- Nonlinear robustness (concave/threshold from original paper)
- K-stakeholder scaling (original paper)
- Functional form fitting (Gao doesn't fit — BT is "sharp Goodhart")
- Per-parameter sensitivity (rank stability = 1.0)
- Stress testing (label noise, sample size)
- LLM-as-annotator compatibility

## Paper Structure

### Title

"When More Data Hurts: A Directional Goodhart Condition for Multi-Stakeholder Preference Learning"

Or shorter: "Multi-Stakeholder Goodhart: The Cosine Condition"

### Abstract (~200 words)

Multi-stakeholder recommendation systems optimize proxy objectives that may harm unobserved stakeholders. We identify a precise condition governing when this occurs: in Bradley-Terry preference learning with K stakeholders, additional training data degrades a hidden stakeholder's utility if and only if the cosine similarity between the optimization target and the hidden stakeholder's utility direction is negative. We validate this direction condition on 6 data points across 3 datasets (synthetic, MovieLens-100K, MovieLens-1M) with zero violations. The condition is non-trivial: it correctly predicts that our synthetic benchmark exhibits no Goodhart effect (all stakeholder pairs have positive cosine), correcting an earlier Hausdorff-based analysis that falsely indicated otherwise.

The condition arises from the low-rank structure of the multi-stakeholder weight space: with K stakeholders, BT-learned weight vectors lie in a K-dimensional subspace (projection cosine > 0.97), making the optimization direction fully determined by the stakeholder geometry. Applied to X's open-sourced recommendation algorithm, we show that the Goodhart risk reduces to one observable: whether the platform treats negative signals (blocks, reports) as positive engagement. We provide an audit toolkit, a data budget analysis (25 preference pairs recover ~50% of hidden stakeholder harm), and practical guidance for platform transparency under the EU Digital Services Act.

### Section 1: Introduction (1.5 pages)

**Opens with:** X's January 2026 transparency transition. Explicit weights (2023) → implicit weights (2026). The audit question: can regulators assess multi-stakeholder welfare from released code?

**The gap:** Goodhart's law is well-formalized for single-objective RL (Skalse et al. 2024) with scaling laws (Gao et al. 2023). But nobody has addressed: with multiple stakeholders of known geometry, when does optimizing for one harm another?

**Our answer:** The cosine condition. cos < 0 → Goodhart. cos > 0 → safe. Validated 6/6.

**Contribution list:**
1. The direction condition (Tier 1)
2. Hausdorff metric correction (Tier 2)
3. Low-rank weight space explanation (Tier 3)
4. Audit toolkit applied to X's action space (Tier 5)
5. Data budget and composition results (Tiers 4-5)

### Section 2: Background and Related Work (1.5 pages)

- BT preference learning (Bradley-Terry 1952, Sun et al. 2025)
- Goodhart formalization (Skalse 2024 taxonomy, Gao 2023 scaling, Kwa 2024 catastrophic)
- Multi-stakeholder recommendation (Burke 2017, Abdollahpouri 2022)
- Platform transparency (DSA, X open-sourcing)
- **What's missing:** the intersection — nobody connects stakeholder geometry to Goodhart onset

### Section 3: Setup (1.5 pages)

- K stakeholders with utility directions w_1, ..., w_K ∈ R^D
- BT preference learning: trains w from N pairwise comparisons
- Three datasets: synthetic (D=18, 500 items), ML-100K (D=19, 1305 items), ML-1M (D=19, 3347 items)
- Stakeholder definitions: user (α=1.0), platform (α=0.3), diversity (α=4.0 or anti-popularity on MovieLens)
- Evaluation: per-stakeholder utility of selected content (not Hausdorff)
- Held-out evaluation throughout (Phase 1.5 contribution)

### Section 4: The Direction Condition (2.5 pages) — CORE

**4.1 Statement**

Proposition: In linear BT preference learning with K stakeholders and finite content pool, if cos(ŵ_target, w_hidden) < 0, then U_hidden is eventually decreasing in N. If cos > 0, eventually increasing.

**4.2 Evidence**

The 6-point validation table (synthetic platform, synthetic society, ML-100K platform, ML-100K diversity, ML-1M platform, ML-1M diversity).

**4.3 The MovieLens Goodhart Curve** (FIGURE)

3-line plot: user utility ↑, platform utility ↑, diversity utility ↓ as N increases. Same experiment, same data, opposite directions predicted by cosine sign. This is the paper's key figure.

**4.4 Why Synthetic Has No Goodhart**

All synthetic stakeholder pairs have cos > 0. Society utility IMPROVES from 3.21 to 5.34. The original paper's Hausdorff-based "Goodhart" was a metric artifact (see §7).

**4.5 Connection to Skalse's Extremal Mechanism**

Our setting is extremal Goodhart, but the channel is BT convergence on a finite pool (not RL policy optimization over distributions). Tail behavior is irrelevant. The cosine condition is the multi-stakeholder extension of "the proxy and truth diverge at the extremes."

### Section 5: Why It Holds — Low-Rank Weight Space (1.5 pages) — EXPLANATION

**5.1 Labels-Not-Loss**

BT convergence is independent of loss variant. Within-loss cosine > 0.95 across 4 losses, 3 datasets. Weight vectors are determined by labels (stakeholder utilities), not training procedure. (Brief — the evidence is in the appendix.)

**5.2 The K-Dimensional Subspace**

With K=3 stakeholders, all BT-learned vectors (from any loss, any scalarization mixing) project onto the 3-vector per-stakeholder span with cosine > 0.97. The weight space IS 3-dimensional in 19-dim feature space.

**5.3 Composition ≈ Scalarization**

Per-stakeholder composition achieves 99% of scalarization hypervolume with 3 training runs instead of 18. The low-rank structure means composition algebraically subsumes scalarization. (The comparison table, briefly.)

**5.4 Implication for Goodhart**

Convergence in K-dim space = movement toward the target direction. If target and hidden are anti-correlated (cos < 0), movement toward target is movement AWAY from hidden. QED (informally).

### Section 6: Practical Implications (1.5 pages) — ACTIONABLE

**6.1 Data Budget**

25 hidden-stakeholder preference pairs recover ~50% of harm. Bootstrap CIs: [34%, 63%]. Validated on ML-100K and ML-1M. Cost: 5 annotators × 5 comparisons.

**6.2 Audit Toolkit Applied to X**

The 18-action classification. The scenario table (pure engagement → GOODHART; any negative penalty → safe). The one observable: does the platform treat blocks as engagement? The 5-step audit procedure.

**6.3 Design Principle: Selection Mechanism Independence**

The diversity weight δ is orthogonal to stakeholder utilities → individual misspecification doesn't change the optimal operating point (rank stability = 1.0). Recommendation: maintain at least one structural control dimension.

### Section 7: Methodological Note — Hausdorff False Positive (1 page) — SECONDARY FINDING

The original analysis used Hausdorff distance between Pareto frontiers. This metric conflates three things: weight scale, scorer precision, and actual stakeholder harm. Only the third is Goodhart.

On synthetic data: Hausdorff increases with N (appears to show Goodhart). Utility-based metrics show ALL stakeholders improve (no Goodhart). The Hausdorff result was measuring optimization precision, not harm.

Recommendation for the alignment community: when evaluating Goodhart in multi-stakeholder settings, measure per-stakeholder utility on selected content, not frontier distance.

### Section 8: Limitations (0.5 pages)

- **Scale gap:** 19-dim genre features vs 8B transformer parameters. The direction condition is a property of linear BT — whether it transfers to deep models is open.
- **Feature richness:** Binary genre features limit the Goodhart magnitude. Continuous features (the synthetic case) would show stronger effects if stakeholders were opposed.
- **Magnitude not predicted:** The condition predicts DIRECTION (6/6) but not MAGNITUDE. ML-1M has more degradation with less anti-correlation. Content pool geometry matters.
- **Linear utilities only:** All stakeholder utilities are w·φ(x). Nonlinear set-level utilities (genre entropy) were tested but didn't add signal beyond the linear direction condition.
- **Three stakeholders only:** K=3 on real data. The K-stakeholder scaling (K up to 10) is synthetic only.

### Section 9: Conclusion (0.5 pages)

The direction condition. The practical toolkit. The honest assessment of where it applies and where it doesn't.

### Appendix

A. Labels-not-loss full results (within-loss cosine matrices, 87 experiments)
B. Temporal generalization (per-group, per-seed)
C. User-group universality (7 MovieLens genre groups)
D. MovieLens-1M full replication tables
E. Scalarization comparison details (hypervolume, dominated fractions)
F. Functional form fitting (Gao's α√d-βd doesn't fit — "sharp Goodhart")
G. Nonlinear robustness (concave/threshold utilities)
H. K-stakeholder scaling (K=3,5,7,10)
I. Stress testing (label noise, sample size, temperature, correlation)
J. Per-parameter sensitivity (rank stability table)
K. Bootstrap CI methodology
L. Genre correlation matrix

## Estimated Length

Main paper: ~12-13 pages
References: ~1.5 pages
Appendix: ~8-10 pages
Total: ~22-25 pages

## Data Sources for Each Section

| Section | Results file(s) |
|---------|----------------|
| §4 Direction condition | `goodhart_condition_validation.json`, `movielens_goodhart.json`, `ml-1m_goodhart.json`, `synthetic_goodhart_utility.json` |
| §5 Low-rank / composition | `ml-100k_deep_analysis.json`, `ml-1m_deep_analysis.json`, `*_scalarization_baseline.json` |
| §6 Data budget | `movielens_loso.json`, `ml-1m_loso.json`, `*_deep_analysis.json` (bootstrap) |
| §6 Audit toolkit | `audit_toolkit.json` |
| §7 Hausdorff correction | `utility_sensitivity.json` (original), `synthetic_goodhart_utility.json` (corrected) |
| App A Labels-not-loss | `movielens_labels_not_loss.json`, `ml-1m_labels_not_loss.json` |
| App B-C Temporal/groups | `movielens_labels_not_loss.json` (groups C, E) |
| App D ML-1M replication | `ml-1m_*.json` |
| App E Scalarization | `*_scalarization_baseline.json` |
| App F Functional form | `functional_form_fit.json` |

## Figures

1. **The money figure** (§4): 3-line plot of user/platform/diversity utility vs N. User↑, platform↑, diversity↓. Annotated with cosine values.
2. **The direction condition table** (§4): 6-point validation. Could also be a scatter plot (cos vs degradation).
3. **The audit scenario table** (§6): Platform scenarios with cos values and risk levels.
4. **The low-rank projection** (§5): Bar chart of projection cosines for 18 scalarized models.
5. **System pipeline** (§3): TikZ diagram from original paper (adapted).

## Writing Priority

1. §4 (the finding) — write this first, it's the reason the paper exists
2. Abstract — distill §4 into 200 words
3. §3 (setup) — enough for §4 to make sense
4. §6 (practical) — the "so what"
5. §5 (explanation) — the "why"
6. §7 (metric correction) — the methodological contribution
7. §1-2 (intro/background) — framing
8. §8-9 (limitations/conclusion) — honesty
9. Appendix — completeness
