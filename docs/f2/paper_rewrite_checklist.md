# Paper Rewrite Checklist: Goodhart Metric Correction

## Background

The original preprint's Goodhart finding (§6.4) uses Hausdorff distance on synthetic data. Phase 10 showed:
- Hausdorff measures frontier distance, not stakeholder harm
- Under utility-based metrics, synthetic shows NO Goodhart (all stakeholders improve)
- Goodhart IS real on MovieLens (diversity degrades 42-46%) because cos(user, diversity) < 0
- The direction condition (cos < 0 → degrades) is 6/6 validated across 3 datasets

The paper needs to replace the Hausdorff-based Goodhart claim with the direction-condition-based finding.

## Specific Revisions

### Abstract (lines 117-120)
**Current**: "With misspecified weights, additional training data amplifies the error after N > 100 pairs — a Goodhart effect. Data helps when specification is correct; it hurts when specification is wrong."

**Change to**: "With misspecified weights pointing away from a hidden stakeholder (cosine similarity < 0), additional training data degrades that stakeholder's utility — a multi-stakeholder Goodhart effect validated on 6 data points across 3 datasets. The effect does not occur when stakeholders are positively correlated."

### Section 6.4 — Specification vs. Data (lines 1029-1070)
**Current**: Hausdorff-based Goodhart curve (min at N=100, peak at N=500) as the primary evidence. Two strategies compared via Hausdorff.

**Rewrite around**: 
- The direction condition as the organizing principle
- MovieLens utility-based results (diversity peaks at N=50, degrades 42% by N=2000) as primary evidence
- Platform as positive control (improves 14% over same N range)
- Synthetic utility results showing NO Goodhart (all cos > 0)
- Hausdorff result acknowledged as measuring frontier precision, not stakeholder harm

### Figure 6 (spec vs data)
**Current**: Two-panel Hausdorff plot from synthetic

**Replace with** (or supplement with):
- Panel (a): MovieLens user utility ↑ while diversity utility ↓ as N increases (the Goodhart curve)
- Panel (b): 6-point direction condition validation (cos vs degradation, colored by dataset)

Or keep Figure 6 as-is but add a Figure 7 with the MovieLens utility results and direction condition.

### The "productive tension" paragraph (lines 1065-1070)
**Current**: "data is powerful when the specification is correct, but amplifies errors when specification is wrong"

**Refine to**: "data is powerful when specification is aligned with all stakeholders (cos > 0), but amplifies harm to anti-correlated stakeholders (cos < 0). The direction of convergence, not the amount of data, determines whether more training helps or hurts."

### Discussion synthesis (§8.1, lines 1198-1207)
**Current**: "additional training data amplifies misspecified utilities rather than correcting them" as "the paper's deepest finding"

**Reframe**: The deepest finding is the **direction condition**: Goodhart occurs if and only if the optimization direction is anti-correlated with the hidden stakeholder. This is more precise than "data amplifies misspecification" — it predicts exactly WHEN and for WHICH stakeholders data hurts.

### Conclusion (lines 1322-1324)
**Current**: "additional training data amplifies misspecified utilities rather than correcting them"

**Change to**: "additional training data amplifies harm to hidden stakeholders that are anti-correlated with the optimization target (cos < 0), while improving aligned stakeholders (cos > 0) — a directional Goodhart condition validated on 6 data points with zero violations"

### Limitations (§8.3)
**Add**: "The Hausdorff distance metric used in the original synthetic Goodhart analysis measures frontier precision, not stakeholder harm. Under utility-based metrics that directly measure hidden stakeholder degradation, synthetic data shows no Goodhart effect because all stakeholder pairs are positively correlated. The Goodhart finding transfers to MovieLens where user-diversity opposition (cos = -0.3) creates genuine degradation."

### Table 6 — Sensitivity summary (lines 1077-1093)
**Current row**: "Simultaneous (all params) | 6.45 | <1.0 | Errors compound"

**No change needed**: This row is about Hausdorff under simultaneous perturbation (rank stability test), which is a different experiment from the Goodhart N-sweep. The Hausdorff metric is appropriate for measuring frontier shape change.

## New Content to Add

### The direction condition proposition
State formally (as in the study guide Proposition 6.2):
- cos < 0 → eventually degrading
- cos > 0 → eventually improving
- 6/6 validation table

### Cross-dataset comparison table
| Dataset | Hidden | cos | Degradation | Match |
|---------|--------|-----|-------------|-------|
| Synthetic | platform | +0.893 | improves | ✓ |
| Synthetic | society | +0.843 | improves | ✓ |
| ML-100K | platform | +0.956 | +14% | ✓ |
| ML-100K | diversity | -0.313 | -42% | ✓ |
| ML-1M | platform | +0.926 | +14% | ✓ |
| ML-1M | diversity | -0.155 | -46% | ✓ |

### MovieLens Goodhart figure
The user/diversity/platform utility curves vs N — showing user↑, platform↑, diversity↓ simultaneously.

## What Stays the Same

- Labels-not-loss (§4): Unaffected. Uses cosine similarity, not Hausdorff.
- LOSO/data budget (§5): Unaffected. Uses regret metric, not Hausdorff.
- Sensitivity rank stability (§6.1-6.3): Hausdorff is appropriate for measuring frontier shape change under perturbation. The rank stability = 1.0 finding is correct.
- Nonlinear robustness (§7): Unaffected. Uses cosine and Spearman.
- MovieLens validation (§8): Strengthened, not weakened.

## Priority Order for Rewrite

1. Section 6.4 (core Goodhart claim) — highest priority
2. Abstract — must match
3. Figure 6 — needs replacement or supplement
4. Discussion + Conclusion — reframe
5. Limitations — add acknowledgment
