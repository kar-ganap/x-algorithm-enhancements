# RecSys Phase 10: Goodhart Formalization — Retro

**Goal**: Formalize the multi-stakeholder Goodhart condition and validate on all 3 datasets.

**Status**: Complete. Direction condition validated 6/6. Synthetic Goodhart result revealed as metric artifact.

---

## 1. What Worked

### The direction condition is clean and perfect
cos(w_target, w_hidden) < 0 → hidden degrades. cos > 0 → hidden improves. 6/6 data points, zero violations, 3 datasets. This is the kind of result you build a paper section around.

### Platform as positive control was decisive
Same experiment, same data, opposite prediction for two hidden stakeholders. Platform (cos ≈ +0.95) improves. Diversity (cos ≈ -0.3) degrades. The prediction is the cosine sign, nothing else.

## 2. Surprises

### The synthetic benchmark has NO Goodhart under utility metrics
All synthetic stakeholder pairs have cos > 0 (user-society = +0.84, user-platform = +0.89). Society utility IMPROVES from 3.21 to 5.34 with more data. The original paper's Hausdorff-based "Goodhart" was a metric artifact — Hausdorff measures frontier distance (which increases with optimization precision), not stakeholder harm.

This is the single most important finding of the RecSys experiment phases. It means the preprint's central Goodhart claim on synthetic data is incorrect under the proper metric.

### The magnitude condition doesn't hold simply
ML-1M has less anti-correlation (-0.155) but more degradation (46%) than ML-100K (-0.313, 42%). Degradation severity depends on both the angle AND the content pool geometry, not just the angle. Direction is clean; magnitude is complex.

## 3. Implications for Paper Rewrite

See `docs/f2/paper_rewrite_checklist.md` for the specific revision checklist. The paper's Goodhart section (§6.4, Figure 6, and references in abstract/synthesis/conclusion) needs rewriting around the direction condition, with synthetic Hausdorff result reframed as a metric artifact.

## 4. Metrics

| Metric | Value |
|--------|-------|
| Direction condition | 6/6 validated, 0 violations |
| Datasets | Synthetic, ML-100K, ML-1M |
| Synthetic Goodhart (utility) | None (all cos > 0) |
| ML-100K diversity degradation | 42% |
| ML-1M diversity degradation | 46% |
| Study guide | 18 pages (goodhart_formalization.pdf) |
| Files created | 3 (analysis script, synthetic experiment, study guide) |
| Results | goodhart_condition_validation.json, synthetic_goodhart_utility.json |
