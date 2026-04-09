# Preprint Retention Assessment

What to keep, change, and drop from the existing preprint (`docs/f2/paper/main.tex`) when writing the new arXiv paper.

**Companion doc**: `docs/f2/paper_rewrite_checklist.md` — line-by-line revision guide for the Goodhart correction specifically.

## The Structural Shift

Old paper: 3 equal pillars (identifiability, partial observation, sensitivity) on synthetic data, with MovieLens as a footnote validation.

New paper: 1 core finding (direction condition) validated on 3 datasets, with supporting evidence and an applied audit toolkit. Labels-not-loss is the EXPLANATION, not the headline.

## RETAIN (well-written, still valid)

### Figure 1: X transition TikZ (lines 222-281)
The 2023 explicit → 2026 implicit motivation. Perfect as-is. Keep verbatim.

### Section 2: Background (lines 295-419)
All 6 subsections are well-cited and relevant:
- Multi-stakeholder RecSys (Burke, Multi-FR, Abdollahpouri)
- BT preference learning (Bradley-Terry, Sun et al.)
- Goodhart/reward misspecification (Skalse, Casper, Hadfield-Menell)
- MCDA sensitivity (Wieckowski, Farmer, Iancu)
- Platform transparency (DSA, X)
- X's Phoenix architecture

**Additions needed**: Gao et al. (2023) scaling laws and Kwa et al. (2024) catastrophic Goodhart — both are now part of our theoretical context. Brief paragraph each in §2.3.

### Section 3: Methodology (lines 422-647)
Core setup is reused:
- Action space classification and Table 1 (lines 501-550)
- Stakeholder utility model and Eq 1 (lines 552-579)
- BT training pipeline and Eq 2 (lines 599-617)
- Evaluation framework definitions: LOSO, rank stability (lines 619-647)

**Additions needed**: MovieLens D=19 genre feature setup, held-out evaluation, the 3-dataset structure (synthetic + ML-100K + ML-1M).

**Change**: The pipeline TikZ diagram (Fig 2, lines 427-498) should be updated or moved to appendix — the new paper's pipeline is similar but the emphasis shifts from "3 research questions" to "direction condition validation."

### Practical guidance list (§8.2, lines 1214-1230)
The 4-step sequential approach is still valid. Needs refinement:
- Step 1 (specify α) → add: "and verify cos(target, hidden) > 0"
- Step 2 (collect data) → keep: 25 pairs = ~50% recovery, now with bootstrap CIs
- Step 3 (calibrate weights) → keep
- Step 4 (diversity knob) → keep

### Limitations (§8.3, lines 1265-1302)
All 5 limitations are honest and transfer:
1. Synthetic data and scale gap
2. Three stakeholders
3. Simplified society utility
4. Fixed engagement scorer
5. Separable utility families

**Additions needed**:
- Magnitude not predicted (direction only, 6/6)
- Hausdorff metric was inappropriate (corrected)
- Synthetic benchmark has blind spot (all cos > 0, can't test Goodhart)

### Appendix tables (lines 1344-1567)
All move to appendix in the new paper:
- Nonlinear robustness (App A-B in old paper)
- K-stakeholder scaling (App C)
- Disagreement bound derivation (App D)
- MovieLens training details (App E)
- Synthetic Twitter verification (App F)
- Pluralistic models and causal verification (App G)
- Per-parameter sensitivity (App H)

### References (references.bib)
All 25 references are still needed. Add:
- Gao et al. 2023 (scaling laws)
- Kwa et al. 2024 (catastrophic Goodhart)

## CHANGE (claim or framing needs correction)

### Abstract (lines 87-125)
The entire abstract needs rewriting. Currently frames 3 equal pillars. New abstract leads with the direction condition.

**Specific corrections** (see `paper_rewrite_checklist.md`):
- "additional training data amplifies the error after N > 100" → qualified with cos < 0
- "42% improvement" → "46-56% recovery [40%, 64%] CI" (20-seed MovieLens numbers)
- Add: validated on 3 datasets, 6/6 direction condition
- Add: audit toolkit applied to X's action space

### Section 6.4: Specification vs Data (lines 1029-1070)
The Hausdorff-based Goodhart claim. Replace entirely with:
- The direction condition (cos < 0 → degrades)
- MovieLens utility curves (Fig 2)
- Platform as positive control
- Synthetic has NO Goodhart under utility metrics (all cos > 0)

### Figure 6: Spec vs data plot (lines 1046-1056)
**DROP** — this figure shows the Hausdorff metric artifact. Replace with Fig 2 (Goodhart utility curves).

### "Productive tension" paragraph (lines 1065-1070)
"data helps when spec is correct, hurts when wrong" → "data helps when cos > 0, hurts when cos < 0. The cosine between stakeholder utility directions determines whether convergence is beneficial or harmful."

### Discussion synthesis (lines 1179-1207)
"the paper's deepest finding" (Goodhart via Hausdorff) → The direction condition is the deepest finding. Labels-not-loss is the theoretical explanation. The productive tension is resolved by cos, not by spec quality.

### Conclusion (lines 1305-1336)
"additional training data amplifies misspecified utilities" → "additional training data degrades hidden stakeholders that are anti-correlated with the optimization target (cos < 0), while improving aligned stakeholders (cos > 0)"

### Transparency implications (§8.3, lines 1232-1261)
Still valid in spirit but needs the audit toolkit results added. The "structure without weights" discussion now has a concrete answer: the one observable that matters is whether the platform weights negative actions positively or negatively.

## DROP (not needed in main paper narrative)

These all move to appendix:

### §4.2 α-Recovery (lines 702-733)
Supporting evidence for labels-not-loss. In the new paper, labels-not-loss is the explanation for the direction condition, not a finding in itself. Brief mention in §5 ("the weight space is K-dimensional because BT convergence is determined by labels"), full details in appendix.

### §4.3 Stress Testing (lines 735-765)
Appendix. The 1,300-run stress sweep is thorough but tangential to the direction condition.

### §4.4 Disagreement-to-Differentiation Bound (lines 767-783)
Appendix. The R²=0.977 model is neat but not needed for the main narrative.

### §4.5 LLM-as-Annotator (lines 785-796)
Appendix. Tangential.

### §5.2 Proxy Methods Table (lines 837-867)
Appendix. The 6-method comparison is detailed but the new paper's data budget section (§6) makes the key point more directly.

### §6.1-6.2 α-Dominance and Matched Magnitude (lines 974-1008)
Appendix. The rank stability finding (§6.3) is relevant to "selection mechanism independence" but can be stated in one paragraph.

### §7 Nonlinear Robustness (lines 1104-1144)
Appendix entirely. Important for completeness, not for the narrative.

### §8.1 External Validation (lines 1147-1172)
The MovieLens NDCG validation and synthetic Twitter verification are no longer "external validation" — MovieLens is now the primary real-data evidence. MovieLens NDCG and synthetic Twitter → appendix.

## ADD (new content for the new paper)

### The direction condition (NEW §4)
- Proposition statement
- 6/6 validation table (Table 2)
- MovieLens Goodhart utility curves (Fig 2)
- Platform as positive control
- Synthetic null case (all cos > 0, no Goodhart)
- Connection to Skalse's extremal mechanism

### Low-rank weight space (NEW §5)
- Labels-not-loss → weight space is K-dimensional
- Projection cos > 0.97 for scalarized → per-stakeholder span
- Composition ≈ scalarization (3 models = 99%)
- This explains WHY the direction condition holds

### Practical implications (NEW §6)
- Data budget: 25 pairs = 46-56% recovery, 20 seeds, bootstrap CIs (Fig 5)
- Audit toolkit applied to X's 18 actions (Fig 4, Table 3)
- Selection mechanism independence (diversity weight as orthogonal control)

### Hausdorff correction (NEW §7)
- Original Hausdorff result was false positive for Goodhart on synthetic
- Utility-based metrics reveal no Goodhart when cos > 0
- Methodological recommendation for alignment evaluation

### ML-1M scale validation (brief in §4 or appendix)
- All findings replicate at 10× scale
- Table showing synthetic / ML-100K / ML-1M side by side

## Estimated Reuse

| Component | Old paper | New paper | Reuse % |
|-----------|-----------|-----------|:---:|
| Background (§2) | 3 pages | 2 pages (tighter + Gao/Kwa) | 70% |
| Methodology (§3) | 3 pages | 2 pages (+ MovieLens setup) | 60% |
| Results (§4-7) | 8 pages | 5 pages (completely restructured) | 20% |
| Discussion (§8) | 3 pages | 2 pages (reframed) | 50% |
| Appendix | 5 pages | 8 pages (absorbs dropped sections) | 80% |
| **Overall** | **21 pages** | **~20 pages** | **~40%** |
