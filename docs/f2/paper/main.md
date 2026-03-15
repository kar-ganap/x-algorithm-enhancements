# Labels Not Loss: Multi-Stakeholder Differentiation, Partial Observation, and Utility Sensitivity in Recommendation Systems

*Empirical analysis on X's open-source recommendation algorithm*

**Kartik G Bhat**\*

\*Independent Researcher. Contact: gkartik@gmail.com

---

## Abstract

<!-- ~300 words. See paper_plan.md for draft. -->

---

## 1. Introduction

Recommendation systems serve multiple stakeholders with conflicting objectives. A social media platform balances user engagement (did the user enjoy this content?), platform retention (will the user return tomorrow?), and societal welfare (did this content increase polarization?). Multi-stakeholder optimization frameworks formalize these tradeoffs as Pareto frontiers over stakeholder utilities [1, 2, 3], but three questions remain largely unexamined empirically:

1. **Identifiability.** What properties of stakeholder preferences can a Bradley-Terry (BT) reward model actually recover from pairwise comparison data?
2. **Partial observation.** When a stakeholder is unobservable---as societal welfare typically is---how much does the Pareto frontier degrade, and how cheaply can the degradation be mitigated?
3. **Specification sensitivity.** How precisely must stakeholder utility functions be specified before the frontier shifts meaningfully?

These questions are newly urgent. In January 2026, X (formerly Twitter) open-sourced a complete rewrite of its recommendation algorithm [4], replacing the hand-engineered Phoenix heavy ranker [5] with a Grok-based transformer that "eliminated every single hand-engineered feature and most heuristics." The 2023 release exposed *explicit* utility weights---per-action multipliers for favorites, reposts, blocks, and 15 other engagement signals. The 2026 release exposes *structure without weights*: the prediction targets (action types) are visible, but the actual utility weighting is implicit in 8B learned parameters. This transition from explicit to implicit specification raises a concrete audit question: can regulators assess multi-stakeholder welfare from released code when the weights are opaque? [6]

We study all three questions on a realistic synthetic benchmark built on X's 18-action engagement space, using BT preference learning to train separate reward models for three stakeholders (user, platform, society). Our principal findings:

- **Labels, not loss, determine differentiation.** Across 87 training experiments (79 converged) with four BT loss variants, stakeholder weight vectors converge to cosine similarity >0.92 when trained on identical preference labels, regardless of loss function. Differentiation arises entirely from stakeholder-specific training data. The negativity-aversion parameter α is recoverable from learned weights with Spearman ρ = 1.0, robust to ≤20% label noise and ≥250 preference pairs (§4).

- **Hiding society costs 10× more than hiding user.** Leave-One-Stakeholder-Out (LOSO) analysis shows that removing society from the optimization produces average regret of 1.08, versus 0.11 for user---the stakeholder hardest to observe is the most costly to miss. Even 25 preference pairs from the hidden stakeholder reduce regret by 42% (20 seeds, *p* < 0.01), with diminishing returns beyond ~200 pairs (§5).

- **The Pareto frontier is robust to individual weight perturbation but not to simultaneous misspecification.** Perturbing any single utility weight by up to 3× preserves all Pareto-optimal operating points (rank stability = 1.0). But simultaneous perturbation of multiple weights at matched magnitude (σ = 0.3) causes 2× more frontier shift than changing the α ratio alone. With misspecified weights, additional training data *amplifies* the error after N > 100 pairs---a Goodhart effect [7] (§6).

All three directions are validated under nonlinear utility families (prospect-theoretic concave and threshold utilities) and externally on MovieLens-100K (+59% NDCG) and a 648-parameter synthetic Twitter environment. Code and data are available at https://github.com/kar-ganap/x-algorithm-enhancements.

The remainder of the paper is organized as follows. §2 reviews multi-stakeholder recommendation, BT preference learning, and utility sensitivity analysis. §3 describes the synthetic benchmark and evaluation framework. §4--6 present the three research directions. §7 tests robustness under nonlinear utilities. §8 provides external validation. §9 synthesizes practical guidance and discusses limitations.

---

## 2. Background and Related Work

### 2.1 Multi-Stakeholder Recommender Systems

<!-- Burke '17, Multi-FR '22, Eval '25, Lasser '25 -->

### 2.2 Bradley-Terry Preference Learning

<!-- BT model, MAUT (Keeney & Raiffa '76), Sun et al. ICLR 2025 (order consistency) -->

### 2.3 Reward Misspecification and Goodhart's Law

<!-- Skalse '24, Casper '23, Weng '24 -->

### 2.4 Sensitivity Analysis in Multi-Criteria Decision Analysis

<!-- MCDA review '23, weight stability '25, Farmer '87 -->

### 2.5 Platform Transparency and Algorithmic Audit

<!-- EU DSA '24, X open-source '23/'26, Shaped.ai -->

### 2.6 X's Algorithm: Phoenix Architecture

<!-- 18 actions, heavy ranker, Grok transition -->

---

## 3. System, Data, and Methodology

### 3.1 Action Space

<!-- 18 actions (5 positive, 4 negative, 9 neutral). Table 1. -->

### 3.2 Stakeholder Utility Model

<!-- U = w · φ(x), structural family U = pos − α·neg. Three stakeholders: user (α=1.0), platform (α=0.3), society (α=4.0). -->

### 3.3 Synthetic Data Generation

<!-- 600 users × 100 content × 6 topics × 5 archetypes -->

### 3.4 BT Training Pipeline

<!-- Preference pairs → weight vectors. 4 loss variants. Figure 2. -->

### 3.5 Evaluation Framework

<!-- Diversity-weight sweep, Pareto frontier, LOSO projection, regret metrics -->

---

## 4. Direction 1: Identifiability — What BT Training Recovers

### 4.1 Labels vs Loss

<!-- 87 experiments (79 converged), 4 loss types. Table 2. -->

### 4.2 α-Recovery

<!-- Spearman=1.0, 13 α values, affine calibration. Figure 3. -->

### 4.3 Stress Testing

<!-- 4 dimensions, 1300 runs. Table 3. Breaking points: ≤20% noise, ≥250 pairs, β≥0.5, ρ≤0.6 -->

### 4.4 Disagreement-to-Differentiation Bound

<!-- 2-variable model (R²=0.977): cos = 1.098 − 1.127d − 0.088m -->

### 4.5 LLM-as-Annotator Compatibility

<!-- Spearman=0.929 with Claude Haiku. Brief — appendix has details. -->

---

## 5. Direction 3: Partial Observation — Missing Stakeholders

### 5.1 LOSO Geometry and Training

<!-- Society 10× costlier. Table 4. Exps 1-2. -->

### 5.2 Proxy Methods

<!-- 6 methods. DW0.7 best practical (70%). Table 5. -->

### 5.3 Data Budget

<!-- 25 pairs = −42% regret. Plateau at ~200. 20 seeds. Figure 4. -->

### 5.4 Degradation Predictability

<!-- Oracle R²=0.954, proxy R²=0.72. R²=1.0 collapse. Figure 5. -->

### 5.5 Scaling with K Stakeholders

<!-- Correlation > count. F=8 factors, K=3-10. Charter E. -->

---

## 6. Direction 2: Utility Sensitivity — How Precise Must Specification Be?

### 6.1 The α-Dominance Test

<!-- Permutation: ratio=0.057. Selection-level: 0.062. -->

### 6.2 The Matched-Magnitude Reversal

<!-- At σ=0.3, within-group magnitude > α-only (ratio=1.96). -->

### 6.3 Rank Stability vs Hausdorff

<!-- Rank stability=1.0 for ALL individual params. Hausdorff is scale artifact. -->

### 6.4 Specification vs Data

<!-- Goodhart: more data hurts after N=100 with wrong spec. Figure 6. -->

### 6.5 The Pareto Robustness Buffer

<!-- Individual params safe. Simultaneous perturbation dangerous. Synthesis. -->

---

## 7. Nonlinear Robustness

### 7.1 Utility Families

<!-- Concave (prospect theory), threshold (dead zone) -->

### 7.2 Labels-Not-Loss (Exp A)

<!-- All cos_sim > 0.92. Table 7. -->

### 7.3 α-Recovery (Exp B)

<!-- Spearman=1.0 for both families. -->

### 7.4 Proxy Recovery (Exp C)

<!-- DW invariant (0.724). Interp degrades under threshold (0.738→0.499). -->

### 7.5 Stress × Nonlinearity (Exp D)

<!-- Concave tightens label noise 0.30→0.20. All else unchanged. -->

---

## 8. External Validation

### 8.1 MovieLens-100K

<!-- +59% NDCG. 107.5% synergy effect. Table 8. -->

### 8.2 Synthetic Twitter

<!-- 648 parameters. All 5 test suites pass. 78% causal. -->

---

## 9. Discussion

### 9.1 Synthesis

<!-- D1 → D3 → D2 arc: what you can learn → what you lose → how precise you must be -->

### 9.2 Practical Guidance

<!-- Combined prescription: 250 pairs for α, 25 pairs for hidden stakeholder, calibrate top-3 engagement weights, DW0.7 as baseline -->

### 9.3 Implications for Platform Transparency

<!-- Structure partially sufficient. Weight magnitudes matter at matched perturbation. DSA audit question. -->

### 9.4 Limitations

<!-- Synthetic data, 3 stakeholders, simplified society utility, fixed engagement scorer -->

---

## 10. Conclusion

<!-- ~0.5 pages. -->

---

## References

<!-- See references.bib -->

---

## Appendix

### A. Nonlinear Robustness Full Tables

### B. Stress × Nonlinearity (Exp D)

### C. K-Stakeholder Scaling (Charter E)

### D. Disagreement Bound Derivation

### E. MovieLens Training Details

### F. Synthetic Twitter Verification

### G. Pluralistic Models and Causal Verification

### H. Per-Parameter Sensitivity
