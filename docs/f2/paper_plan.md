# F2 Preprint: Multi-Stakeholder Reward Modeling for Open-Source Recommendation Algorithms

## 1. Results Inventory (Aggregated)

### Complete results

| ID | Direction | Key finding | Metric | Seeds | Status |
|---|---|---|---|---|---|
| Phase 4 core | D1 | Labels, not loss, determine differentiation | cos_sim=0.478 (P-S), 79/87 converge | 87 runs | ✅ |
| α-recovery | D1 | α recoverable from BT weights | Spearman=1.0 (13 α values) | 5 | ✅ |
| α-stress | D1 | Recovery robust to noise/sample/temp/corr | Breaking: p_flip=0.30, n=50, β=0.5, ρ=0.8 | 1300 runs | ✅ |
| Disagreement bound | D1 | disagreement→cos_sim (R²=0.977) | 2-var model: cos=1.098−1.127d−0.088m | 34 pts | ✅ |
| LLM margin proxy | D1 | LLM confidence preserves ranking | Spearman=0.929 | 15×200 | ✅ |
| Exp 1 (LOSO geom) | D3 | Society 10× costlier to hide than user | regret: soc=1.082, plat=0.369, user=0.111 | 5 | ✅ |
| Exp 2 (training LOSO) | D3 | Training-based matches geometric | regret: soc=1.100, consistent ranking | 5 | ✅ |
| Exp 3 (proxy) | D3 | DW0.7 best practical; interp 73.8% | 6 methods compared | 5 | ✅ |
| Exp 4 (sampling) | D3 | 25 pairs cuts regret 42%; plateau ~200 | Full frontier regret, 20 seeds | 20 | ✅ |
| Exp 5 (degradation bound) | D3 | Oracle R²=0.954; ranking Spearman=0.96 | Power law: regret≈35×std(resid)^1.8 | 5 | ✅ |
| Charter B (closed) | D3 | w_hidden exactly reconstructible if α known | R²=1.0 in structural family | via Exp 5 | ✅ |
| Charter E (K scaling) | D3 | Correlation > K; modest 10% benefit at K=5 | F=8, 101 dw, 10 configs×5 seeds | 200 | ✅ |
| Nonlinear Exp A | Robustness | Labels-not-loss holds under nonlinear | All cos_sim>0.92 (3 families) | 36 | ✅ |
| Nonlinear Exp B | Robustness | α-recovery Spearman=1.0 under nonlinear | Concave + threshold | 120 | ✅ |
| Nonlinear Exp C | Robustness | DW invariant; interp degrades under threshold | 0.724 everywhere; 0.738→0.499 | 135 | ✅ |
| D2.B α-dominance | D2 | Permutation: α dom (0.057); selection: dom (0.062) | Matched magnitude: REVERSES (1.96) | 125+825 | ✅ |
| D2.A param sweep | D2 | Top-3: favorite, repost, repost. Rank stability=1.0 for ALL | Hausdorff is scale artifact; rank stability is primary metric | 560 | ✅ |
| D2.C spec vs data | D2 | Goodhart: more data + wrong spec hurts after N=100 | Data cannot compensate for misspecification | 1300 | ✅ |
| MovieLens (Phase 6) | Validation | +59% NDCG; 107.5% synergy effect | NDCG@3=0.4112 | 1 | ✅ |
| Synthetic Twitter (Phase 7) | Validation | All 5 test suites pass; 648 params recovered | behavioral 100%, causal 78% | 5 | ✅ |

### In-flight results

| ID | What | Expected | Impact on paper |
|---|---|---|---|
| **Exp D** (stress × nonlinear) | Stress thresholds under concave/threshold | Thresholds tighten slightly | Strengthens robustness story; if label_noise breaks at 0.20 vs 0.30, note in limitations |
| **Exp D** (stress × nonlinear) | Stress thresholds under concave/threshold | Thresholds tighten slightly | Strengthens robustness story |

---

## 2. Rank-Ordering by Generality, Robustness, Relevance, Prominence

### Tier 1 — Headline findings (main body, prominent display)

1. **"It's the labels, not the loss"** (Phase 4 core): Most general — applies to ANY BT-family loss, ANY stakeholder. 87 experiments. This is the paper's central theorem.

2. **α-dominance is partial** (D2): The nuanced finding — permutation and selection are robust, but matched-magnitude perturbation shows individual weights matter 2×. This has direct DSA/audit implications for the X open-source transition.

3. **25 preference pairs cuts regret 42%** (Exp 4): Most actionable. Value-of-information result that practitioners can immediately use.

4. **Society is 10× costlier to hide** (Exp 1): Frames the entire Direction 3 — the stake is clear and large.

### Tier 2 — Supporting results (main body, secondary display)

5. **Degradation bound: ranking predictable, magnitude not** (Exp 5): The oracle/proxy gap (R²=0.954 vs 0.721) and the R²=1.0 collapse — theoretically rich.

6. **Nonlinear robustness: core claims survive** (Exp A-C): Validates under prospect theory and threshold utilities. DW invariant at 0.724.

7. **K-stakeholder scaling: correlation > count** (Charter E): Concentration matters 4× more than K. Pareto fraction increases with K but regret is flat.

8. **α-recovery: Spearman=1.0 with stress tolerance** (D1): Foundation for everything else. Breaking points: ≤20% noise, ≥250 pairs.

### Tier 3 — Appendix material

9. Disagreement bound 2-variable model (R²=0.977)
10. LLM margin proxy (Spearman=0.929)
11. Per-parameter rank stability table (D2 Angle A — complete, rank stability=1.0)
12. Specification vs data Goodhart curve (D2 Angle C — complete)
13. MovieLens validation (+59% NDCG, 107.5% synergy)
14. Synthetic Twitter verification (648 params, 78% causal)
15. Exp D stress × nonlinear table (in flight)
16. Phase 2 pluralistic models (100% cluster purity)
17. Phase 3 causal verification (100% block/follow, 50% history)

---

## 3. Paper Structure

### Title (working)

**"Labels Not Loss: Multi-Stakeholder Differentiation, Partial Observation, and Utility Sensitivity in Recommendation Systems"**

Subtitle: *Empirical analysis on X's open-source recommendation algorithm*

### Format

Full technical report, ~15-20 pages + appendix. Arxiv preprint. X as motivating example (§1) but the work stands on its own technical merit. All three directions in main body.

### Abstract (~300 words)

Opening: Multi-stakeholder recommendation systems must balance user engagement, platform retention, and societal welfare — but the utility functions that define these objectives are hand-specified, partially observable, and potentially misspecified. We present a systematic empirical study of these three challenges using X's open-source recommendation algorithm (Phoenix, 18 engagement actions, 3 stakeholders), motivated by the platform's transition from explicit utility weights (2023) to an implicit Grok-based transformer (2026).

Three research directions, each building on the previous:

**Direction 1 — Identifiability**: Across 87 training experiments (79 converged; 8 constrained-BT divergences traced to a numerical bug, since fixed) with 4 Bradley-Terry loss variants, we show that stakeholder differentiation is determined entirely by training *labels*, not the loss function (cosine similarity >0.92 within any loss type, 0.478 across stakeholders). The negativity-aversion parameter α is recoverable from learned weight vectors with Spearman=1.0, robust to ≤20% label noise and ≥250 preference pairs.

**Direction 2 — Utility sensitivity**: We test the "Pareto robustness buffer" hypothesis — whether multi-objective structure absorbs specification errors. Weight permutation and selection-level tests confirm α-dominance (variance ratio <0.06), but matched-magnitude perturbation reveals that individual weight magnitudes cause 2× more frontier shift than α alone. The frontier is not fully robust to specification error.

**Direction 3 — Partial observation**: Leave-One-Stakeholder-Out analysis shows hiding society costs 10× more regret than hiding user. Even 25 preference pairs from the hidden stakeholder cuts regret by 42% (20 seeds), with diminishing returns beyond 200 pairs. A degradation bound predicts which stakeholder is most dangerous to miss (Spearman=0.96) but not by how much (R²=0.72), with the gap attributable to the information-theoretic cost of not observing the hidden stakeholder.

We validate on MovieLens-100K (+59% NDCG) and a 648-parameter synthetic Twitter environment.  All code available at [repo].

### Section outline (~18 pages + appendix)

**§1 Introduction** (~2 pages)
- The multi-stakeholder recommendation problem: user, platform, society
- X as motivating example: 2023 open-source (explicit weights) → 2026 Grok (implicit weights)
- Three research questions: identifiability, partial observation, specification sensitivity
- Contributions summary (4 bullets matching abstract)
- Paper roadmap
- **Figure 1: X algorithm transition schematic** — 2023 explicit weights → 2026 Grok implicit. Annotate what's visible (action types, architecture) vs opaque (learned parameters).

**§2 Background and Related Work** (~2.5 pages)
- §2.1 Multi-stakeholder recommender systems (Burke '17, Multi-FR '22, Eval '25, Lasser '25)
- §2.2 Bradley-Terry preference learning and multi-attribute utility theory (Keeney & Raiffa '76)
- §2.3 Reward misspecification and Goodhart's law (Skalse '24, Casper '23, Weng '24)
- §2.4 Sensitivity analysis in multi-criteria decision analysis (MCDA review '23, weight stability '25)
- §2.5 Platform transparency and algorithmic audit (EU DSA '24, Shaped.ai analysis '23)
- §2.6 X's algorithm: Phoenix heavy ranker architecture

**§3 System, Data, and Methodology** (~2 pages)
- §3.1 Action space: 18 actions (5 positive, 4 negative, 9 neutral), `ACTION_INDICES`
- §3.2 Stakeholder utility model: U = w · φ(x), with structural family U = pos − α·neg
- §3.3 Synthetic data generation: 600 users × 100 content × 6 topics × 5 archetypes
- §3.4 BT training pipeline: preference pairs → weight vectors → Pareto frontier
- §3.5 Evaluation framework: diversity-weight sweep, LOSO projection, regret metrics
- **Figure 2: System schematic** — content actions → BT preference learning → 18-dim weight vectors → stakeholder utilities → Pareto frontier. Show the diversity knob operating on the frontier.
- **Table 1**: Action space — 18 actions with their positive/negative/neutral classification and default weights

**§4 Direction 1: Identifiability — What BT Training Recovers** (~2.5 pages)
- §4.1 Labels vs loss: 87 experiments (79 converged), 4 loss types, the convergence result
- §4.2 α-recovery: Spearman=1.0 across 13 α values, affine calibration curve
- §4.3 Stress testing: 4 dimensions, 1300 runs, breaking points and practical thresholds
- §4.4 Disagreement-to-differentiation bound: 2-variable model (R²=0.977)
- §4.5 LLM-as-annotator compatibility: Spearman=0.929 with Claude Haiku
- **Table 2**: Cosine similarity matrix — pairwise for 3 stakeholders, across 4 loss types
- **Figure 3**: α-recovery plot — true vs recovered α (13 points), with stress contours overlaid
- **Table 3**: Stress test breaking points (4 dimensions × threshold + practical guidance)

**§5 Direction 3: Partial Observation — Missing Stakeholders** (~3 pages)
- §5.1 LOSO geometry and training: society 10× costlier (Exps 1-2)
- §5.2 Proxy methods: 6 methods, diversity knob best practical at 70% (Exp 3)
- §5.3 Data budget: 25 pairs cuts regret 42%, plateau at ~200 (Exp 4, 20 seeds)
- §5.4 Degradation bound: oracle R²=0.954, proxy R²=0.72, the R²=1.0 collapse (Exp 5)
- §5.5 Scaling with K stakeholders: correlation > count (Charter E, F=8, 101 dw)
- **Table 4**: LOSO degradation — 3 stakeholders × geometry + training regret
- **Table 5**: Proxy comparison — 6 methods, recovery rate, type
- **Figure 4 (2-panel)**: Exp 4 — frontier shape vs data budget (left) + regret vs N (right). EXISTS.
- **Figure 5**: Degradation bound — predicted vs actual regret, oracle vs proxy, log scale

**§6 Direction 2: Utility Sensitivity — How Precise Must Specification Be?** (~3 pages)
- §6.1 The α-dominance test: permutation (ratio=0.057), selection-level (0.062)
- §6.2 The matched-magnitude reversal: at σ=0.3, individual weights matter 2× more
- §6.3 Parameter sweep: rank stability=1.0 for all individual params (scale artifact resolved)
- §6.4 Specification vs data: Goodhart effect — more data hurts with wrong spec after N=100
- §6.5 The Pareto robustness buffer: individual params safe, simultaneous perturbation dangerous
- **Table 6**: α-dominance across 4 test conditions (shuffle, magnitude, selection, matched)
- **Table 6b**: Parameter sensitivity ranking (14 params, absolute + normalized) with rank stability
- **Figure 6**: Specification vs data — Hausdorff vs σ (left) + Hausdorff vs N showing Goodhart reversal (right)

**§7 Nonlinear Robustness** (~1.5 pages)
- §7.1 Concave (prospect theory) and threshold (dead zone) utility families
- §7.2 Labels-not-loss holds (Exp A): all cos_sim > 0.92
- §7.3 α-recovery holds (Exp B): Spearman=1.0 for both families
- §7.4 Proxy recovery degrades under threshold (Exp C): 0.738 → 0.499
- §7.5 Stress × nonlinearity: thresholds tighten or hold (Exp D, in flight)
- **Table 7**: Nonlinear robustness summary — Exp A (3×3), B (2×6), C (3×4 metrics)

**§8 External Validation** (~1 page)
- §8.1 MovieLens-100K: +59% NDCG, 107.5% synergy effect
- §8.2 Synthetic Twitter: 648-parameter ground truth, all 5 test suites pass
- **Table 8**: MovieLens ablation — full model vs components vs random baseline

**§9 Discussion** (~2 pages)
- §9.1 Synthesis: the three directions form a coherent story
  - D1 (identifiability) tells you WHAT you can learn from BT
  - D3 (partial observation) tells you WHAT YOU LOSE when stakeholders are hidden
  - D2 (sensitivity) tells you HOW PRECISELY you need to specify what you haven't lost
- §9.2 Practical guidance: the combined prescription
  - Identify your stakeholders' α values (250 preference pairs suffice)
  - If a stakeholder is hidden, collect even 25 pairs (42% regret reduction)
  - Calibrate your top-3 sensitive parameters to ±20%
  - The diversity knob at dw=0.7 is a robust baseline if you can't do any of the above
- §9.3 Implications for platform transparency
  - Structure-without-weights is partially sufficient: α and action-type classification carry most information
  - But weight magnitudes matter at matched perturbation — full transparency requires approximate weight disclosure
  - The EU DSA audit question: can you assess multi-stakeholder welfare from released code?
- §9.4 Limitations
  - Synthetic data (648-param ground truth ≠ real user behavior)
  - 3 stakeholders (Charter E tests K>3 but with synthetic factors)
  - Society utility simplified to pos-α·neg (real: diversity-polarization)
  - Fixed engagement scorer for selection (real systems use learned rankers)

**§10 Conclusion** (~0.5 pages)

**Appendix** (~5-7 pages)
- A: Full nonlinear robustness tables (Exp A-C detailed)
- B: Stress × nonlinearity (Exp D, when complete)
- C: Charter E scaling — K sweep + concentration sweep full tables
- D: Disagreement bound 2-variable model derivation
- E: MovieLens training details and ablation
- F: Synthetic Twitter verification suites
- G: Phase 2 (pluralistic models) and Phase 3 (causal verification) summaries
- H: Per-parameter tolerance radii (Angle A full table, when complete)

---

## 4. Key Exhibits

### Figures needed

| # | Type | Content | Section | Status |
|---|------|---------|---------|--------|
| 1 | **Schematic** | System overview: actions → preferences → BT training → weights → Pareto frontier | §3 | Need to create |
| 2 | **Plot** | α-recovery: 13 α values, fitted line, stress contours overlaid | §4 | Need to create |
| 3 | **Plot (2-panel)** | Exp 4: frontier curves (left) + regret vs N (right), 20 seeds, bootstrap CI | §5 | EXISTS: `results/exp4_partial_sampling.png` |
| 4 | **Plot** | Degradation bound: oracle vs proxy prediction, log-space | §5 | Need to create |
| 5 | **Bar chart** | Parameter rank stability (all 1.0) + sensitivity ranking (normalized). Shows scale artifact: high Hausdorff but perfect rank stability | §6 | Ready (data exists) |
| 6 | **Plot (2-panel)** | Spec vs data: Hausdorff vs σ (left) + Hausdorff vs N showing Goodhart reversal at N=100 (right) | §6 | Ready (data exists) |
| 7 | **Schematic** | X algorithm transition: 2023 (explicit weights) → 2026 (Grok implicit) | §1 or §6 | Need to create |

### Tables needed

| # | Content | Section | Status |
|---|---------|---------|--------|
| 1 | Phase 4 cosine similarity matrix (4 losses × 3 stakeholders) | §4 | Ready (data exists) |
| 2 | LOSO degradation (3 stakeholders × geometry + training) | §5 | Ready |
| 3 | Proxy method comparison (6 methods, hiding society) | §5 | Ready |
| 4 | α-dominance across 4 test conditions | §6 | Ready |
| 5 | α-stress breaking points (4 dimensions) | §4 appendix | Ready |
| 6 | K-scaling results (K=3,5,7,10) | §5 appendix | Ready |

---

## 5. What-Ifs for In-Flight Results

### Exp D (stress × nonlinear)
- **If thresholds tighten by 1 level** (e.g., label noise breaks at 0.20 vs 0.30 for concave): Report as "nonlinearity tightens safety margins by ~30%; practitioners should derate the linear thresholds by one stress level." This strengthens the paper — it's a nuanced result.
- **If thresholds are unchanged**: Report as "stress tolerance is independent of utility shape" — a strong robustness finding. Shorter to report but equally valuable.

### D2 Angle A — RESOLVED
Top-3 are engagement weights (favorite, repost), not negative weights. But rank stability=1.0 everywhere — the Hausdorff was measuring scale change, not shape change. Narrative: "individual parameters are fully forgiving; the specification challenge is in simultaneous multi-parameter error."

### D2 Angle C — RESOLVED
Data does NOT beat specification — the opposite. With wrong weights, more data makes things worse (Goodhart, N>100). The prescription shifts from Exp 4's "just collect data" to: "fix the specification first, then add data." This creates a productive tension with Exp 4 that the paper should highlight: data is powerful when specification is correct, counterproductive when it's wrong.

---

---

## 6. Literature Review: Missing Citations and Positioning

### Must-cite (directly overlapping, would be noticed if absent)

| Paper | Why | How we differ |
|-------|-----|---------------|
| **Sun et al., "Rethinking BT Models" (ICLR 2025)** | Prove BT's "order consistency" property theoretically — our "labels not loss" is the empirical version | **Dialogue framing**: their theory covers our D1; our D2+D3 extend beyond what order consistency addresses |
| **Sorensen et al., "Roadmap to Pluralistic Alignment" (2024)** | Our multi-stakeholder setup IS pluralistic alignment | They aggregate diverse preferences into one model; we maintain separate utilities + Pareto frontier |
| **Stray et al., "Building Human Values into Recommender Systems" (ACM Trans RecSys 2024)** | Interdisciplinary synthesis on value-embedding in RecSys | We provide empirical sensitivity analysis of their framework's parameters |
| **Agarwal et al., "System-2 Recommenders" (FAccT 2024)** | Disentangles utility vs engagement — our user/platform/society is the 3-way version | They use temporal point processes; we use BT preference learning |

### Should-cite (adjacent, strengthens positioning)

| Paper | Why |
|-------|-----|
| **"Beyond BT Models" (ICLR 2025)** | Shows BT assumptions limit preference modeling; our nonlinear audit tests this empirically |
| **"PAD: Personalized Alignment" (ICLR 2025)** | Personalized reward modeling; relates to our per-stakeholder BT training |
| **"Federated RLHF for Pluralistic Alignment" (2025)** | Aggregation of diverse preferences; our Pareto approach is an alternative |
| **"Robust Recommender System Survey" (ACM Computing Surveys 2025)** | Our Direction 2 falls within their robustness taxonomy |
| **Audit of Twitter's friend recommender (2024)** | Empirical audit finding algorithm produces less polarization than organic behavior |
| **Bail et al. (2018)** on exposure to opposing views increasing polarization | Context for why society utility (diversity-polarization) matters |

### Currently cited but still relevant

Burke & Abdollahpouri (2017), Multi-FR (2022), Multistakeholder Eval (2025), Casper et al. (2023), Skalse et al. (ICLR 2024), Iancu & Trichakis (2014), Keeney & Raiffa (1976), Farmer (1987), Lasser et al. (2025), Hadfield-Menell (2019), MCDA sensitivity review (2023)

### Key positioning gap to address

The **pluralistic alignment** literature (Sorensen, Bakker, PAD) addresses diverse preferences through reward model personalization or aggregation. We address it through **separate stakeholder models + Pareto frontier optimization**. The paper should explicitly contrast these approaches: pluralistic alignment assumes all preferences can be captured in a single (possibly personalized) model; our approach assumes stakeholders have fundamentally different objective functions that must be traded off on a frontier. The LOSO analysis tests what happens when one stakeholder's objective is absent from the frontier entirely.

---

## 7. Adversarial Review: Claim-by-Claim Assessment

### Claims we can stand behind (strong evidence, main body)

| Claim | Evidence | Potential attack | Defense |
|-------|----------|-----------------|---------|
| **Labels not loss** | 87 run, 79 converged, 4 losses, cos>0.92 across all converged | "Toy system, 18 actions" | Scope to "BT preference learning on the Phoenix action space." Report 87/79 transparently (8 constrained-BT-society diverged due to numerical bug, since fixed). |
| **Society 10× costlier to hide** | 5 seeds, consistent across Exp 1+2, rank matches correlation structure | "Artifact of your society utility definition" | Acknowledge dependency on diversity-polarization operationalization. Report as "for this operationalization." |
| **25 pairs cuts regret 42%** | 20 seeds, bootstrap CIs, VoI framework | "Changed metric mid-experiment" | **Be transparent**: report both metrics (2D Pareto recovery saturated; full-frontier regret is more appropriate). Explain why. |
| **α-recovery Spearman=1.0** | 13 α values, 5 seeds, 1300 stress runs | Solid. No obvious attack. | Report as-is. |

### Claims that need hedging (evidence mixed or scope-limited)

| Claim | Issue | Hedge |
|-------|-------|-------|
| **"α-dominance"** | Holds for permutation/selection but reverses at matched magnitude. Individual params have rank stability=1.0 (Hausdorff was scale artifact). | Report three-level finding: (1) individual params safe (rank stability=1.0), (2) α dominates permutation/selection, (3) simultaneous perturbation is dangerous. The Goodhart result (data hurts with wrong spec) is the strongest D2 claim. |
| **Degradation "bound"** | R²=0.72 proxy, R²=0.954 oracle. Not a mathematical bound. | Call it "predictive model" not "bound." Lead with ranking (Spearman=0.96). The R²=1.0 collapse is the theoretical contribution, not the regression. |
| **K-scaling: correlation > count** | Synthetic factor model, specific Dirichlet distribution | "Under the factor-based stakeholder model." Note real stakeholders may not follow this structure. |
| **Nonlinear robustness** | Concave + threshold are still separable. No interaction effects tested. | "Within the separable utility family." Note non-separable interactions as future work. |

### Claims to REMOVE or move to appendix

| Claim | Reason | Disposition |
|-------|--------|-------------|
| "Structure-without-weights suffices for audit" | Premature; D2 test #3 reverses at matched magnitude | **Remove from main narrative.** Report the nuanced finding instead. |
| "14 parameters collapse to 3" | Directly refuted by test #3 | **Remove entirely.** |
| LLM margin proxy (Spearman=0.929) | Tangential, go/no-go criterion changed | **Appendix only.** |
| Phase 2 pluralistic models (100% purity) | Earlier phase, doesn't contribute to D1-D3 arc | **Appendix only.** |
| Phase 3 causal verification (100% block, 50% history) | Important for F2 but not for this paper | **Appendix only.** |
| MovieLens 107.5% synergy | Validates architecture, not multi-stakeholder analysis | **Brief mention in §8, details in appendix.** |

### Structural vulnerability: all results are on synthetic data

**This is the biggest meta-criticism.** Every single result depends on synthetic data with known ground truth. A reviewer will say: "you created a world, studied it, and reported findings about the world you created. How do I know any of this transfers to real user behavior?"

**Defense:**
1. MovieLens validation shows the architecture works on real data (+59% NDCG)
2. Synthetic Twitter verification recovers 648 known parameters — the pipeline is faithful
3. The 18-action space comes from X's actual production system (Phoenix/Grok)
4. The synthetic data design (archetypes, topics, engagement patterns) is calibrated to known platform distributions
5. But ultimately, acknowledge this limitation prominently in §9.4

**Mitigation for the paper:** Frame the contribution as "a systematic methodology and a set of empirical insights on a realistic synthetic benchmark" rather than "results about real platform behavior." The methodology (LOSO, α-recovery, sensitivity analysis) generalizes; the specific numbers are benchmark-specific.

---

## 8. Prioritized Next Steps

1. **Wait for in-flight results** (D2 A+C: ~1 hour; Exp D: overnight)
2. **Create paper directory**: `docs/f4/paper/` with `main.tex`, `references.bib`, `figures/`
3. **Write §1-3 first** (intro, background, system) — these don't depend on in-flight results
4. **Write §4** (labels not loss) — all data ready
5. **Write §5** (partial observation) — all data ready
6. **Write §6 skeleton** with placeholders for A+C results
7. **Create figures** 1, 2, 4, 7 (schematics + existing data plots)
8. **Fill §6** when A+C complete
9. **Write §7-8** (discussion, conclusion)
10. **Write appendix** with full tables
