# F4 Retrospective: Reward Modeling

**Feature**: F4 — Reward Modeling for Multi-Stakeholder Recommendations
**Phases**: 1 (BT Learning) → 2 (Pluralistic Models) → 3 (Causal Verification) → 4 (Multi-Stakeholder) → 6 (MovieLens) → 7 (Synthetic Twitter)
**Experiments**: 87 training + 30 post-hoc experiments (Phase 4), 8 pluralistic approaches (Phase 2), 3 MovieLens configs (Phase 6), multi-fix synthetic verification (Phase 7)
**Core Insight**: Stakeholder differentiation comes from the training labels, not the loss function.

---

## The Journey at a Glance

| Phase | Objective | Key Win | Key Struggle |
|-------|-----------|---------|--------------|
| 1 | BT preference learning | 99.3% val accuracy | Label sensitivity: 30% flip rate → 68% accuracy |
| 2 | Pluralistic reward models | 100% cluster purity (topic×action features) | Weight correlation stuck at 0.554 (gate was 0.8) |
| 3 | Causal verification | Block/follow 100%, 7/7 stress tests | History intervention only 50% |
| 4 | Multi-stakeholder differentiation | Cosine sim 0.478 (target was <0.95) | 87 experiments to discover "it's the labels" |
| 6 | MovieLens validation | NDCG 0.4112 (+59% vs untrained) | BCE overfitting; had to pivot to BPR |
| 7 | Synthetic Twitter verification | All 5 test suites pass | Three separate fixes needed for causal learning |

The narrative arc: starts with "99% accuracy, easy" → discovers weight recovery is fundamentally limited → spends 87 experiments trying to fix differentiation via loss functions → realizes the fix was always in the data.

---

## 1. What Worked

### Two-stage architecture beat everything (Phase 2)

Simple k-means clustering on user features + per-cluster BT training beat every end-to-end approach: EM, auxiliary loss, hybrid, supervised classification, learned embeddings.

| Approach | Accuracy | Weight Correlation | Cluster Purity |
|----------|----------|-------------------|----------------|
| EM | 99.3% | 0.387 | collapsed |
| Hybrid | 97.3% | 0.510 | — |
| Supervised classification | ~98% | 0.559 | — |
| **Two-stage (topic-aware)** | **99.3%** | **0.604** | **100%** |

Source: `results/f4_phase2_two_stage/two_stage_comparison.json` — Topic-aware avg_purity: 1.0, Topic×Action avg_purity: 1.0. The lesson: feature engineering > model complexity. Using per-topic engagement features solved clustering perfectly, making complex end-to-end models unnecessary.

### Systematic experimentation made the core insight discoverable (Phase 4)

87 experiments across 4 loss functions (BT, Margin-BT, Calibrated-BT, Constrained-BT), each with multiple hyperparameter settings, each producing a JSON with accuracy, weight vectors, cosine similarities, and topic scores. Every result is in `results/loss_experiments/`. This exhaustive approach is what made it possible to conclude definitively that no alternative loss improves on baseline BT — because we tried them all.

### Multi-level verification caught subtle failures (Phase 7)

The 4-level verification hierarchy (representations → actions → ranking → causal interventions) was essential. After initial training, the model passed L1-L3 tests (user embeddings clustered correctly, behavioral predictions were reasonable) but failed L4: only 4% archetype flip rate, meaning the transformer was ignoring history content entirely.

A single accuracy metric would not have caught this. The hierarchy forced us to check causal understanding, not just correlational accuracy.

Source: `results/synthetic_verification.json` — final results after fixes: user_silhouette 0.369, behavioral_accuracy 1.0, action_tests 6/6, block_effect 0.78, flip_rate 0.857.

### Ground truth design was unambiguous (Phase 7)

648 explicit probability parameters across (6 archetypes × 6 topics × 18 actions). Every expected behavior was specified in `enhancements/data/ground_truth.py`. This made verification deterministic — there was no ambiguity about what the model should learn.

### Causal stress tests were thorough (Phase 3)

7/7 stress tests passed for both default and trained weights: effect scaling (monotonic), compound interventions (2-3x amplification), conflicting signals (block overrides high engagement), cross-preference consistency, reversibility (zero error on restore), noise robustness, and threshold sensitivity (no cliff-edge).

Source: `results/f4_phase3_causal/stress_test_results.json`, `results/f4_phase3_causal/causal_verification_results.json`.

---

## 2. Surprises

### The core insight was hiding in Phase 1

Phase 1's cross-archetype transfer experiment showed 0.997 cosine similarity across archetypes. At the time, this was interpreted as "the model generalizes well." In retrospect, it was the first signal that identical preference labels produce identical weights regardless of the user group. The clue was there — we just didn't recognize it until Phase 4 forced us to.

The retrospective annotation at `docs/results.md:1425` was added after Phase 4 made this clear:
> "The 0.997 cosine similarity across archetypes is a consequence of the preference label generation, not a model property."

### Standard BT beat every alternative (Phase 4)

After implementing Margin-BT, Calibrated-BT, Constrained-BT, and Post-Hoc reranking in `enhancements/reward_modeling/alternative_losses.py`, standard BT with stakeholder-specific labels achieved the **lowest** cosine similarity:

| Loss | Min Cosine Sim (P-S) |
|------|---------------------|
| **Standard BT** | **0.478** |
| Margin-BT (m=0.05) | 0.541 |
| Margin-BT (m=0.5) | 0.551 |
| Calibrated-BT (λ=0.05) | 0.512 |

Source: Cosine similarities computed from weight vectors in `results/loss_experiments/`. Margin-BT makes differentiation *worse* — every alternative produces higher cosine similarity than standard BT. (Note: `docs/results.md:2109-2117` records slightly different values from the original experiment run; the Margin-BT JSONs were regenerated in a later training run. The directional conclusion is unchanged.)

### Transformer-embedding synergy was 107.5% (Phase 6)

MovieLens ablation showed neither component works alone. Learned embeddings barely beat random (+0.4% NDCG). Transformer with random embeddings performed *worse* than random. All improvement comes from the interaction effect. Source: `docs/results.md:998-1004`.

This means the model's architecture is fundamentally a joint system — you can't attribute value to individual components. This has implications for interpretability and ablation-based debugging.

### Platform scorer resists diversification (Pareto analysis)

The platform-trained BT weight vector produces such concentrated scores (all topics score 2.9-4.3, narrow range) that the serving-time diversity knob can't overcome them until diversity_weight reaches ~0.7. The platform frontier in `results/pareto_comparison.json` (lines 140-205) is essentially flat for diversity_weight 0.0 through 0.6 — user_utility and society_utility barely change.

Contrast with the user-trained scorer, where the frontier peaks at diversity_weight 0.1 (user_utility 1.149) and declines to 1.0 (user_utility 0.685). The slight increase from 0.0 (1.145) to 0.1 suggests a small amount of diversity marginally benefits even user-centric utility.

### Society scorer gets dominated in combination

When the society-trained scorer is combined with the diversity knob, its frontier sits below and to the left of both the hardcoded and user-scorer frontiers. The reason: double-penalization. The society scorer already down-ranks divisive content (politics scores: -4.29, -4.22). The diversity knob then penalizes topic concentration. The two mechanisms partially duplicate each other. Source: `results/pareto_comparison.json` lines 207-273.

---

## 3. Deviations from Plan

### Phase 2: 8 approaches before finding one that works

Plan was "implement pluralistic reward models." Actual trajectory:

1. EM training → collapsed (all value systems identical)
2. Auxiliary loss with diversity → wrong systems (high diversity, low correlation)
3. Hybrid EM+diversity → best unsupervised, still failed (correlation 0.510)
4. Supervised classification loss → improved but still failed (correlation 0.559)
5. Learned user embeddings (unsupervised) → correlation ~0.3
6. Learned user embeddings (supervised) → correlation ~0.5
7. Oracle one-hot input → correlation ~0.5 (critical: even perfect clustering doesn't fix weights)
8. **Two-stage (k-means + per-cluster BT)** → worked (purity 100%, correlation 0.554)

Then GMM soft clustering (equivalent to k-means with good features), plus 4 stress tests. The 0.8 weight correlation gate was never passed — it was accepted as a fundamental BT limitation.

### Phase 4: Two failed hypotheses before the real fix

1. **Penalty-based differentiation** (failed): Added stakeholder-specific loss terms. All three models converged to identical weights (cosine sim ~1.0). Incorrectly diagnosed as "BT loss dominates penalties."

2. **Alternative loss functions** (failed): 87 experiments across 4 loss functions (standard BT baseline, Margin-BT, Calibrated-BT, Constrained-BT). All produced identical weights when trained on the same preference pairs. This proved the issue wasn't the loss.

3. **Stakeholder-specific preference labels** (succeeded): Different utility functions per stakeholder → different (preferred, rejected) pairs → different learned weights. Cosine sim dropped from 1.0 to 0.478. The winning approach was conceptually the simplest.

### Phase 7: Three separate fixes for three failure modes

Initial training produced a model that passed correlational tests but failed causal ones:

| Fix | Before | After | What It Addressed |
|-----|--------|-------|-------------------|
| Binary → soft labels | 12% behavioral accuracy | 100% | Model couldn't predict archetype-specific action rates |
| Synthetic block contrastive training | 24% block effect | 78% | Model memorized specific block pairs instead of learning the semantic |
| History-topic contrastive loss | 4% archetype flip rate | 86% | User embeddings dominated, transformer ignored history |

None of these three fixes was anticipated in the Phase 7 plan. Each was discovered through the verification suite catching a specific failure mode.

### Phase 6: BCE → BPR pivot

Original MovieLens training used BCE loss with 7 random negatives. Severe overfitting: val NDCG 0.3157 vs test NDCG 0.2410. Pivoting to BPR + in-batch negatives fixed generalization: val 0.4112, test 0.4183. Also discovered Phoenix's transformer used `Constant(0)` initialization, completely bypassing attention.

---

## 4. Implicit Assumptions Made Explicit

### "The loss function is the lever"

Phases 1-3 assumed model differentiation would come from changing the loss function — adding penalties, regularizers, constraints. Phase 4 proved this wrong across 87 experiments. The lever is the training data: which content pair is labeled (preferred, rejected) determines what the model learns. Different labels → different models. Same labels → same models, regardless of loss function.

This is arguably F4's most important lesson and corrects the Phase 2 narrative that attributed weight convergence to "fundamental BT limitations." BT's scale invariance is real (you can't recover exact weight *values*), but weight *differentiation* across groups was always solvable — it just required different preference labels per group.

### "Weight recovery matters"

Phase 2 invested heavily in recovering ground truth weights (target: 0.8 correlation). Best result: 0.554 (`results/f4_phase2_two_stage/two_stage_summary.json`, mean_correlation). This was declared a fundamental limitation and accepted.

In retrospect, weight recovery was the wrong metric. What matters for production is: (a) ranking accuracy (99.3% — excellent), (b) cluster purity (100% — excellent), and (c) weight *differentiation* between stakeholders (cosine sim 0.478 — excellent). The fact that learned weights don't match ground truth numerically is a property of BT's scale invariance, not a deficiency.

### "High accuracy = good model"

Phase 7 proved this false. After initial training, the model achieved high ranking accuracy but only 4% archetype flip rate — meaning swapping a user's history barely changed predictions. The model was using user embeddings (which encode archetype) as a shortcut, bypassing the transformer entirely.

Multi-level verification was essential to catch this. Accuracy alone would have declared the model correct.

### "Feature noise = label noise"

Phase 1 sensitivity analysis showed they're fundamentally different:
- Feature noise (std=0.15) acts as data augmentation: accuracy 82.0% on noisy test
- Label flips (15%) teach wrong preferences: accuracy 70.8% on noisy test
- Combined: 75.2%

Source: `docs/results.md:1388-1401`. Best strategy: add feature noise, keep labels clean. This directly foreshadowed the core Phase 4 insight — clean, stakeholder-specific labels are the critical variable.

### "Diversity knob coefficients are justified"

The serving-time scoring formula `favorite + 0.8*repost + 0.5*follow_author` has no empirical basis. No experiment determined that reposts should be weighted 0.8 or follows 0.5. The Pareto frontier analysis in `scripts/compare_pareto_frontiers.py` showed the learned BT scorer (18D weight vector per stakeholder) produces a shifted frontier — user-trained scorer achieves user_utility 1.149 at diversity_weight=0.1, vs hardcoded max of 1.087 at diversity_weight=0.3. The knob's *mechanism* (greedy diversity-aware selection) is valuable; its *scoring* is the weak link.

Source: `results/pareto_comparison.json` — hardcoded max user_utility 1.087 vs learned user scorer max 1.149.

---

## 5. Research Directions

F4's original question — "given stakeholder utility definitions, can we train differentiated models?" — is answered. The natural next questions are about identifiability and robustness: what can and can't be recovered from preference data, and how sensitive are the results to the choices we made?

### Central research question

*What are the identifiability limits of multistakeholder reward functions learned from pairwise preferences — and how sensitive are Pareto-optimal policies to utility misspecification and unobserved stakeholders?*

### Direction 1: Identifiability limits of Bradley-Terry (core contribution)

**Question**: Given pairwise preference labels from K stakeholders, what can we recover about their individual reward functions?

BT's scale invariance means absolute weight values are unrecoverable. But Phase 4 showed weight *differentiation* IS recoverable (cosine sim 0.478). The open question: what's the full identifiability frontier? Specifically:

- **Rank-order recovery** *(resolved)*: Kendall's τ = 0.612, Spearman ρ = 0.767, Pearson = 0.604 (two-stage model, K=6). Spearman is notably higher (+0.163), indicating partial nonlinear/scale suppression of Pearson. But Kendall ≈ Pearson, meaning pairwise concordance isn't much better than linear fit. **Conclusion**: the low Pearson is partially but not primarily a scale artifact — the model recovers broad rank ordering (ρ=0.767) but has genuine weight recovery limitations. See `results/f4_rank_recovery.json`.
- **Disagreement → differentiation bound**: P-S disagreement 35% → cosine sim 0.478. U-S disagreement 12% → cosine sim 0.884. Is there a formal relationship? Can we prove: "given X% label disagreement, the minimum achievable cosine similarity is Y"?
- **What's recoverable vs not**: Weight magnitudes (no), weight ordering (likely yes), relative stakeholder importance (unknown). Characterize each.

Connects to: Skalse et al. (2025) on partial identifiability in reward learning; rethinking BT in preference-based reward modeling (ICLR 2025).

**Infrastructure**: Existing 87 experiments + synthetic ground truth directly support empirical measurement. Need to add rank-correlation metrics and theoretical analysis.

### Direction 2: Utility function sensitivity

**Question**: How sensitive is the Pareto frontier to utility function parameterization?

The `UtilityWeights` dataclass in `stakeholder_utilities.py` defines utility functions with ~14 free parameters (action weights for user positive/negative, platform weights). We used one set of values. What happens if you perturb them?

- If you change society's negativity penalty from 4.0 to 2.0, how much does the frontier shift?
- Is there a "utility function tolerance" — a perturbation radius within which the Pareto-optimal policy doesn't change?
- Which parameters matter most? (Sensitivity analysis: partial derivatives of frontier position with respect to each utility parameter.)

This answers a practitioner question: *"how precisely must I specify my utility function before deploying a multistakeholder system?"* If the answer is "very precisely," that's a serious limitation. If it's "the frontier is robust to ±30% perturbations," that's reassuring.

**Infrastructure**: `compute_pareto_frontier()` exists. Need to add utility parameter sweep (grid over `UtilityWeights` dimensions) and frontier distance metrics.

### Direction 3: Partial observation (missing stakeholders)

**Question**: If you can observe user and platform preferences but society objectives are hidden, how wrong is your Pareto frontier?

In practice, societal impact is the hardest stakeholder to observe — there's no direct engagement signal for "this content increased polarization." Platforms optimize what they can measure (user engagement, platform retention) and hope societal outcomes follow.

- Train on K-1 of K stakeholders. Compute the Pareto frontier. Then reveal the Kth stakeholder and measure frontier degradation.
- Does the frontier shift predictably? Is the shift bounded by the correlation structure between observed and unobserved stakeholders?
- Which stakeholder is most dangerous to miss? (Hypothesis: society, because it's most anti-correlated with platform.)

Connects to: robustness under partial identifiability (Skalse et al., 2025); multistakeholder evaluation of recommender systems (de Vrieze et al., 2025).

**Infrastructure**: Train on subsets of `{user, platform, society}`, compare frontiers via existing Pareto machinery.

### How these connect

All three directions share F4's infrastructure and build on each other:
1. **Identifiability** tells you what's theoretically recoverable
2. **Sensitivity** tells you how much specification error matters in practice
3. **Partial observation** tells you what happens when you can't specify at all

Together they answer: *"If I'm a practitioner building a multistakeholder recommender, what do I need to get right, what can I get approximately right, and what am I safe to ignore?"*

---

## 6. Metrics

### Experiment volume

| Category | Count |
|----------|-------|
| Loss function experiments (Phase 4) | 87 |
| Pluralistic approaches tried (Phase 2) | 8 |
| MovieLens training configs (Phase 6) | 3 |
| Synthetic verification fix iterations (Phase 7) | 3 |
| Causal stress tests (Phase 3) | 7 |
| Result JSON files in `loss_experiments/` | 119 |

### Test coverage

| Scope | Count |
|-------|-------|
| Total test functions | 211 across 9 files |
| F4-scoped pass (`make test`) | 70 |
| F2 pre-existing failures | 9 in `tests/test_optimization/` |
| **Analysis scripts with no tests** | `compare_pareto_frontiers.py`, `analyze_stakeholder_utilities.py`, `run_loss_experiments.py` |

### Code volume

| Directory | Files | Purpose |
|-----------|-------|---------|
| `enhancements/reward_modeling/` | 12 | Core reward model, training, BT variants, stakeholder utilities, causal verification |
| `enhancements/data/` | 3 | Ground truth, synthetic Twitter data, Phoenix adapter |
| `enhancements/verification/` | 4 | Embedding probes, behavioral tests, action tests, counterfactual tests |
| `scripts/` (F4-related) | 15+ | Training, evaluation, analysis, comparison scripts |

### Key quantitative results

| Metric | Source File | Value |
|--------|------------|-------|
| Phase 1 val accuracy | `results/f4_phase1/training_metrics.json` | 99.3% |
| Phase 1 standard accuracy | `results/f4_phase1/comprehensive_evaluation.json` | 100% |
| Phase 1 label flip impact (30%) | `docs/results.md:1383` | 68.7% accuracy |
| Phase 2 accuracy (topic-aware) | `results/f4_phase2_two_stage/two_stage_comparison.json` | 99.3% |
| Phase 2 cluster purity (topic-aware) | same | 100% (avg_purity: 1.0) |
| Phase 2 weight correlation | `results/f4_phase2_two_stage/two_stage_summary.json` | 0.554 |
| Phase 3 block/follow pass rate | `results/f4_phase3_causal/causal_verification_results.json` | 100% |
| Phase 3 history pass rate | same | 50% |
| Phase 3 stress tests | `results/f4_phase3_causal/stress_test_results.json` | 7/7 |
| Phase 4 cosine sim P-S (BT) | `results/loss_experiments/bradley_terry_{platform,society}.json` | 0.478 |
| Phase 4 cosine sim U-P (BT) | same | 0.830 |
| Phase 4 cosine sim U-S (BT) | same | 0.884 |
| Phase 4 label disagreement P-S | `docs/results.md:2056` | 35% |
| Phase 4 politics gap P-S | `docs/results.md:2131-2132` | 7.7 points |
| Phase 6 val NDCG@3 (BPR) | `docs/results.md:981` | 0.4112 (+59%) |
| Phase 6 test NDCG@3 (BPR) | `docs/results.md:981` | 0.4183 (+73% vs BCE) |
| Phase 6 synergy effect | `docs/results.md:1004` | 107.5% |
| Phase 7 behavioral accuracy | `results/synthetic_verification.json` | 100% |
| Phase 7 action tests | same | 6/6 |
| Phase 7 block effect rate | same | 78% |
| Phase 7 archetype flip rate | same | 86% |
| Phase 7 user embedding silhouette | same | 0.369 |
| Phase 7 topic embedding silhouette | same | 1.000 |
| Pareto: hardcoded max user_utility | `results/pareto_comparison.json` | 1.087 (at div_weight=0.3) |
| Pareto: learned user scorer max | same | 1.149 (at div_weight=0.1) |

---

## Harsh Assessment: What's Actually Weak

### 1. All results are on synthetic data

Phase 1 data: synthetic. Phase 2: synthetic. Phase 3: synthetic. Phase 4: synthetic. Phase 7: synthetic with 648 known parameters. MovieLens (Phase 6) is real, but it's a movie rating dataset — not social media engagement. We have produced zero evidence that any of this works on real Twitter/X data.

The synthetic data is *designed* to be learnable. Ground truth archetypes have clean separation. Engagement patterns are noise-free (before we add noise ourselves). Real engagement data has overlapping user behaviors, noisy signals, temporal drift, and ambiguous preferences. The impressive numbers (99.3% accuracy, 100% purity, 0.478 cosine sim) are achieved under ideal conditions.

### 2. The "core insight" is obvious in hindsight

"If you train all stakeholders on the same preference labels, you get the same model." This is not surprising once stated. It's supervised learning 101 — the model learns what the data teaches. The 87 experiments were valuable for *proving* that no clever loss function can overcome identical training signals, but the insight itself is table stakes for any ML practitioner.

The harder, unanswered question: how do you *design* stakeholder-specific utility functions for real data? What combination of engagement signals constitutes "user utility" vs "platform utility" vs "society utility"? This is fundamentally a product/policy question, not a modeling question, and F4 doesn't address it.

### 3. Analysis scripts have no tests

The scripts that produce the key results cited throughout this retro — `scripts/compare_pareto_frontiers.py`, `scripts/analyze_stakeholder_utilities.py`, `scripts/run_loss_experiments.py` — have zero test coverage. They are the source of truth for claims like "user scorer shifts frontier rightward" and "87 experiments show standard BT is best." If these scripts had bugs in their utility calculations or Pareto computations, we wouldn't know from the test suite.

### 4. Weight correlation gate was relaxed, not passed

Phase 2 set a quality gate: weight correlation > 0.8. The best result was 0.554 (`results/f4_phase2_two_stage/two_stage_summary.json`). This was accepted as a fundamental BT limitation, which is intellectually honest but means we have never passed our own quality bar for weight recovery. The argument that "weight recovery doesn't matter for production" may be correct, but it's also convenient.

### 5. History-level causality remains partially solved

Phase 3: 50% history intervention pass rate. Phase 7: 86% after adding history-topic contrastive loss. Neither is fully solved. The model can learn "blocking an author reduces their posts' scores" (100%) but only partially learns "a user who reads sports content should see more sports" (86%). Content-level personalization — arguably the most user-facing application of reward modeling — is the weakest link.

### 6. Diversity knob coefficients remain arbitrary

The production scoring formula `favorite + 0.8*repost + 0.5*follow_author` was never empirically validated. We identified this weakness during Pareto frontier analysis and showed the learned scorer improves on it, but the learned scorer isn't integrated. The arbitrary formula is still what would be used in practice.

### 7. No path from experiments to production

F4 produces 18-dimensional weight vectors stored in JSON files. F2 produces an optimized inference pipeline with JIT, KV-cache, and quantization. There is no system that combines them into a serving-time recommendation pipeline. No A/B testing framework. No real-data ingestion. No monitoring or feedback loop.

The entire F4 body of work is a research prototype with rigorous internal validation but zero production readiness.

### 8. We searched under the lamppost

The 87 experiments exploring loss function variants were a productive detour that yielded a useful negative result: no loss function can overcome identical training signals. But the reason we went down that path was a framing error. We assumed the loss function was the lever because that's what we knew how to modify. The actual lever — training labels — was in the data pipeline, which we weren't examining. Phase 1's 0.997 cross-archetype cosine similarity was the clue, but we read it as "good generalization" rather than "identical signal." In any eventual report, this journey should be acknowledged honestly. The negative result has independent value (it's a rigorous proof of a general principle), but the path was a course correction, not a deliberate strategy.

---

## 7. Discussion

Deeper analysis of the F4 journey, organized thematically.

### Weight recovery: fundamental or worth revisiting?

The 0.554 Pearson correlation was declared a fundamental BT limitation and accepted. We later ran rank-order recovery analysis to test whether this was a BT scale artifact.

**Result**: Kendall's τ = 0.612, Spearman ρ = 0.767, Pearson = 0.604 (two-stage model rerun). Spearman is higher by +0.163, confirming that nonlinear/scale effects partially suppress Pearson. The model recovers the *broad* rank ordering of action weights (ρ = 0.767) better than magnitudes. But Kendall τ ≈ Pearson, meaning pairwise concordance (how many pairs are correctly ordered) is similar to the linear correlation — many individual action pairs are still misordered.

**Interpretation**: The 0.554 Pearson (or 0.604 in this rerun) is *partially* a scale artifact but the model has genuine recovery limitations. The 0.8 Pearson gate was too strict for BT-learned weights, but a hypothetical 0.85 Spearman gate would also fail (0.767). Single-stakeholder BT does better: user BT achieves Pearson 0.944, Spearman 0.840 — the pluralistic clustering step introduces the main recovery loss, not BT's scale invariance.

### The penalty-based hypothesis: reasonable but falsified

The Phase 4 expectation — "adding stakeholder-specific penalty terms to BT will shift models to different weight optima" — was grounded in standard ML intuition. In most supervised learning settings, changing the loss *does* change the model: BCE, MSE, and hinge produce different decision boundaries on the same data.

Where this breaks down for BT: the ranking loss operates on *all 50,000 preference pairs*, while penalty terms (discomfort, diversity) operate on derived signals from those same pairs. The ranking gradient overwhelms the penalty gradient. And more fundamentally: when preference pairs are identical across stakeholders, the ranking-optimal weights are identical — penalties try to push away from this shared optimum but swim upstream against a much stronger signal.

It IS fair to call this a failed hypothesis. The hypothesis was stated, tested rigorously, and falsified. The falsification is the contribution — "no loss function modification can overcome identical training signals" is a stronger, more general claim than "our penalties worked on the first try." The 87 experiments weren't wasted; they produced a proof by exhaustion.

### Phoenix zero-initialization: current status

Phase 6 discovered that Phoenix's vendored transformer code (`phoenix/grok.py`) uses `Constant(0)` for weight initialization, causing the attention mechanism to compute zero. The model effectively bypasses the transformer and relies only on embeddings.

We fixed this to Xavier initialization directly in the vendored file (`phoenix/grok.py`, commit `0e16441`). Current state:
- `phoenix/grok.py` now uses `VarianceScaling` (Xavier) for weights and `Constant(1)` for RMSNorm scale
- Our trained models (MovieLens, synthetic Twitter) have working transformers
- Open question: was the original `Constant(0)` intentional upstream (perhaps a placeholder for pre-trained weights) or a bug?

### The diversity knob: scoring vs mechanism

The serving-time diversity system has two distinct parts:

**Scoring** (the weak link): `favorite + 0.8*repost + 0.5*follow_author` in `enhancements/reward_modeling/stakeholder_utilities.py:399-402`. Three actions out of 18. Coefficients chosen by hand with no empirical basis. This reduces 18-dimensional action probabilities to a scalar engagement score.

**Mechanism** (the valuable part): Greedy diversity-aware selection in `enhancements/reward_modeling/stakeholder_utilities.py:415-430`. Interpolates between engagement and diversity bonus: `(1 - α) * engagement + α * (1 / (topic_count + 1))`. The α parameter (diversity_weight) controls the tradeoff. This mechanism is sound — it naturally trades off engagement for topic diversity.

F4's contribution: the learned 18D BT weight vector replaces the 3-coefficient scoring function while reusing the same mechanism. Result: user_utility 1.149 vs 1.087 hardcoded — a 5.7% improvement from better scoring alone (`results/pareto_comparison.json`).

### End-to-end integration: what would prove the value?

The Pareto comparison already gives a figure of merit: 5.7% user utility improvement by swapping 3 hardcoded coefficients for 18 learned ones, achievable at lower diversity sacrifice (div_weight 0.1 vs 0.3).

Per-archetype analysis *(resolved)*: The user-trained BT scorer benefits **all 6 archetypes** — no losers. At div_weight=0.1: tech_bro gains most (+34.4%), political archetypes gain modestly (+0.9-1.2%), lurker/power_user gain +7.1%. Mean improvement across diversity weights ranges from +0.062 (lurker) to +0.195 (tech_bro). The aggregate 7.9% improvement is not masking harm to any group. Platform scorer hurts most archetypes (trades user utility for platform engagement). See `results/archetype_pareto_analysis.json`.

### Pushing history-level causality

Two fronts, with concrete next steps:

**Phase 3 (50% → ?)**: The two-stage model uses cluster assignment to encode user type, then applies identical per-cluster weights regardless of content topic. It can't model topic-conditional preferences.

- **Topic-conditional weights**: `w[cluster, topic]` instead of `w[cluster]`. A 6×6 matrix of 18D weight vectors — 648 parameters, matching ground truth dimensionality exactly. This directly models the missing interaction.

**Phase 7 (86% → ?)**: The contrastive learning approach works but 14% of archetype flips fail.

- **Embedding dropout**: Zero out user embeddings during 50% of training steps. Forces the model to rely on transformer history processing instead of the embedding shortcut. This directly targets the known root cause — user embeddings were so informative that the transformer was bypassed.
- **Hard-negative mining**: Find history-candidate pairs where the model currently gets the wrong answer. Oversample these in contrastive training batches.
- **Attention supervision**: Add a loss encouraging attention weights on history tokens to correlate with the candidate's topic.

Embedding dropout is the highest-leverage, lowest-effort approach.

### Calibrated-BT accuracy hit: why it happens

Calibrated-BT adds a secondary objective: predicted scores should match observed engagement rates (MSE term). This conflicts with the primary ranking objective.

For Platform specifically, the utility function values all engagement, so most content has high engagement rates. Calibration pushes all scores toward high values, compressing the score distribution. With compressed scores, the model has less room for fine-grained ranking distinctions. The degradation is monotonic: λ=0 (standard BT) → 91.9% accuracy; λ=0.5 → 86.0%. Source: `results/loss_experiments/calibrated_bt_platform_cal*.json`.

User and Society utilities are more discriminating (they penalize negative actions), so calibration targets are more varied and accuracy degrades less.

### Was utility function design the original F4 question?

No. The design doc (`docs/design_doc.md:365-448`) framed F4 as three tiers:
1. **Tier 1**: Pluralistic causal rewards (discover K value systems, verify causality)
2. **Tier 2**: Multi-stakeholder framework (define stakeholder utility functions, compute Pareto frontiers)
3. **Tier 3**: Game-theoretic analysis (optional)

Tier 2 lists "stakeholder utility functions" as a deliverable but *assumes* they can be specified. The design doc never asks "how do you define user utility?" — it assumes you hand it a definition and asks "given these definitions, can we train differentiated models?"

F4 answered that question: yes, with stakeholder-specific preference labels. The deeper question — *how to design the right utility functions for real data* — emerges from F4 as the natural next-order problem. It's not a gap in F4; it's the question F4 unlocks.

### Why we searched under the lamppost

Four factors, in order of importance:

1. **Wrong abstraction level.** We were thinking "loss function determines model behavior." When stakeholders didn't differentiate, we changed the loss. We never questioned the input because the loss felt like the natural lever. Classic case of searching where the light is.

2. **Misinterpreted early evidence.** Phase 1's 0.997 cosine similarity across archetypes was read as "great generalization" — a positive signal. In retrospect, it was "identical training signal" — a diagnostic signal we missed. The data was telling us the answer; our framing prevented us from hearing it.

3. **Label generation was buried.** Preference pairs were generated by a formula inside the data pipeline. It wasn't a first-class concept in the design. You had to trace: `ground_truth.py` → engagement probabilities → preference pairs → BT labels → training. The fact that all stakeholders shared this pipeline wasn't architecturally visible.

4. **Standard ML intuition misled.** In most settings, changing the loss DOES change the model. BT is special: it only cares about *which item of a pair is preferred*, not by how much. When all stakeholders agree on every pair, no loss modification can introduce disagreement. This property of pairwise ranking losses isn't intuitive from experience with pointwise losses.

These four factors compounded: we started at the wrong abstraction level, misread early evidence, couldn't see the data pipeline as a first-class component, and our ML intuition didn't transfer to pairwise ranking losses.

---

## Summary

F4 is a genuine research contribution: it discovered (and rigorously proved through 87 experiments) that Bradley-Terry preference learning differentiates stakeholders through training labels, not loss functions. It built a complete verification stack from synthetic ground truth through causal intervention tests. It validated on MovieLens (real data) and a 648-parameter synthetic Twitter ground truth.

But it is *research*, not product. Every result is on synthetic or proxy data. The central unanswered question — how to design stakeholder utility functions for real engagement data — is a product/policy problem that F4's modeling infrastructure can support but cannot answer.

The natural next step is not more modeling but deeper investigation of the identifiability and robustness properties of what we've built: what can BT actually recover from preference data, how sensitive are Pareto frontiers to utility specification, and what happens when stakeholders go unobserved? These are questions practitioners will face when deploying any multistakeholder system, and F4's infrastructure — 648-parameter ground truth, verified training pipeline, Pareto frontier computation — is well-positioned to answer them.
