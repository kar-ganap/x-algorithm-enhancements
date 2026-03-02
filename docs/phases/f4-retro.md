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
| **Two-stage (topic×action)** | **99.5%** | **0.554** | **100%** |

Source: `results/f4_phase2_two_stage/two_stage_comparison.json` — Topic-aware avg_purity: 1.0, Topic×Action avg_purity: 1.0. The lesson: feature engineering > model complexity. Choosing the right 108D feature set (topic × action cross-features) solved clustering perfectly, making complex end-to-end models unnecessary.

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
| Margin-BT (m=0.05) | 0.613 |
| Margin-BT (m=0.5) | 0.534 |
| Calibrated-BT (λ=0.05) | 0.512 |

Source: `docs/results.md:2109-2117`. Margin-BT actually made differentiation *worse* by regularizing weight directions — the margin constraint forces larger magnitudes but more similar orientations.

### Transformer-embedding synergy was 107.5% (Phase 6)

MovieLens ablation showed neither component works alone. Learned embeddings barely beat random (+0.4% NDCG). Transformer with random embeddings performed *worse* than random. All improvement comes from the interaction effect. Source: `docs/results.md:998-1004`.

This means the model's architecture is fundamentally a joint system — you can't attribute value to individual components. This has implications for interpretability and ablation-based debugging.

### Platform scorer resists diversification (Pareto analysis)

The platform-trained BT weight vector produces such concentrated scores (all topics score 2.9-4.3, narrow range) that the serving-time diversity knob can't overcome them until diversity_weight reaches ~0.7. The platform frontier in `results/pareto_comparison.json` (lines 140-205) is essentially flat for diversity_weight 0.0 through 0.6 — user_utility and society_utility barely change.

Contrast with the user-trained scorer, where the frontier moves smoothly from diversity_weight 0.0 (user_utility 1.145) to 1.0 (user_utility 0.685).

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

2. **Alternative loss functions** (failed): 87 experiments across Margin-BT, Calibrated-BT, Constrained-BT. All produced identical weights when trained on the same preference pairs. This proved the issue wasn't the loss.

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

In retrospect, weight recovery was the wrong metric. What matters for production is: (a) ranking accuracy (99.5% — excellent), (b) cluster purity (100% — excellent), and (c) weight *differentiation* between stakeholders (cosine sim 0.478 — excellent). The fact that learned weights don't match ground truth numerically is a property of BT's scale invariance, not a deficiency.

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

## 5. Scope Changes for Next Phase

### Priority 1: Real preference data pipeline

**Why**: Everything so far uses synthetic data with known ground truth (Phases 1-4, 7) or movie ratings (Phase 6). Zero evidence this works on real Twitter/X engagement data.

**The hard part**: On synthetic data, stakeholder utility functions are given. On real data, they must be *designed*. How do you define "user utility" vs "society utility" from engagement logs? What signals indicate societal harm vs legitimate controversy? The core insight says "it's the labels" — but designing the right labels for real data is the open research question.

### Priority 2: End-to-end integration

**Why**: F4 produces 18-dimensional weight vectors per stakeholder. F2 produces an optimized inference pipeline (JIT, KV-cache, quantization). Nothing connects them. The Pareto analysis script (`scripts/compare_pareto_frontiers.py`) is an ad-hoc prototype, not a production scoring pipeline.

**What's needed**: A serving-time pipeline that takes (user, candidate_set) → scores → diversity-aware ranking → final feed, using the learned weight vectors as the scorer.

### Priority 3: Replace hardcoded scorer with learned weights

**Why**: The Pareto frontier analysis proved the learned scorer outperforms the hardcoded `favorite + 0.8*repost + 0.5*follow` formula. This is the most concrete, immediately actionable improvement from F4.

**What it looks like**: Load the BT weight vectors from `results/loss_experiments/bradley_terry_{user,platform,society}.json`, use them as the scoring function in the diversity-aware selection mechanism.

### Priority 4: History-level causality

**Why**: Phase 3 causal verification: block/follow pass 100%, but history intervention only 50%. Phase 7 improved to 86% with contrastive learning. Still not fully solved. Content-level personalization ("user likes sports → show more sports") only partially works.

**Possible approaches**: Topic-aware reward weights, content-user interaction features, attention mechanisms that explicitly condition on history topics.

### Priority 5: Weight interpretability

**Why**: Weight correlation is 0.554 — meaning learned weights don't match ground truth numerically (BT scale invariance). Rankings are correct, but you can't point to a weight and say "the model learned that blocking is negative because w_block = -2.9." The weight magnitudes are arbitrary, only relative order matters.

**Mitigation**: Use the weight vectors for what they're good at (scoring, differentiation) and don't try to interpret individual values. Alternatively, explore calibrated loss functions that anchor scores to observable engagement rates, accepting the accuracy hit (Platform drops to 86% at λ=0.5).

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
| Total result JSON files in `results/` | 119 (in `loss_experiments/` alone) |

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
| Phase 2 overall accuracy (topic×action) | `results/f4_phase2_two_stage/two_stage_comparison.json` | 99.5% (stress variant) |
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
| Phase 7 archetype flip rate | same | 85.7% |
| Phase 7 user embedding silhouette | same | 0.369 |
| Phase 7 topic embedding silhouette | same | 1.000 |
| Pareto: hardcoded max user_utility | `results/pareto_comparison.json` | 1.087 (at div_weight=0.3) |
| Pareto: learned user scorer max | same | 1.149 (at div_weight=0.1) |

---

## Harsh Assessment: What's Actually Weak

### 1. All results are on synthetic data

Phase 1 data: synthetic. Phase 2: synthetic. Phase 3: synthetic. Phase 4: synthetic. Phase 7: synthetic with 648 known parameters. MovieLens (Phase 6) is real, but it's a movie rating dataset — not social media engagement. We have produced zero evidence that any of this works on real Twitter/X data.

The synthetic data is *designed* to be learnable. Ground truth archetypes have clean separation. Engagement patterns are noise-free (before we add noise ourselves). Real engagement data has overlapping user behaviors, noisy signals, temporal drift, and ambiguous preferences. The impressive numbers (99.5% accuracy, 100% purity, 0.478 cosine sim) are achieved under ideal conditions.

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

---

## Summary

F4 is a genuine research contribution: it discovered (and rigorously proved through 87+ experiments) that Bradley-Terry preference learning differentiates stakeholders through training labels, not loss functions. It built a complete verification stack from synthetic ground truth through causal intervention tests. It validated on two external datasets.

But it is *research*, not product. Every result is on synthetic or proxy data. The central unanswered question — how to design stakeholder utility functions for real engagement data — is a product/policy problem that F4's modeling infrastructure can support but cannot answer. The most valuable next step is not more modeling experiments but rather: bring real data in and see what breaks.
