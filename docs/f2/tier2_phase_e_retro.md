# Tier 2 Phase E: Cross-Dataset Analysis + Paper Rewrite — Retro

**Goal**: integrate Phases A–D experiment results into the paper. Write all remaining sections, regenerate figures, resolve the trained-vs-raw cosine and selection mechanism findings into the paper's proposition and framing.

**Status**: Complete 2026-04-17. Paper has no remaining placeholder sections. 17 pages including appendix. The intellectual arc of Phase E reshaped the paper's central claim from "100% on MovieLens with raw cosines" to "100% across 4 datasets with trained cosines under softmax selection, with a characterized failure mode under hard top-K."

---

## 1. What Worked

### The selection mechanism finding was discovered during Phase E, not planned

The plan entering Phase E was "update numbers, regenerate figures, write stubs." Instead, the session produced the paper's most important methodological contribution: the direction condition is a theorem about score distributions (softmax), not about hard top-K selection. This finding emerged from a chain of investigation:

1. Re-scored Method A with trained cosines → 89% pooled, not 100% (MIND journalist failures persisted)
2. Hypothesized within-loss convergence as precondition → ran convergence ablation → **falsified** (ρ = -0.006)
3. Investigated WHY journalist fails → full-pool vs top-K diagnostic → **found the mechanism** (top-K reverses full-pool direction for high-variance stakeholders)
4. Temperature sweep confirmed the transition at T ≈ 1.0 → **28/28 under softmax, 23/28 under top-K**

None of this was in the Phase E plan. The plan said "update §4 with 4-dataset numbers." What actually happened was a research investigation that changed the paper's proposition statement.

### The convergence ablation negative result saved us from a bad claim

Had the ablation shown a clean correlation, we would have published a "within-loss > 0.85 is a necessary precondition" claim that was post-hoc curve-fitting to one data point (journalist). The ablation (ρ = -0.006, 2.7 hours of compute) falsified this and forced us to find the real mechanism. The negative result is not in the paper (correct decision — no need to show wrong hypotheses) but it prevented an embarrassing post-publication retraction.

### Review feedback on the draft caught real issues

An external agent's review identified three concrete problems: (1) intro named 3 datasets not 4, (2) audit toolkit didn't address trained-vs-raw tension, (3) the paper needed to distinguish multi-stakeholder Goodhart from single-stakeholder Goodhart as qualitatively new. All three were fixed in a single commit.

---

## 2. Surprises

### The 0.85 threshold was completely wrong

Going into Phase E, I was confident that BT non-convergence explained journalist's direction condition failures. The within-loss values mapped cleanly: journalist at 0.847 (below 0.85, fails), everyone else above 0.89 (passes). The convergence ablation destroyed this narrative in 2.7 hours. Journalist's within-loss stayed at ~0.86 regardless of label noise σ from 0.05 to 1.50 — its convergence issue is structural, not noise-induced. And civic_diversity's within-loss dropped to 0.45 with no effect on its direction condition match rate. Zero correlation.

### The real mechanism was hiding in the evaluation metric

The full-pool vs top-K diagnostic was a ~20-line Python script that took 60 minutes to run and produced the cleanest result of the entire tier 2 expansion: journalist's full-pool utility follows the direction condition perfectly (all 3 failing pairs switch from violation to match), while top-K reverses it. The mechanism was in the selection, not the geometry. This was discoverable at any point during Phases C or D — we had all the data — but nobody thought to check whether the utility evaluation method was the variable.

### ml-1m was missing from the figure

Phase B refactored scripts were only re-run on ml-100k (regression baseline), mind-small (Phase C), and amazon-kindle (Phase D). ml-1m was never re-run with the `--dataset` flag, so `ml-1m_expanded_direction_validation.json` didn't exist. The user caught this when reviewing Fig 3. Fix: ran expanded_direction on ml-1m (942s) and added it to the selection mechanism ablation. Result: ml-1m contributes 4/4 strong matches, bringing the pooled total from 28/28 to 32/32.

### The paper grew from 10 to 17 pages

Pre-Phase-E: 10 pages (3 written sections + 6 stubs). Post-Phase-E: 17 pages (9 sections + 6 appendix sections). The appendix is compact (tables + methodology descriptions) but the main text grew substantially from the new §5 (low-rank as spectrum) and the expanded §4 (selection mechanism subsection, 4-dataset validation).

---

## 3. Deviations from Plan

### Commit count: 6, not the planned 5

The plan called for 5 commits (ablation, §4, §3+§5+§6, remaining sections, retro). Actual: 6 commits before the retro, because review feedback and codebase-reference cleanup each warranted their own commit. The extra commits are small and focused.

### Phase E included original research, not just paper writing

The plan scoped Phase E as "cross-dataset analysis and figure/table updates." The convergence ablation (new experiment, 2.7 hours compute), selection mechanism ablation (new experiment, 10 min compute + 20 min for 4-dataset version), and the full-pool vs top-K diagnostic were all original experiments not in the plan. These produced the paper's strongest contribution (the selection mechanism finding with 32/32 under softmax). In hindsight, the Phase E plan should have anticipated that integrating cross-dataset results would surface new research questions.

### No Phase E plan existed until mid-session

The user caught that we were "winging it" after Commit 1. The plan was written retroactively, which is backwards from the PLAN → TEST → IMPLEMENT → RETRO lifecycle. In this case it worked because the research questions were driving the investigation and planning prematurely would have locked us into the (wrong) convergence-threshold framing. But it's a process deviation worth noting.

---

## 4. Implicit Assumptions Made Explicit

### "The direction condition is about top-K selected content"

False. It's about expected utility under the score distribution. The original paper's evaluation used top-K without questioning whether the selection mechanism was load-bearing. It was — and the 3 MIND violations were entirely caused by it.

### "Raw cosines work as a first approximation everywhere"

False on high-dim sparse features. On MovieLens they agree within ±0.05. On MIND, 3 of 10 flip sign. The paper now states the condition with trained cosines and notes raw cosines as a valid proxy only when the feature space is low-dimensional and dense.

### "A convergence diagnostic will explain stakeholder-level failures"

False. The convergence ablation showed zero correlation. The failure mechanism is selection-related, not convergence-related. This assumption cost 2.7 hours of compute to falsify, but the falsification was necessary — without it, we would have published a wrong precondition.

### "Phase E is just paper writing"

False. Integrating cross-dataset results surfaced research questions that required new experiments. Phase E was 50% writing, 50% investigation.

---

## 5. Scope Changes for Next Phase (Phase F — Tests + Docs)

Phase F is scoped in the tier2 expansion plan as tests + docs (1 day). No changes to this scope from Phase E findings. The paper is written; Phase F is the quality gate.

The one addition: Phase F should run `make all` to verify the new convergence ablation script doesn't break lint/typecheck, and should verify that `pdflatex main.tex` produces a clean PDF from a fresh clone.

Learned representation spaces (extending the direction condition to neural embeddings) is explicitly deferred to future work (§9 of the paper).

---

## 6. Metrics

| Metric | Value |
|--------|-------|
| Files created | 2 (run_convergence_ablation.py, tier2_phase_e_retro.md) |
| Files modified | 3 (main.tex, references.bib, generate_paper_figures.py) |
| Result JSONs created | 3 (convergence_ablation, selection_mechanism_ablation, ml-1m_expanded_direction) |
| Paper sections written from scratch | 6 (§1, §2, §5, §7, §8, §9) |
| Paper sections rewritten | 3 (abstract, §4, §6) |
| Paper sections extended | 1 (§3: added MIND + Amazon) |
| Appendix sections | 6 (A–F) |
| Paper length | 17 pages (was 10) |
| Commits on `recsys/phase-e-analysis` | 7 (including this retro) |
| Convergence ablation result | ρ = -0.006, p = 0.98 (hypothesis falsified) |
| Direction condition match rate (trained cos, softmax T=1.0, \|cos\|>0.2) | 32/32 = 100% across 4 datasets |
| Direction condition match rate (trained cos, hard top-K, \|cos\|>0.2) | 29/32 = 91% (3 MIND journalist violations) |
| Selection mechanism transition | T = 0.5 → T = 1.0 on MIND |
| Full-pool vs top-K divergences | 5/20 on MIND, 3/20 on Amazon, 0/6 on ml-100k |
| Wall clock (convergence ablation) | 9610s (~2.7 hours) |
| Wall clock (selection mechanism ablation, 4 datasets) | ~770s (~13 min) |
| Wall clock (ml-1m expanded direction) | 942s (~16 min) |
