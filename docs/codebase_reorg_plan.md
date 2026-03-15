# Codebase Cleanup & Reorg Plan

*Created: 2026-03-14. Reference doc for the restructure — keep until reorg is verified.*

## Pre-Reorg State Snapshot

### Git status before reorg
```
Modified (8):
  CLAUDE.md
  docs/f4/experimental_design.md
  docs/f4/retro.md
  docs/results.md
  enhancements/reward_modeling/alternative_losses.py
  scripts/analyze_partial_observation.py
  scripts/analyze_nonlinear_robustness.py
  tasks/lessons.md

Untracked (12):
  docs/f4/paper_plan.md
  enhancements/reward_modeling/factor_stakeholders.py
  enhancements/reward_modeling/k_stakeholder_frontier.py
  scripts/analyze_utility_sensitivity.py
  scripts/compute_exp4_frontiers.py
  scripts/exp4_diagnostic.py
  scripts/run_charter_e.py
  scripts/visualize_exp4.py
  tests/test_analysis/test_nonlinear_robustness.py
  tests/test_analysis/test_utility_sensitivity.py
  tests/test_reward_modeling/test_factor_stakeholders.py
  docs/codebase_reorg_plan.md (this file)
```

### Current scripts/ layout (48 files, flat)
```
scripts/
├── ablation_study.py
├── analyze_alpha_recovery.py
├── analyze_alpha_stress.py
├── analyze_archetype_pareto.py
├── analyze_disagreement_bound.py
├── analyze_learned_model.py
├── analyze_nonlinear_robustness.py
├── analyze_partial_observation.py
├── analyze_rank_recovery.py
├── analyze_stakeholder_utilities.py
├── analyze_utility_sensitivity.py
├── compare_pareto_frontiers.py
├── compare_pluralistic_approaches.py
├── compute_exp4_frontiers.py
├── download_movielens.py
├── evaluate_reward_model.py
├── exp4_diagnostic.py
├── generate_synthetic.py
├── llm_margin_proxy.py
├── run_charter_e.py
├── run_loss_experiments.py
├── run_phase3b_analysis.py
├── sensitivity_analysis.py
├── test_causal_verification.py
├── test_gmm_rich_features.py
├── test_learned_embeddings.py
├── test_two_stage.py
├── train_and_compare_stakeholder_models.py
├── train_movielens.py
├── train_reward_model.py
├── train_synthetic.py
├── tune_cls_weight.py
├── verify_synthetic.py
├── visualize_exp4.py
```

### Known import dependencies (scripts loading other scripts via importlib)

These are the critical cross-references that must be updated during restructure:

```
analyze_partial_observation.py loads:
  - scripts/run_loss_experiments.py (generate_content_pool)
  - scripts/analyze_stakeholder_utilities.py (generate_synthetic_data)

analyze_nonlinear_robustness.py loads:
  - scripts/run_loss_experiments.py
  - scripts/analyze_partial_observation.py
  - scripts/analyze_stakeholder_utilities.py
  - scripts/analyze_alpha_stress.py

analyze_utility_sensitivity.py loads:
  - scripts/run_loss_experiments.py
  - scripts/analyze_partial_observation.py
  - scripts/analyze_stakeholder_utilities.py

run_charter_e.py loads:
  - scripts/analyze_partial_observation.py
  - scripts/analyze_stakeholder_utilities.py

compute_exp4_frontiers.py loads:
  - scripts/run_loss_experiments.py
  - scripts/analyze_partial_observation.py
  - scripts/analyze_stakeholder_utilities.py

visualize_exp4.py loads: (none — reads from JSON)

run_loss_experiments.py loads:
  - (enhancements only, no cross-script deps)

analyze_alpha_stress.py loads:
  - scripts/run_loss_experiments.py

analyze_disagreement_bound.py loads:
  - scripts/run_loss_experiments.py
  - scripts/analyze_stakeholder_utilities.py

analyze_alpha_recovery.py loads:
  - scripts/run_loss_experiments.py

analyze_rank_recovery.py loads:
  - scripts/run_loss_experiments.py

analyze_archetype_pareto.py loads:
  - scripts/run_loss_experiments.py
  - scripts/analyze_stakeholder_utilities.py
```

### Target scripts/ layout (post-reorg)
```
scripts/
├── training/
│   ├── train_reward_model.py
│   ├── train_synthetic.py
│   ├── train_movielens.py
│   └── train_and_compare_stakeholder_models.py
├── analysis/
│   ├── analyze_alpha_recovery.py
│   ├── analyze_alpha_stress.py
│   ├── analyze_partial_observation.py
│   ├── analyze_nonlinear_robustness.py
│   ├── analyze_utility_sensitivity.py
│   ├── analyze_stakeholder_utilities.py
│   ├── analyze_disagreement_bound.py
│   ├── analyze_rank_recovery.py
│   ├── analyze_archetype_pareto.py
│   └── analyze_learned_model.py
├── experiments/
│   ├── run_loss_experiments.py
│   ├── run_charter_e.py
│   ├── compute_exp4_frontiers.py
│   └── llm_margin_proxy.py
├── visualization/
│   ├── visualize_exp4.py
│   └── visualize_charter_e.py
├── evaluation/
│   ├── evaluate_reward_model.py
│   ├── verify_synthetic.py
│   └── compare_pareto_frontiers.py
├── data/
│   ├── generate_synthetic.py
│   └── download_movielens.py
└── _archive/
    ├── exp4_diagnostic.py
    ├── run_phase3b_analysis.py
    ├── ablation_study.py
    ├── tune_cls_weight.py
    ├── sensitivity_analysis.py
    ├── test_causal_verification.py
    ├── test_learned_embeddings.py
    ├── test_two_stage.py
    ├── test_gmm_rich_features.py
    └── compare_pluralistic_approaches.py
```

### Import path update pattern

When moving `scripts/foo.py` to `scripts/subdir/foo.py`, all references of the form:
```python
os.path.join(project_root, "scripts/foo.py")
```
must become:
```python
os.path.join(project_root, "scripts/subdir/foo.py")
```

And `project_root` definitions using `__file__` must account for the extra directory level:
```python
# Before (script in scripts/):
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# After (script in scripts/subdir/):
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### Rollback

If the reorg breaks things:
1. `git stash` the reorg changes
2. Refer to the "Pre-Reorg State Snapshot" above
3. All original file locations are documented here
