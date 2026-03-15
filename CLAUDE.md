# x-algorithm-enhancements

Enhancements to xAI's open-sourced recommendation algorithm (Phoenix/Grok). Two features: KV-cache optimization (F1) and multi-stakeholder reward modeling (F2).

## Current State

**Feature F1 (KV-Cache Optimization):** Complete, dormant. 10.3x JIT speedup, 9.6x KV-cache, 58% INT8 memory reduction. 166 tests (9 failures, environment-related). See `docs/f1/retro.md`.

**Feature F2 (Reward Modeling):** All 7 phases complete.

| Phase | What | Key Result |
|-------|------|------------|
| Phase 1 | Bradley-Terry preference learning | 99.3% accuracy; label quality > feature noise |
| Phase 2 | Pluralistic reward models | 100% cluster purity with two-stage approach |
| Phase 3 | Causal verification | 5/5 test suites pass (action-level); 50% history-level |
| Phase 4 | Multi-stakeholder differentiation | Cosine sim 0.478 with standard BT |
| Phase 5 | MovieLens validation | +59% NDCG; 107.5% synergy effect |
| Phase 6 | Synthetic Twitter verification | All 5 test suites pass; 648 params recovered |
| Phase 7 | Research directions (D1-D3) | Labels-not-loss; partial observation; sensitivity |

**Core insight:** Stakeholder differentiation comes from the training labels, not the loss function. 79 of 87 experiments across 4 loss functions converged: when all stakeholders train on identical preference pairs, they converge to identical weights regardless of loss (8 constrained-BT-society experiments diverged due to numerical instability in the diversity constraint, since fixed). Different utility functions → different labels → different models.

**Nonlinear robustness:** Tested under Concave (prospect theory) and Threshold (dead zone) utility families. Labels-not-loss holds (cos sim >0.92), α-recovery Spearman=1.0 for both. α-interpolation proxy degrades under threshold (0.738→0.499) but diversity knob is invariant (0.724). See `results/nonlinear_robustness.json`.

Full narrative: `docs/results.md`

## Ground Rules

### Workflow
- **Plan mode** for ANY non-trivial task (3+ steps). If things go sideways, STOP and re-plan.
- **Phase lifecycle**: PLAN → TEST → IMPLEMENT → RETRO. See `docs/process.md`.
- **TDD**: Tests first. Tests define "done."
- **Objective before subjective**: Run `make all` before manual review.
- **Subagents** for research/exploration. One task per subagent. Keep main context clean.
- **Self-improvement loop**: After ANY correction, update `tasks/lessons.md`. Review at session start.
- **Autonomous bug fixing**: Just fix it. Zero context switching from the user.

### Code
- **Simplicity first**: Minimal code, minimal impact. Don't over-engineer.
- **No laziness**: Root causes, not temporary fixes.
- **Minimal impact**: Touch only what's necessary.

### Quality Gates

All three must pass before merging:

```bash
make test       # pytest
make lint       # ruff
make typecheck  # mypy
make all        # all three in sequence
```

### Git
- Prefix: `feat(fN):`, `fix(fN):`, `refactor(fN):`, `test(fN):`, `docs(fN):`
- Phase branches: `fN-phaseM-description` off main, merged via PR
- No force pushes to main. No Co-Authored-By lines.

### Phase Docs
- **Plan**: `docs/fN/plan.md` or `docs/fN/phaseM_plan.md` — before code
- **Retro**: `docs/fN/retro.md` — after code (6-section format, see `docs/process.md`)

## Key Paths

```
enhancements/reward_modeling/   # F2 core: reward model, training, utilities
  reward_model.py               # ContextualRewardModel
  training.py                   # Bradley-Terry loss
  pluralistic.py                # Phase 2: pluralistic models
  two_stage.py                  # Phase 2: two-stage approach (best)
  causal_verification.py        # Phase 3: intervention tests
  stakeholder_utilities.py      # Phase 4: user/platform/society utilities
  alternative_losses.py         # Phase 4: margin-BT, calibrated-BT, etc.
  weights.py                    # ACTION_INDICES, NUM_ACTIONS (18)

scripts/                        # Organized into subdirectories:
  training/                     # train_reward_model, train_synthetic, train_movielens
  analysis/                     # analyze_partial_observation, analyze_nonlinear_robustness, etc.
  experiments/                  # run_loss_experiments, run_charter_e, compute_exp4_frontiers
  visualization/                # visualize_exp4 (reads JSON, instant)
  evaluation/                   # evaluate_reward_model, verify_synthetic, compare_pareto
  data/                         # generate_synthetic, download_movielens
  _archive/                     # one-off diagnostics (exp4_diagnostic, ablation_study, etc.)

tests/                          # pytest suite
tasks/lessons.md                # Self-improvement log (review at session start)
docs/results.md                 # Full results narrative (all phases)
docs/process.md                 # PLAN→TEST→IMPLEMENT→RETRO lifecycle
docs/design_doc.md              # Architecture and vision
docs/implementation_plan.md     # Risk-tiered implementation strategy
docs/f2/                        # F2-specific: retro, plans, reports
results/loss_experiments/       # 87+ Phase 4 experiment JSONs
results/pareto_comparison.*     # Pareto frontier analysis
```

## Known Technical Debt

Lint and type errors exist in pre-existing code (baseline as of Feb 2026):
- **136 ruff errors** — mostly E501 (line too long), F841 (unused vars), E701 (one-line ifs)
- **365 mypy errors** — mostly missing type annotations, numpy type mismatches, dynamic module loading
- **9 test failures** — all in `tests/test_optimization/` (F1 code, not actively developed)

Rule: don't add new violations. Fix existing ones opportunistically when touching a file.

## Gotchas

- **Module loading:** Scripts use `importlib.util` for module loading due to package structure. See `analyze_stakeholder_utilities.py` for the pattern.
- **pyproject.toml ordering:** hatchling requires `[build-system]` after `[project]`. Don't reorder.
- **Phoenix code:** `phoenix/` and `home-mixer/` are vendored from xAI. Don't lint or modify.
- **NUM_ACTIONS = 18:** All weight vectors are 18-dimensional (one per action). `ACTION_INDICES` in `weights.py` is the source of truth.
