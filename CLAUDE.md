# x-algorithm-enhancements

Enhancements to xAI's open-sourced recommendation algorithm (Grok/Phoenix). Focus areas: KV-cache optimization, reward modeling, and multimodal retrieval.

## Current State

**Feature F4 (Reward Modeling):** All 4 phases complete.

| Phase | What | Key Result |
|-------|------|------------|
| Phase 1 | Bradley-Terry preference learning | 99.3% accuracy; label quality > feature noise |
| Phase 2 | Pluralistic reward models | 100% cluster purity with two-stage approach |
| Phase 3 | Causal verification | 5/5 test suites pass (action-level); 50% history-level |
| Phase 4 | Multi-stakeholder differentiation | Cosine sim 0.478 with standard BT |

Validated on MovieLens (Phase 6, +59% NDCG) and 648-parameter synthetic Twitter ground truth (Phase 7).

**Core insight:** Stakeholder differentiation comes from the training labels, not the loss function. 87 experiments across 4 loss functions confirmed: when all stakeholders train on identical preference pairs, they converge to identical weights regardless of loss. Different utility functions → different labels → different models.

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
- **Plan**: `docs/phases/phase-N-plan.md` — before code
- **Retro**: `docs/phases/phase-N-retro.md` — after code (6-section format, see `docs/process.md`)

## Key Paths

```
enhancements/reward_modeling/   # F4 core: reward model, training, utilities
  reward_model.py               # ContextualRewardModel
  training.py                   # Bradley-Terry loss
  pluralistic.py                # Phase 2: pluralistic models
  two_stage.py                  # Phase 2: two-stage approach (best)
  causal_verification.py        # Phase 3: intervention tests
  stakeholder_utilities.py      # Phase 4: user/platform/society utilities
  alternative_losses.py         # Phase 4: margin-BT, calibrated-BT, etc.
  weights.py                    # ACTION_INDICES, NUM_ACTIONS (18)

scripts/                        # Training, evaluation, analysis scripts
tests/                          # pytest suite
tasks/lessons.md                # Self-improvement log (review at session start)
docs/results.md                 # Full results narrative (all phases)
docs/process.md                 # PLAN→TEST→IMPLEMENT→RETRO lifecycle
docs/design_doc.md              # Architecture and vision
docs/implementation_plan.md     # Risk-tiered implementation strategy
results/loss_experiments/       # 87+ Phase 4 experiment JSONs
results/pareto_comparison.*     # Pareto frontier analysis
```

## Known Technical Debt

Lint and type errors exist in pre-existing code (baseline as of Feb 2026):
- **136 ruff errors** — mostly E501 (line too long), F841 (unused vars), E701 (one-line ifs)
- **365 mypy errors** — mostly missing type annotations, numpy type mismatches, dynamic module loading
- **9 test failures** — all in `tests/test_optimization/` (F2 code, not actively developed)

Rule: don't add new violations. Fix existing ones opportunistically when touching a file.

## Gotchas

- **Module loading:** Scripts use `importlib.util` for module loading due to package structure. See `analyze_stakeholder_utilities.py` for the pattern.
- **pyproject.toml ordering:** hatchling requires `[build-system]` after `[project]`. Don't reorder.
- **Phoenix code:** `phoenix/` and `home-mixer/` are vendored from xAI. Don't lint or modify.
- **NUM_ACTIONS = 18:** All weight vectors are 18-dimensional (one per action). `ACTION_INDICES` in `weights.py` is the source of truth.
