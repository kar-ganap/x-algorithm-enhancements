# Lessons Learned

Self-improvement log. After any correction from the user, record the pattern and a rule to prevent the same mistake. Review at session start.

## Rules

1. **Don't sell the work short.** When summarizing project status, include the intellectual journey — the wrong hypotheses, course corrections, and key insights — not just the final metrics. A 4-phase project that discovered "it's the labels, not the loss" through 87 experiments is fundamentally different from "99% accuracy, done." *Source: F2 status recap, user correction.*

2. **Verify claims against actual code/docs before stating them.** Stated Phase 3 trained 3 models (it trained 1). Stated BT baseline was not considered effective (it was the winner). Both were wrong and contradicted what we'd just read. When uncertain, re-read the source rather than relying on memory. *Source: F2 recap discussion, user caught both errors.*

3. **Don't contradict your own earlier findings.** Called the diversity knob "a simple slider that gets you most of the way there" after previously establishing that its hardcoded coefficients (favorite + 0.8*repost + 0.5*follow) have no empirical basis. Recognize when a new framing contradicts an earlier one and reconcile them. *Source: Pareto frontier discussion, user flagged the contradiction.*

4. **Read source docs thoroughly, not just agent summaries.** Agent summaries can miss or mischaracterize details. When accuracy matters (project status, phase history), read the actual files. *Source: multiple corrections in F2 recap.*

5. **NumpyEncoder must handle ALL numpy types from day one.** Every `NumpyEncoder` class must include `np.bool_`, `np.floating`, `np.integer`, and `np.ndarray`. This has caused serialization failures multiple times. When writing any JSON encoder for numpy data, always include the full set — don't wait for it to break. *Source: Exp 4 partial sampling, repeated serialization failures across sessions. Happened again in RecSys Phase 1.*

6. **Success criteria need both tests AND scripts.** Tests (pytest) verify correctness with pass/fail assertions. Scripts produce inspectable artifacts (JSON, printed tables) that humans review. Phase verification needs both: tests catch regressions, scripts produce the numbers you'll reference in the paper. Don't put all verification in tests/ alone. *Source: RecSys Phase 1, user correction.*

7. **Follow the full lifecycle even under deadline pressure.** PLAN → TEST → IMPLEMENT → RETRO. Skipping retro loses the learnings (e.g., the user-platform near-alignment surprise in RecSys Phase 1 directly affects Phase 2 framing). Skipping verification scripts means no inspectable output. The discipline exists because each step catches things the others miss. *Source: RecSys Phase 1, user correction.*

8. **Use importlib for modules behind Phoenix/grok import chains.** `enhancements/data/__init__.py` and `enhancements/reward_modeling/__init__.py` eagerly import modules that depend on vendored Phoenix/grok. Any new code or tests that only needs a subset (e.g., `movielens.py`, `alternative_losses.py`) must use `importlib.util.spec_from_file_location()` to load modules directly, bypassing `__init__.py`. *Source: RecSys Phase 1, import chain failures in both module and tests.*

9. **Branch per phase, commit before moving on.** Each phase gets its own branch (or at minimum a commit on a feature branch) before starting the next. Don't pile 3 phases of uncommitted work onto the wrong branch. The convention is in CLAUDE.md: `fN-phaseM-description` branches off main, merged via PR. This was violated across RecSys Phases 1-3 — all work accumulated uncommitted on `paper/f2-preprint` (a paper branch, not an experiment branch). The cost: messy stash/checkout to separate paper fixes from experiment code, conflict resolution on files that don't belong on the branch. *Source: RecSys Phase 3, user caught that we'd been working on the wrong branch with no commits.*
