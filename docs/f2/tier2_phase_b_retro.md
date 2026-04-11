# Tier 2 Phase B: Dataset-Agnostic Refactor — Retro

**Goal**: refactor 8 existing experiment/analysis scripts to run on all 3 dataset families (MovieLens, MIND, Amazon Kindle) via a single `--dataset` flag, without changing any numerical behavior on MovieLens. Phase B stops when the refactor is clean AND a byte-equivalence regression test on `ml-100k` passes.

**Status**: Complete 2026-04-10. All 8 scripts refactored. All 7 numerical-output scripts byte-equivalent against ml-100k baselines (excluding `total_time_seconds`).

---

## 1. What Worked

### The dataset registry as single source of truth

`scripts/_dataset_registry.py` collapses what would have been ~60 lines of duplicated importlib boilerplate (per script × 5 scripts) into one ~310-line file with a clean public API. The `LoadedDataset` bundle exposes adapter methods (`rating_item_id()`, `item_dict()`, `generate_preferences()`) that hide the `movie_id`/`item_id` and `movies`/`items` discrepancy without forcing every script to do `getattr` lookups. The `DatasetSpec` frozen dataclass makes per-dataset metadata grep-able and centrally maintainable.

### `primary_stakeholder_order` as a load-bearing field

The plan flagged RNG drift from iteration-order changes as the highest-risk failure mode. `DatasetSpec.primary_stakeholder_order` (a tuple, NOT `sorted(configs.keys())`) preserves the canonical MovieLens ordering `("user", "platform", "diversity")` exactly. Combined with `combinations(2)` for pair generation, the across-stakeholder pair list `[("user","platform"), ("user","diversity"), ("platform","diversity")]` is bit-identical to the original hardcoded version. No iteration-order RNG drift was observed in any of the 7 regression-tested scripts.

### Capturing the regression baseline first, in serial

The plan called for capturing baselines on main (pre-refactor) so the refactor branch could diff against them. The first attempt at capturing 7 baselines in parallel deadlocked (7 simultaneous `uv run` invocations all sleeping at 0% CPU after 5 minutes — uv cache contention). Switching to serial capture worked cleanly: ~3 hours total, every script produced a clean output, every snapshot landed in `results/baseline/`. The serial constraint forced patience but eliminated the entire class of "flaky parallel run" failures.

### The 4-commit gating strategy

Splitting Phase B into 4 commits — registry foundation → drop-in scripts → heavy scripts → retro — meant each gate had a smaller blast radius. When the labels_not_loss drift appeared in Commit 3, only one script's refactor needed re-debugging. The byte-equivalence regression diff (jq -S with `total_time_seconds` exclusion) caught the bug within 2 minutes of the test finishing.

### `hasattr` feature flag for Method B

`run_expanded_direction_validation.py` Method B (named stakeholders) is MovieLens-only. Rather than make MIND/Amazon stakeholder modules implement empty `build_named_stakeholder_configs` stubs (which would have violated the "no MIND/Amazon module changes in Phase B" constraint), the script feature-flags Method B with `hasattr(ds.stakeholders_mod, "build_named_stakeholder_configs")`. Future Phase C could add Method B to MIND/Amazon by simply implementing the function — no script changes needed.

---

## 2. Surprises

### labels_not_loss Group C drift caught a real bug, not a false alarm

The first labels_not_loss refactor attempt produced byte-equivalent output for Groups A, B, D, E, F, but Group C drifted by ~0.005 on cosine similarities and temporal eval accuracy. The byte-equivalence test caught it immediately.

**Root cause**: my refactored `_train(ds, features, ...)` function called `ds.generate_preferences(weights, n, seed)`, but `LoadedDataset.generate_preferences` always uses `self.pool` — the FULL content pool — not the `features` parameter passed into `_train`. Group C trains on `early_pool` (a temporal subset of the pool), so the silent fallback to `ds.pool` meant Group C was training on the WRONG data.

**Fix**: bypass the adapter and call the preferences function with the explicit `features` argument:
```python
pref_fn = getattr(ds.stakeholders_mod, ds.spec.preferences_fn)
pref, rej = pref_fn(features, stakeholder_weights, n_pairs, seed)
```

Without the regression test, this bug would have shipped silently and corrupted every Group C result on every dataset. The byte-equivalence guarantee is doing real work here.

### `LoadedDataset.generate_preferences` is a misleading abstraction

The fix above means `LoadedDataset.generate_preferences` is unsafe to use in any code path that wants to use a custom pool. It's only correct for callers that operate on `ds.pool` directly. The 5 drop-in scripts (Commit 2) do use `ds.pool`, so they were unaffected. But `labels_not_loss` Group C and any future code that constructs alternative pools needs to know to bypass the adapter.

I should consider either: (a) removing `LoadedDataset.generate_preferences` entirely so callers always pass `features` explicitly, or (b) renaming it to `generate_preferences_on_pool()` to make the implicit pool argument visible. **Adding to lessons.md for follow-up.**

### Pyright "missing import" warnings on `_dataset_registry` are false positives

Every refactored script gets `Import "_dataset_registry" could not be resolved` from pyright. The runtime import works fine (because we add `scripts/` to `sys.path` before importing), but pyright doesn't statically resolve `sys.path.insert` mutations. This is a known pyright limitation. Decision: ignore — the pre-existing baseline already has 365 mypy errors and 136 ruff violations, and these new 4-per-script pyright warnings don't graduate to mypy.

### `run_movielens_goodhart.py`'s "metrics" config field had a quirky 2-of-3 stakeholder list

The original goodhart config metadata listed `["user_utility", "diversity_utility", "genre_entropy", "n_unique_genres"]` — only TWO of the three stakeholder utilities, with `platform_utility` missing. I initially "fixed" this in the refactor to include all 3, then realized that change would break byte-equivalence. Reverted to the legacy 2-element pattern using `target_stakeholder` + `diversity_stakeholder` from the spec. Output is bit-identical to the original quirk.

### Parallel test execution worked fine after baselines

The Phase B regression tests run faster than baselines because there's no contention from the 7-way parallel uv invocations that broke baseline capture. I ran 3 refactor tests in parallel (goodhart, data_budget, extremal) without any issues, and another 3 in parallel later (scalarization, expanded_direction, labels_not_loss). The earlier parallel-baseline failure was specific to the first cold start of 7 simultaneous uv runs — once the uv cache was warm, parallelism worked.

---

## 3. Deviations from Plan

### Goodhart moved from "heavy" to "drop-in" mid-execution

The plan's Commit 3 (heavy refactor) listed `run_movielens_goodhart.py` as a heavy refactor target because the explore agent flagged `compute_genre_entropy` and `select_top_k` as MovieLens-specific. But on closer reading, both functions take the per-item `content_topics` integer index, which all 3 dataset families produce identically from their content_pool functions. The script became a drop-in once I deleted the `STAKEHOLDER = "user"` constant and replaced it with `ds.spec.primary_stakeholder_order[0]`. I documented the swap in the user-facing exchange before committing, so the user could push back if they disagreed.

### `run_scalarization_baseline.py` swapped INTO heavy refactor

The mirror of the above: `run_scalarization_baseline.py` was originally drop-in but had a pre-existing `--dataset {synthetic, movielens}` flag. Resolving the flag namespace collision (extending choices to `{synthetic, ml-100k, ml-1m, mind-small, amazon-kindle}` and removing `"movielens"`) is a breaking API change, which justified the heavier classification.

### labels_not_loss required a second pass

Plan estimated 1.5 days for the entire refactor. The refactor itself took ~3 hours of editing, but the labels_not_loss bug forced a re-test (which costs ~28 min wall clock per run). Net cost: one extra full re-test cycle. Total elapsed wall-clock for Phase B (including baseline capture) was ~6 hours, well under the 1.5-day budget.

### No new tests written

Plan said "no new tests in Phase B" — the refactor is verified by the regression test, not by pytest. That held: the only test file modified was `test_expanded_validation.py`, which was DRY-updated in Commit 1 to import `build_named_stakeholder_configs` from `movielens_stakeholders.py` instead of duplicating the function bodies. All 4 test classes still pass; full reward_modeling suite (107 tests) passes unchanged.

---

## 4. Implicit Assumptions Made Explicit

### "Every script that consumes a content pool uses `ds.pool`"

False. `labels_not_loss` Group C constructs `early_pool` (a temporal subset) and trains on it. Any script that wants to train on a non-default pool needs to bypass `LoadedDataset.generate_preferences` and call the preferences function directly with the desired pool. This is now documented in `_train`'s docstring but should also be documented on `LoadedDataset.generate_preferences` itself.

### "Pre-refactor scripts produce ml-100k results in `results/ml-100k_*.json`"

Mostly true, but with 3 exceptions: `run_data_budget_all_hidden.py`, `run_extremal_evidence.py`, and `run_expanded_direction_validation.py` previously iterated over both ML variants internally and wrote one combined JSON (`data_budget_all_hidden.json`, `extremal_evidence.json`, `expanded_direction_validation.json`) — no `ml-100k_` prefix. The Phase B refactor changes them to one-dataset-per-invocation, so the ml-100k subtree of the old combined files is what we diff against.

### "The `metrics` config field is informational and order-insensitive"

False. It's literally serialized into the output JSON as a list with a specific order, and byte-equivalence requires the EXACT same list in the EXACT same order. I had to revert my "fix" to the goodhart metrics field to match the legacy 2-element layout exactly.

### "`ds.pool` is identical to what `generate_movielens_content_pool` would return"

True only if both are called with the same `min_ratings` and `seed`. The registry's `load_dataset(name, min_ratings=5, seed=42)` defaults match the original Phase A and B scripts' values. If anyone changes the registry defaults later, all baseline regressions break silently.

### "Commit ordering matches plan = scripts can be tested independently"

Mostly true, but Commit 1 must land first because Commits 2 and 3 import from `_dataset_registry` (which doesn't exist before Commit 1) and from `movielens_stakeholders.build_named_stakeholder_configs` (which is added in Commit 1). Bisection across commits 2 and 3 is fine; bisection across commit 1 is not.

---

## 5. Scope Changes for Phase C

### Document the `LoadedDataset.generate_preferences` pitfall

Phase C will run experiment scripts on MIND and Amazon. Any future script (or any future modification of `labels_not_loss`-style scripts) that constructs an alternative pool needs to know to bypass `LoadedDataset.generate_preferences`. Either:
- Add a docstring warning to `LoadedDataset.generate_preferences`
- Or rename it to `generate_preferences_on_default_pool()` to make the implicit pool visible
- Or remove the adapter entirely and force callers to always pass `features`

This is a small follow-up that should land before Phase C runs.

### `STAKEHOLDER_MAP` fallback to USER is correct but worth verifying

The `_stakeholder_type(name)` helper falls back to `StakeholderType.USER` for unknown stakeholder names. The plan asserted this is safe because the enum is only used as a label inside `LossConfig`, never for semantic dispatch. Phase B's regression test verified this on MovieLens (where the names ARE in the map), but Phase C will be the first time non-MovieLens stakeholder names hit the fallback. If `train_with_loss` does ANY semantic dispatch on the stakeholder enum, MIND/Amazon results will be wrong.

Action: before running labels_not_loss on MIND/Amazon in Phase C, search `enhancements/reward_modeling/alternative_losses.py` for any reference to `StakeholderType.{USER,PLATFORM,SOCIETY}` and verify there's no semantic dispatch. (The plan claims this is already verified; trust but verify.)

### Output filename change is a small Phase C surprise

`run_data_budget_all_hidden.py` and `run_extremal_evidence.py` now produce `{dataset}_data_budget_all_hidden.json` instead of the un-prefixed `data_budget_all_hidden.json`. Any Phase C analysis script that consumes these files needs to be aware. `analyze_recsys_deep.py` already takes a `--dataset` argument and reads `f"{dataset_name}_..."`, so it should be fine. But shell wrappers (e.g., `scripts/experiments/run_all_movielens.sh` if it exists and references the old name) will break.

### Phase C will need `--data` deprecation timeline

The deprecated `--data` flag still works (with a DeprecationWarning) for backward compat. At some point in Phase F we'll want to remove it entirely. Tracking as a future cleanup, not a Phase C blocker.

---

## 6. Metrics

| Metric | Value |
|--------|-------|
| Wall-clock time (excluding baseline capture) | ~3 hours |
| Wall-clock time (including baseline capture) | ~6 hours |
| Plan estimate | 1.5 days |
| Files created | 1 (`scripts/_dataset_registry.py`) |
| Files modified | 9 (8 scripts + `movielens_stakeholders.py` + 1 test file) |
| Files deleted | 0 |
| Lines added (code) | ~700 |
| Lines deleted (code) | ~290 |
| Net LOC change | +410 (mostly registry + retained legacy compat) |
| Commits on `recsys/phase-b-refactor` | 4 (registry, drop-ins, heavy scripts, retro) |
| Tests passing before refactor | 107 |
| Tests passing after refactor | 107 (unchanged) |
| Tests modified | 1 (`test_expanded_validation.py`, DRY update) |
| Bugs caught by regression test | 1 (labels_not_loss Group C used wrong pool) |
| Bugs that would have shipped without regression test | 1 |
| Refactored scripts byte-equivalent on ml-100k | 7/7 (loso, goodhart, data_budget, extremal, scalarization, expanded_direction, labels_not_loss) |
| Refactored scripts not regression-tested | 1 (`analyze_recsys_deep.py` — read-only JSON analysis, smoke-tested only) |
| New CLI flag | `--dataset {ml-100k,ml-1m,mind-small,amazon-kindle}` (and `synthetic` for scalarization) |
| Deprecated CLI flag | `--data` (still works with DeprecationWarning) |
| Out-of-scope as planned | 0 (no MIND/Amazon stakeholder module changes; no `enhancements/reward_modeling/` changes; no new tests beyond `test_expanded_validation.py`) |
