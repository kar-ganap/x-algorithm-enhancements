# Tier 2 Dataset Expansion: MIND + Amazon Reviews

**Status**: Phase A (dry run) not yet started
**Last updated**: 2026-04-10

## Context

The paper's central claim — the directional Goodhart condition (cos(w_target, w_hidden) < 0 predicts degradation) — is validated on 42 data points across MovieLens-100K and MovieLens-1M. But both datasets share the **same 19-dim genre feature space** and derive stakeholder definitions from the same rating data. It's effectively one dataset at two scales, not two distinct datasets.

To claim meaningful generalizability, we need to reproduce all paper results on datasets with:
1. Different domains (not just entertainment media)
2. Different feature construction methods
3. Domain-native stakeholder definitions (not copies of MovieLens's user/platform/diversity mapped onto new data)

**Chosen datasets**: MIND (Microsoft News) and Amazon Reviews (Kindle Store 5-core). MIND is the closest public analog to social media content ranking (directly relevant to the paper's X motivation); Amazon Kindle diversifies into e-commerce entirely.

**Approach**: Start with a half-day Phase A dry run that verifies stakeholder cosine geometry *before* running any BT training. If either dataset's stakeholder geometry doesn't span the needed cosine range ([-0.2, +0.9]), iterate on stakeholder definitions until it does, or abandon that dataset. Only after Phase A greenlights do we commit to the ~8-day Phases B–F.

**Outcome**: A 4-dataset paper (~100 data points for §4, 4 panels for §§4/5/6 figures) with meaningfully different domains, making the direction condition claim much harder to dismiss as an artifact of MovieLens geometry.

---

## Phased Plan

| Phase | What | Duration | Status |
|---|---|---|---|
| **A** | Dry run: download, load, define stakeholders, verify cosine geometry. **Gate before Phase B.** | 4 hours | Not started |
| **B** | Refactor 3 existing scripts to be dataset-agnostic; regression-test on MovieLens | 1.5 days | Deferred (flesh out after A) |
| **C** | Run all experiments on MIND | 2 days | Deferred |
| **D** | Run all experiments on Amazon Kindle | 2 days | Deferred |
| **E** | Cross-dataset analysis and figure/table updates | 1 day | Deferred |
| **F** | Tests, docs, retro | 1 day | Deferred |
| Buffer | | 1 day | |

**Total: ~9 days** (under 10-day budget).

---

## Phase A: Dry Run Spec (IMMEDIATE NEXT STEP)

**Goal**: answer one question — "Does the stakeholder cosine geometry on each new dataset span roughly [-0.2, +0.9]?" If yes, proceed. If no, iterate on stakeholder definitions. Never run BT training on broken geometry.

### A.1 Files to create

All under existing directories; no new package structure.

| File | Purpose | Est. LOC |
|---|---|---|
| `scripts/data/download_mind.py` | Download MIND-small train + dev, extract | ~60 |
| `scripts/data/download_amazon.py` | Download Amazon Kindle 5-core + metadata, subset to top-20k items | ~80 |
| `enhancements/data/mind.py` | `MINDDataset` loader, mirrors MovieLensDataset interface (duck-typed) | ~350 |
| `enhancements/data/mind_stakeholders.py` | Stakeholder weight functions, content pool, preference generator | ~250 |
| `enhancements/data/amazon.py` | `AmazonDataset` loader (Kindle Store) | ~350 |
| `enhancements/data/amazon_stakeholders.py` | Stakeholder weight functions, content pool, preference generator | ~250 |
| `scripts/analysis/phase_a_geometry_check.py` | The dry-run verification script | ~200 |

**Reference pattern**: `enhancements/data/movielens_stakeholders.py` — use `importlib.util` module loading to avoid Phoenix/grok imports.

### A.2 Data to download

**MIND-small** (~1 GB):
- `https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip`
- `https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip`
- Target: `data/mind-small/`
- Key files: `news.tsv` (article metadata), `behaviors.tsv` (impressions/clicks)

**Amazon Reviews 2018 Kindle Store 5-core** (UCSD McAuley, ~2 GB raw):
- `https://jmcauley.ucsd.edu/data/amazon/amazonReviews/Kindle_Store_5.json.gz`
- `https://jmcauley.ucsd.edu/data/amazon/metaFiles/meta_Kindle_Store.json.gz`
- Target: `data/amazon-kindle/`
- **Subsetting**: filter to top-20k items by review count + users with ≥5 reviews on those items (~100k reviews, 10k users, 20k items)

### A.3 Feature spaces

**MIND (35-dim)**:
- 18 top-level categories (binary one-hot)
- 3 article quality signals (title length, abstract length, entity count — normalized)
- 14 top subcategories (one-hot across ~75% coverage)

**Amazon Kindle (32-dim)**:
- 20 top-level Kindle genres (Fiction, Nonfiction, Mystery, Romance, Sci-Fi/Fantasy, …)
- 4 price tier buckets (free, <$5, $5–15, >$15) — one-hot
- 3 review-volume buckets (niche/mid/popular) — one-hot
- 3 review-sentiment buckets (1–3, 3–4, 4–5 mean stars) — one-hot
- 2 book length indicators (<200 pages, ≥200 pages)

Different dims from MovieLens (19) by design — visibly signals "different dimensionality" in the paper.

### A.4 Domain-native stakeholders (5 per dataset)

#### MIND
| Name | Definition | Expected role |
|---|---|---|
| **reader** | CTR-weighted per category from `behaviors.tsv` click logs | "User" anchor |
| **publisher** | Category volume (articles published per category) | Correlates with reader but over-weights political news |
| **advertiser** | Engagement × non-controversial; negative on `newscrime`, `politics`; positive on `autos`, `travel`, `foodanddrink` | **Opposed to reader on hard news — primary cos < 0 candidate** |
| **journalist** | Flat across serious news (newsworld, newspolitics, financenews, healthnews); negative on `video`, `kids`, `games` | Opposed to advertiser on hard news |
| **civic_diversity** | Zero-sum anti-concentration (same structure as MovieLens diversity) | Cross-dataset control |

#### Amazon Kindle
| Name | Definition | Expected role |
|---|---|---|
| **reader** | Purchase volume × star rating per category | "User" anchor |
| **publisher** | Category × book count × mean review volume | Correlates with reader, over-weights Romance/Mystery |
| **indie_author** | Negative on top-3 populated categories; positive on niche (Travel, Crafts, Politics) | **Opposed to publisher — primary cos < 0 candidate** |
| **premium_seller** | Positive on ">$15" × high-rating; negative on "free" tier | Tests price as orthogonal axis |
| **diversity** | Zero-sum anti-concentration (same structure as MovieLens diversity) | Cross-dataset control |

**Rationale for `diversity` in both**: gives a direct cross-dataset control. If `diversity` behaves consistently across 4 datasets, that's strong geometric evidence independent of domain specifics.

### A.5 Geometry check script

```python
# scripts/analysis/phase_a_geometry_check.py
for each dataset in [mind, amazon]:
    dataset = load()
    pool, _ = generate_content_pool(dataset)
    stakeholder_weights = build_stakeholder_configs(dataset)  # 5 vectors

    # 1. Full cosine matrix (5x5)
    cosine_matrix = pairwise_cosine(stakeholder_weights)

    # 2. Label disagreement rate across 10k random pairs
    disagreement = pairwise_label_disagreement(pool, stakeholder_weights, n_pairs=10000)

    # 3. Label balance per stakeholder (catches degenerate utilities)
    label_balance = per_stakeholder_label_balance(pool, stakeholder_weights)

    # 4. Cosine spread for Method A + B pairs
    spread = {min, max, count_negative, count_transition, count_positive_strong}

    print_pass_fail_report(...)
```

**Runtime: ~2 minutes total**. No BT training in Phase A.

### A.6 Go/no-go criteria

The script prints explicit PASS/FAIL for each:

| ID | Criterion | Threshold |
|---|---|---|
| G1 | At least one pair with cos < -0.1 | Required |
| G2 | Cosine range spans ≥ 0.5 units | max − min ≥ 0.5 |
| G3 | At least one pair with \|cos\| < 0.2 | Required (borderline case) |
| G4 | At least 2 pairs with cos > 0.5 | Required (positive control) |
| G5 | No stakeholder has label imbalance > 90/10 | Required (not degenerate) |
| G6 | No stakeholder pair has cos > 0.95 | Required (not redundant) |

**Decision rule:**
- **All pass** → greenlight Phase B
- **G1 or G2 fails** → iterate on stakeholder definitions (2-hour budget)
- **Still failing after iteration** → abandon that dataset, fall back to Tier 1.5 (the other dataset only)

### A.7 Phase A deliverable

- `results/phase_a_geometry.json` — cosine matrices, disagreement rates, label balance, PASS/FAIL verdict per dataset
- Printed summary fitting on one terminal screen
- **User reviews, then greenlights Phase B**

**Time budget: 4 hours.** Breakdown:
- 1h: download scripts + actual download + file structure verification
- 1.5h: MIND loader + stakeholder module
- 1.5h: Amazon loader + stakeholder module
- included: geometry check script + iteration buffer

---

## Phases B–F (Summary — flesh out after Phase A results)

Intentionally kept light. Detailed specs to be added incrementally after Phase A results inform decisions.

### Phase B: Refactor (1.5 days)

**Decision: refactor once, don't duplicate.** Three scripts need changes:

1. **`run_expanded_direction_validation.py`**: Move the 5 named-stakeholder functions (creator, advertiser, niche, mainstream, moderator) from the script into `movielens_stakeholders.py` as `build_named_stakeholder_configs(dataset)`. Add analogous function to `mind_stakeholders.py` and `amazon_stakeholders.py`. Script becomes a loop over dataset modules.

2. **`run_movielens_goodhart.py`**: Add `--dataset {movielens,mind,amazon}` flag. Dispatch via shared `scripts/_dataset_registry.py` helper that returns `(dataset_module, stakeholder_module, data_path)`. Core loop already generic.

3. **`run_movielens_labels_not_loss.py`**: Group D (downstream prediction) needs special handling — MIND has binary clicks (report AUC), Amazon has 1–5 stars (report Spearman). Group E (user groups) needs per-dataset `get_user_category_groups` / `get_user_genre_groups` functions.

**Drop-in (just add `--dataset` flag)**: `run_movielens_loso.py`, `run_scalarization_baseline.py`, `run_data_budget_all_hidden.py`, `run_extremal_evidence.py`, `analyze_recsys_deep.py`.

**Regression test**: rerun refactored scripts on ML-100K, diff against existing `results/*.json` (allowing only timestamp differences). Any numerical delta = bug.

### Phase C: MIND experiments (2 days)

Run in order (schedule long runs overnight):
1. `run_expanded_direction_validation.py --dataset mind` → 27 new data points for §4
2. `run_movielens_goodhart.py --dataset mind` → §4 utility curves
3. `run_movielens_labels_not_loss.py --dataset mind --all` → §5 labels-not-loss (3 losses × 5 stakeholders × 5 seeds = 75 runs)
4. `run_scalarization_baseline.py --dataset mind` → §5 composition
5. `run_data_budget_all_hidden.py --dataset mind` → §6 data budget
6. `run_movielens_loso.py --dataset mind` → §6 selection independence
7. `run_extremal_evidence.py --dataset mind` → §5 stretch

Outputs: `results/mind-small_*.json`.

### Phase D: Amazon experiments (2 days)

Exact mirror of Phase C with `--dataset amazon`. Outputs: `results/amazon-kindle_*.json`.

### Phase E: Cross-dataset analysis (1 day)

New script `scripts/analysis/cross_dataset_summary.py` loads all 4 datasets' JSONs and produces:
- **Updated §4 scatter plot**: cos vs utility change, 4 colors (~100 points)
- **Updated §4 main-text table**: 4 rows per dataset with match rate, cosine range, violations
- **Updated §5 table**: labels-not-loss within-loss cos per dataset
- **Updated §6 data budget plot**: 4 curves

Paper-side edits (LaTeX): update sections referencing dataset counts/point counts.

### Phase F: Tests + docs (1 day)

New tests:
- `tests/test_data/test_mind_loader.py`
- `tests/test_data/test_mind_stakeholders.py`
- `tests/test_data/test_amazon_loader.py`
- `tests/test_data/test_amazon_stakeholders.py`
- Update `tests/test_reward_modeling/test_expanded_validation.py` for dataset dispatch

Update `docs/f2/plan.md`, `docs/results.md`, write phase retro.

---

## Key Decisions (defaults in **bold**)

1. **MIND-small** (not large). MIND-large is 17 GB with no added geometric benefit.
2. **Amazon Kindle Store 5-core** (not Books/Electronics). Books too large; Electronics has weak category geometry; Kindle has rich nested categories + price tiers.
3. **Duck-typed interface, no `DatasetProtocol` ABC**. Matches existing codebase pattern of `importlib.util` loading.
4. **Refactor once, don't duplicate scripts**. Prevents 3× maintenance burden on every paper revision.
5. **5 stakeholders per dataset** (listed above). User should sanity-check domain naturalness.
6. **2-hour Phase A iteration budget** before abandoning a dataset.
7. **Feature dims: 35 MIND, 32 Amazon** (deliberately different from MovieLens's 19).
8. **Use Bradley-Terry + Margin-BT + Calibrated-BT** for labels-not-loss (Constrained-BT is hardcoded for D=18 Phoenix actions, skip).

---

## Risks and Mitigations

| ID | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | MIND/Amazon cosine geometry too narrow (all stakeholders aligned) | Moderate for MIND, low for Amazon | Phase A gates this. Iterate on definitions. Hard fallback: abandon that dataset. |
| R2 | Download sizes prohibitive | Low | Both well within budget. Subset Amazon at download time. |
| R3 | Stakeholder definitions don't map to linear utilities (conjunctive logic needed) | Moderate for Amazon `premium_seller` | Define as correlated sub-vectors, not AND-logic. G5 label-balance check catches degenerate cases. |
| R4 | Existing MovieLens tests regress during refactor | Moderate | Phase B regression test (diff JSONs) is cheap insurance. |
| R5 | Paper revision scope creep | High | Phase E is scoped to figure/table updates only — 1 day hard cap. |
| R6 | MIND CTR extraction noisy (sparse clicks) | Moderate | Log-smoothed CTR: `(clicks + 1) / (impressions + 10)`. Drop categories with <100 impressions. |

---

## Success Criteria

1. **Phase A**: both datasets pass all 6 go/no-go criteria (G1–G6)
2. **Phase B**: all existing MovieLens results reproduce byte-equivalent (modulo timestamps)
3. **Phase C/D**: direction condition holds across 3+ domains at ~100% rate for \|cos\| > 0.2
4. **Phase E**: paper §§4/5/6 updated with 4-dataset figures and tables, abstract claim broadened
5. **Phase F**: all new tests pass; `make all` clean; phase retro written

---

## Verification (end-to-end test plan)

After Phase F:
1. `make test` — all tests pass
2. `make lint` — no new violations
3. `make typecheck` — no new errors
4. Spot check: `uv run python scripts/experiments/run_expanded_direction_validation.py --dataset movielens` produces output matching existing `results/expanded_direction_validation.json`
5. Spot check: `uv run python scripts/analysis/cross_dataset_summary.py` produces populated figures at `docs/f2/paper/figures/`
6. Paper compiles: `cd docs/f2/paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`

---

## Critical Files Referenced

**For implementation**:
- `enhancements/data/movielens_stakeholders.py` — reference pattern for new stakeholder modules
- `enhancements/data/movielens.py` — reference dataset loader
- `scripts/data/download_movielens.py` — template for download scripts
- `scripts/experiments/run_expanded_direction_validation.py` — main Phase B refactor target (lines 76–143 have 5 genre-specific named stakeholders)
- `scripts/experiments/run_movielens_labels_not_loss.py` — Phase B target (groups C/D/E have dataset-specific paths)
- `enhancements/reward_modeling/k_stakeholder_frontier.py` — fully dimension-agnostic, no changes needed
- `enhancements/reward_modeling/alternative_losses.py` — `Constrained-BT` is D=18 only; use BT/Margin-BT/Calibrated-BT on new datasets

**For Phase A dry run, the only file modified per run is**:
- `results/phase_a_geometry.json` (new)

---

## Changelog

- **2026-04-10**: Initial plan created. Phase A not yet started.
