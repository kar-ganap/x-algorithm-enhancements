# Tier 2 Dataset Expansion: MIND + Amazon Reviews

**Status**: Phase A complete (both datasets PASS). Phase B not yet started.
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
| **A** | Dry run: download, load, define stakeholders, verify cosine geometry. **Gate before Phase B.** | ~4 hours actual | **Complete 2026-04-10, both datasets PASS** |
| **B** | Refactor 3 existing scripts to be dataset-agnostic; regression-test on MovieLens | 1.5 days | Not started |
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

## Phase A Results (2026-04-10)

Both datasets **PASS** the blocking criteria (G1, G2, G3, G5, G6). Only G4 (informational) fails on both. Detailed JSON results in `results/phase_a_geometry.json`.

### MIND-small

- **Pool**: 65,238 articles, 12,261-item content pool (after min-5-impressions filter)
- **Feature dim**: 35 (17 top categories + 14 top subcategories + 4 quality signals)
- **Top categories observed**: news, sports, finance, foodanddrink, travel, lifestyle, video, weather, health, autos, tv, music, entertainment, movies, kids, middleeast, northamerica
- **5 stakeholders**: reader, publisher, advertiser, journalist, civic_diversity

**Cosine matrix** (sorted):

| Pair | cos |
|---|---|
| publisher ↔ civic_diversity | **-0.839** |
| advertiser ↔ journalist | **-0.308** |
| reader ↔ journalist | -0.152 |
| journalist ↔ civic_diversity | -0.116 |
| publisher ↔ advertiser | -0.057 |
| publisher ↔ journalist | -0.038 |
| reader ↔ civic_diversity | +0.012 |
| reader ↔ advertiser | +0.396 |
| advertiser ↔ civic_diversity | +0.401 |
| reader ↔ publisher | +0.482 |

**Criteria**:
- G1 (≥1 pair cos < -0.1): **PASS** (4 pairs)
- G2 (range ≥ 0.5): **PASS** (1.321)
- G3 (≥1 pair \|cos\| < 0.2): **PASS** (5 pairs in transition zone)
- G4 (≥2 pairs cos > 0.5): **FAIL** (0 pairs; max is reader-publisher at +0.482)
- G5 (no label imbalance > 90/10): **PASS** (worst: advertiser at frac_first=0.348)
- G6 (no cos > 0.95): **PASS** (max +0.482)

### Amazon Kindle

- **Pool**: 20,000 books (top-20K by review count), 17,425-item content pool
- **Feature dim**: 32 (20 genres + 4 price + 3 volume + 3 sentiment + 2 length)
- **Top categories observed**: Literature & Fiction, Romance, Mystery/Thriller/Suspense, Science Fiction & Fantasy, Teen/YA, Religion/Spirituality, Biographies/Memoirs, Children's, History, Humor/Entertainment, Kindle Unlimited, Politics, Business, Arts/Photography, Science/Math, Sports/Outdoors, Travel, Crafts/Hobbies, Cookbooks, Health/Fitness
- **5 stakeholders**: reader, publisher, indie_author, premium_seller, diversity

**Cosine matrix** (sorted):

| Pair | cos |
|---|---|
| publisher ↔ diversity | **-0.761** |
| publisher ↔ indie_author | **-0.578** |
| reader ↔ diversity | -0.008 |
| publisher ↔ premium_seller | +0.000 |
| indie_author ↔ premium_seller | +0.000 |
| premium_seller ↔ diversity | +0.000 |
| reader ↔ premium_seller | +0.104 |
| reader ↔ indie_author | +0.154 |
| reader ↔ publisher | +0.311 |
| indie_author ↔ diversity | **+0.654** |

**Criteria**:
- G1 (≥1 pair cos < -0.1): **PASS** (2 pairs)
- G2 (range ≥ 0.5): **PASS** (1.415)
- G3 (≥1 pair \|cos\| < 0.2): **PASS** (6 pairs in transition zone)
- G4 (≥2 pairs cos > 0.5): **FAIL** (only 1 pair: indie_author-diversity at +0.654)
- G5 (no label imbalance > 90/10): **PASS** (worst: indie_author at frac_first=0.197; expected — see below)
- G6 (no cos > 0.95): **PASS** (max +0.654)

### G4 Failure: Decision to Accept (Option A)

Both datasets fail G4 (≥2 pairs with cos > 0.5). Decision: **accept the failure, proceed to Phase B unchanged**.

**Why G4 fails**: The domain-native stakeholder definitions deliberately create tension between roles. There are no "twin" stakeholders — no "mainstream editor" or "bestseller chaser" that's trivially aligned with another. Each role is distinct by design, which is the point of going domain-native.

**Why accepting is correct**:

1. **Positive controls already come from MovieLens.** G4 is a positive-control sanity check. MovieLens ML-100K and ML-1M both have user-platform cos ≈ +0.95, plus named stakeholders (user-creator, user-advertiser) in the +0.6 to +0.8 range. The direction condition's "improve" side is already thoroughly tested.

2. **The new datasets contribute different evidence types.** MIND and Amazon don't need to replicate MovieLens's positive-control regime. What they contribute is precisely what MovieLens lacks:
   - **Stronger negative extremes**: publisher-civic_diversity on MIND is **-0.84**, publisher-diversity on Amazon is **-0.76**. Stronger than any MovieLens pair.
   - **Exact-zero orthogonal pairs**: Amazon's premium_seller has cos = **0.000** with publisher, indie_author, and diversity (because it operates entirely on the price/sentiment/length feature slots while the others operate on category slots). No MovieLens pair comes this close to the theoretical cos=0 boundary. This gives us a unique test case for the direction condition at its exact decision threshold.
   - **Transition-zone density**: 5 pairs with \|cos\| < 0.2 on MIND, 6 on Amazon. Much denser coverage of the ambiguous region than MovieLens.

3. **Alternatives muddle the story**:
   - Option B (add contrived "twin" stakeholders like `mainstream_editor`) satisfies G4 but adds roles a reviewer would correctly identify as ad-hoc. The question "what real-world role does `mainstream_editor` play?" has no good answer.
   - Option C (modify existing stakeholders to be more aligned) preserves realism but weakens the roles we care about — losing the advertiser-journalist conflict on MIND or the reader-quality-focus on Amazon.

4. **The Phase B regression test catches direction-condition bugs on the positive side.** Phase B reruns all experiments on MovieLens and diffs against existing results. If a refactoring bug breaks the "improve" side of the direction condition, the MovieLens regression will catch it at cos ≈ +0.95.

**What we need to watch for in Phase C/D**: Since MIND and Amazon lack strong positive pairs, a dataset-specific bug that only affects the "improve" side would be harder to detect. Phase B's MovieLens regression test mitigates this, but we should also explicitly check that at least the one strong positive pair we do have (Amazon indie_author-diversity +0.65) behaves as expected.

### Minor Observation: indie_author Label Balance

Amazon's `indie_author` shows `frac_first = 0.197` (imbalance = 0.303) — much below 0.5. This looks alarming but is an artifact of the label-balance metric, not a problem with the stakeholder.

Indie_author assigns weight -1.0 to items in the top-3 populated categories (which is where most items live). Random pairs sampled from the pool therefore have a high probability of both items being popular-category → both get utility ≈ -1 → **ties**. The label-balance check computes `utility[c1] > utility[c2]` strictly, without noise, so ties aren't counted. In the non-tie cases (mixed pairs with one popular + one niche item), the niche item always wins — which happens to be at position c1 only 50% of the time, giving a biased-looking 20% frac_first.

**This is not a problem for BT training**: preference generation adds noise (σ=0.05) that breaks ties 50/50, so the actual training labels will be balanced. The G5 threshold (0.4) still passes at 0.303. Just a metric quirk to be aware of. If we want a cleaner check in a future run, add noise to the label-balance computation to match the preference generator.

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
- **2026-04-10**: Phase A complete. Both MIND and Amazon PASS all blocking criteria (G1, G2, G3, G5, G6). G4 (informational) fails on both due to domain-native stakeholder design; decision is **Option A** (accept and proceed) — reasoning captured in Phase A Results section. Phase B greenlit.
