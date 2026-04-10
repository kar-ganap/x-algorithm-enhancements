# Tier 2 Phase A: Geometry Dry Run — Retro

**Goal**: Verify that MIND and Amazon Kindle produce stakeholder cosine geometry suitable for direction-condition validation, *before* committing to ~8 days of BT experiments in Phases B–F.

**Status**: Complete 2026-04-10. Both datasets PASS blocking criteria (G1, G2, G3, G5, G6). G4 (informational) fails on both; decision Option A (accept, proceed).

---

## 1. What Worked

### Gating BT training behind a geometry check was the right call
Phase A ran in ~4 hours wall-clock and produced a definitive go/no-go answer without a single BT training run. The cosine matrix + label disagreement + label balance is a ~2-minute compute that catches the catastrophic failure mode (all stakeholders aligned → no Goodhart testable) before it wastes days. On a ~10-day budget, this is excellent leverage.

### Using HuggingFace mirrors for both datasets
The canonical sources were dead (MIND Azure blob → 409, Amazon UCSD 2018 → 404). The Recommenders/MIND and McAuley-Lab/Amazon-Reviews-2023 HF mirrors worked cleanly with `requests` + streaming — no API keys, no auth. The 2023 Amazon release is actually *better* than 2018: schema includes explicit `price`, `average_rating`, `rating_number`, and hierarchical `categories`. This richer metadata enabled the price-tier and sentiment-bucket features in the 32-dim Amazon space.

### Duck-typed dataset interface (not a formal protocol)
Keeping `MINDDataset` and `AmazonDataset` as structural twins of `MovieLensDataset` (same attribute names: `train_ratings`, `items`, `num_items`, `num_features`) means Phase B can add a single `--dataset` flag and dispatch via a small registry. No inheritance, no protocol class, no rewrite of the reward-modeling stack. The dimension-agnostic core (`k_stakeholder_frontier.py`, BT/Margin-BT/Calibrated-BT losses) was already ready to accept `D=35` or `D=32` without changes.

### Domain-native stakeholders produced interpretable geometry
The 5 MIND stakeholders (reader, publisher, advertiser, journalist, civic_diversity) and 5 Amazon stakeholders (reader, publisher, indie_author, premium_seller, diversity) map to real-world roles. The cosine structure tells a coherent story: in news, advertisers oppose journalists on hard news (-0.31); in Kindle, publishers oppose indie authors on top-3 populated categories (-0.58). These aren't synthetic test cases — a reviewer can read the weights and understand why each pair is aligned or opposed.

### Streaming subset construction for Amazon
The Amazon ratings file is 932 MB and the metadata is 2.27 GB. Rather than loading everything into memory, the download script does 4 streaming passes: (1) count reviews per item, (2) pick top-20K, (3) count reviews per user on those items, (4) write filtered JSONL. Peak memory stays in tens of MB while processing gigabytes. Final subset: 12 MB metadata + 608 MB reviews, loadable in ~30s.

## 2. Surprises

### Python 3.11 dataclass + PEP-604 unions + dynamic imports is a silent foot-gun
`@dataclass` with `float | None` annotations + loading the module via `importlib.util.spec_from_file_location` + NOT registering the module in `sys.modules` = `AttributeError: 'NoneType' object has no attribute '__dict__'` from deep inside the stdlib dataclass machinery. The error surfaces from `_is_type` trying to resolve `cls.__module__` which isn't registered yet. Fix: `sys.modules[name] = mod` BEFORE `spec.loader.exec_module(mod)`. The existing `movielens_stakeholders.py` pattern escaped this because MovieLens dataclasses don't use PEP-604 unions. Worth a new lesson since it cost ~15 min of debugging on two modules.

### The 2023 Amazon categories field has a 2-level fixed prefix
My first-pass `_extract_top_category` took the first non-empty string from `categories`, which yielded "Kindle Store" for all books (because the list always starts with `['Kindle Store', 'Kindle eBooks', <actual_genre>, ...]`). Result: only 2 "categories" detected across 20K items → catastrophic feature collapse. Fix: skip the two known wrappers and return the first real category. Yielded 20 meaningful genres (Literature & Fiction, Romance, Mystery/Thriller, etc.) — exactly what the feature space needed.

The lesson generalizes: when onboarding a new structured dataset, *print raw records before trusting field shapes*. The schema documentation lied by omission — it listed the fields but didn't mention the fixed-prefix convention.

### G4 is harder to satisfy with domain-native stakeholders than expected
The plan assumed G4 (≥2 pairs with cos > 0.5) would pass easily on both datasets. It didn't. MIND's max cosine is +0.482; Amazon has exactly 1 pair above 0.5. The reason: when you define 5 distinct domain roles, there *aren't* two that should share 50°+ alignment. On MovieLens the user-platform alignment (+0.95) is a geometry quirk of the rating-derived definitions, not a universal property of multi-stakeholder systems. Forcing G4 to pass would have meant adding contrived "twin" stakeholders, which defeats the domain-native story.

Resolution captured in the plan doc's "G4 Failure Decision (Option A)" section. Short version: positive controls come from MovieLens; new datasets contribute stronger negatives and exact-zero orthogonal cases that MovieLens can't provide.

### Amazon's `premium_seller` is exactly orthogonal (cos = 0.000) to 3 category-based stakeholders
Because `premium_seller` has weights only in the price, sentiment, and length feature slots while `publisher`, `indie_author`, and `diversity` have weights only in the category slots, their dot products are exactly zero. No other pair in the tier 2 geometry (and no pair in MovieLens) comes this close to cos=0. This is a *uniquely useful* test case for the direction condition: it probes the exact theoretical decision boundary where the condition's prediction flips. Worth calling out in Phase C as a special case to monitor.

### indie_author label balance metric was misleading
The label-balance check computes `utility[c1] > utility[c2]` strictly without noise. For `indie_author`, which assigns the same utility (-1) to all top-3-category items (most of the pool), most random pairs tie. Non-ties only happen when one item is niche — and then the niche item always wins, giving a suspicious frac_first=0.197. With noise (σ=0.05) during preference generation, ties break 50/50 and actual training labels will be balanced. Still passes G5 (0.303 < 0.40), so no action. But the metric should be improved in a future iteration: add noise matching the preference generator.

## 3. Deviations from Plan

### Download sources changed mid-execution
Plan specified Microsoft Azure blob for MIND and UCSD 2018 for Amazon. Both were dead on the first attempt. Switched to HuggingFace mirrors (Recommenders/MIND and McAuley-Lab/Amazon-Reviews-2023) during Phase A, not as pre-planned work. Cost: ~20 min of research via a subagent. No change to the overall plan.

### Amazon is 2023 release, not 2018
Consequence of the URL fix. Schema differences from 2018:
- `rating` instead of `overall`, `user_id` instead of `reviewerID`
- `parent_asin` is the join key (not `asin`)
- Metadata includes `price`, `average_rating`, `rating_number`, `categories` hierarchy

The 2023 schema is *richer* and enabled more feature dimensions. No downside beyond needing to parse a different format.

### Branch discipline lapse, caught mid-phase
Initial download scripts and tier2 plan doc were written on `recsys/paper-writing` instead of a new `recsys/tier2-datasets` branch. User caught the lapse with the exact message lesson 9 was written about. Fixed by committing paper work to `recsys/paper-writing`, then branching `recsys/tier2-datasets` off it. Added **lesson 10** to `tasks/lessons.md`: "Branch at the moment scope changes, not when you notice."

### Retro discipline lapse, caught after code
Phase A was complete but I moved directly to asking about next steps without writing this retro file. User caught the lapse. Lesson 7 already covers retro discipline — this is the second time in RecSys phases I've had to be reminded. Worth reinforcing in lessons.

## 4. Implicit Assumptions Made Explicit

### The direction condition is testable on any dataset with positive cosine spread
*False*. If all pairs on a dataset have cos > 0, you can't test the "degrade" direction on that dataset alone (see original synthetic benchmark). MIND and Amazon deliberately span both signs, which is what makes them contribute to the generalizability claim.

### Domain-native stakeholders would give us G4 passage for free
*False*. My mental model was that "reader" and "publisher" on a news site are aligned enough to have cos > 0.5. They're not — cos = 0.48 on MIND. The categories readers click aren't exactly the categories publishers produce. Each domain is its own geometry problem; there's no universal "aligned pair" template.

### Label balance = absence of noise is the right metric
*False for rare-positive-class stakeholders*. The metric needs to match what BT sees during training (noisy preferences), not what the raw utility function gives for exact-tie pairs. Fix is straightforward; logged as a future improvement.

### MovieLens's positive-control role is obvious to reviewers
*Probably false*. A reviewer reading only §4 of the paper might not immediately see that "MovieLens provides positive controls at cos ≈ +0.95, MIND and Amazon provide negative controls and transition-zone coverage." This needs to be stated explicitly in the paper — the division of labor across datasets is part of the evidence structure.

## 5. Scope Changes for Next Phase (Phase B)

### Phase B regression test on MovieLens must verify positive-side direction condition
Since MIND and Amazon lack strong positive pairs (cos > 0.5), a refactoring bug that only breaks the "improve" prediction could hide on those datasets. The existing Phase B plan already runs all refactored scripts on MovieLens and diffs against committed JSONs — that diff will catch bugs at cos ≈ +0.95. Just explicitly note this in the Phase B plan so the regression test isn't treated as optional.

### Phase C should explicitly monitor the Amazon orthogonal cases
`premium_seller` vs `publisher`/`indie_author`/`diversity` all have cos = 0.000 exactly. This is a *unique* test case — theoretically the boundary where the direction condition's prediction is undefined. Phase C should log these cases separately and report what the learned models do: do they randomly improve or degrade? Does the condition break down at exactly zero, or does one direction dominate?

### Phase C's MIND runs should check for label skew at Method B's 5 named stakeholders
The advertiser frac_first=0.348 (imbalance 0.152) is the highest on MIND. That's within G5 but non-trivial. If any of the 5 named stakeholders from the existing MovieLens test suite (creator, advertiser, niche, mainstream, moderator) turn out to have even worse balance on MIND, BT convergence could suffer. Add a pre-flight check in Phase C that runs `label_balance` with noise on every stakeholder before committing to a full run.

### Phase F (docs) should update `docs/results.md` with the cross-dataset summary, not just a retro
The existing pattern (phase retros + a rolling `results.md`) is the right one. Don't let Phase F skip the `results.md` update again.

## 6. Metrics

| Metric | Value |
|--------|-------|
| Wall-clock time | ~4 hours (matches plan budget) |
| Files created | 7 (`download_mind.py`, `download_amazon.py`, `mind.py`, `mind_stakeholders.py`, `amazon.py`, `amazon_stakeholders.py`, `phase_a_geometry_check.py`) |
| Lines of code added | ~2400 |
| MIND pool size | 65,238 articles → 12,261 content pool (min-5-impressions filter) |
| MIND feature dim | 35 (17 categories + 14 subcategories + 4 quality signals) |
| MIND stakeholders | 5 (reader, publisher, advertiser, journalist, civic_diversity) |
| MIND cosine range | [-0.839, +0.482] |
| MIND negative pairs | 4 (cos < -0.1) |
| MIND transition-zone pairs | 5 (\|cos\| < 0.2) |
| Amazon pool size | 20,000 books → 17,425 content pool |
| Amazon feature dim | 32 (20 genres + 4 price + 3 volume + 3 sentiment + 2 length) |
| Amazon stakeholders | 5 (reader, publisher, indie_author, premium_seller, diversity) |
| Amazon cosine range | [-0.761, +0.654] |
| Amazon negative pairs | 2 (cos < -0.1) |
| Amazon transition-zone pairs | 6 (\|cos\| < 0.2) |
| Amazon orthogonal pairs (cos = 0.000) | 3 (unique to Amazon) |
| G1 (≥1 negative) | PASS both |
| G2 (range ≥ 0.5) | PASS both (MIND 1.32, Amazon 1.42) |
| G3 (≥1 transition) | PASS both |
| G4 (≥2 cos > 0.5) | FAIL both (informational only) |
| G5 (no 90/10 imbalance) | PASS both |
| G6 (no cos > 0.95) | PASS both |
| Tests added | 0 (Phase A is exploratory, tests added in Phase F) |
| Downloads | MIND 80 MB, Amazon ~3.2 GB |
| Branches touched | 1 new (`recsys/tier2-datasets`) |
| Commits on tier2 branch | 4 |
| Lessons added | 1 (lesson 10: branch at scope change trigger) |
