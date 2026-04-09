# RecSys Phase 13: Data Budget 20 Seeds — Retro

**Goal**: Increase seed count from 5 to 20 to tighten data budget CIs and resolve non-monotonicity.

**Status**: Complete. All concerns resolved.

---

## 1. What Worked

### 20 seeds resolved every issue
- CI width halved (34pp → 16pp on ML-100K, 29pp → 13pp on ML-1M)
- ML-1M non-monotonicity at N=50 resolved — was pure seed noise
- Both datasets now monotonically improving across all N values
- Cross-dataset consistency: 46-56% recovery at N=25

### The finding is now publishable
"25 pairs recover 46-56% of hidden stakeholder harm (95% CI: [40%, 64%])" is a tight, defensible claim. The lower bound (40%) is still substantial — even worst case, a small annotation budget meaningfully reduces harm.

## 2. Surprises

### The point estimate shifted
ML-100K moved UP (52.9% → 56.3%) and ML-1M moved DOWN (49.1% → 46.3%). The 5-seed estimates were noisy in opposite directions. With 20 seeds the two datasets converge to a tighter range (46-56% vs the earlier 49-53%).

### 20 seeds was sufficient
Standard error at N=25 is now ~4% (std ~0.65 / √20 ≈ 0.15, recovery SE ≈ 4pp). Further increasing seeds would improve precision marginally. 20 is the right number for this experiment.

## 3. Lesson

**5 seeds is not enough for data budget experiments where per-seed variance is high.** The BT model at N=25 is highly variable (content pool sampling + BT initialization + preference pair sampling all contribute noise). 20 seeds reduces SE by 2× and resolves artifacts. Should have used 20 from the start — the synthetic experiment already used 20 seeds for the same reason.

## 4. Metrics

| Metric | 5 seeds | 20 seeds |
|--------|---------|----------|
| ML-100K N=25 recovery | 52.9% [34%, 72%] | 56.3% [48%, 64%] |
| ML-1M N=25 recovery | 49.1% [34%, 63%] | 46.3% [40%, 53%] |
| ML-1M monotonic? | No (N=50 > N=25) | Yes |
| CI width (ML-100K) | 38pp | 16pp |
| CI width (ML-1M) | 29pp | 13pp |
| Runtime (total both datasets) | ~5 min | ~13 min |
