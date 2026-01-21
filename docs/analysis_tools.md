# Analysis Tools for Phoenix Recommendation System

This document describes the analysis tools built to understand recommendation dynamics, filter bubble effects, and model behavior.

---

## Overview

| Tool | Purpose | Status |
|------|---------|--------|
| **Trajectory Simulation** | How do rankings evolve as users engage? | ✅ Works with random weights |
| **Real Re-ranking** | Actual model inference per engagement step | ✅ Ready for trained weights |
| **Diversity Metrics** | Detect filter bubble / echo chamber effects | ✅ Works (coverage metrics meaningful) |
| **Sensitivity Analysis** | Quantify outcome space constraints | ✅ Works with random weights |
| **Counterfactual Analysis** | Which history items influence rankings? | ⏳ Needs trained weights |

**Key Finding:** With randomly initialized weights, the model produces uniform scores (~0.5 for all candidates). Filter bubble dynamics and history sensitivity are **learned behaviors**, not architectural properties. These tools are ready to reveal real dynamics once trained weights are available.

---

## Quick Start

```bash
# Trajectory simulation (simulated perturbation)
uv run python enhancements/analysis/ranking_dynamics.py

# Path divergence analysis
uv run python enhancements/analysis/path_divergence.py

# Sensitivity analysis
uv run python enhancements/analysis/sensitivity_analysis.py

# Diversity metrics (simulated)
uv run python enhancements/analysis/diversity_metrics.py --candidates 100

# Diversity metrics (real re-ranking - use with trained weights)
uv run python enhancements/analysis/diversity_metrics.py --candidates 100 --real

# Real trajectory simulation
uv run python enhancements/analysis/real_trajectory_simulation.py

# Counterfactual analysis (use with trained weights)
uv run python enhancements/analysis/counterfactual_analysis.py --history 32
```

---

## Tool Details

### 1. Trajectory Simulation

**File:** `enhancements/analysis/trajectory_simulation.py`

**Purpose:** Simulate how rankings change as a user engages with content over time.

**How it works:**
1. Initial ranking of all candidates
2. User "engages" with a candidate (removes it from pool)
3. Re-rank remaining candidates
4. Repeat

**Two modes:**
- **Simulated:** Score perturbation (`score * 0.98 + noise`) - fast but artificial
- **Real:** Actual model inference with extended history - slower but accurate

**Key classes:**
```python
from enhancements.analysis import TrajectorySimulator, RealTrajectorySimulator

# Simulated (fast)
simulator = TrajectorySimulator(runner, batch, embeddings)
simulator.engage_top_n(5)
trajectory = simulator.get_trajectory()

# Real re-ranking (accurate)
real_sim = RealTrajectorySimulator(runner, batch, embeddings, num_item_hashes, num_author_hashes)
real_sim.engage_top_n(5)
real_trajectory = real_sim.get_trajectory()
```

---

### 2. Diversity Metrics

**File:** `enhancements/analysis/diversity_metrics.py`

**Purpose:** Detect filter bubble / echo chamber effects by measuring:
- **Intra-list diversity:** Embedding distance between candidates
- **Coverage:** What fraction of candidates ever get engaged
- **Gini coefficient:** Concentration of engagement (0=equal, 1=concentrated)

**Key findings (with 100 candidates):**

| Metric | Top Strategy | Random Strategy |
|--------|--------------|-----------------|
| Unique engaged | 20/100 (20%) | 100/100 (100%) |
| Gini coefficient | 0.80 | 0.16 |
| Coverage gap | 80% fewer | - |

**Interpretation:** The filter bubble manifests as **coverage**, not embedding similarity. Users following recommendations only see 20% of the catalog.

**Usage:**
```bash
# Simulated mode (fast)
uv run python enhancements/analysis/diversity_metrics.py --candidates 100

# Real re-ranking mode (use with trained weights)
uv run python enhancements/analysis/diversity_metrics.py --candidates 100 --real
```

---

### 3. Sensitivity Analysis

**File:** `enhancements/analysis/sensitivity_analysis.py`

**Purpose:** Quantify how sensitive outcomes are to early choices.

**Compares:**
- **Random strategy:** Choose randomly at each step
- **Top-biased strategy:** 70% choose top, 30% random

**Key metrics:**
- Position entropy (predictability of each engagement position)
- Outcome diversity (unique sequences / total runs)
- Engagement concentration (Gini)

**Finding:** With 8 candidates, top-biased strategy shows 40% reduction in outcome diversity and 68% first-choice predictability.

---

### 4. Counterfactual Analysis

**File:** `enhancements/analysis/counterfactual_analysis.py`

**Purpose:** Understand which aspects of user history influence rankings.

**Questions answered:**
1. **Ablation:** If we remove history item X, how do rankings change?
2. **Recency:** Does recent history matter more than old history?
3. **Action sensitivity:** Does engagement type (favorite vs retweet) matter?

**⚠️ Requires trained weights** to produce meaningful results. With random weights, all Kendall's τ = 1.0 (no history influence).

**Key classes:**
```python
from enhancements.analysis import CounterfactualAnalyzer

analyzer = CounterfactualAnalyzer(runner, batch, embeddings)

# Which history items matter most?
ablation_results = analyzer.ablate_all_history()

# Does recent history dominate?
recency_sensitivity = analyzer.analyze_recency_sensitivity()

# Truncate to last N items
snapshot = analyzer.truncate_history(keep_last_n=10)
```

**Expected results with trained weights:**
- Some history positions have lower τ when removed (more influential)
- Recent items likely more influential (recency bias)
- Different action types should produce different rankings

---

## When to Use Each Tool

| Question | Tool | Notes |
|----------|------|-------|
| "Do small choices compound?" | Path Divergence | Works now |
| "Does the system create filter bubbles?" | Diversity Metrics | Coverage/Gini meaningful now |
| "How predictable are outcomes?" | Sensitivity Analysis | Works now |
| "Which history items matter?" | Counterfactual | **Needs trained weights** |
| "How does actual model respond to history?" | Real Re-ranking | **Needs trained weights** |

---

## Architecture Notes

### Why caching doesn't help counterfactual analysis

**Trajectory simulation (caching helps):**
```
Cache: user + history K/V tensors
Vary: candidates
Flow: context → candidates (candidates attend TO cached context)
```

**Counterfactual (no caching benefit):**
```
Would need to cache: candidate K/V tensors
But: candidates DEPEND on context (they attend to it)
So: can't pre-compute candidate K/V independently
```

The asymmetry comes from Phoenix's attention pattern:
- Context tokens: causal self-attention (don't see candidates)
- Candidates: attend to ALL context + self

### What IS context-independent

Candidate **input embeddings** (hash lookups) don't depend on context. Only transformer representations do. So counterfactual analysis:
1. Fixes candidate input embeddings
2. Varies context input embeddings
3. Re-runs transformer each time

---

## Future Work (with trained weights)

1. **Load trained weights** into `FullKVCachedRunner`
2. **Re-run analyses** to observe real dynamics:
   - Do rankings actually shift with engagement?
   - Which history items are most influential?
   - Is there recency bias?
3. **Compare to random baseline** to isolate learned vs architectural effects

---

## File Summary

```
enhancements/analysis/
├── __init__.py                    # Exports
├── trajectory_simulation.py       # Simulated trajectory (perturbation-based)
├── real_trajectory_simulation.py  # Real trajectory (actual model inference)
├── ranking_dynamics.py            # Visualize score evolution
├── path_divergence.py             # Compare diverging paths
├── sensitivity_analysis.py        # Quantify outcome sensitivity
├── diversity_metrics.py           # Filter bubble detection
└── counterfactual_analysis.py     # History item importance

tests/test_analysis/
└── test_trajectory_simulation.py  # 25 tests
```

---

## Test Coverage

```bash
# Run all analysis tests
uv run pytest tests/test_analysis/ -v

# Run all tests (87 total)
uv run pytest tests/ -v
```
