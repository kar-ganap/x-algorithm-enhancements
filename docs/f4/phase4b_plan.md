# Phase 4b: Per-Stakeholder Model Training

## Objective

Train three separate reward models, each optimized for a different stakeholder's utility:
1. **User-Optimized Model**: Maximizes engagement - discomfort
2. **Platform-Optimized Model**: Maximizes total engagement
3. **Society-Optimized Model**: Maximizes diversity - polarization

Then compare how these models behave differently.

---

## Training Objectives

### User-Optimized Model
```
Loss = BradleyTerry(R_pref, R_rej) + λ * discomfort_penalty

Where:
  R = w_favorite * favorite + w_repost * repost + ...
      - w_block * block - w_report * report

  discomfort_penalty = Σ(negative_action_probs)
```

**Goal**: Learn weights that maximize positive engagement while heavily penalizing content that triggers blocks/reports.

### Platform-Optimized Model
```
Loss = BradleyTerry(R_pref, R_rej)

Where:
  R = Σ(all action probs)  # All engagement counts equally
```

**Goal**: Learn weights that maximize total engagement regardless of sentiment.

### Society-Optimized Model
```
Loss = BradleyTerry(R_pref, R_rej) + λ_div * diversity_loss + λ_pol * polarization_penalty

Where:
  diversity_loss = -entropy(topic_distribution)
  polarization_penalty = same_side_ratio for political users
```

**Goal**: Learn weights that promote diverse content and reduce political echo chambers.

---

## Test Plan

### Test 1: Learned Weight Comparison

**Question**: What weights does each model learn for each action?

**Method**:
- Extract learned weights from all 3 models
- Compare weights for key actions: favorite, repost, block, report, follow

**Expected Findings**:
| Action | User Model | Platform Model | Society Model |
|--------|------------|----------------|---------------|
| favorite | High (+) | High (+) | Moderate (+) |
| repost | High (+) | High (+) | Lower (viral spread) |
| block | **Very negative** | Near zero | Negative |
| report | **Very negative** | Near zero | Negative |
| follow | Positive | Positive | Moderate |

**Success Criteria**: Models learn meaningfully different weights.

---

### Test 2: Ranking Correlation Analysis

**Question**: How much do the models agree/disagree on rankings?

**Method**:
- For 100 users × 50 content items, get rankings from all 3 models
- Compute pairwise Kendall's τ correlation
- Compute Jaccard similarity of top-10 recommendations

**Expected Findings**:
```
Correlation Matrix (Kendall's τ):
              User    Platform   Society
User          1.00    0.7-0.8    0.5-0.7
Platform      -       1.00       0.4-0.6
Society       -       -          1.00
```

**Success Criteria**:
- User-Platform correlation > 0.6 (some overlap)
- Platform-Society correlation < 0.7 (meaningful divergence)

---

### Test 3: Contested Content Identification

**Question**: Which content items do models rank most differently?

**Method**:
- For each content item, compute rank variance across models
- Identify top-10 most contested items
- Analyze: What type of content is contested?

**Expected Findings**:
- Political content: High variance (Platform loves engagement, Society penalizes)
- Outrage-bait: High variance (Platform loves, User hates)
- Educational content: Low variance (all agree it's good)

**Success Criteria**: Can identify and explain contested content.

---

### Test 4: Content Type Analysis

**Question**: How does each model rank different content types?

**Method**:
- Group content by topic (sports, tech, politics_L, politics_R, entertainment, news)
- Compute average rank per topic per model

**Expected Findings**:
| Topic | User Model | Platform Model | Society Model |
|-------|------------|----------------|---------------|
| Sports | Mid | Mid | Mid |
| Politics | Lower | **High** | **Low** |
| Entertainment | High | High | Mid |
| News | Mid | Mid | High |

**Success Criteria**: Society model ranks political content lower.

---

### Test 5: Cross-Partisan Exposure Analysis

**Question**: Do political users see content from "the other side"?

**Method**:
- For political_L users: What % of top-10 is politics_R content?
- For political_R users: What % of top-10 is politics_L content?
- Compare across models

**Expected Findings**:
| Model | Cross-Exposure Rate |
|-------|---------------------|
| Platform | 5-10% (echo chamber) |
| User | 10-15% |
| Society | **30-50%** (intentional diversity) |

**Success Criteria**: Society model has significantly higher cross-exposure.

---

### Test 6: Win-Win Content Identification

**Question**: Is there content that ALL stakeholders agree is good?

**Method**:
- Find content ranked in top-20 by ALL three models
- Analyze characteristics of this "universally good" content

**Expected Findings**:
- High-quality educational content
- Entertainment without controversy
- Content that generates positive engagement (favorites) without negative (blocks)

**Success Criteria**: Can identify win-win content characteristics.

---

### Test 7: Policy Simulation

**Question**: What happens if we switch optimization objectives?

**Method**:
Simulate: "Platform currently uses Platform-Optimized model. What if they switched to Society-Optimized?"
- Compute: % of recommendations that would change
- Compute: Change in each utility metric
- Identify: Who wins/loses from the switch

**Expected Findings**:
```
Switch from Platform → Society Model:
  - Recommendations changed: 35-50%
  - User utility: -5% to +5% (minimal impact)
  - Platform utility: -15% to -25% (engagement drop)
  - Society utility: +40% to +60% (big improvement)
```

**Success Criteria**: Can quantify policy change impact.

---

### Test 8: Causal Verification

**Question**: Do all models pass causal tests?

**Method**:
- Run block intervention test on all 3 models
- Run follow intervention test on all 3 models
- Compare effect sizes

**Expected Findings**:
| Model | Block Effect | Follow Effect |
|-------|--------------|---------------|
| User | **Large negative** | Positive |
| Platform | Small negative | Positive |
| Society | Moderate negative | Positive |

**Success Criteria**: User model has strongest block effect.

---

## Implementation Plan

### Step 1: Define Stakeholder Loss Functions
- `user_bradley_terry_loss()` with discomfort penalty
- `platform_bradley_terry_loss()` standard
- `society_bradley_terry_loss()` with diversity/polarization terms

### Step 2: Train Three Models
- Use same training data for all three
- Same architecture (two-stage with rich features)
- Different loss functions

### Step 3: Implement Comparison Metrics
- Ranking correlation computation
- Contested content identification
- Cross-exposure measurement

### Step 4: Run All Tests
- Generate comprehensive comparison report

### Step 5: Visualize Results
- Weight comparison bar charts
- Ranking correlation heatmap
- Policy simulation impact table

---

## Files to Create

| File | Purpose |
|------|---------|
| `stakeholder_models.py` | Training with stakeholder-specific losses |
| `stakeholder_comparison.py` | Comparison metrics and analysis |
| `train_stakeholder_models.py` | Script to train all 3 models |
| `compare_stakeholder_models.py` | Script to run all comparisons |

---

## Success Metrics

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Weight divergence | Cosine sim < 0.9 | Models learned different things |
| Ranking correlation (Platform-Society) | τ < 0.7 | Meaningful disagreement |
| Contested content identified | ≥ 10 items | Can explain disagreement |
| Cross-exposure difference | ≥ 20% gap | Society model reduces echo chambers |
| Policy simulation complete | All metrics computed | Can inform decisions |
| All models pass causal tests | 100% block/follow | Still valid reward models |
