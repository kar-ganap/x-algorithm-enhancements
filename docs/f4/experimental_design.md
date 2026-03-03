# F4 Experimental Design: Multi-Stakeholder Reward Modeling

## Abstract

This report documents the experimental design behind F4, a multi-stakeholder reward modeling system built on xAI's Phoenix recommendation ranker. The core problem: a social media platform serves multiple stakeholders — users, the platform itself, and society — whose objectives often conflict. A post that maximizes engagement (platform's goal) might increase political polarization (society's concern) or surface content the user would rather not see (user's concern). We ask: given distinct definitions of what each stakeholder values, can we train meaningfully different reward models from the same underlying action probabilities?

The answer is yes — but the mechanism is not what we initially expected. Across 87 experiments with 4 different loss functions, we discovered that stakeholder differentiation comes entirely from the training labels (which content pairs each stakeholder prefers), not from the loss function used to train the model. This report details the full experimental setup: how we generate synthetic data with 648 explicit behavioral parameters, how we define stakeholder utility functions, how preference pairs are constructed, and the mathematical formulation of each loss function we tested.

All experiments use a 648-parameter synthetic ground truth (6 user archetypes, 6 content topics, 18 user actions), enabling deterministic verification that would be impossible with real data. The system is validated through a 4-level verification hierarchy and Pareto frontier analysis comparing learned and hardcoded reward scorers.

---

## 1. Problem Setup

### 1.1 The recommendation setting

Phoenix, xAI's recommendation ranker, processes each (user, content) pair and outputs a vector of 18 action probabilities — the model's estimate of how likely the user is to take each possible action on that content. The 18 actions, in Phoenix's canonical order, are:

| Index | Action | Category |
|-------|--------|----------|
| 0 | favorite | Positive |
| 1 | reply | Positive (ambiguous) |
| 2 | repost | Positive |
| 3 | `photo_expand` | Neutral |
| 4 | click | Neutral |
| 5 | `profile_click` | Neutral |
| 6 | vqv (video quality view) | Neutral |
| 7 | share | Positive |
| 8 | `share_via_dm` | Positive |
| 9 | `share_via_copy_link` | Neutral |
| 10 | dwell | Neutral |
| 11 | quote | Positive |
| 12 | `quoted_click` | Neutral |
| 13 | `follow_author` | Positive |
| 14 | `not_interested` | Negative |
| 15 | `block_author` | Negative |
| 16 | `mute_author` | Negative |
| 17 | report | Negative |

A reward model converts this 18-dimensional probability vector into a single scalar score that determines ranking. The simplest form is a linear reward:

$$R(u, c) = \mathbf{w}^\top \mathbf{P}(u, c)$$

where $\mathbf{w} \in \mathbb{R}^{18}$ is a weight vector and $\mathbf{P}(u, c) \in \mathbb{R}^{18}$ is the action probability vector for user $u$ on content $c$.

This is implemented as `rewards = jnp.einsum("bca,a->bc", probs, self.weights.weights)` in `enhancements/reward_modeling/reward_model.py:130`.

### 1.2 Three stakeholders, one action space

The key observation: the same 18 action probabilities mean different things to different stakeholders.

**User**: A favorite (action 0) is positive — the user enjoyed the content. A block (action 15) is strongly negative — the user had a bad experience. The user wants content they'll enjoy and wants to avoid content that triggers negative reactions.

**Platform**: A favorite is positive (engagement), but a block is *also* informative — it's a signal the platform can use to improve future recommendations. For the platform, almost all actions represent engagement, and engagement drives revenue. Even negative actions are better than the user leaving the app.

**Society**: Neither individual favorites nor blocks capture societal impact. What matters is the aggregate pattern: are users exposed to diverse viewpoints, or are they trapped in echo chambers? Society cares about the distribution of content across topics, not individual interactions.

### 1.3 The core question

Given three stakeholder utility functions $U_\text{user}$, $U_\text{platform}$, $U_\text{society}$, each mapping the same action probabilities to different scalar values, can we train three reward models with meaningfully different weight vectors?

"Meaningfully different" is operationalized as cosine similarity below 0.95:

$$\cos(\mathbf{w}_i, \mathbf{w}_j) = \frac{\mathbf{w}_i \cdot \mathbf{w}_j}{\|\mathbf{w}_i\| \|\mathbf{w}_j\|} < 0.95 \quad \text{for } i \neq j$$

This threshold was chosen because prior experiments with identical training labels produced cosine similarities of ~0.999 — so anything below 0.95 represents genuine differentiation.

### 1.4 Formal notation

We use the following notation throughout this report:

| Symbol | Meaning |
|--------|---------|
| $a \in \{1, \ldots, 18\}$ | Action index |
| $k \in \{\text{user, platform, society}\}$ | Stakeholder |
| $P_a(u, c)$ | Probability user $u$ takes action $a$ on content $c$ |
| $\mathbf{P}(u, c) \in \mathbb{R}^{18}$ | Full action probability vector |
| $\mathbf{w}_k \in \mathbb{R}^{18}$ | Stakeholder $k$'s reward weights |
| $R_k(u, c) = \mathbf{w}_k^\top \mathbf{P}(u, c)$ | Stakeholder $k$'s reward score |
| $\sigma(x) = 1/(1 + e^{-x})$ | Sigmoid function |
| $U_k(u, c)$ | Stakeholder $k$'s utility for showing content $c$ to user $u$ |

---

## 2. Synthetic Data Generation

### 2.1 Why synthetic data?

All experiments use synthetic data with explicitly specified behavioral parameters. This is a deliberate design choice: with synthetic data, we know the ground truth. If the model says a sports fan has a 0.68 probability of favoriting a sports post, we can compare against the ground truth value of 0.70. With real data, we would never know whether a model failure reflects a bug, a data quality issue, or a genuine limitation of the approach.

The cost is external validity — synthetic data is designed to be learnable, with clean archetype separation and noise-free engagement patterns. Real social media data has overlapping behaviors, temporal drift, and ambiguous preferences. Results on synthetic data are upper bounds on what's achievable in practice.

### 2.2 User archetypes

Six user archetypes capture distinct behavioral patterns. Each archetype represents a cluster of users with similar engagement tendencies.

| Archetype | Share | Behavioral signature |
|-----------|-------|---------------------|
| `SPORTS_FAN` | 15% | High engagement with sports; ignores politics |
| `POLITICAL_L` | 15% | Engages with left-leaning content; hostile to right-leaning content |
| `POLITICAL_R` | 15% | Engages with right-leaning content; hostile to left-leaning content |
| `TECH_BRO` | 15% | Tech/startup content enthusiast |
| `LURKER` | 20% | Passive consumer — likes occasionally, rarely shares or replies |
| `POWER_USER` | 20% | High engagement across all action types and topics |

Source: `UserArchetype` enum and `ARCHETYPE_DISTRIBUTION` in `enhancements/data/ground_truth.py:14-42`.

The distribution is intentional: `LURKER` and `POWER_USER` are the two most common archetypes (20% each), reflecting that most platforms have more passive consumers and power users than niche-topic enthusiasts.

### 2.3 Content topics

Six content topics cover the major content categories on a social media platform.

| Topic | Share | Notes |
|-------|-------|-------|
| SPORTS | 25% | Largest category |
| `POLITICS_L` | 12.5% | Left-leaning political content |
| `POLITICS_R` | 12.5% | Right-leaning political content |
| TECH | 20% | Technology and startups |
| ENTERTAINMENT | 20% | General entertainment |
| NEWS | 10% | Mixed/neutral news |

Source: `ContentTopic` enum and `TOPIC_DISTRIBUTION` in `enhancements/data/ground_truth.py:24-52`.

Political content is split into left and right to enable modeling of cross-partisan dynamics — a critical test case for societal impact.

### 2.4 The 648 parameters

The ground truth specifies `P(action | archetype, topic)` for every combination: 6 archetypes × 6 topics × 18 actions = 648 parameters. In practice, `LURKER` and `POWER_USER` use wildcard rules (same probabilities regardless of topic), so the specification has 26 rule entries in the `ENGAGEMENT_RULES` dictionary, each defining up to 18 action probabilities.

Here are the key engagement patterns that drive the experimental results:

**Sports fan on sports content** — strong topic affinity:
```
favorite: 0.70    repost: 0.30    reply: 0.10
click: 0.50       dwell: 0.60     follow_author: 0.15
```

**Political-left user on right-leaning content** — hostile cross-partisan interaction:
```
block_author: 0.25    mute_author: 0.15    not_interested: 0.30
reply: 0.10           report: 0.05         dwell: 0.15
```

**Lurker on any content** — passive consumption:
```
favorite: 0.20    repost: 0.01    reply: 0.00
click: 0.15       dwell: 0.35     follow_author: 0.02
```

**Power user on any content** — active across all types:
```
favorite: 0.45    repost: 0.35    reply: 0.25
click: 0.50       dwell: 0.55     quote: 0.15
share: 0.12       profile_click: 0.20
```

Source: `ENGAGEMENT_RULES` in `enhancements/data/ground_truth.py:142-338`.

The cross-partisan hostility pattern is especially important: when a left-leaning user encounters right-leaning content, the dominant actions are block (0.25), `not_interested` (0.30), and mute (0.15). This creates the divergence between stakeholder utilities — the platform sees engagement (even negative engagement counts), while society sees polarization.

### 2.5 Generating preference data

The engagement rules produce a 3D tensor of action probabilities: users × content × 18 actions. From this, preference pairs are generated for Bradley-Terry training. A typical dataset has 600 users × 100 content items = 60,000 (user, content) pairs, from which 2,000-5,000 preference triples (user, `preferred_content`, `rejected_content`) are sampled.

The wildcard mechanism works as follows: `get_engagement_probs(archetype, topic)` first checks for a specific `(archetype, topic)` rule. If none exists, it falls back to `(archetype, "*")` — the wildcard rule. This means `LURKER` and `POWER_USER` have the same behavior regardless of topic, while the other four archetypes have topic-specific engagement patterns.

---

## 3. Stakeholder Utility Functions

Each stakeholder's utility function maps the 18-dimensional action probability vector to a scalar value. These functions determine which content each stakeholder considers "better" and therefore which preference pairs are generated for training.

All utility functions are defined in `enhancements/reward_modeling/stakeholder_utilities.py` via the `UtilityWeights` dataclass (lines 52-102).

### 3.1 User utility: engagement minus discomfort

The user wants content they enjoy and wants to avoid content that triggers negative reactions.

$$U_\text{user}(\mathbf{P}) = \sum_{a \in \text{positive}} w_a^+ \cdot P_a - \sum_{a \in \text{negative}} |w_a^-| \cdot P_a$$

**Positive weights** (actions the user considers good):

| Action | Weight | Rationale |
|--------|--------|-----------|
| `follow_author` | 1.2 | Strongest positive — user wants more from this author |
| favorite | 1.0 | User liked the content |
| share | 0.9 | User recommended it to others |
| repost | 0.8 | User endorsed it publicly |
| quote | 0.6 | User engaged with commentary |
| reply | 0.5 | Engaged, but replies can be negative |

**Negative weights** (actions indicating discomfort):

| Action | Weight | Rationale |
|--------|--------|-----------|
| report | -2.5 | Strongest negative — content was harmful |
| `block_author` | -2.0 | User never wants to see this author again |
| `mute_author` | -1.5 | User wants less of this |
| `not_interested` | -1.0 | Mild negative signal |

The asymmetry is deliberate: one report (weight -2.5) cancels more than two favorites (weight +1.0). Showing a user content bad enough to trigger a report is a much bigger failure than showing them content good enough to favorite.

Implementation: `compute_user_utility()` in `stakeholder_utilities.py:136-182`.

### 3.2 Platform utility: total engagement plus retention

The platform wants maximum engagement of any kind. Even negative actions are signals — a block tells the platform to stop showing that author's content, and a report provides content moderation data.

$$U_\text{platform}(\mathbf{P}) = \sum_{a=1}^{18} w_a^\text{plat} \cdot P_a + \text{retention\_proxy}$$

where the retention proxy models user return rate:

$$\text{retention\_proxy} = r \cdot \left(1 - 0.5 \cdot \frac{\text{negative\_signals}}{\text{positive\_signals} + 0.1}\right)$$

with $r = 0.8$ (base return rate), clamped to $[0, 1]$.

**Platform weights** for all 18 actions:

| Action | Weight | Notes |
|--------|--------|-------|
| repost | 1.5 | Highest — viral spread |
| `share_via_dm` | 1.4 | Personal recommendation |
| share | 1.3 | Content distribution |
| quote | 1.3 | Creates derivative content |
| reply | 1.2 | Creates content threads |
| `share_via_copy_link` | 1.2 | External sharing |
| favorite | 1.0 | Standard engagement |
| `follow_author` | 1.0 | Builds creator relationships |
| `profile_click` | 0.6 | User exploration |
| click | 0.5 | Content consumption |
| vqv | 0.4 | Video engagement |
| `quoted_click` | 0.4 | Derivative content |
| `photo_expand` | 0.3 | Mild interest |
| dwell | 0.2 | Time on platform |
| report | 0.2 | Moderation signal |
| `not_interested` | 0.1 | Preference signal |
| `block_author` | 0.1 | Preference signal |
| `mute_author` | 0.1 | Preference signal |

The critical difference from user utility: negative actions still have positive (small) platform weights. A block is worth 0.1 to the platform but -2.0 to the user. This structural difference in how the same actions are valued is what makes different preference labels possible.

Implementation: `compute_platform_utility()` in `stakeholder_utilities.py:185-237`.

### 3.3 Society utility: diversity minus polarization

Society's utility doesn't depend on individual action probabilities in the same way. Instead, it measures aggregate patterns across a user's recommended content set.

$$U_\text{society} = \text{diversity} - \text{polarization}$$

**Diversity**: the average fraction of unique topics each user sees in their top-K recommendations. If a user's feed contains 4 out of 6 possible topics, diversity = 4/6 = 0.67. Perfect diversity (all 6 topics) = 1.0; complete homogeneity (one topic) = 1/6 = 0.17.

**Polarization**: computed only for political users (archetypes `POLITICAL_L` and `POLITICAL_R`). For a left-leaning user, polarization measures the fraction of political content from their own side:

$$\text{polarization} = \frac{\text{same\_side\_political}}{\text{same\_side\_political} + \text{other\_side\_political}}$$

A polarization of 1.0 means the user only sees political content they already agree with (pure echo chamber). 0.5 means balanced exposure. The society utility function penalizes polarization because echo chambers are considered harmful regardless of political direction.

Implementation: `compute_society_utility()` in `stakeholder_utilities.py:240-310`.

### 3.4 The free parameters

The `UtilityWeights` dataclass contains approximately 14 free parameters: 6 user positive action weights, 4 user negative action weights, and the 18 platform weights (though many are constrained by the "all engagement is valuable" design principle, leaving perhaps 4-5 truly free choices for the platform).

These parameters are hand-specified based on action semantics, not empirically validated. This is acknowledged as a limitation: no experiment determined that reposts should be weighted 0.8 for users or 1.5 for platforms. The sensitivity of results to these choices is an open research question (see the retro's research directions).

---

## 4. Preference Pair Generation

This section describes the mechanism that turned out to be the critical lever for stakeholder differentiation. The entire F4 investigation ultimately comes down to how preference pairs are constructed.

### 4.1 The algorithm

For each stakeholder $k$, preference pairs are generated as follows:

1. For each user $u$, sample two content items $c_1, c_2$ uniformly at random.
2. Compute $U_k(u, c_1)$ and $U_k(u, c_2)$ using stakeholder $k$'s utility function.
3. If $U_k(u, c_1) > U_k(u, c_2)$: the pair is (preferred = $c_1$, rejected = $c_2$).
4. Otherwise: the pair is (preferred = $c_2$, rejected = $c_1$).
5. Repeat to generate $N$ preference pairs (typically $N$ = 2,000-5,000).

Because different stakeholders have different utility functions, the same content pair $(c_1, c_2)$ can receive opposite preference labels. When a left-leaning user encounters one sports post and one right-leaning political post:

- **User utility** prefers the sports post (positive engagement, no discomfort).
- **Platform utility** might prefer the political post (higher total engagement — even angry engagement counts).
- **Society utility** depends on the user's current feed diversity.

### 4.2 Label disagreement rates

We measured how often each pair of stakeholders disagrees on the preference ordering — i.e., for what fraction of content pairs does one stakeholder prefer $c_1$ while the other prefers $c_2$?

| Stakeholder pair | Agreement | Disagreement |
|-----------------|-----------|--------------|
| Platform vs Society | 65% | **35%** |
| User vs Platform | 77% | 23% |
| User vs Society | ~88% | ~12% |

Source: `docs/results.md:2056`.

The highest disagreement is between Platform and Society (35%). This makes intuitive sense: the platform values all engagement including hostile cross-partisan interactions, while society penalizes the polarization those interactions represent. The lowest disagreement is between User and Society (12%), reflecting that users generally don't want to see content that makes them block or report — and neither does society.

### 4.3 Why preference pairs are THE lever

This is the central finding of the entire F4 project, established through 87 experiments:

**When all stakeholders train on identical preference pairs, they learn identical weight vectors regardless of loss function.**

**When stakeholders train on different preference pairs (from different utility functions), they learn different weight vectors even with standard Bradley-Terry loss.**

The formal argument is as follows. The Bradley-Terry loss for a single preference pair is:

$$\mathcal{L} = -\log \sigma\!\big(\mathbf{w}^\top \mathbf{P}^\text{pref} - \mathbf{w}^\top \mathbf{P}^\text{rej}\big)$$

The gradient with respect to $\mathbf{w}$ is:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = -\sigma(-\Delta R) \cdot \big(\mathbf{P}^\text{pref} - \mathbf{P}^\text{rej}\big)$$

where $\Delta R = \mathbf{w}^\top(\mathbf{P}^\text{pref} - \mathbf{P}^\text{rej})$.

The key observation: the gradient direction is determined by $(\mathbf{P}^\text{pref} - \mathbf{P}^\text{rej})$. When $(\mathbf{P}^\text{pref}, \mathbf{P}^\text{rej})$ are identical across stakeholders — because the preference labels are the same — the gradient direction is identical. The optimizer converges to the same $\mathbf{w}$ regardless of:

- Additional penalty terms (constrained BT)
- Margin requirements (margin BT)
- Calibration objectives (calibrated BT)

These modifications affect the gradient magnitude and add secondary gradient signals, but the dominant signal comes from the ranking term over all preference pairs. With 50,000 identical ranking pairs, penalty terms operating on derived signals swim upstream against a much stronger gradient.

This is a property specific to pairwise ranking losses. In pointwise supervised learning (e.g., regression with MSE), changing the loss function directly changes the optimal parameters. But BT only cares about *which item of a pair is preferred*, not by how much. When all stakeholders agree on every pair, no loss modification can introduce disagreement.

### 4.4 The implication

If you want different stakeholder models, change the labels. Different utility functions produce different preference orderings for the same content pairs, and those different orderings are sufficient to produce differentiated weight vectors with any standard loss function. The loss function is not the lever; the data pipeline is.

---

## 5. Loss Functions

This section provides the mathematical formulation of each loss function tested in the 87-experiment sweep. All implementations are in `enhancements/reward_modeling/alternative_losses.py`.

### 5.1 Standard Bradley-Terry (baseline)

The Bradley-Terry model assumes the probability of preferring item $A$ over item $B$ follows a logistic model:

$$P(A \succ B) = \sigma(R(A) - R(B)) = \frac{1}{1 + \exp(-(R(A) - R(B)))}$$

The corresponding negative log-likelihood loss is:

$$\mathcal{L}_\text{BT} = -\frac{1}{N} \sum_{i=1}^{N} \log \sigma\!\left(\mathbf{w}^\top \mathbf{P}_i^\text{pref} - \mathbf{w}^\top \mathbf{P}_i^\text{rej}\right)$$

**Scale invariance**: For any positive constant $c$, the weights $c \cdot \mathbf{w}$ produce the same preference ordering as $\mathbf{w}$. This means BT cannot recover absolute weight magnitudes — only the direction of $\mathbf{w}$ matters for ranking. This property is fundamental: it means the Pearson correlation between learned and ground-truth weights can be low (we observe 0.554) even when ranking performance is excellent (99.3% accuracy).

Implementation: `bradley_terry_loss()` in `alternative_losses.py:101-109`.

Training uses Adam optimizer with learning rate 0.01, L2 regularization of 0.001, batch size 64, and 150 epochs.

**Result with stakeholder-specific labels**: Platform-Society cosine similarity = **0.478** (best differentiation of any loss function).

### 5.2 Margin-BT

Margin-BT attempts to break scale invariance by requiring a minimum score gap between preferred and rejected items.

**Hard margin** (hinge loss):

$$\mathcal{L}_\text{margin} = \frac{1}{N} \sum_{i=1}^{N} \max\!\left(0,\; m - \big(\mathbf{w}^\top \mathbf{P}_i^\text{pref} - \mathbf{w}^\top \mathbf{P}_i^\text{rej}\big)\right)$$

**Smooth margin** (used in practice, better gradient properties):

$$\mathcal{L}_\text{smooth} = \frac{1}{N} \sum_{i=1}^{N} \tau \cdot \text{softplus}\!\left(\frac{m - \Delta R_i}{\tau}\right)$$

where $\text{softplus}(x) = \log(1 + e^x)$, $m$ is the margin, and $\tau$ is a temperature parameter. As $\tau \to 0$, the smooth version approaches the hard hinge loss. We use $\tau = 0.1$.

**Motivation**: By requiring $R(\text{pref}) - R(\text{rej}) \geq m$, the model must learn absolute score magnitudes, not just orderings. This was expected to break scale invariance and enable stakeholder-specific constraints to take effect.

**Hyperparameters tested**: $m \in \{0.05, 0.1, 0.5\}$.

Implementation: `margin_bt_loss_smooth()` in `alternative_losses.py:145-165`.

**Result**: Cosine similarity 0.541 (m=0.05) and 0.551 (m=0.5) — *worse* than standard BT (0.478). The margin constraint forces larger weight magnitudes but more similar weight directions, actually reducing differentiation.

### 5.3 Calibrated-BT

Calibrated-BT adds a secondary objective: predicted scores should match observed engagement rates.

$$\mathcal{L}_\text{cal} = \mathcal{L}_\text{BT} + \lambda \cdot \left[\text{MSE}\!\big(\sigma(\mathbf{w}^\top \mathbf{P}^\text{pref}),\; e^\text{pref}\big) + \text{MSE}\!\big(\sigma(\mathbf{w}^\top \mathbf{P}^\text{rej}),\; e^\text{rej}\big)\right]$$

where $e^\text{pref}$ and $e^\text{rej}$ are the ground-truth engagement rates (from the synthetic data), and $\sigma(\cdot)$ maps scores to the $[0, 1]$ probability space.

**Motivation**: Anchoring scores to absolute engagement rates should break scale invariance (the sigmoid maps the scale of $\mathbf{w}$ to the shape of the calibration curve) and create stakeholder-specific anchoring targets.

**Hyperparameters tested**: $\lambda \in \{0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0\}$.

Implementation: `calibrated_bt_loss()` in `alternative_losses.py:172-214`.

**Result**: Cosine similarity 0.512 at $\lambda = 0.05$, with a monotonic accuracy-calibration tradeoff. For the Platform stakeholder (which values all engagement uniformly), calibration pushes all scores toward high values, compressing the score distribution and degrading ranking accuracy:

| $\lambda$ | Platform accuracy |
|-----------|------------------|
| 0 (standard BT) | 91.9% |
| 0.05 | ~90% |
| 0.5 | 86.0% |

The calibration objective conflicts with ranking: trying to predict absolute engagement rates correctly interferes with the model's ability to distinguish which of two items is better.

### 5.4 Constrained-BT

Constrained-BT adds stakeholder-specific penalty terms to the base BT loss:

$$\mathcal{L}_\text{constrained} = \mathcal{L}_\text{BT} + \lambda_c \cdot C_k(\mathbf{w})$$

where $C_k$ depends on the stakeholder:

**User constraint** — penalize positive weights on negative actions:

$$C_\text{user}(\mathbf{w}) = \frac{1}{|\mathcal{N}|}\sum_{a \in \mathcal{N}} \max\!\big(0,\; w_a + \epsilon\big)$$

where $\mathcal{N} = \{\text{block, mute, report, not\_interested}\}$ and $\epsilon = 0.1$ is the maximum allowed weight on negative actions. If $w_\text{block} > -0.1$, the penalty activates.

**Society constraint** — encourage diverse weight distribution:

$$C_\text{society}(\mathbf{w}) = \max\!\big(0,\; \sigma_\text{target} - \text{std}(\mathbf{w}_\text{positive})\big)$$

where $\mathbf{w}_\text{positive}$ are the weights on positive actions (favorite, repost, `follow_author`, share, reply) and $\sigma_\text{target} = 0.3$. This penalizes uniform positive weights, encouraging the model to value some positive actions more than others (which could produce more diverse recommendations).

**Platform constraint**: none ($C_\text{platform} = 0$). The platform has no structural constraints beyond maximizing engagement.

**Hyperparameters**: $\lambda_c = 10.0$, $\sigma_\text{target} = 0.3$, $\epsilon = 0.1$.

Implementation: `constrained_bt_loss()` in `alternative_losses.py:221-277`.

**Result**: No meaningful improvement over standard BT when trained on identical preference labels. The constraint terms are overwhelmed by the ranking gradient from the BT loss, which operates on all 50,000 preference pairs.

### 5.5 Post-hoc reranking

Post-hoc reranking takes a completely different approach: train a single shared model, then apply stakeholder-specific adjustments at serving time. This sidesteps the BT scale invariance issue entirely, since differentiation happens after training.

**User adjustment**: subtract a penalty for predicted negative actions.

$$\text{score}_\text{user} = \mathbf{w}^\top \mathbf{P} - \alpha \sum_{a \in \mathcal{N}} P_a$$

**Society adjustment**: add a bonus for action diversity (measured by entropy).

$$\text{score}_\text{society} = \mathbf{w}^\top \mathbf{P} + \alpha \cdot \frac{H(\mathbf{P})}{\log 18}$$

where $H(\mathbf{P}) = -\sum_a P_a \log P_a$ is the entropy of the action distribution, normalized by $\log 18$ to lie in $[0, 1]$.

**Platform**: no adjustment (base scores only).

Implementation: `PostHocReranker` class in `alternative_losses.py:316-392`.

**Advantage**: No training required. Any base model works. **Disadvantage**: Adjustments are heuristic, not learned from data.

### 5.6 Summary of loss function results

All cosine similarities are Platform-Society (the most differentiated pair):

| Loss | P-S Cosine Sim | Best Accuracy | Parameters |
|------|---------------|---------------|------------|
| **Standard BT** | **0.478** | 91.9% | — |
| Margin-BT (m=0.05) | 0.541 | ~90% | $m$, $\tau$ |
| Margin-BT (m=0.5) | 0.551 | ~89% | $m$, $\tau$ |
| Calibrated-BT ($\lambda$=0.05) | 0.512 | ~90% | $\lambda$ |
| Constrained-BT | ~0.5 | ~91% | $\lambda_c$, $\sigma_\text{target}$, $\epsilon$ |

Standard BT with stakeholder-specific labels achieves the best differentiation. Every alternative loss produces higher (worse) cosine similarity.

---

## 6. Model Architectures

### 6.1 PhoenixRewardModel (single weight vector)

The simplest architecture: a single weight vector $\mathbf{w} \in \mathbb{R}^{18}$ shared across all users.

$$R(u, c) = \mathbf{w}^\top \mathbf{P}(u, c)$$

Implementation: `PhoenixRewardModel` in `enhancements/reward_modeling/reward_model.py:35-184`. The default weights are hand-tuned: `favorite=+1.0, follow_author=+1.2, block_author=-1.5, report=-2.0`, etc.

**Limitation**: One-size-fits-all. Cannot capture that sports fans and political users value different actions differently.

### 6.2 ContextualRewardModel (per-archetype weights)

Extends the single-vector model to per-archetype weights: a matrix $\mathbf{W} \in \mathbb{R}^{K \times 18}$ where $K$ is the number of archetypes (6 in our setup).

$$R(u, c) = \mathbf{W}[\text{archetype}(u)]^\top \mathbf{P}(u, c)$$

Implementation: `ContextualRewardModel` in `reward_model.py:186-408`. Uses `rewards = jnp.einsum("bca,ba->bc", probs, user_weights)` where `user_weights` are selected per-user from the weight matrix.

This enables different reward functions per user group — sports fans can have different weights than power users. Total parameters: 6 × 18 = 108.

### 6.3 Two-stage model (K-means + per-cluster BT)

The two-stage approach won the Phase 2 comparison across 8 approaches. It separates clustering from reward learning:

**Stage 1: Cluster users**. Run K-means ($K = 6$) on user interaction features (per-topic engagement vectors). Users are grouped by behavioral similarity, not demographics. Configuration: `n_init=10` (10 random seeds, pick best).

**Stage 2: Train per-cluster BT models**. For each cluster $k$, collect the preference pairs belonging to users in that cluster. Train an independent BT model on each cluster's data, producing cluster-specific weights $\mathbf{w}_k$.

**Inference**:
```
cluster_id = kmeans.predict(user_features)
R(u, c) = w[cluster_id]^T @ P(u, c)
```

Implementation: `TwoStageConfig` and training functions in `enhancements/reward_modeling/two_stage.py`.

**Why it works**: Feature engineering maps directly to ground truth structure. When user features include per-topic engagement rates (a 6-dimensional vector per user), K-means on this feature space recovers the 6 archetypes perfectly — 100% cluster purity. The key insight from Phase 2: the right features make clustering trivial, eliminating the need for complex end-to-end architectures.

**Result**: 99.3% accuracy, 100% cluster purity (with topic-aware features), mean weight correlation 0.554.

### 6.4 Pluralistic mixture model

A more principled approach that jointly learns clustering and reward weights as a mixture model.

$$R(u, c) = \sum_{k=1}^{K} \pi_k(u) \cdot \big(\mathbf{w}_k^\top \mathbf{P}(u, c)\big)$$

where $\pi_k(u) = \text{softmax}(\text{MLP}(\text{embedding}(u)))$ are user-specific mixture weights over $K$ value systems.

The MLP maps user embeddings to mixture weights: input → 64-dimensional hidden layer → ReLU → $K$-dimensional output → softmax.

Implementation: `PluralConfig` and training functions in `enhancements/reward_modeling/pluralistic.py`.

Three training approaches were tested:

**EM**: Alternate between computing responsibilities (soft cluster assignments) and updating weights given responsibilities. Result: collapsed — all value systems became identical.

**Auxiliary losses**: End-to-end training with additional loss terms to encourage diversity:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{BT} + \lambda_\text{div} \cdot \mathcal{L}_\text{diversity} + \lambda_\text{ent} \cdot \mathcal{L}_\text{entropy}$$

where $\mathcal{L}_\text{diversity}$ penalizes high cosine similarity between value system weight vectors and $\mathcal{L}_\text{entropy}$ encourages peaky (low-entropy) mixture assignments. Result: wrong systems — high diversity but low correlation with ground truth.

**Hybrid**: EM structure with diversity regularization in the M-step. Result: best unsupervised approach (correlation 0.510), but still worse than two-stage.

**Conclusion**: The two-stage approach wins because it decouples two fundamentally different problems — "which users are similar?" (clustering) and "what do similar users value?" (weight learning). Coupling them in a single optimization makes both harder.

---

## 7. Verification Framework

A 4-level verification hierarchy tests progressively deeper properties of the trained model. Each level catches failure modes invisible to simpler metrics.

### 7.1 Level 1: Embedding probes

**Question**: Do users of the same archetype cluster together in the model's learned representation?

**Method**: Extract user embeddings from the trained model. Compute silhouette score — a measure of how well-separated clusters are, ranging from -1 (worst) to +1 (best).

For each point $i$ in cluster $C$:
- $a(i)$ = mean distance to other points in the same cluster
- $b(i)$ = minimum mean distance to any other cluster
- $s(i) = (b(i) - a(i)) / \max(a(i), b(i))$

Overall silhouette = mean of $s(i)$ across all points. Threshold: $\geq 0.2$.

Implementation: `enhancements/verification/embedding_probes.py`.

**Result**: `user_silhouette` = 0.369, `topic_silhouette` = 1.000. Users cluster reasonably well; topic embeddings cluster perfectly.

### 7.2 Level 2: Behavioral tests

**Question**: Does the model predict the correct engagement rates for each (archetype, topic) pair?

**Method**: For each (archetype, topic) combination in the ground truth, compare the model's predicted action probabilities against the specified ground truth values. For each action with expected probability > 0.05, check that the absolute error is within tolerance (0.15).

**Result**: 100% behavioral accuracy (after fixing a binary-vs-soft label issue in Phase 7).

Implementation: `enhancements/verification/behavioral_tests.py`.

### 7.3 Level 3: Action differentiation tests

**Question**: Does the model capture the behavioral differences between archetypes?

**Method**: Six specific behavioral checks:

1. `LURKER` repost ratio < 5% (ground truth: 1%)
2. `POWER_USER` repost ratio > 30% (ground truth: 35%)
3. `POWER_USER` repost > 2× `LURKER` repost
4. `LURKER` reply < 5% (ground truth: 0%)
5. `POLITICAL_L` blocks `POLITICS_R` content at > 20% (ground truth: 25%)
6. `POLITICAL_R` blocks `POLITICS_L` content at > 20%

Implementation: `enhancements/verification/action_tests.py`.

**Result**: 6/6 tests pass.

### 7.4 Level 4: Counterfactual tests

**Question**: Does the model understand *causal* relationships, not just correlations?

This is the most important level. Levels 1-3 test correlational properties — "do the right users have the right engagement patterns?" Level 4 tests whether the model understands *why* — if we intervene on the input, does the output change in the expected way?

**Block effect test**: For 50 users, compute the model's reward score for an author's posts. Then inject a block action (`block_author = 1.0`) into the user's engagement history with that author. The model should produce a lower reward score after the block. Pass criterion: score decreases for $\geq$ 50% of tests.

**Archetype flip test**: Take a user of one archetype (e.g., sports fan) and replace their engagement history with a different archetype's history (e.g., tech enthusiast). The model's predictions should change to match the new archetype's patterns. Pass criterion: flip rate $\geq$ 50%.

Implementation: `enhancements/verification/counterfactual_tests.py`.

**Why this matters**: A model can achieve 99% ranking accuracy by memorizing user embeddings (which encode archetype identity) without ever reading the content or history. Such a model would pass Levels 1-3 perfectly but fail Level 4 — swapping a user's history wouldn't change predictions because the model ignores history entirely. This is exactly what happened in Phase 7's initial training, where the model achieved high accuracy but only 4% archetype flip rate.

**Result**: `block_effect` = 78%, `archetype_flip` = 86% (after three successive fixes: soft labels, block contrastive training, and history-topic contrastive loss).

---

## 8. Pareto Frontier Analysis

### 8.1 The tradeoff problem

No single recommendation policy simultaneously maximizes user utility, platform utility, and society utility. Improving diversity (society's goal) typically reduces engagement (platform's goal). Reducing polarizing content (society's goal) may reduce engagement from politically active users (harming user utility for those users). The question is: what are the achievable tradeoffs?

### 8.2 The diversity knob

The serving-time mechanism for navigating the tradeoff is a greedy diversity-aware selection algorithm. Given a set of candidate content items, the algorithm selects items one at a time:

$$\text{score}(c) = (1 - \alpha) \cdot \text{engagement}(c) + \alpha \cdot \text{diversity\_bonus}(c)$$

where:
- $\text{engagement}(c)$ is the reward score from either the hardcoded or learned scorer
- $\text{diversity\_bonus}(c) = 1 / (\text{topic\_count}[\text{topic}(c)] + 1)$ — topics already well-represented in the selected set get penalized
- $\alpha \in [0, 1]$ is the diversity weight

At $\alpha = 0$: pure engagement optimization (show whatever scores highest).
At $\alpha = 1$: pure diversity optimization (maximize topic variety).

Implementation: `compute_pareto_frontier()` in `enhancements/reward_modeling/stakeholder_utilities.py:365-486`.

### 8.3 Hardcoded vs learned scorers

The **hardcoded scorer** uses a hand-picked 3-coefficient formula for engagement:

$$\text{engagement}_\text{hardcoded}(c) = P_\text{favorite}(c) + 0.8 \cdot P_\text{repost}(c) + 0.5 \cdot P_\text{follow}(c)$$

This uses 3 of 18 available actions with arbitrary coefficients. No experiment determined that reposts should be weighted 0.8 or follows 0.5.

The **learned scorer** replaces this with the full 18-dimensional BT weight vector:

$$\text{engagement}_\text{learned}(c) = \mathbf{w}^\top \mathbf{P}(c)$$

where $\mathbf{w}$ comes from the BT training on stakeholder-specific labels. Three variants: user-trained, platform-trained, and society-trained weight vectors.

Implementation: `compute_learned_frontier()` in `scripts/compare_pareto_frontiers.py:151-225`.

### 8.4 Sweeping the frontier

For each value of $\alpha \in \{0.0, 0.1, 0.2, \ldots, 1.0\}$, we run the greedy selection algorithm and compute all three utilities. This produces 11 points in the 3D utility space. A point $p$ is Pareto-optimal if no other point dominates it — i.e., no other point is at least as good in all three objectives and strictly better in at least one.

### 8.5 Key results

| Scorer | Max `user_utility` | At $\alpha$ | Max `society_utility` | At $\alpha$ |
|--------|-----------------|-------------|--------------------|----|
| Hardcoded | 1.087 | 0.3 | 0.500 | 0.9 |
| Learned (user) | **1.149** | **0.1** | 0.500 | 1.0 |
| Learned (platform) | 1.024 | 0.8 | 0.500 | 1.0 |
| Learned (society) | 1.059 | 0.6 | 0.500 | 1.0 |

All scorers achieve the same maximum society utility (0.500) at near-full diversity ($\alpha \geq 0.9$), because at high $\alpha$ the diversity-aware selection overwhelms scorer differences — every scorer converges to the same maximally diverse content set. The differentiating factor is user utility.

The user-trained learned scorer achieves 5.7% higher user utility than the hardcoded scorer (1.149 vs 1.087), and achieves it at a lower diversity cost ($\alpha = 0.1$ vs $\alpha = 0.3$). This is the primary quantitative evidence that learned scorers improve on handcrafted ones.

The platform-trained scorer produces concentrated scores (narrow range across topics), making the diversity knob ineffective until very high $\alpha$ values. Its frontier is essentially flat for $\alpha \in [0, 0.6]$, and its peak user utility (1.024 at $\alpha = 0.8$) requires heavy diversity weighting to break the concentration.

The society-trained scorer suffers from double-penalization: it already down-ranks divisive content, and then the diversity knob penalizes topic concentration. The two mechanisms partially duplicate each other, causing its frontier to sit below the hardcoded baseline for user utility (1.059 vs 1.087).

Source: `results/pareto_comparison.json`.

---

## 9. Key Experimental Results

### 9.1 Phase-by-phase summary

| Phase | Objective | Key metric | Result |
|-------|-----------|------------|--------|
| 1 | BT preference learning | Val accuracy | 99.3% |
| 2 | Pluralistic reward models | Cluster purity (topic-aware) | 100% |
| 3 | Causal verification | Block/follow pass rate | 100% |
| 4 | Multi-stakeholder differentiation | Min cosine sim (P-S) | 0.478 |
| 6 | MovieLens validation | NDCG@3 | 0.4112 (+59%) |
| 7 | Synthetic Twitter verification | All test suites | Pass |

### 9.2 The 87-experiment result

87 experiments across 4 loss functions (Standard BT, Margin-BT, Calibrated-BT, Constrained-BT), each trained for 3 stakeholders (user, platform, society), with multiple hyperparameter settings. Every experiment produces a JSON file with weights, accuracy, cosine similarities, and topic scores, stored in `results/loss_experiments/`.

The definitive finding:

| Stakeholder pair | Cosine similarity | Label disagreement |
|-----------------|-------------------|-------------------|
| Platform-Society | 0.478 | 35% |
| User-Platform | 0.830 | 23% |
| User-Society | 0.884 | ~12% |

Higher label disagreement maps directly to lower cosine similarity (more differentiation). This monotonic relationship is the empirical foundation of the core insight.

### 9.3 External validation

**MovieLens (Phase 6)**: Real movie rating data. The reward model architecture (learned embeddings + transformer) achieves NDCG@3 of 0.4112, a 59% improvement over an untrained baseline. Key discovery: the transformer and embeddings exhibit a 107.5% synergy effect — neither component works alone; all improvement comes from their interaction.

**Synthetic Twitter (Phase 7)**: Full pipeline test with 648-parameter ground truth. All 5 test suites pass after three successive fixes (soft labels, block contrastive training, history-topic contrastive loss). Final metrics: `behavioral_accuracy` 100%, `action_tests` 6/6, `block_effect` 78%, `archetype_flip` 86%.

---

## 10. Discussion and Limitations

### 10.1 The synthetic data caveat

All F4 results except MovieLens are on synthetic data designed to be learnable. Ground truth archetypes have clean separation; engagement patterns are noise-free before we add noise ourselves. Real engagement data has overlapping user behaviors, noisy signals, temporal drift, and ambiguous preferences. The reported numbers (99.3% accuracy, 100% purity, 0.478 cosine sim) are upper bounds on real-world performance.

### 10.2 Utility function sensitivity

The ~14 free parameters in `UtilityWeights` are hand-specified. We do not know how sensitive the results are to these choices. If changing society's negativity penalty from 4.0 to 2.0 dramatically shifts the Pareto frontier, that's a serious practical limitation — practitioners need to know how precisely they must specify their utility functions. This is an open research question.

### 10.3 Weight recovery

BT training achieves 0.554 Pearson correlation between learned and ground-truth weights. This was initially declared a fundamental limitation, but may be an artifact of the metric: BT's scale invariance means Pearson correlation is not the right measure. Rank-order correlation (Kendall's $\tau$, Spearman $\rho$) on the weight vector has never been computed. If rank correlation is high, the 0.554 Pearson is purely a scale artifact and the model is learning the correct structure.

### 10.4 History-level causality

The model achieves 86% archetype flip rate — meaning 14% of the time, swapping a user's history doesn't change the model's predictions. The root cause: user embeddings are so informative that the transformer sometimes bypasses history content. Embedding dropout during training (zeroing out user embeddings 50% of the time) is the most promising fix, forcing the transformer to rely on history processing.

### 10.5 No production path

F4 produces 18-dimensional weight vectors stored in JSON files. There is no system that combines them with F2's optimized inference pipeline (JIT, KV-cache, quantization) into a serving-time recommendation pipeline. No A/B testing framework, no real-data ingestion, no monitoring. The entire F4 body of work is a research prototype with rigorous internal validation but zero production readiness.

### 10.6 The lamppost

The 87 experiments exploring loss function variants were valuable — they produced a rigorous proof that no loss function can overcome identical training signals. But the reason we explored loss functions was a framing error: we assumed the loss was the lever because that's what we knew how to modify. The actual lever was in the data pipeline (preference pair generation), which we weren't examining. Phase 1's cross-archetype cosine similarity of 0.997 was the clue — but we read it as "good generalization" rather than "identical training signal."

---

## Appendix: Code References

| Component | File | Key lines |
|-----------|------|-----------|
| Action names and indices | `enhancements/reward_modeling/weights.py` | 32-56 |
| Utility weights | `enhancements/reward_modeling/stakeholder_utilities.py` | 52-102 |
| User utility | same | 136-182 |
| Platform utility | same | 185-237 |
| Society utility | same | 240-310 |
| Pareto frontier | same | 365-486 |
| Hardcoded scoring | same | 399-402 |
| Diversity knob | same | 415-430 |
| Standard BT loss | `enhancements/reward_modeling/alternative_losses.py` | 101-109 |
| Margin-BT loss | same | 145-165 |
| Calibrated-BT loss | same | 172-214 |
| Constrained-BT loss | same | 221-277 |
| Post-hoc reranker | same | 316-392 |
| Training loop | same | 453-564 |
| BT training (original) | `enhancements/reward_modeling/training.py` | 39-78 |
| Ground truth rules | `enhancements/data/ground_truth.py` | 142-338 |
| Archetype enum | same | 14-21 |
| Topic enum | same | 24-31 |
| PhoenixRewardModel | `enhancements/reward_modeling/reward_model.py` | 35-184 |
| ContextualRewardModel | same | 186-408 |
| Two-stage model | `enhancements/reward_modeling/two_stage.py` | full file |
| Pluralistic model | `enhancements/reward_modeling/pluralistic.py` | full file |
| Embedding probes | `enhancements/verification/embedding_probes.py` | full file |
| Behavioral tests | `enhancements/verification/behavioral_tests.py` | full file |
| Action tests | `enhancements/verification/action_tests.py` | full file |
| Counterfactual tests | `enhancements/verification/counterfactual_tests.py` | full file |
| Pareto comparison script | `scripts/compare_pareto_frontiers.py` | 151-225 |
| BT results (user) | `results/loss_experiments/bradley_terry_user.json` | — |
| BT results (platform) | `results/loss_experiments/bradley_terry_platform.json` | — |
| BT results (society) | `results/loss_experiments/bradley_terry_society.json` | — |
| Pareto frontier data | `results/pareto_comparison.json` | — |
