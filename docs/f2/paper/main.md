# Introduction {#sec:intro}

Recommendation systems serve multiple stakeholders with conflicting objectives. A social media platform balances user engagement (did the user enjoy this content?), platform retention (will the user return tomorrow?), and societal welfare (did this content increase polarization?). Multi-stakeholder optimization frameworks formalize these tradeoffs as Pareto frontiers over stakeholder utilities [@burke2017multistakeholder; @multifr2022; @eval2025multistakeholder], but three questions remain largely unexamined empirically:

1.  **Identifiability.** What properties of stakeholder preferences can a Bradley-Terry (BT) reward model actually recover from pairwise comparison data?

2.  **Partial observation.** When a stakeholder is unobservable---as societal welfare typically is---how much does the Pareto frontier degrade, and how cheaply can the degradation be mitigated?

3.  **Specification sensitivity.** How precisely must stakeholder utility functions be specified before the frontier shifts meaningfully?

These questions are newly urgent. In January 2026, X (formerly Twitter) open-sourced a complete rewrite of its recommendation algorithm [@xalgorithm2026], replacing the hand-engineered Phoenix heavy ranker [@xalgorithm2023] with a Grok-based transformer that "eliminated every single hand-engineered feature and most heuristics." The 2023 release exposed *explicit* utility weights---per-action multipliers for favorites, reposts, blocks, and 15 other engagement signals. The 2026 release exposes *structure without weights*: the prediction targets (action types) are visible, but the actual utility weighting is implicit in 8B learned parameters. This transition ([1]) from explicit to implicit specification raises a concrete audit question: can regulators assess multi-stakeholder welfare from released code when the weights are opaque? [@shaped2023secrets]

We cannot study these questions directly on the production Grok model: the trained parameters and user data are proprietary, and our methodology requires known ground-truth utilities to measure regret and recovery. Instead, we build a controlled synthetic benchmark on X's 18-action engagement space, using BT preference learning to train separate reward models for three stakeholders (user, platform, society). The benchmark shares the production action space but uses synthetic utilities with known ground truth---the only setting where questions about *how much* is lost can be answered precisely. Our principal findings:

- **Labels, not loss, determine differentiation.** Across 87 training experiments (79 converged) with four BT loss variants, stakeholder weight vectors converge to cosine similarity $>0.92$ when trained on identical preference labels, regardless of loss function. Differentiation arises entirely from stakeholder-specific training data. The negativity-aversion parameter $\alpha$ is recoverable from learned weights with Spearman $\rho = 1.0$, robust to $\leq$`<!-- -->`{=html}20% label noise and $\geq$`<!-- -->`{=html}250 preference pairs ([4]).

- **Hiding society costs 10$\times$ more than hiding user.** Leave-One-Stakeholder-Out (LOSO) analysis shows that removing society from the optimization produces average regret of 1.08, versus 0.11 for user---the stakeholder hardest to observe is the most costly to miss. Even 25 preference pairs from the hidden stakeholder reduce regret by 42% (20 seeds, $p < 0.01$), with diminishing returns beyond $\sim$`<!-- -->`{=html}200 pairs ([5]).

- **The Pareto frontier is robust to individual weight perturbation but not to simultaneous misspecification.** Perturbing any single utility weight by up to $3\times$ preserves all Pareto-optimal operating points (rank stability $= 1.0$). But simultaneous perturbation of multiple weights at matched magnitude ($\sigma = 0.3$) causes $2\times$ more frontier shift than changing the $\alpha$ ratio alone. With misspecified weights, additional training data *amplifies* the error after $N > 100$ pairs---a Goodhart effect [@skalse2024goodhart] ([6]).

All results are validated under nonlinear utility families (prospect-theoretic concave and threshold utilities) and externally on MovieLens-100K (+59% NDCG) and a 648-parameter synthetic Twitter environment. Code and data are available at <https://github.com/kar-ganap/x-algorithm-enhancements>.

<figure id="fig:transition" data-latex-placement="t">

<figcaption>X’s algorithm transition. The 2023 Phoenix release exposed explicit per-action weights (left). The 2026 Grok release exposes prediction targets but not weights (right). Our study tests how much of the multi-stakeholder frontier is recoverable from structure alone.</figcaption>
</figure>

The remainder of the paper is organized as follows. [2] reviews multi-stakeholder recommendation, BT preference learning, and utility sensitivity analysis. [3] describes the synthetic benchmark and evaluation framework. [\[sec:identifiability,sec:partial,sec:sensitivity\]][7] present results on identifiability, partial observation, and sensitivity. [7] tests robustness under nonlinear utilities. [8] provides external validation. [9] synthesizes practical guidance and discusses limitations.

# Background and Related Work {#sec:background}

## Multi-Stakeholder Recommender Systems

Burke and Abdollahpouri [@burke2017multistakeholder] introduced a utility-based framework for representing stakeholder values in recommendation, distinguishing consumers, providers, and system operators as distinct principals. Subsequent work formalized multi-stakeholder optimization using Pareto frontiers: Multi-FR [@multifr2022] applies multiple gradient descent to generate Pareto-optimal solutions balancing accuracy and fairness across stakeholders, while recent surveys [@eval2025multistakeholder] propose a four-stage evaluation methodology (stakeholder identification, value articulation, metric selection, aggregate evaluation). Lasser et al. [@lasser2025designing] argue for designing recommendation algorithms through the lens of healthy civic discourse, foregrounding societal welfare as a first-class stakeholder objective. Stray et al. [@stray2024building] provide an interdisciplinary synthesis of how to embed human values in recommender systems.

A parallel line of work on *pluralistic alignment* addresses diverse preferences within reinforcement learning from human feedback (RLHF). Sorensen et al. [@sorensen2024roadmap] map the design space for handling preference heterogeneity, ranging from personalized reward models [@pad2025personalized] to federated aggregation. These approaches assume preferences can be captured in a single (possibly personalized) model. Our work takes a different stance: stakeholders have fundamentally different objective functions that must be traded off on a frontier, Agarwal et al. [@agarwal2024system2] formalize the utility-engagement distinction via temporal point processes. Our work asks the complementary question: what happens when one stakeholder's objective is absent entirely.

## Bradley-Terry Preference Learning

The Bradley-Terry (BT) model [@bradley1952rank] converts pairwise comparisons into scalar scores via $P(\text{prefer } a \succ b) = \sigma(r_a - r_b)$, where $r_i$ are learned reward values. BT preference learning underpins most modern RLHF pipelines, though the theoretical justification for this application has only recently been examined. Sun et al. [@sun2025rethinking] prove that BT models possess an *order consistency* property: any monotonic transformation of the true reward preserves the correct ranking, making BT sufficient for downstream optimization even when absolute reward values are unrecoverable. Extensions beyond the BT assumption [@beyondbt2025] address intransitive and context-dependent preferences, though these remain less common in deployed systems.

Our utility model is a special case of multi-attribute utility theory (MAUT) [@keeney1976decisions], which decomposes preferences into weighted sums of single-attribute utilities. The structural family $U = \text{pos} - \alpha \cdot \text{neg}$ parameterizes each stakeholder by a single negativity-aversion coefficient $\alpha$, varying from $\alpha = 0.3$ (platform, tolerant of negative signals) to $\alpha = 4.0$ (society, highly penalizing).

## Reward Misspecification and Goodhart's Law

Goodhart's Law---"when a measure becomes a target, it ceases to be a good measure"---manifests in RL as reward hacking: agents exploit misspecified proxy rewards to achieve high proxy scores while degrading true objectives [@skalse2024goodhart; @weng2024reward]. Casper et al. [@casper2023open] catalog open problems in RLHF, arguing that condensing diverse human preferences into a single reward model is "fundamentally misspecified." This concern applies *a fortiori* to multi-stakeholder systems, where each stakeholder's utility may be independently misspecified. Hadfield-Menell and Hadfield [@hadfield2019incomplete] draw an analogy to incomplete contracting: reward functions inevitably fail to specify all contingencies.

The Goodhart taxonomy distinguishes regressional, extremal, causal, and adversarial failure modes [@skalse2024goodhart]. Our Our sensitivity analysis ([6]) tests the *misweighting* variant---same goals, different priorities---and finds a novel interaction: additional training data amplifies misspecified utilities rather than correcting them.

## Sensitivity Analysis in MCDA

Sensitivity analysis in multi-criteria decision analysis (MCDA) assesses how robust a ranking is to changes in decision weights [@mcda2023sensitivity]. Classical MAUT sensitivity analysis [@farmer1987testing] computes "weight stability intervals"---the range within which each weight can vary without changing the top-ranked alternative [@weightstability2025]. A recent survey [@robust2025survey] catalogs robustness approaches for recommender systems. Iancu and Trichakis [@iancu2014pareto] extend this to multi-objective optimization, defining *robust Pareto optimality*: solutions that remain Pareto-optimal under worst-case parameter uncertainty.

We adapt this framework from single rankings to Pareto frontiers, introducing *rank stability* (the fraction of operating points that survive perturbation) as the primary metric rather than Hausdorff distance (which we show conflates scale change with shape change in our setting).

## Platform Transparency

The EU Digital Services Act (DSA), effective 2024, mandates that large platforms assess algorithmic risks to democratic values and provide transparency about recommender systems. X was fined 140M in 2025 for transparency violations and subsequently open-sourced its algorithm [@xalgorithm2026]. However, the release provides *structure without weights*: the model architecture and prediction targets are public, but learned parameters are not [@shaped2023secrets]. Whether this level of transparency is sufficient for meaningful multi-stakeholder audit is an open question that our sensitivity analysis directly addresses.

## X's Algorithm: Phoenix Architecture

X's recommendation system has undergone two public releases. The March 2023 release [@xalgorithm2023] exposed the Phoenix heavy ranker: a MaskNet-based neural network predicting engagement probabilities for 18 action types (favorites, reposts, replies, blocks, reports, etc.), combined via explicit per-action weights into a final relevance score. The January 2026 release [@xalgorithm2026] replaced this with a Grok-based transformer that learns all weighting implicitly from engagement sequences, eliminating hand-engineered features entirely.

Our synthetic benchmark is built on the 2023 Phoenix action space (18 actions, 5 positive, 4 negative, 9 neutral) and explicitly parameterizes the utility weights that the 2026 system learns implicitly. This allows controlled experimentation on sensitivity, partial observation, and identifiability questions that cannot be studied on the opaque production system.

# System, Data, and Methodology {#sec:system}

<figure id="fig:pipeline" data-latex-placement="t">

<figcaption>System pipeline. Each step flows strictly downward: data generation, utility specification, preference pair construction, BT training, and Pareto frontier evaluation. Braces indicate the scope of each research question.</figcaption>
</figure>

## Action Space

The Phoenix action space defines 18 user actions on content, inherited from X's production system [@xalgorithm2023]. We classify these into three groups ([1][10]):

- **Positive** (5): favorite, repost, follow_author, share, reply---signals of genuine user value.

- **Negative** (4): block_author, mute_author, report, not_interested---signals of harm or discomfort.

- **Neutral** (9): click, dwell, photo_expand, vqv (video quality view), profile_click, share_via_dm, share_via_copy_link, quote, quoted_click---engagement signals of ambiguous valence.

Each content item $c$ viewed by user $u$ produces an action probability vector $\mathbf{p}(u, c) \in [0,1]^{18}$, representing the predicted probability of each action. The full dataset is a tensor $\mathbf{P} \in \mathbb{R}^{N \times M \times 18}$ for $N$ users and $M$ content items.

::: {#tab:actions}
+----------------+---------------+-------------------------+------------------+
| **Action**     | **Class**     | **User wt.**            | **Platform wt.** |
+:===============+:==============+========================:+:=================+
| favorite       | Positive      | 1.0                     | 1.0              |
+----------------+---------------+-------------------------+------------------+
| repost         | Positive      | 0.8                     | 1.5              |
+----------------+---------------+-------------------------+------------------+
| follow_author  | Positive      | 1.2                     | 1.0              |
+----------------+---------------+-------------------------+------------------+
| share          | Positive      | 0.9                     | 1.3              |
+----------------+---------------+-------------------------+------------------+
| reply          | Positive      | 0.5                     | 1.2              |
+----------------+---------------+-------------------------+------------------+
| block_author   | Negative      | $-$`<!-- -->`{=html}2.0 | 0.1              |
+----------------+---------------+-------------------------+------------------+
| mute_author    | Negative      | $-$`<!-- -->`{=html}1.5 | 0.1              |
+----------------+---------------+-------------------------+------------------+
| report         | Negative      | $-$`<!-- -->`{=html}2.5 | 0.2              |
+----------------+---------------+-------------------------+------------------+
| not_interested | Negative      | $-$`<!-- -->`{=html}1.0 | 0.1              |
+----------------+---------------+-------------------------+------------------+
| click          | Neutral       | ---                     | 0.5              |
+----------------+---------------+-------------------------+------------------+
| dwell          | Neutral       | ---                     | 0.2              |
+----------------+---------------+-------------------------+------------------+
| *+ 7 others (photo_expand, vqv, profile_click, ...)*                        |
+-----------------------------------------------------------------------------+

: Action space: 18 Phoenix actions with positive/negative/neutral classification and default `UtilityWeights` values.
:::

## Stakeholder Utility Model

Each stakeholder $k$ has a utility function $U_k(c) = \mathbf{w}_k^\top \mathbf{p}(c)$, where $\mathbf{w}_k \in \mathbb{R}^{18}$ is a weight vector over actions. We study the structural family $$\begin{equation}
  U_k = \sum_{a \in \text{pos}} w_a \cdot p_a
      - \alpha_k \sum_{a \in \text{neg}} |w_a| \cdot p_a,
  \label{eq:utility}
\end{equation}$$ where $\alpha_k$ is the *negativity aversion* parameter. The three stakeholders differ primarily in $\alpha$:

- **User** ($\alpha = 1.0$): Equal weight on positive engagement and discomfort avoidance.

- **Platform** ($\alpha = 0.3$): Tolerant of negative signals; all engagement has value.

- **Society** ($\alpha = 4.0$): Heavily penalizes (motivated by evidence that algorithmic exposure affects polarization [@bail2018exposure]) block/report/mute signals as proxies for polarization and harm.

This family is a special case of additive MAUT [@keeney1976decisions], where the full 18-parameter weight vector is summarized by a single ratio. Whether $\alpha$ alone is sufficient or individual action weights matter independently is tested in [6].

## Synthetic Data Generation

We generate synthetic evaluation data with known ground truth: 600 users, 100 content items, 6 topics (technology, sports, politics-left, politics-right, entertainment, science), and 5 user archetypes (casual, power user, political-left, political-right, niche enthusiast). Each user--content pair produces an 18-dimensional action probability vector based on archetype--topic affinity rules (e.g., political-left users have elevated block rates on politics-right content).

For BT training, a separate content pool of 500 items is generated per seed using `generate_content_pool()`, which draws action probabilities from calibrated distributions. Preference pairs are constructed by sampling two content items, computing stakeholder utility, adding Gaussian noise ($\sigma = 0.05$), and labeling the higher-utility item as preferred.

## Bradley-Terry Training Pipeline

For each stakeholder, we train a BT reward model on 2,000 preference pairs for 50 epochs with Adam optimization (learning rate 0.01). The model learns an 18-dimensional weight vector $\mathbf{w} \in \mathbb{R}^{18}$ by minimizing the standard BT loss: $$\begin{equation}
  \mathcal{L}_{\text{BT}} = -\log \sigma\!\left(
    \mathbf{w}^\top \mathbf{p}_{\text{pref}}
    - \mathbf{w}^\top \mathbf{p}_{\text{rej}}
  \right).
  \label{eq:btloss}
\end{equation}$$

We test four loss variants: standard BT, Margin-BT (adds a fixed margin to the preference gap), Calibrated-BT (regresses on engagement magnitude), and Constrained-BT (adds a diversity regularizer). The central identifiability result is that these variants produce nearly identical weight vectors when trained on identical preference data.

## Evaluation Framework {#sec:evalframework}

The full data flow is shown in [2][11]. Stakeholder tradeoffs are measured via the *Pareto frontier*: we sweep a diversity weight $\delta \in \{0.0, 0.05, \ldots, 1.0\}$ (21 points) that controls the tradeoff between engagement-based content scoring and topic diversity. At each $\delta$, we select top-$K$ content per user using a greedy diversity-aware algorithm, then evaluate all three stakeholder utilities on the selected content. The resulting 21-point frontier in $(U_{\text{user}},
U_{\text{platform}}, U_{\text{society}})$ space defines the achievable tradeoffs.

**LOSO projection.** To simulate hiding a stakeholder, we extract the 2D Pareto front in the observed dimensions and measure hidden-dimension *regret*: $\text{regret} = \max_\pi U_{\text{hidden}}(\pi) - U_{\text{hidden}}(\pi_{\text{LOSO}})$, averaged across operating points.

**Rank stability.** For sensitivity analysis, we define rank stability as the fraction of Pareto-optimal operating points that remain optimal after utility weight perturbation. We show ([6]) that this is more informative than Hausdorff distance, which conflates utility-axis rescaling with frontier shape change.

All experiments use 5 seeds unless otherwise noted; Experiment 4 (partial sampling) uses 20 seeds for tighter confidence intervals.

# What Drives Stakeholder Differentiation? {#sec:identifiability}

## Labels vs. Loss

We train BT reward models for all three stakeholders under four loss variants: standard BT ([\[eq:btloss\]][12]), Margin-BT (adds a fixed margin $m$ to the preference gap), Calibrated-BT (regresses on engagement magnitude), and Constrained-BT (adds a diversity regularizer penalizing weight concentration). Each variant is swept across multiple hyperparameter settings, yielding 87 total training runs.

Of these, 79 converge (8 Constrained-BT society runs diverge due to numerical instability in the diversity constraint, since fixed). Among the converged experiments, the key finding is that *all four loss types produce near-identical weight vectors when trained on the same preference data*. Pairwise cosine similarity between weight vectors trained with different losses on the same stakeholder exceeds $0.92$ in every case ([2][13]). The differentiation that appears in the final models---cosine similarity of 0.478 between platform and society, 0.830 between user and platform, 0.884 between user and society---arises entirely from *different training labels*, not different loss functions.

This confirms Sun et al.'s [@sun2025rethinking] theoretical order-consistency result in the multi-stakeholder recommendation setting: BT training faithfully preserves the preference ordering in the data, and no loss variant can overcome identical training signals. The practical implication is that practitioners should invest in stakeholder-specific labeling, not loss engineering.

::: {#tab:cosine}
                         User--Platform   User--Society   Platform--Society
  --------------------- ---------------- --------------- -------------------
  Within any loss           $>0.99$          $>0.97$           $>0.92$
  Across stakeholders        0.830            0.884             0.478

  : Pairwise cosine similarity between stakeholder weight vectors. *Within-loss*: same labels, different losses ($>0.92$ everywhere). *Across-stakeholder*: different labels, same loss.
:::

## $\alpha$-Recovery

<figure id="fig:alpha" data-latex-placement="t">
<embed src="figures/fig3_alpha_recovery.pdf" style="width:65.0%" />
<figcaption><span class="math inline"><em>α</em></span>-recovery from BT weight vectors. 13 <span class="math inline"><em>α</em></span> values, 5 seeds each. Perfect rank ordering (Spearman <span class="math inline"> = 1.0</span>) with affine amplification (<span class="math inline"><em>α̂</em> ≈ −0.06 + 1.32<em>α</em></span>). Three real stakeholders annotated.</figcaption>
</figure>

Given that labels determine differentiation, we ask: what structural properties of the underlying utility function are recoverable from the learned weight vector? For the structural family ([\[eq:utility\]][14]) $U = \text{pos} - \alpha \cdot \text{neg}$, we test whether the negativity-aversion parameter $\alpha$ can be extracted from trained BT weights via the ratio $\hat\alpha = -\bar{w}_{\text{neg}} / \bar{w}_{\text{pos}}$.

We sweep $\alpha \in \{0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0,
3.0, 4.0, 5.0, 7.0, 8.0\}$ (13 values, 5 seeds each). The recovered $\hat\alpha$ values ([3][15]) preserve perfect monotonic ordering: Spearman $\rho = 1.0$ across all methods tested (ratio, sum-ratio, regression). The relationship is affine: $\hat\alpha \approx -0.06 + 1.32 \cdot \alpha_{\text{true}}$ ($R^2 = 0.997$), with systematic amplification (slope $> 1$) but perfect rank preservation. For the three real stakeholders, the recovered values are: user $\hat\alpha = 1.35$ (true 1.0), platform $\hat\alpha = 0.34$ (true 0.3), society $\hat\alpha = 5.59$ (true 4.0)---the ordering society $>$ user $>$ platform is correctly recovered.

## Stress Testing {#sec:stress}

We stress-test $\alpha$-recovery across four dimensions ([3][16]) (1,300 total training runs, 5 seeds $\times$ 13 $\alpha$ values per condition):

::: {#tab:stress}
  **Stress dimension**           **Breaking point**         **Practical threshold**
  ------------------------------ -------------------------- ---------------------------------------------
  Label noise (random flip)      $p_{\text{flip}} = 0.30$   $\leq$`<!-- -->`{=html}20% annotation error
  Sample size                    $n = 50$ pairs             $\geq$`<!-- -->`{=html}250 pairs sufficient
  BT temperature ($\beta$)       $\beta = 0.5$              $\beta \geq 0.5$
  Content correlation ($\rho$)   $\rho = 0.8$               $\rho \leq 0.6$

  : $\alpha$-recovery breaking points: the condition at which Spearman $\rho$ drops below 0.95.
:::

Key findings: (1) recovery is robust to 20% label noise and works with 250 pairs; (2) BT temperature is the strongest stressor---soft preferences ($\beta < 0.5$) destroy the signal; (3) moderate content correlation ($\rho = 0.3$) actually *improves* recovery by creating more informative preference pairs; (4) Spearman degrades much slower than Pearson---rank ordering survives conditions that destroy the linear fit.

## Disagreement-to-Differentiation Bound

To predict stakeholder differentiation *before* training, we measure pairwise label disagreement rates on 50,000 preference pairs. The relationship between disagreement rate $d$ and cosine similarity is well fit by a two-variable model incorporating the mean margin $m$ (average utility gap on disagreed pairs): $$\begin{equation}
  \cos(\mathbf{w}_i, \mathbf{w}_j) \approx 1.098 - 1.127\,d - 0.088\,m
  \qquad (R^2 = 0.977).
  \label{eq:disagreement}
\end{equation}$$ Disagreement rate alone achieves $R^2 = 0.90$; adding margin resolves reference-dependence (the platform-fixed and user-fixed sweeps collapse onto a single curve). Practical thresholds: $\geq$`<!-- -->`{=html}10% disagreement for $\cos < 0.95$; $\geq$`<!-- -->`{=html}19% for $\cos < 0.80$; $\geq$`<!-- -->`{=html}37% for $\cos < 0.50$.

## LLM-as-Annotator Compatibility

We test whether an LLM can substitute for analytic stakeholder definitions. Claude Haiku, given natural-language stakeholder descriptions (e.g., "you are a safety-first content moderator") and content as engagement/negativity scores, generates preference labels across 15 sweep points $\times$ 200 pairs. The LLM's implicit utility preserves the ranking structure: Spearman $\rho = 0.929$ between LLM-based and analytic disagreement rates. LLM confidence is too compressed (range \[1.40, 1.66\]) to serve as a precise margin proxy, but rank-ordering---the practically relevant quantity---is preserved. Details in [18].

# How Much Stakeholder Data Is Needed? {#sec:partial}

## LOSO Geometry and Training

For each of the three stakeholders, we hide it and compute the 2-stakeholder Pareto frontier by projecting the full 3D frontier onto the observed dimensions. Two variants: *geometric* (uses the hardcoded scorer, hides utility at evaluation) and *training-based* (trains BT models only on the observed stakeholders' preference pairs, uses learned weight vectors as scorers).

::: {#tab:loso}
  **Hidden**        **Geom. regret**       **Training regret**   **Dom. frac.**   **$\alpha$**
  ------------ -------------------------- --------------------- ---------------- --------------
  Society       $\mathbf{1.08 \pm 0.04}$     $1.10 \pm 0.05$          0.0%            4.0
  Platform          $0.37 \pm 0.06$          $0.31 \pm 0.01$        0--8.5%           0.3
  User              $0.11 \pm 0.01$          $0.14 \pm 0.03$       6.4--14.4%         1.0

  : LOSO degradation: average regret on the hidden dimension (5 seeds, 21 diversity weights).
:::

Society is 10$\times$ more costly to hide than user ([4][17]). The degradation ranking matches the pairwise correlation structure: society has the lowest cosine similarity (0.478) with the observed pair. Geometric and training-based LOSO produce consistent rankings, validating the geometric analysis. The 2D Pareto frontiers are rarely dominated in 3D (0% for society)---The regret metric follows the minimax criterion of Savage [@savage1951theory]: the cost of partial observation is entirely in the hidden dimension.

## Proxy Methods

Can we recover the hidden stakeholder's utility using proxies constructed from the observed stakeholders' weight vectors? We test six methods, focusing on hiding society (the highest-stakes case) ([5][19]):

::: {#tab:proxy}
+------------------------------------------------+-------------------+-----------------------+
| **Proxy**                                      | **Recovery**      | **Type**              |
+:===============================================+==================:+:======================+
| $\alpha$-interpolation (oracle $\alpha = 4.0$) | 1.000$^\dagger$   | Structural, oracle    |
+------------------------------------------------+-------------------+-----------------------+
| $\alpha$-interpolation (blind $\alpha = 2.2$)  | 0.738             | Structural, heuristic |
+------------------------------------------------+-------------------+-----------------------+
| Diversity knob ($\delta = 0.7$)                | **0.699**         | Practical, no oracle  |
+------------------------------------------------+-------------------+-----------------------+
| Oracle linear (LS proxy)                       | 0.566             | Oracle, ceiling       |
+------------------------------------------------+-------------------+-----------------------+
| Diversity knob ($\delta = 0.5$)                | 0.415             | Practical, no oracle  |
+------------------------------------------------+-------------------+-----------------------+
| Structural synthesis                           | 0.153             | Structural, oracle    |
+------------------------------------------------+-------------------+-----------------------+
| $^\dagger$At 100-item pool; 0.738 at 500 items.                                            |
+--------------------------------------------------------------------------------------------+

: Proxy recovery rates for hidden society utility (5 seeds, 500-item evaluation pool).
:::

The diversity knob at $\delta = 0.7$ is the best practical method (70% recovery, no oracle knowledge required). Weight-space $\alpha$-interpolation outperforms the oracle linear proxy (0.738 vs 0.566) because it preserves action-level BT structure that least-squares regression dilutes.

## Data Budget {#sec:databudget}

If a platform can collect $N$ society preference pairs (rather than the full 2,000), how much regret is recovered? We sweep $N \in
\{0, 25, 50, 100, 200, 500, 1000, 2000\}$ across 20 seeds, measuring average regret across all 21 diversity-weight operating points on the frontier scored by the trained society model.

Even 25 preference pairs reduce regret from 1.11 (LOSO baseline) to 0.64---a 42% improvement ([4][20]). The marginal value of additional pairs diminishes rapidly: the regret plateau at $\sim$`<!-- -->`{=html}0.55 is statistically indistinguishable across $N = 200$--$2{,}000$ (overlapping 95% bootstrap confidence intervals). This is a Value of Information (VoI) result in the sense of Howard [@howard1966information]: the first few bits of stakeholder information are the most valuable.

<figure id="fig:exp4" data-latex-placement="t">
<embed src="figures/fig4_partial_sampling.pdf" />
<figcaption>Partial observation sampling (20 seeds, 95% bootstrap CI). <strong>(a)</strong> Frontier shape evolves from chaotic (LOSO, gray) to near-optimal (n=500+, dark) as society data increases. CI bands separate cleanly in the <span class="math inline"><em>δ</em> = 0.3</span>–<span class="math inline">0.7</span> range. <strong>(b)</strong> Average regret drops sharply at <span class="math inline"><em>N</em> = 25</span> (<span class="math inline">−42%</span>) and plateaus at <span class="math inline"><em>N</em> ≈ 200</span>.</figcaption>
</figure>

## Degradation Predictability

Can we predict LOSO degradation from observable correlation features *without* running the experiment? We sweep $\alpha_{\text{hidden}} \in [0.1, 10.0]$ (13 synthetic hidden stakeholders, 5 seeds each) and fit predictive models in log space.

The oracle feature ([5][21])---standard deviation of the residual from regressing structural $u_{\text{hidden}}$ on *actual* $u_{\text{user}}$ and $u_{\text{platform}}$---achieves near-perfect prediction: $\text{regret} \approx 35 \times
\text{std}(u_\perp)^{1.8}$ ($R^2_{\text{log}} = 0.954$, Spearman $= 0.962$). The *proxy* feature (weight-space cosine similarity) achieves $R^2_{\text{log}} = 0.72$, Spearman $= 0.69$---adequate for ranking but not magnitudes.

A key theoretical observation: within the structural utility family ($U = \text{pos} - \alpha \cdot \text{neg}$), regressing $u_{\text{hidden}}$ on structural $u_{\text{user}}$ and $u_{\text{platform}}$ yields $R^2 = 1.0$ for *every* $\alpha$. This is inevitable---three linear functions of two variables span a 2D space. The theoretical bound predicts zero regret, yet actual regret ranges from 0.04 to 4.7. The gap arises from the discrepancy between structural utilities and actual utility functions (`compute_user_utility`, etc.), which include engagement weighting and retention proxies beyond the simple pos/neg model. Regressing structural $u_{\text{hidden}}$ on *actual* utilities yields $R^2 = 0.957$--$0.994$---a small residual that generates all observed regret.

<figure id="fig:degradation" data-latex-placement="t">
<embed src="figures/fig5_degradation_bound.pdf" style="width:70.0%" />
<figcaption>Degradation vs. stakeholder correlation. Each point is a synthetic hidden stakeholder at a different <span class="math inline"><em>α</em></span>. Color encodes <span class="math inline"><em>α</em><sub>hidden</sub></span>. Society (highest <span class="math inline"><em>α</em></span>) has lowest cosine similarity and highest regret. Log scale on <span class="math inline"><em>y</em></span>-axis.</figcaption>
</figure>

**Practical implication.** Correlation features predict the degradation *ranking* (which stakeholder is most dangerous to miss) with Spearman $= 0.96$. Predicting *magnitude* requires knowledge of the hidden utility's sensitivity, which is inherently unobservable. The $0.23$ $R^2$ gap between oracle and proxy quantifies the information cost of not observing the hidden stakeholder---connecting directly to [5.3]'s finding that even 25 pairs dramatically reduces this gap.

## Scaling with $K$ Stakeholders

Does adding more observed stakeholders reduce the cost of hiding one? We generate $K \in \{3, 5, 7, 10\}$ synthetic stakeholders using a factor model: 8 latent factors (engagement, safety, virality, retention, passive consumption, link sharing, discovery, content quality) over the 18-action space, with Dirichlet-sampled factor loadings. Each configuration is evaluated with 10 random stakeholder sets $\times$ 5 data seeds, using 101 diversity weights.

Average regret is roughly invariant with $K$ (0.19 at $K=3$, 0.17 at $K=5$, 0.18 at $K=10$)---a modest 10% improvement at $K=5$ that plateaus. The Pareto fraction (fraction of frontier points that are Pareto-optimal) increases from 0.48 ($K=3$) to 0.76 ($K=10$), confirming the dimensionality effect: higher-dimensional objectives make dominance harder, preserving more operating points. A secondary sweep over Dirichlet concentration shows that the correlation structure between stakeholders matters $4\times$ more than $K$ itself: regret drops from 0.27 (diverse, concentration $= 0.5$) to 0.16 (similar, concentration $= 5.0$) at fixed $K=5$.

# How Sensitive Is the Frontier to Specification? {#sec:sensitivity}

## The $\alpha$-Dominance Test

Does the pos/neg ratio $\alpha$ dominate over individual action weights? We compare two perturbation conditions, each applied to all three stakeholders' weight vectors and evaluated via Hausdorff distance between the perturbed and baseline Pareto frontiers:

- **Within-group shuffle** ($\alpha$-preserving): randomly permute positive weights among positive actions and negative weights among negative actions, preserving $\alpha = \bar{w}_{\text{neg}} / \bar{w}_{\text{pos}}$.

- **$\alpha$-perturbation**: uniformly scale all negative weights by factors $\times\{0.5, 0.8, 1.2, 1.5, 2.0\}$, changing $\alpha$ while preserving within-group weight structure.

The $\alpha$-dominance ratio (within-group variance / between-group variance) is 0.057---$\alpha$ perturbation causes $\sim$`<!-- -->`{=html}18$\times$ more frontier variability than within-group shuffling. At the *selection level* (perturbed weights drive content ranking, not just evaluation), the ratio is 0.062. By these measures, $\alpha$ dominates decisively.

## The Matched-Magnitude Reversal

However, the above comparison is unfair: the $\alpha$-perturbation condition ($\times 0.5$--$2.0$) involves much larger absolute weight changes than permutation. When we match perturbation magnitude ($\sigma = 0.3$ budget, allocated either entirely to within-group magnitude noise or entirely to $\alpha$ scaling), the result *reverses*: within-group magnitude perturbation causes $2\times$ more frontier shift than $\alpha$-only perturbation (Hausdorff 4.12 vs 2.10, ratio 1.96). At matched budget, individual weight magnitudes matter more than the group ratio.

## Rank Stability vs. Hausdorff Distance

This apparent contradiction is resolved by examining *what* changes. We perturb each of the 14 `UtilityWeights` parameters individually at factors $\times\{0.5, \ldots, 3.0\}$ and measure both Hausdorff distance and rank stability (fraction of Pareto-optimal operating points that survive).

The Hausdorff sensitivity ranking (normalized by baseline weight magnitude) identifies engagement weights as most sensitive: platform repost (6.6), user favorite (5.5), user repost (6.5). Negative weights (report: 0.27, mute: 0.30) rank last. But rank stability is $1.0$ for *every parameter at every perturbation level*---the same 7 diversity-weight operating points remain Pareto-optimal even under $3\times$ perturbation. The Hausdorff distance is detecting utility-axis *rescaling* (linear in perturbation magnitude), not frontier *shape change*. Individual parameter misspecification does not change the optimal policy.

## Specification vs. Data {#sec:goodhart}

If individual parameters don't change the policy, when does specification error matter? We compare two strategies for improving frontier quality with misspecified weights ($\sigma = 0.3$ Gaussian noise on all weights, *not* preserving $\alpha$):

- **Better specification** (fixed $N = 2{,}000$ pairs): reduce $\sigma$ from 0.5 to 0.0. Hausdorff decreases monotonically, as expected.

- **More data** (fixed $\sigma = 0.3$): increase $N$ from 25 to 2,000. Hausdorff *increases* after $N = 100$ ([6][22])---from 2.27 at $N = 100$ to 6.82 at $N = 500$.

<figure id="fig:goodhart" data-latex-placement="t">
<embed src="figures/fig6_spec_vs_data.pdf" />
<figcaption>Specification vs. data. <strong>(a)</strong> Better specification (reducing <span class="math inline"><em>σ</em></span>) monotonically improves frontier quality at fixed <span class="math inline"><em>N</em> = 2, 000</span>. <strong>(b)</strong> More data at fixed misspecification (<span class="math inline"><em>σ</em> = 0.3</span>) <em>worsens</em> quality after <span class="math inline"><em>N</em> = 100</span>—a Goodhart effect where the model more precisely learns the wrong utility.</figcaption>
</figure>

This is a Goodhart effect [@skalse2024goodhart]: with more training data, the BT model more faithfully learns the *wrong* utility function, producing a weight vector that is more precisely misaligned with the true objective. With fewer pairs ($N \leq 100$), the model is undertrained and noise partially masks the misspecification.

This finding creates a productive tension with [5.3]'s result that 25 pairs cuts regret by 42%. The resolution: data is powerful when the specification is correct ([5]), but *amplifies errors* when specification is wrong ([6]). The practical prescription is sequential: fix the specification first, then collect data.

## The Pareto Robustness Buffer

Synthesizing across all sensitivity experiments ([6][23]):

::: {#tab:sensitivity}
  **Perturbation**                     **Hausdorff**       **Rank stab.**       **Interpretation**
  ----------------------------------- --------------- ------------------------- -----------------------
  Single param ($\times 3$)              3.5--6.6              **1.0**          Scale change only
  Within-group shuffle                     2.24                  1.0            Assignment irrelevant
  $\alpha$ perturbation                    7.02        $<$`<!-- -->`{=html}1.0  Shape change
  Matched $\sigma = 0.3$ (within)          4.12                  ---            Magnitudes matter
  Matched $\sigma = 0.3$ ($\alpha$)        2.10                  ---            $\alpha$ cheaper
  Simultaneous (all params)                6.45        $<$`<!-- -->`{=html}1.0  Errors compound

  : Utility sensitivity summary: the Pareto frontier's response to different perturbation types.
:::

The frontier exhibits a *partial* robustness buffer: individual parameter perturbation is absorbed (rank stability $= 1.0$), but simultaneous misspecification of multiple parameters changes the frontier shape. The Pareto structure is more robust than single-objective RL (where any misspecification can be Goodharted [@skalse2024goodhart]), but less robust than the $\alpha$-dominance result initially suggested.

# Nonlinear Robustness {#sec:nonlinear}

All results above assume linear utility ($U = \text{pos} - \alpha
\cdot \text{neg}$). Real preferences may exhibit diminishing returns (prospect theory) or dead zones (ignoring low-level negativity). We test two nonlinear departures:

- **Concave** (prospect theory): $U = \text{pos}^\gamma -
      \alpha \cdot \text{neg}^\gamma$, with $\gamma \in \{0.3, \ldots, 1.0\}$.

- **Threshold** (dead zone): $U = \text{pos} - \alpha
      \cdot \max(\text{neg} - \tau, 0)$, with $\tau \in \{0.0, \ldots, 0.5\}$.

Four experiments across three utility families (linear, concave, threshold):

**Labels vs. Loss:** All mean cosine similarities exceed 0.92 across families---the labels-not-loss claim generalizes to nonlinear utilities.

**$\alpha$-Recovery:** Both families preserve Spearman $= 1.0$ for $\alpha$ rank-ordering. Nonlinear parameters bias absolute $\hat\alpha$ (concave inflates up to 57%, threshold deflates up to 40%) but never disrupt the ranking.

**Proxy Recovery:** The diversity knob is invariant (0.724 recovery across all families). $\alpha$-interpolation degrades under threshold utility ($0.738 \to 0.499$) because the dead zone creates preferences that the linear proxy cannot reproduce, but holds or improves under concave.

**Stress $\times$ Nonlinearity:** The same 4D stress sweep from [4.3] under all three families. Concave tightens the label-noise breaking point from $p = 0.30$ to $p = 0.20$ (practitioners using prospect-theoretic utilities should allow $\leq$`<!-- -->`{=html}15% annotation error rather than 20%). All other thresholds---sample size, temperature, content correlation---are unchanged across families. Full table in [12].

# External Validation {#sec:validation}

## MovieLens-100K

To validate the BT training architecture on real preference data, we train on MovieLens-100K movie ratings. The best configuration (BPR loss with in-batch negatives, 64-dim embeddings, 4 transformer layers) achieves Normalized Discounted Cumulative Gain (NDCG)@3 of 0.411, a 59% improvement over the untrained baseline (0.259). An ablation study reveals a 107.5% synergy effect: neither learned embeddings nor the transformer improves performance alone; all gain comes from their interaction. Details in [15].

## Synthetic Twitter

We construct a 648-parameter synthetic Twitter environment (6 archetypes $\times$ 6 topics $\times$ 18 actions) with known ground truth. All 5 verification suites pass: behavioral accuracy 100% (mean error $< 10^{-15}$), action differentiation 6/6 tests (lurker--power user repost ratio 15.6$\times$), block causal effect 78%, archetype flip 86%. The pipeline faithfully recovers the synthetic ground truth, confirming that observed multi-stakeholder phenomena are properties of the data, not artifacts of the training procedure. Details in [16].

# Discussion {#sec:discussion}

## Synthesis

The three research questions form a progressive arc:

**Identifiability** ([4]) establishes what BT training *can* recover: the preference ordering encoded in training labels, the negativity-aversion parameter $\alpha$ (Spearman $= 1.0$), and a disagreement-to-differentiation bound ($R^2 = 0.977$). Sun et al.'s [@sun2025rethinking] theoretical order-consistency result predicts these findings; our contribution extends the analysis to the multi-stakeholder setting and provides empirical stress bounds.

**Partial observation** ([5]) quantifies what is *lost* when a stakeholder is hidden. Society is $10\times$ costlier to miss than user, but even 25 preference pairs recover 42% of the lost utility---a VoI finding with immediate practical implications.

**Sensitivity** ([6]) reveals how *precisely* the non-hidden stakeholders' utilities must be specified. The answer is nuanced: individual parameters are forgiving (rank stability $= 1.0$), but correlated specification errors compound, and more training data amplifies misspecification rather than correcting it. This Goodhart result [@skalse2024goodhart] is the paper's deepest finding: it resolves an apparent contradiction between "data helps" ([5]) and "data hurts" ([6]) by showing that data is only as good as the utility specification it learns from.

## Practical Guidance

For practitioners deploying multi-stakeholder recommendation systems, our results suggest a sequential approach:

1.  **Specify $\alpha$ values.** The pos/neg ratio is the dominant structural parameter. It can be estimated from 250 preference pairs (Spearman $= 1.0$ under stress testing) or from pairwise disagreement rates without any training ([\[eq:disagreement\]][24]).

2.  **Collect hidden-stakeholder data.** If a stakeholder (typically society) is unobservable, invest in collecting even a small sample of preference labels. 25 pairs cut regret by 42%; 200 pairs reach the plateau.

3.  **Calibrate engagement weights.** The top-3 sensitive parameters are engagement weights (repost, favorite), not negative weights. Calibrate these to within $\pm$`<!-- -->`{=html}20%.

4.  **Use the diversity knob as a baseline.** If none of the above is feasible, setting the diversity weight to $\delta = 0.7$ recovers 70% of society's utility without any oracle knowledge.

## Implications for Platform Transparency

X's January 2026 algorithm release provides prediction targets (action types) but not learned parameters---*structure without weights*. In our synthetic benchmark, this is *partially* sufficient for multi-stakeholder audit: the ranking of which stakeholder is most dangerous to miss is predictable from action-type classification and pairwise disagreement rates (Spearman $= 0.96$). However, magnitude prediction requires approximate weight disclosure, and the Goodhart result shows that audits based on misspecified proxy utilities can be worse than no audit at all if the proxies systematically amplify specification errors.

These findings are established on an 18-parameter linear model, not on a production-scale transformer. Whether the same degradation bounds hold when utility is distributed across 8B learned parameters is an open question. Nevertheless, the *structural* results---the ordering of stakeholder sensitivity, the diminishing returns of data collection, and the Goodhart mechanism---depend on properties of the multi-stakeholder optimization problem (Pareto geometry, BT convergence) rather than model scale, and we conjecture they generalize.

If these results transfer, they suggest that regulators implementing the EU Digital Services Act [@lasser2025designing] should require platforms to disclose at minimum (1) the action space (positive/negative classification), (2) approximate negativity-aversion ratios per stakeholder, and (3) the engagement weights for the top-3 most sensitive actions. This would enable meaningful third-party LOSO analysis at modest transparency cost.

## Limitations

**Synthetic data and scale gap.** All multi-stakeholder findings depend on synthetic data with known ground truth (600 users, 100 content items, 648-parameter Twitter simulation). MovieLens validates the BT architecture on real data but not the stakeholder analysis. The methodology (LOSO, $\alpha$-recovery, sensitivity analysis) generalizes; the specific numbers are benchmark-specific. Critically, our benchmark uses 18-dimensional linear weight vectors, while the production system distributes utility across 8B transformer parameters. The production model's trained weights and user data are not publicly available, and ground-truth stakeholder utilities do not exist for real platforms---both prerequisites for our regret-based evaluation. Whether the degradation bounds and sensitivity thresholds established here transfer to production-scale models remains an open empirical question.

**Three stakeholders.** The $K$-stakeholder scaling analysis ([13]) tests $K > 3$ but with synthetic factor-based stakeholders. Real stakeholder populations may not follow Dirichlet factor loadings.

**Simplified society utility.** Society utility is operationalized as $\text{pos} - 4 \cdot \text{neg}$, a structural proxy for diversity minus polarization. The actual `compute_society_utility` function measures topic diversity and cross-partisan exposure, which is structurally different (set-level, not per-item). The structural $R^2 = 1.0$ collapse explicitly reveals this gap.

**Fixed engagement scorer.** Content selection uses a fixed engagement formula ($\text{fav} + 0.8 \times \text{repost} + 0.5
\times \text{follow}$), not the stakeholder-specific weight vectors. In production, the scorer co-evolves with the utility specification. Our selection-level sensitivity test partially addresses this ($\alpha$-dominance holds at $0.062$), but a full co-evolution study remains future work.

**Separable utility families.** The nonlinear robustness audit tests concave and threshold utilities, both of which decompose additively over actions. Non-separable interaction effects (e.g., repost $\times$ block) are not tested.

# Conclusion {#sec:conclusion}

We present a systematic empirical study of multi-stakeholder reward modeling on X's open-source recommendation algorithm, addressing identifiability, partial observation, and utility sensitivity through 87 training experiments, 20-seed value-of-information analysis, and six sensitivity test conditions.

The central finding is that stakeholder differentiation is determined by training labels, not loss functions---an empirical confirmation of BT order consistency [@sun2025rethinking] in the multi-stakeholder setting. Building on this, we show that hiding the least-observable stakeholder (society) costs $10\times$ more than hiding the most-observable (user), but even 25 preference pairs recover 42% of the lost utility. The Pareto frontier absorbs individual parameter perturbation (rank stability $= 1.0$) but not correlated specification errors, and additional training data amplifies misspecified utilities rather than correcting them.

These findings have direct implications for platform transparency under the EU Digital Services Act: structure-without-weights disclosure enables degradation *ranking* but not magnitude prediction, and proxy-based audits can amplify specification errors through a Goodhart mechanism. The practical prescription is sequential---fix the specification, then collect data---and the diversity knob at $\delta = 0.7$ serves as a robust baseline when neither is feasible.

All code and experiment protocols are publicly available.[^2]

# Nonlinear Robustness Full Tables {#app:nonlinear}

::: {#tab:nonlinear-labels}
  Family                              User    Platform   Society
  ---------------------------------- ------- ---------- ---------
  Linear                              0.989    0.929      0.976
  Concave ($\gamma = 0.5$--$0.9$)     0.992    0.933      0.971
  Threshold ($\tau = 0.05$--$0.3$)    0.989    0.924      0.976

  : Labels vs. Loss under nonlinear utilities (mean cosine similarity across 4 loss types).
:::

::: {#tab:nonlinear-alpha}
  Family                      $\alpha$-Spearman   Sensitivity                 Bias
  -------------------------- ------------------- ------------- -----------------------------------
  Concave ($\gamma = 0.7$)           1.0             0.169               +57% inflation
  Threshold ($\tau = 0.1$)           1.0             0.154      $-$`<!-- -->`{=html}40% deflation

  : $\alpha$-recovery under nonlinear utilities.
:::

::: {#tab:nonlinear-proxy}
  Family       Oracle   $\alpha$-Interp   DW 0.7   Interp cosine
  ----------- -------- ----------------- -------- ---------------
  Linear       0.264         0.738        0.724        0.964
  Concave      0.431         1.000        0.724        0.961
  Threshold    0.259         0.499        0.724        0.966

  : Proxy recovery under nonlinear utilities (hiding society, 500-item pool).
:::

# Stress $\times$ Nonlinearity {#app:stress}

::: {#tab:stress-nonlinear}
  Dimension                          Linear   Concave    Threshold
  --------------------------------- -------- ---------- -----------
  Label noise ($p_{\text{flip}}$)     0.30    **0.20**     0.30
  Sample size ($n$)                    50        50         50
  BT temperature ($\beta$)            0.5       0.5         0.5
  Content correlation ($\rho$)        0.8       0.8         0.8

  : Breaking points (Spearman $< 0.95$) under nonlinear stress.
:::

Concave utility tightens the label-noise threshold from 0.30 to 0.20. All other thresholds are invariant across utility families. Practitioners using prospect-theoretic preferences should derate the linear noise threshold by one level ($\leq$`<!-- -->`{=html}15% annotation error).

# $K$-Stakeholder Scaling {#app:kscaling}

::: {#tab:kscaling}
  $K$    Avg regret   Pareto frac.   Eff. rank   Ratio vs. $K\!=\!3$
  ----- ------------ -------------- ----------- ---------------------
  3        0.189         0.484           3              1.000
  5        0.171         0.594           5              0.902
  7        0.179         0.749           7              0.947
  10       0.182         0.757           8              0.960

  : Scaling with $K$ stakeholders (8 factors, 101 diversity weights, 10 configs $\times$ 5 seeds).
:::

::: {#tab:concentration}
  Concentration     Avg regret   Mean cosine
  ---------------- ------------ -------------
  0.5 (diverse)       0.272         0.202
  2.0 (moderate)      0.171         0.280
  5.0 (similar)       0.157         0.266

  : Concentration sweep at $K = 5$.
:::

Correlation structure (Dirichlet concentration) produces a $4\times$ larger effect on regret than the number of stakeholders.

# Disagreement Bound Derivation {#app:disagreement}

The two-variable model ([\[eq:disagreement\]][24]) is fit via ordinary least squares on 34 data points from two sweeps: one fixing Platform ($\alpha = 0.3$) and sweeping a partner, another fixing User ($\alpha = 1.0$). Label disagreement rate $d$ is computed on 50,000 preference pairs. Mean total margin $m$ is the average $|U_i(c_1) - U_i(c_2)|$ on disagreed pairs. The univariate model ($d$ only) achieves $R^2 = 0.90$ but is reference-dependent; adding margin resolves this. For the utility family $U = \text{pos} - \alpha \cdot \text{neg}$, mean margin decomposes as $|\Delta\alpha| \cdot \mathbb{E}[\Delta\text{neg} \mid \text{disagreement}]$, cleanly separating parameter distance from content structure.

# MovieLens Training Details {#app:movielens}

Best configuration: BPR loss with in-batch negatives (batch size 32), 64-dimensional learned embeddings, 4 transformer layers, learning rate $5 \times 10^{-4}$ (halved from $10^{-3}$), weight decay $10^{-4}$. Training stopped at epoch 9 (best validation NDCG@3). The BCE baseline achieved val NDCG@3 of 0.316 but test NDCG@3 of only 0.241 (severe overfitting due to score margins $\sim 10^{-8}$). BPR + in-batch negatives produces margins $\sim$`<!-- -->`{=html}0.1--0.2, eliminating the generalization gap (val 0.411, test 0.418).

::: {#tab:movielens-ablation}
  Configuration                 NDCG@3          Contribution
  ---------------------------- -------- ----------------------------
  Full model                    0.410             Baseline
  Learned emb. + dot-product    0.286              +1.4%
  Random emb. + transformer     0.268     $-$`<!-- -->`{=html}5.2%
  Random baseline               0.282               ---
  Synergy effect                 ---     $+$`<!-- -->`{=html}107.5%

  : MovieLens ablation study.
:::

# Synthetic Twitter Verification {#app:synthetic}

The synthetic environment defines 648 ground-truth parameters: $6 \times 6 \times 18$ (archetypes $\times$ topics $\times$ actions). Five verification suites test progressively deeper model properties: (1) embedding probes (silhouette scores: topic 1.00, archetype 0.369); (2) behavioral tests (100% accuracy, mean error $< 10^{-15}$); (3) action differentiation (6/6 tests pass, lurker--power user repost ratio 15.6$\times$); (4) block causal effect (78% pass rate after adding synthetic block pairs during training); (5) archetype flip via history swapping (85.7% pass rate).

# Pluralistic Models and Causal Verification {#app:pluralistic}

We tested four pluralistic approaches for multi-group reward modeling: single BT (baseline), GMM clustering, soft assignment, and two-stage (cluster-then-train). The two-stage approach achieved 100% cluster purity using topic$\times$action features, with overall accuracy 99.4%. Weight correlation (0.554) fell below the target gate (0.8), a limitation attributed to BT's scale invariance rather than model failure.

Causal verification tested intervention: replacing a user's engagement history with a different archetype's history and measuring score changes. Action-level interventions (block, follow) achieve 100% pass rate. History-level interventions achieve 50% initially, improving to 86% with history-topic contrastive loss.

# Per-Parameter Sensitivity {#app:sensitivity}

::: {#tab:per-param}
  Parameter               Category     Rel. sensitivity          Tolerance
  ----------------------- ---------- ------------------ ---------------------------
  user:favorite           Positive                 8.25       $< \times 0.8$
  platform:repost         Positive                 6.56       $< \times 0.8$
  user:repost             Positive                 6.45       $< \times 0.8$
  user:reply              Positive                 3.23  $\pm$`<!-- -->`{=html}20%
  platform:reply          Positive                 3.22  $\pm$`<!-- -->`{=html}20%
  user:not_interested     Negative                 1.72  $\pm$`<!-- -->`{=html}50%
  platform:block_author   Negative                 1.32  $\pm$`<!-- -->`{=html}50%
  user:block_author       Negative                 1.31  $\pm$`<!-- -->`{=html}30%
  user:share              Positive                 1.15  $\pm$`<!-- -->`{=html}50%
  platform:share          Positive                 1.15  $\pm$`<!-- -->`{=html}50%
  user:follow_author      Positive                 0.33        $> \times 2$
  user:quote              Positive                 0.32        $> \times 2$
  user:mute_author        Negative                 0.30        $> \times 2$
  user:report             Negative                 0.27        $> \times 2$

  : Per-parameter sensitivity ranking (normalized by baseline weight magnitude). Rank stability = 1.0 for all parameters at all perturbation levels.
:::

The LLM margin proxy experiment (Claude Haiku, 15 sweep points $\times$ 200 pairs) achieves Spearman $\rho = 0.929$ between LLM-based and analytic disagreement rates. LLM confidence is too compressed (range \[1.40, 1.66\]) for precise margin estimation but preserves rank-ordering. The go/no-go criterion was revised from $R^2$ to Spearman during analysis: rank-ordering is the practically relevant quantity for predicting which stakeholder pairs will be most differentiated.

[^1]: Independent Researcher. Contact: `gkartik@gmail.com`

[^2]: <https://github.com/kar-ganap/x-algorithm-enhancements>

  [1]: #fig:transition {reference-type="ref+label" reference="fig:transition"}
  [4]: #sec:identifiability {reference-type="ref+label" reference="sec:identifiability"}
  [5]: #sec:partial {reference-type="ref+label" reference="sec:partial"}
  [6]: #sec:sensitivity {reference-type="ref+label" reference="sec:sensitivity"}
  [2]: #sec:background {reference-type="ref+Label" reference="sec:background"}
  [3]: #sec:system {reference-type="ref+Label" reference="sec:system"}
  [7]: #sec:identifiability,sec:partial,sec:sensitivity {reference-type="ref+Label" reference="sec:identifiability,sec:partial,sec:sensitivity"}
  [7]: #sec:nonlinear {reference-type="ref+Label" reference="sec:nonlinear"}
  [8]: #sec:validation {reference-type="ref+Label" reference="sec:validation"}
  [9]: #sec:discussion {reference-type="ref+Label" reference="sec:discussion"}
  [10]: #tab:actions {reference-type="ref+label" reference="tab:actions"}
  [11]: #fig:pipeline {reference-type="ref+label" reference="fig:pipeline"}
  [12]: #eq:btloss {reference-type="ref+label" reference="eq:btloss"}
  [13]: #tab:cosine {reference-type="ref+label" reference="tab:cosine"}
  [14]: #eq:utility {reference-type="ref+label" reference="eq:utility"}
  [15]: #fig:alpha {reference-type="ref+label" reference="fig:alpha"}
  [16]: #tab:stress {reference-type="ref+label" reference="tab:stress"}
  [18]: #app:sensitivity {reference-type="ref+label" reference="app:sensitivity"}
  [17]: #tab:loso {reference-type="ref+label" reference="tab:loso"}
  [19]: #tab:proxy {reference-type="ref+label" reference="tab:proxy"}
  [20]: #fig:exp4 {reference-type="ref+label" reference="fig:exp4"}
  [21]: #fig:degradation {reference-type="ref+label" reference="fig:degradation"}
  [5.3]: #sec:databudget {reference-type="ref+label" reference="sec:databudget"}
  [22]: #fig:goodhart {reference-type="ref+label" reference="fig:goodhart"}
  [23]: #tab:sensitivity {reference-type="ref+label" reference="tab:sensitivity"}
  [4.3]: #sec:stress {reference-type="ref+label" reference="sec:stress"}
  [12]: #app:stress {reference-type="ref+label" reference="app:stress"}
  [15]: #app:movielens {reference-type="ref+label" reference="app:movielens"}
  [16]: #app:synthetic {reference-type="ref+label" reference="app:synthetic"}
  [24]: #eq:disagreement {reference-type="ref+label" reference="eq:disagreement"}
  [13]: #app:kscaling {reference-type="ref+label" reference="app:kscaling"}
