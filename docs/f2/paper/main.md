# Labels Not Loss: Multi-Stakeholder Differentiation, Partial Observation, and Utility Sensitivity in Recommendation Systems

*Empirical analysis on X's open-source recommendation algorithm*

**Kartik G Bhat**

Independent Researcher. Contact: gkartik@gmail.com

---
## Abstract

Multi-stakeholder recommendation systems balance user engagement,
platform retention, and societal welfare, but the utility functions
defining these objectives are hand-specified, partially observable, and
potentially misspecified. We present a systematic empirical study of
these three challenges on X's open-source recommendation algorithm
(Phoenix, 18 engagement actions, 3 stakeholders), motivated by the
platform's transition from explicit utility weights (2023) to an
implicit Grok-based transformer (2026).

**Identifiability.** Across 87 training experiments with four
Bradley-Terry loss variants, stakeholder differentiation is determined
entirely by training labels, not the loss function (cosine
similarity $>0.92$ within any loss type, 0.478 across stakeholders).
The negativity-aversion parameter $$ is recoverable from learned
weight vectors with Spearman $ = 1.0$, robust to $$20
label noise and $$250 preference pairs.

**Partial observation.** Leave-One-Stakeholder-Out analysis shows
hiding society costs $10$ more regret than hiding user. Even 25
preference pairs from the hidden stakeholder reduce regret by 42
(20 seeds), with diminishing returns beyond $$200 pairs. A
degradation model predicts which stakeholder is most dangerous to miss
(Spearman $= 0.96$) but not by how much ($R^2 = 0.72$).

**Utility sensitivity.** The Pareto frontier absorbs individual
weight perturbation (rank stability $= 1.0$ at all levels tested) but
not simultaneous misspecification. With misspecified weights,
additional training data amplifies the error after $N > 100$
pairs—a Goodhart effect. Data helps when specification is correct;
it hurts when specification is wrong.

Results are validated under nonlinear utility families (prospect theory,
threshold) and externally on MovieLens-100K (+59
648-parameter synthetic Twitter environment.


## Introduction


Recommendation systems serve multiple stakeholders with conflicting
objectives. A social media platform balances user engagement (did the
user enjoy this content?), platform retention (will the user return
tomorrow?), and societal welfare (did this content increase
polarization?). Multi-stakeholder optimization frameworks formalize
these tradeoffs as Pareto frontiers over stakeholder
utilities [burke2017multistakeholder, multifr2022,
eval2025multistakeholder], but three questions remain largely
unexamined empirically:


  - **Identifiability.** What properties of stakeholder
    preferences can a Bradley-Terry (BT) reward model actually recover
    from pairwise comparison data?
  - **Partial observation.** When a stakeholder is
    unobservable—as societal welfare typically is—how much does the
    Pareto frontier degrade, and how cheaply can the degradation be
    mitigated?
  - **Specification sensitivity.** How precisely must
    stakeholder utility functions be specified before the frontier
    shifts meaningfully?


These questions are newly urgent. In January 2026, X (formerly
Twitter) open-sourced a complete rewrite of its recommendation
algorithm [xalgorithm2026], replacing the hand-engineered
Phoenix heavy ranker [xalgorithm2023] with a Grok-based
transformer that "eliminated every single hand-engineered feature and
most heuristics." The 2023 release exposed *explicit* utility
weights—per-action multipliers for favorites, reposts, blocks, and
15 other engagement signals. The 2026 release exposes
*structure without weights*: the prediction targets (action
types) are visible, but the actual utility weighting is implicit in
8B learned parameters. This transition from explicit to implicit
specification raises a concrete audit question: can regulators assess
multi-stakeholder welfare from released code when the weights are
opaque? [shaped2023secrets]

We study all three questions on a realistic synthetic benchmark built
on X's 18-action engagement space, using BT preference learning to
train separate reward models for three stakeholders (user, platform,
society). Our principal findings:


  - **Labels, not loss, determine differentiation.** Across
     training experiments ( converged) with four BT
    loss variants, stakeholder weight vectors converge to cosine
    similarity $>$ when trained on identical preference
    labels, regardless of loss function. Differentiation arises
    entirely from stakeholder-specific training data. The
    negativity-aversion parameter $$ is recoverable from learned
    weights with Spearman $ = $, robust to
    $$20
    pairs ([sec:identifiability]).

  - Hiding society costs 10$$ more than hiding
    user. Leave-One-Stakeholder-Out (LOSO) analysis shows that
    removing society from the optimization produces average regret of
    1.08, versus 0.11 for user—the stakeholder hardest to observe
    is the most costly to miss. Even 25 preference pairs from the
    hidden stakeholder reduce regret by 42
    with diminishing returns beyond $$200
    pairs ([sec:partial]).

  - The Pareto frontier is robust to individual weight
    perturbation but not to simultaneous misspecification. Perturbing
    any single utility weight by up to $3$ preserves all
    Pareto-optimal operating points (rank stability $= 1.0$). But
    simultaneous perturbation of multiple weights at matched magnitude
    ($ = 0.3$) causes $2$ more frontier shift than
    changing the $$ ratio alone. With misspecified weights,
    additional training data *amplifies* the error after
    $N > 100$ pairs—a Goodhart
    effect [skalse2024goodhart] ([sec:sensitivity]).


All three directions are validated under nonlinear utility
families (prospect-theoretic concave and threshold utilities) and
externally on MovieLens-100K (+59
synthetic Twitter environment. Code and data are available at
https://github.com/kar-ganap/x-algorithm-enhancements.


The remainder of the paper is organized as follows.
[sec:background] reviews multi-stakeholder recommendation, BT
preference learning, and utility sensitivity analysis.
[sec:system] describes the synthetic benchmark and evaluation
framework. [sec:identifiability,sec:partial,sec:sensitivity]
present the three research directions. [sec:nonlinear] tests
robustness under nonlinear utilities. [sec:validation] provides
external validation. [sec:discussion] synthesizes practical
guidance and discusses limitations.


## Background and Related Work


### Multi-Stakeholder Recommender Systems

Burke and Abdollahpouri [burke2017multistakeholder] introduced a
utility-based framework for representing stakeholder values in
recommendation, distinguishing consumers, providers, and system
operators as distinct principals. Subsequent work formalized
multi-stakeholder optimization using Pareto frontiers:
Multi-FR [multifr2022] applies multiple gradient descent to
generate Pareto-optimal solutions balancing accuracy and fairness
across stakeholders, while recent surveys [eval2025multistakeholder]
propose a four-stage evaluation methodology (stakeholder identification,
value articulation, metric selection, aggregate evaluation).
Lasser et al. [lasser2025designing] argue for designing
recommendation algorithms through the lens of healthy civic discourse,
foregrounding societal welfare as a first-class stakeholder objective. Stray et al. [stray2024building] provide an interdisciplinary synthesis of how to embed human values in recommender systems.

A parallel line of work on *pluralistic alignment* addresses
diverse preferences within reinforcement learning from human feedback
(RLHF). Sorensen et al. [sorensen2024roadmap] map the design
space for handling preference heterogeneity, ranging from personalized
reward models [pad2025personalized] to federated
aggregation. These approaches assume preferences can be captured in a
single (possibly personalized) model. Our work takes a different
stance: stakeholders have fundamentally different objective functions
that must be traded off on a frontier, Agarwal et al. [agarwal2024system2] formalize the utility-engagement distinction via temporal point processes. Our work asks the complementary question:
what happens when one stakeholder's objective is absent entirely.

### Bradley-Terry Preference Learning

The Bradley-Terry (BT) model [bradley1952rank] converts pairwise
comparisons into scalar scores via
$P(prefer  a  b) = (r_a - r_b)$, where $r_i$ are
learned reward values. BT preference learning underpins most modern
RLHF pipelines, though the theoretical justification for this
application has only recently been examined. Sun
et al. [sun2025rethinking] prove that BT models possess an
*order consistency* property: any monotonic transformation of
the true reward preserves the correct ranking, making BT sufficient
for downstream optimization even when absolute reward values are
unrecoverable. Extensions beyond the BT
assumption [beyondbt2025] address intransitive and
context-dependent preferences, though these remain less common in
deployed systems.

Our utility model is a special case of multi-attribute utility theory
(MAUT) [keeney1976decisions], which decomposes preferences into
weighted sums of single-attribute utilities. The structural family
$U = pos -   neg$ parameterizes each
stakeholder by a single negativity-aversion coefficient $$,
varying from $ = 0.3$ (platform, tolerant of negative signals)
to $ = 4.0$ (society, highly penalizing).

### Reward Misspecification and Goodhart's Law

Goodhart's Law—"when a measure becomes a target, it ceases to be a
good measure"—manifests in RL as reward hacking: agents exploit
misspecified proxy rewards to achieve high proxy scores while
degrading true objectives [skalse2024goodhart, weng2024reward].
Casper et al. [casper2023open] catalog open problems in RLHF,
arguing that condensing diverse human preferences into a single reward
model is "fundamentally misspecified." This concern applies
*a fortiori* to multi-stakeholder systems, where each
stakeholder's utility may be independently misspecified. Hadfield-Menell and Hadfield [hadfield2019incomplete] draw an analogy to incomplete contracting: reward functions inevitably fail to specify all contingencies.

The Goodhart taxonomy distinguishes regressional, extremal, causal,
and adversarial failure modes [skalse2024goodhart]. Our
Direction 2 ([sec:sensitivity]) tests the *misweighting*
variant—same goals, different priorities—and finds a novel
interaction: additional training data amplifies misspecified utilities
rather than correcting them.

### Sensitivity Analysis in MCDA

Sensitivity analysis in multi-criteria decision analysis (MCDA)
assesses how robust a ranking is to changes in decision
weights [mcda2023sensitivity]. Classical MAUT sensitivity
analysis [farmer1987testing] computes "weight stability
intervals"—the range within which each weight can vary without
changing the top-ranked alternative [weightstability2025].
A recent survey [robust2025survey] catalogs robustness approaches for recommender systems. Iancu and Trichakis [iancu2014pareto] extend this to
multi-objective optimization, defining robust Pareto
optimality: solutions that remain Pareto-optimal under worst-case
parameter uncertainty.

We adapt this framework from single rankings to Pareto frontiers,
introducing *rank stability* (the fraction of operating points
that survive perturbation) as the primary metric rather than
Hausdorff distance (which we show conflates scale change with shape
change in our setting).

### Platform Transparency

The EU Digital Services Act (DSA), effective 2024, mandates that
large platforms assess algorithmic risks to democratic values and
provide transparency about recommender systems. X was fined
140M in 2025 for transparency violations and subsequently
open-sourced its algorithm [xalgorithm2026]. However, the
release provides *structure without weights*: the model
architecture and prediction targets are public, but learned
parameters are not [shaped2023secrets]. Whether this level of
transparency is sufficient for meaningful multi-stakeholder audit is
an open question that our sensitivity analysis directly addresses.

### X's Algorithm: Phoenix Architecture

X's recommendation system has undergone two public releases. The
March 2023 release [xalgorithm2023] exposed the Phoenix heavy
ranker: a MaskNet-based neural network predicting engagement
probabilities for 18 action types (favorites, reposts, replies,
blocks, reports, etc.), combined via explicit per-action weights into
a final relevance score. The January 2026
release [xalgorithm2026] replaced this with a Grok-based
transformer that learns all weighting implicitly from engagement
sequences, eliminating hand-engineered features entirely.

Our synthetic benchmark is built on the 2023 Phoenix action space (18
actions, 5 positive, 4 negative, 9 neutral) and explicitly
parameterizes the utility weights that the 2026 system learns
implicitly. This allows controlled experimentation on sensitivity,
partial observation, and identifiability questions that cannot be
studied on the opaque production system.


## System, Data, and Methodology


### Action Space

The Phoenix action space defines 18 user actions on content, inherited
from X's production system [xalgorithm2023]. We classify these
into three groups ([tab:actions]):


  - **Positive** (5): favorite, repost, follow\_author,
    share, reply—signals of genuine user value.
  - **Negative** (4): block\_author, mute\_author, report,
    not\_interested—signals of harm or discomfort.
  - **Neutral** (9): click, dwell, photo\_expand, vqv
    (video quality view), profile\_click, share\_via\_dm,
    share\_via\_copy\_link, quote, quoted\_click—engagement signals
    of ambiguous valence.


Each content item $c$ viewed by user $u$ produces an action
probability vector $p(u, c)  [0,1]^{18}$, representing
the predicted probability of each action. The full dataset is a
tensor $P  R^{N  M  18}$ for $N$
users and $M$ content items.


[See PDF for table/figure]

### Stakeholder Utility Model

Each stakeholder $k$ has a utility function
$U_k(c) = w_k^ p(c)$, where
$w_k  R^{18}$ is a weight vector over actions.
We study the structural family
[See PDF for table/figure]
where $_k$ is the *negativity aversion* parameter. The
three stakeholders differ primarily in $$:


  - **User** ($ = 1.0$): Equal weight on positive
    engagement and discomfort avoidance.
  - **Platform** ($ = 0.3$): Tolerant of negative
    signals; all engagement has value.
  - **Society** ($ = 4.0$): Heavily penalizes (motivated by evidence that algorithmic exposure affects polarization [bail2018exposure])
    block/report/mute signals as proxies for polarization and harm.


This family is a special case of additive MAUT [keeney1976decisions],
where the full 18-parameter weight vector is summarized by a single
ratio. Whether $$ alone is sufficient or individual action
weights matter independently is tested in Direction 2
([sec:sensitivity]).

### Synthetic Data Generation

We generate synthetic evaluation data with known ground truth: 600
users, 100 content items, 6 topics (technology, sports,
politics-left, politics-right, entertainment, science), and 5 user
archetypes (casual, power user, political-left, political-right,
niche enthusiast). Each user–content pair produces an 18-dimensional
action probability vector based on archetype–topic affinity rules
(e.g., political-left users have elevated block rates on
politics-right content).

For BT training, a separate content pool of 500 items is generated
per seed using `generate\_content\_pool()`, which draws action
probabilities from calibrated distributions. Preference pairs are
constructed by sampling two content items, computing stakeholder
utility, adding Gaussian noise ($ = 0.05$), and labeling the
higher-utility item as preferred.

### Bradley-Terry Training Pipeline

For each stakeholder, we train a BT reward model on 2{,}000
preference pairs for 50 epochs with Adam optimization (learning rate
0.01). The model learns an 18-dimensional weight vector
$w  R^{18}$ by minimizing the standard BT loss:
[See PDF for table/figure]

We test four loss variants: standard BT, Margin-BT (adds a fixed
margin to the preference gap), Calibrated-BT (regresses on engagement
magnitude), and Constrained-BT (adds a diversity regularizer). The
central Direction 1 result is that these variants produce nearly
identical weight vectors when trained on identical preference data.

### Evaluation Framework


Stakeholder tradeoffs are measured via the *Pareto frontier*: we
sweep a diversity weight $  \{0.0, 0.05, , 1.0\}$ (21
points) that controls the tradeoff between engagement-based content
scoring and topic diversity. At each $$, we select top-$K$
content per user using a greedy diversity-aware algorithm, then
evaluate all three stakeholder utilities on the selected content. The
resulting 21-point frontier in $(U_{user},
U_{platform}, U_{society})$ space defines the
achievable tradeoffs.

**LOSO projection.** To simulate hiding a stakeholder, we
extract the 2D Pareto front in the observed dimensions and measure
hidden-dimension *regret*:
$regret = _ U_{hidden}() - U_{hidden}(_{LOSO})$,
averaged across operating points.

**Rank stability.** For sensitivity analysis, we define rank
stability as the fraction of Pareto-optimal operating points that
remain optimal after utility weight perturbation. We show
([sec:sensitivity]) that this is more informative than Hausdorff
distance, which conflates utility-axis rescaling with frontier shape
change.

All experiments use 5 seeds unless otherwise noted; Experiment 4
(partial sampling) uses 20 seeds for tighter confidence intervals.


## Direction 1: Identifiability


### Labels vs.\ Loss

We train BT reward models for all three stakeholders under four loss
variants: standard BT ([eq:btloss]), Margin-BT (adds a fixed
margin $m$ to the preference gap), Calibrated-BT (regresses on
engagement magnitude), and Constrained-BT (adds a diversity
regularizer penalizing weight concentration). Each variant is swept
across multiple hyperparameter settings, yielding  total
training runs.

Of these,  converge (8 Constrained-BT society runs diverge
due to numerical instability in the diversity constraint, since
fixed). Among the converged experiments, the key finding is that
all four loss types produce near-identical weight vectors when
trained on the same preference data. Pairwise cosine similarity
between weight vectors trained with different losses on the same
stakeholder exceeds $$ in every case ([tab:cosine]). The differentiation
that appears in the final models—cosine similarity of 0.478 between
platform and society, 0.830 between user and platform, 0.884 between
user and society—arises entirely from different training
labels, not different loss functions.

This confirms Sun et al.'s [sun2025rethinking] theoretical
order-consistency result in the multi-stakeholder recommendation
setting: BT training faithfully preserves the preference ordering
in the data, and no loss variant can overcome identical training
signals. The practical implication is that practitioners should
invest in stakeholder-specific labeling, not loss engineering.


[See PDF for table/figure]

### $$-Recovery

[See PDF for table/figure]

Given that labels determine differentiation, we ask: what structural
properties of the underlying utility function are recoverable from
the learned weight vector? For the structural family ([eq:utility])
$U = pos -   neg$, we test whether the
negativity-aversion parameter $$ can be extracted from trained
BT weights via the ratio
$ = -w_{neg} / w_{pos}$.

We sweep $  \{0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0,
3.0, 4.0, 5.0, 7.0, 8.0\}$ (13 values, 5 seeds each). The
recovered $$ values ([fig:alpha]) preserve perfect monotonic ordering:
Spearman $ = $ across all methods tested (ratio,
sum-ratio, regression). The relationship is affine:
$  -0.06 + 1.32  _{true}$
($R^2 = 0.997$), with systematic amplification (slope $> 1$) but
perfect rank preservation. For the three real stakeholders, the
recovered values are: user $ = 1.35$ (true 1.0), platform
$ = 0.34$ (true 0.3), society $ = 5.59$ (true
4.0)—the ordering society $>$ user $>$ platform is correctly
recovered.

### Stress Testing


We stress-test $$-recovery across four dimensions ([tab:stress]) (1{,}300
total training runs, 5 seeds $$ 13 $$ values per
condition):

[See PDF for table/figure]

Key findings: (1) recovery is robust to 20
with 250 pairs; (2) BT temperature is the strongest stressor—soft
preferences ($ < 0.5$) destroy the signal; (3) moderate content
correlation ($ = 0.3$) actually *improves* recovery by
creating more informative preference pairs; (4) Spearman degrades
much slower than Pearson—rank ordering survives conditions that
destroy the linear fit.

### Disagreement-to-Differentiation Bound

To predict stakeholder differentiation *before* training, we
measure pairwise label disagreement rates on 50{,}000 preference
pairs. The relationship between disagreement rate $d$ and cosine
similarity is well fit by a two-variable model incorporating the mean
margin $m$ (average utility gap on disagreed pairs):
[See PDF for table/figure]
Disagreement rate alone achieves $R^2 = 0.90$; adding margin resolves
reference-dependence (the platform-fixed and user-fixed sweeps
collapse onto a single curve). Practical thresholds: $$10
disagreement for $ < 0.95$; $$19
$$37

### LLM-as-Annotator Compatibility

We test whether an LLM can substitute for analytic stakeholder
definitions. Claude Haiku, given natural-language stakeholder
descriptions (e.g., "you are a safety-first content moderator") and
content as engagement/negativity scores, generates preference labels
across 15 sweep points $$ 200 pairs. The LLM's implicit
utility preserves the ranking structure: Spearman $ = 0.929$
between LLM-based and analytic disagreement rates. LLM confidence is
too compressed (range [1.40, 1.66]) to serve as a precise margin
proxy, but rank-ordering—the practically relevant quantity—is
preserved. Details in [app:sensitivity].


## Direction 3: Partial Observation


### LOSO Geometry and Training

For each of the three stakeholders, we hide it and compute the
2-stakeholder Pareto frontier by projecting the full 3D frontier onto
the observed dimensions. Two variants: *geometric* (uses the
hardcoded scorer, hides utility at evaluation) and
*training-based* (trains BT models only on the observed
stakeholders' preference pairs, uses learned weight vectors as
scorers).

[See PDF for table/figure]

Society is 10$$ more costly to hide than user ([tab:loso]).
The degradation ranking matches the pairwise correlation structure:
society has the lowest cosine similarity (0.478) with the observed
pair. Geometric and training-based LOSO produce consistent
rankings, validating the geometric analysis. The 2D Pareto frontiers
are rarely dominated in 3D (0
observation is entirely in the hidden dimension.

### Proxy Methods

Can we recover the hidden stakeholder's utility using proxies
constructed from the observed stakeholders' weight vectors? We test
six methods, focusing on hiding society (the highest-stakes case) ([tab:proxy]):

[See PDF for table/figure]

The diversity knob at $ = 0.7$ is the best practical method
(70
$$-interpolation outperforms the oracle linear proxy (0.738 vs
0.566) because it preserves action-level BT structure that
least-squares regression dilutes.

### Data Budget


If a platform can collect $N$ society preference pairs (rather than
the full 2{,}000), how much regret is recovered? We sweep $N 
\{0, 25, 50, 100, 200, 500, 1000, 2000\}$ across 20 seeds,
measuring average regret across all 21 diversity-weight operating
points on the frontier scored by the trained society model.

Even 25 preference pairs reduce regret from 1.111 (LOSO baseline) to
0.644—a 42
diminishes rapidly: the regret plateau at $$0.55 is statistically
indistinguishable across $N = 200$–$2{,}000$ (overlapping 95
bootstrap confidence intervals). This is a Value of Information
(VoI) result in the sense of Howard [howard1966information]:
the first few bits of stakeholder information are the most valuable.

[See PDF for table/figure]

### Degradation Predictability

Can we predict LOSO degradation from observable correlation features
*without* running the experiment? We sweep
$_{hidden}  [0.1, 10.0]$ (13 synthetic hidden
stakeholders, 5 seeds each) and fit predictive models in log space.

The oracle feature ([fig:degradation])—standard deviation of the residual from
regressing structural $u_{hidden}$ on *actual*
$u_{user}$ and $u_{platform}$—achieves near-perfect
prediction: $regret  35 
std(u_)^{1.8}$ ($R^2_{log} = 0.954$,
Spearman $= 0.962$). The *proxy* feature (weight-space cosine
similarity) achieves $R^2_{log} = 0.72$,
Spearman $= 0.69$—adequate for ranking but not magnitudes.

A key theoretical observation: within the structural utility family
($U = pos -   neg$), regressing
$u_{hidden}$ on structural $u_{user}$ and
$u_{platform}$ yields $R^2 = 1.0$ for *every* $$.
This is inevitable—three linear functions of two variables span a
2D space. The theoretical bound predicts zero regret, yet actual
regret ranges from 0.036 to 4.675. The gap arises from the
discrepancy between structural utilities and actual utility functions
(`compute\_user\_utility`, etc.), which include engagement
weighting and retention proxies beyond the simple pos/neg model.
Regressing structural $u_{hidden}$ on *actual* utilities
yields $R^2 = 0.957$–$0.994$—a small residual that generates all
observed regret.

[See PDF for table/figure]

**Practical implication.** Correlation features predict the
degradation *ranking* (which stakeholder is most dangerous to
miss) with Spearman $= 0.96$. Predicting *magnitude* requires
knowledge of the hidden utility's sensitivity, which is inherently
unobservable. The $0.23$ $R^2$ gap between oracle and proxy
quantifies the information cost of not observing the hidden
stakeholder—connecting directly to [sec:databudget]'s finding
that even 25 pairs dramatically reduces this gap.

### Scaling with $K$ Stakeholders

Does adding more observed stakeholders reduce the cost of hiding one?
We generate $K  \{3, 5, 7, 10\}$ synthetic stakeholders using a
factor model: 8 latent factors (engagement, safety, virality,
retention, passive consumption, link sharing, discovery, content
quality) over the 18-action space, with Dirichlet-sampled factor
loadings. Each configuration is evaluated with 10 random stakeholder
sets $$ 5 data seeds, using 101 diversity weights.

Average regret is roughly invariant with $K$ (0.189 at $K=3$, 0.171
at $K=5$, 0.182 at $K=10$)—a modest 10
that plateaus. The Pareto fraction (fraction of frontier points that
are Pareto-optimal) increases from 0.48 ($K=3$) to 0.76 ($K=10$),
confirming the dimensionality effect: higher-dimensional objectives
make dominance harder, preserving more operating points. A secondary
sweep over Dirichlet concentration shows that the correlation
structure between stakeholders matters $4$ more than $K$
itself: regret drops from 0.272 (diverse, concentration $= 0.5$) to
0.157 (similar, concentration $= 5.0$) at fixed $K=5$.


## Direction 2: Utility Sensitivity


### The $$-Dominance Test

Does the pos/neg ratio $$ dominate over individual action
weights? We compare two perturbation conditions, each applied to all
three stakeholders' weight vectors and evaluated via Hausdorff
distance between the perturbed and baseline Pareto frontiers:


  - **Within-group shuffle** ($$-preserving): randomly
    permute positive weights among positive actions and negative
    weights among negative actions, preserving
    $ = w_{neg} / w_{pos}$.
  - **$$-perturbation**: uniformly scale all negative
    weights by factors $\{0.5, 0.8, 1.2, 1.5, 2.0\}$, changing
    $$ while preserving within-group weight structure.


The $$-dominance ratio (within-group variance / between-group
variance) is 0.057—$$ perturbation causes $$18$$
more frontier variability than within-group shuffling. At the
*selection level* (perturbed weights drive content ranking, not
just evaluation), the ratio is 0.062. By these measures, $$
dominates decisively.

### The Matched-Magnitude Reversal

However, the above comparison is unfair: the $$-perturbation
condition ($ 0.5$–$2.0$) involves much larger absolute weight
changes than permutation. When we match perturbation magnitude
($ = 0.3$ budget, allocated either entirely to within-group
magnitude noise or entirely to $$ scaling), the result
*reverses*: within-group magnitude perturbation causes
$2$ more frontier shift than $$-only perturbation
(Hausdorff 4.12 vs 2.10, ratio 1.96). At matched budget, individual
weight magnitudes matter more than the group ratio.

### Rank Stability vs.\ Hausdorff Distance

This apparent contradiction is resolved by examining *what*
changes. We perturb each of the 14 `UtilityWeights`
parameters individually at factors $\{0.5, , 3.0\}$ and
measure both Hausdorff distance and rank stability (fraction of
Pareto-optimal operating points that survive).

The Hausdorff sensitivity ranking (normalized by baseline weight
magnitude) identifies engagement weights as most sensitive: platform
repost (6.6), user favorite (5.5), user repost (6.5). Negative
weights (report: 0.27, mute: 0.30) rank last. But rank stability is
$1.0$ for *every parameter at every perturbation level*—the
same 7 diversity-weight operating points remain Pareto-optimal even
under $3$ perturbation. The Hausdorff distance is detecting
utility-axis *rescaling* (linear in perturbation magnitude), not
frontier *shape change*. Individual parameter misspecification
does not change the optimal policy.

### Specification vs.\ Data


If individual parameters don't change the policy, when does
specification error matter? We compare two strategies for improving
frontier quality with misspecified weights ($ = 0.3$ Gaussian
noise on all weights, *not* preserving $$):


  - **Better specification** (fixed $N = 2{,}000$ pairs):
    reduce $$ from 0.5 to 0.0. Hausdorff decreases
    monotonically, as expected.
  - **More data** (fixed $ = 0.3$): increase $N$ from
    25 to 2{,}000. Hausdorff *increases* after $N = 100$ ([fig:goodhart])—from
    2.27 at $N = 100$ to 6.82 at $N = 500$.


[See PDF for table/figure]

This is a Goodhart effect [skalse2024goodhart]: with more
training data, the BT model more faithfully learns the *wrong*
utility function, producing a weight vector that is more precisely
misaligned with the true objective. With fewer pairs ($N  100$),
the model is undertrained and noise partially masks the
misspecification.

This finding creates a productive tension with
[sec:databudget]'s result that 25 pairs cuts regret by 42
The resolution: data is powerful when the specification is correct
(Direction 3), but *amplifies errors* when specification is
wrong (Direction 2). The practical prescription is sequential: fix the
specification first, then collect data.

### The Pareto Robustness Buffer

Synthesizing across all Direction 2 experiments ([tab:sensitivity]):

[See PDF for table/figure]

The frontier exhibits a *partial* robustness buffer: individual
parameter perturbation is absorbed (rank stability $= 1.0$), but
simultaneous misspecification of multiple parameters changes the
frontier shape. The Pareto structure is more robust than
single-objective RL (where any misspecification can be
Goodharted [skalse2024goodhart]), but less robust than the
$$-dominance result initially suggested.


## Nonlinear Robustness


All results above assume linear utility ($U = pos - 
 neg$). Real preferences may exhibit diminishing returns
(prospect theory) or dead zones (ignoring low-level negativity). We
test two nonlinear departures:


  - **Concave** (prospect theory): $U = pos^ -
      neg^$, with $  \{0.3, , 1.0\}$.
  - **Threshold** (dead zone): $U = pos - 
     (neg - , 0)$, with $  \{0.0, , 0.5\}$.


Four experiments across three utility families (linear, concave,
threshold):

**Exp A (Labels vs.\ Loss):** All mean cosine similarities
exceed 0.92 across families—the labels-not-loss claim generalizes
to nonlinear utilities.

**Exp B ($$-Recovery):** Both families preserve
Spearman $= 1.0$ for $$ rank-ordering. Nonlinear parameters
bias absolute $$ (concave inflates up to 57
deflates up to 40

**Exp C (Proxy Recovery):** The diversity knob is invariant
(0.724 recovery across all families). $$-interpolation degrades
under threshold utility ($0.738  0.499$) because the dead zone
creates preferences that the linear proxy cannot reproduce, but holds
or improves under concave.

**Exp D (Stress $$ Nonlinearity):** The same 4D stress
sweep from [sec:stress] under all three families. Concave
tightens the label-noise breaking point from $p = 0.30$ to $p = 0.20$
(practitioners using prospect-theoretic utilities should allow
$$15
thresholds—sample size, temperature, content correlation—are
unchanged across families. Full table in [app:stress].


## External Validation


### MovieLens-100K

To validate the BT training architecture on real preference data, we
train on MovieLens-100K movie ratings. The best configuration (BPR
loss with in-batch negatives, 64-dim embeddings, 4 transformer
layers) achieves Normalized Discounted Cumulative Gain (NDCG)@3 of
0.4112, a 59
ablation study reveals a 107.5
embeddings nor the transformer improves performance alone; all gain
comes from their interaction. Details in [app:movielens].

### Synthetic Twitter

We construct a 648-parameter synthetic Twitter environment (6
archetypes $$ 6 topics $$ 18 actions) with known ground
truth. All 5 verification suites pass: behavioral accuracy 100
(mean error $< 10^{-15}$), action differentiation 6/6 tests
(lurker–power user repost ratio 15.6$$), block causal effect
78
synthetic ground truth, confirming that observed multi-stakeholder
phenomena are properties of the data, not artifacts of the
training procedure. Details in [app:synthetic].


## Discussion


### Synthesis

The three research directions form a progressive arc:

**Direction 1 (Identifiability)** establishes what BT training
*can* recover: the preference ordering encoded in training labels,
the negativity-aversion parameter $$ (Spearman $= 1.0$), and a
disagreement-to-differentiation bound ($R^2 = 0.977$). Sun
et al.'s [sun2025rethinking] theoretical order-consistency result
predicts our Direction 1 findings; our contribution extends the
analysis to the multi-stakeholder setting and provides empirical stress
bounds.

**Direction 3 (Partial Observation)** quantifies what is
*lost* when a stakeholder is hidden. Society is $10$
costlier to miss than user, but even 25 preference pairs recover 42
of the lost utility—a VoI finding with immediate practical
implications.

**Direction 2 (Sensitivity)** reveals how *precisely* the
non-hidden stakeholders' utilities must be specified. The answer is
nuanced: individual parameters are forgiving (rank stability $= 1.0$),
but correlated specification errors compound, and more training data
amplifies misspecification rather than correcting it. This Goodhart
result [skalse2024goodhart] is the paper's deepest finding: it
resolves an apparent contradiction between "data helps" (Direction 3)
and "data hurts" (Direction 2) by showing that data is only as good
as the utility specification it learns from.

### Practical Guidance

For practitioners deploying multi-stakeholder recommendation systems,
our results suggest a sequential approach:


  - **Specify $$ values.** The pos/neg ratio is the
    dominant structural parameter. It can be estimated from 250
    preference pairs (Spearman $= 1.0$ under stress testing) or from
    pairwise disagreement rates without any training
    ([eq:disagreement]).
  - **Collect hidden-stakeholder data.** If a stakeholder
    (typically society) is unobservable, invest in collecting even a
    small sample of preference labels. 25 pairs cut regret by 42
    200 pairs reach the plateau.
  - **Calibrate engagement weights.** The top-3 sensitive
    parameters are engagement weights (repost, favorite), not negative
    weights. Calibrate these to within $$20
  - **Use the diversity knob as a baseline.** If none of the
    above is feasible, setting the diversity weight to $ = 0.7$
    recovers 70


### Implications for Platform Transparency

X's January 2026 algorithm release provides prediction targets (action
types) but not learned parameters—*structure without weights*.
Our analysis suggests this is *partially* sufficient for
multi-stakeholder audit: the ranking of which stakeholder is most
dangerous to miss is predictable from action-type classification and
pairwise disagreement rates (Spearman $= 0.96$). However, magnitude
prediction requires approximate weight disclosure, and the Goodhart
result shows that audits based on misspecified proxy utilities can be
worse than no audit at all if the proxies systematically amplify
specification errors.

For regulators implementing the EU Digital Services
Act [lasser2025designing], the practical recommendation is:
require platforms to disclose at minimum (1) the action space
(positive/negative classification), (2) approximate negativity-aversion
ratios per stakeholder, and (3) the engagement weights for the top-3
most sensitive actions. This would enable meaningful third-party LOSO
analysis at modest transparency cost.

### Limitations

**Synthetic data.** All multi-stakeholder findings depend on
synthetic data with known ground truth (600 users, 100 content items,
648-parameter Twitter simulation). MovieLens validates the BT
architecture on real data but not the stakeholder analysis. The
methodology (LOSO, $$-recovery, sensitivity analysis)
generalizes; the specific numbers are benchmark-specific.

**Three stakeholders.** Charter E tests $K > 3$ but with
synthetic factor-based stakeholders. Real stakeholder populations may
not follow Dirichlet factor loadings.

**Simplified society utility.** Society utility is operationalized
as $pos - 4  neg$, a structural proxy for
diversity minus polarization. The actual `compute\_society\_utility`
function measures topic diversity and cross-partisan exposure, which is
structurally different (set-level, not per-item). The Exp 5 $R^2 = 1.0$
collapse explicitly reveals this gap.

**Fixed engagement scorer.** Content selection uses a fixed
engagement formula ($fav + 0.8  repost + 0.5
 follow$), not the stakeholder-specific weight vectors.
In production, the scorer co-evolves with the utility specification.
Our Direction 2 selection-level test partially addresses this
($$-dominance holds at $0.062$), but a full co-evolution study
remains future work.

**Separable utility families.** The nonlinear robustness audit
tests concave and threshold utilities, both of which decompose
additively over actions. Non-separable interaction effects (e.g.,
repost $$ block) are not tested.


## Conclusion


We present a systematic empirical study of multi-stakeholder reward
modeling on X's open-source recommendation algorithm, addressing
identifiability, partial observation, and utility sensitivity through
87 training experiments, 20-seed value-of-information analysis, and
six sensitivity test conditions.

The central finding is that stakeholder differentiation is determined
by training labels, not loss functions—an empirical confirmation of
BT order consistency [sun2025rethinking] in the multi-stakeholder
setting. Building on this, we show that hiding the least-observable
stakeholder (society) costs $10$ more than hiding the
most-observable (user), but even 25 preference pairs recover 42
the lost utility. The Pareto frontier absorbs individual parameter
perturbation (rank stability $= 1.0$) but not correlated
specification errors, and additional training data amplifies
misspecified utilities rather than correcting them.

These findings have direct implications for platform transparency
under the EU Digital Services Act: structure-without-weights
disclosure enables degradation *ranking* but not magnitude
prediction, and proxy-based audits can amplify specification errors
through a Goodhart mechanism. The practical prescription is
sequential—fix the specification, then collect data—and the
diversity knob at $ = 0.7$ serves as a robust baseline when
neither is feasible.

All code and experiment protocols are publicly available. (
  https://github.com/kar-ganap/x-algorithm-enhancements)


---

# Appendix


## Nonlinear Robustness Full Tables


[See PDF for table/figure]

[See PDF for table/figure]

[See PDF for table/figure]

## Stress $$ Nonlinearity (Exp D)


[See PDF for table/figure]

Concave utility tightens the label-noise threshold from 0.30 to 0.20.
All other thresholds are invariant across utility families. Practitioners
using prospect-theoretic preferences should derate the linear noise
threshold by one level ($$15

## $K$-Stakeholder Scaling


[See PDF for table/figure]

[See PDF for table/figure]

Correlation structure (Dirichlet concentration) produces a $4$
larger effect on regret than the number of stakeholders.

## Disagreement Bound Derivation


The two-variable model ([eq:disagreement]) is fit via ordinary
least squares on 34 data points from two sweeps: one fixing Platform
($ = 0.3$) and sweeping a partner, another fixing User
($ = 1.0$). Label disagreement rate $d$ is computed on 50{,}000
preference pairs. Mean total margin $m$ is the average
$|U_i(c_1) - U_i(c_2)|$ on disagreed pairs. The univariate model
($d$ only) achieves $R^2 = 0.90$ but is reference-dependent; adding
margin resolves this. For the utility family
$U = pos -   neg$, mean margin decomposes as
$||  E[  disagreement]$,
cleanly separating parameter distance from content structure.

## MovieLens Training Details


Best configuration: BPR loss with in-batch negatives (batch size 32),
64-dimensional learned embeddings, 4 transformer layers,
learning rate $5  10^{-4}$ (halved from $10^{-3}$), weight
decay $10^{-4}$. Training stopped at epoch 9 (best validation
NDCG@3). The BCE baseline achieved val NDCG@3 of 0.3157 but test
NDCG@3 of only 0.2410 (severe overfitting due to score margins
$ 10^{-8}$). BPR + in-batch negatives produces margins $$0.1–0.2,
eliminating the generalization gap (val 0.4112, test 0.4183).

[See PDF for table/figure]

## Synthetic Twitter Verification


The synthetic environment defines 648 ground-truth parameters:
$6  6  18$ (archetypes $$ topics $$ actions).
Five verification suites test progressively deeper model properties:
(1) embedding probes (silhouette scores: topic 0.9999, archetype 0.369);
(2) behavioral tests (100
(3) action differentiation (6/6 tests pass, lurker–power user
repost ratio 15.6$$);
(4) block causal effect (78
pairs during training);
(5) archetype flip via history swapping (85.7

## Pluralistic Models and Causal Verification


Phase 2 tested four pluralistic approaches for multi-group reward
modeling: single BT (baseline), GMM clustering, soft assignment, and
two-stage (cluster-then-train). The two-stage approach achieved 100
cluster purity using topic$$action features, with overall
accuracy 99.4
gate (0.8), a limitation attributed to BT's scale invariance rather
than model failure.

Phase 3 tested causal intervention: replacing a user's engagement
history with a different archetype's history and measuring score
changes. Action-level interventions (block, follow) achieve 100
pass rate. History-level interventions achieve 50
improving to 86

## Per-Parameter Sensitivity


[See PDF for table/figure]

The LLM margin proxy experiment (Claude Haiku, 15 sweep points
$$ 200 pairs) achieves Spearman $ = 0.929$ between LLM-based
and analytic disagreement rates. LLM confidence is too compressed
(range [1.40, 1.66]) for precise margin estimation but preserves
rank-ordering. The go/no-go criterion was revised from $R^2$ to
Spearman during analysis: rank-ordering is the practically relevant
quantity for predicting which stakeholder pairs will be most
differentiated.