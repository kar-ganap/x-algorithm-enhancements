# RecSys Phase 6: Goodhart with Set-Level Metrics — Retro

**Goal**: Clean Goodhart test using true stakeholder utilities and genre entropy instead of Hausdorff distance.

**Status**: Complete. **Goodhart effect confirmed on MovieLens.** Diversity utility peaks at N=50 then degrades 42% by N=2000 while user utility monotonically improves.

---

## 1. What Worked

### The right metrics reveal a clear signal
Replacing Hausdorff distance with direct utility measurement eliminates both confounds (scorer coupling, weight scale). The signal is unambiguous:

| N | User utility | Diversity utility | Interpretation |
|---|---|---|---|
| 25 | 18.68 | -6.70 | Noisy model, suboptimal for both |
| 50 | 19.44 | **-6.34** (peak) | Sweet spot — learning helps diversity |
| 100 | 20.08 | -7.24 | Over-optimization begins |
| 2000 | 21.33 | -8.98 | Precise wrong optimization |

User utility ↑ by 14% while diversity utility ↓ by 42% from peak. This is textbook Goodhart.

### Strategy 1 validates the mechanism
Better specification (lower σ) at fixed N=2000 makes diversity WORSE (not better): -8.79 (σ=0.5) → -9.45 (σ=0.0). This confirms the mechanism: more precise optimization for user preference (even with correct specification) hurts the diversity stakeholder. The Goodhart effect in Strategy 2 is an amplified version of this fundamental tension.

### Genre entropy is a secondary signal
Entropy is relatively flat (3.05-3.29) because sparse binary features and top-10 limit the entropy range. The utility metrics capture the Goodhart signal more directly.

## 2. Surprises

### The Goodhart peak is at N=50, not N=100
Synthetic had the minimum Hausdorff at N=100. On MovieLens, the diversity peak is at N=50. The peak is earlier because genre features are sparser — BT needs fewer pairs to learn the (wrong) genre-level preference signal.

### Strategy 1 is "anti-monotonic" by design, not by error
In Phase 5/5b, Strategy 1 non-monotonicity was a confound. With utility metrics, it's an expected result: better specification means more precise user optimization, which by construction harms the opposed diversity stakeholder. This is not a measurement artifact — it's the user-diversity tension manifesting through the scorer.

### The paper's "fix the specification first" advice is nuanced
The paper says: "fix the specification, then collect data." On MovieLens, fixing the specification (σ → 0) makes user utility better but diversity utility worse. The advice applies only when the specification is CORRECTLY aligned with all stakeholders, not when it's aligned with one stakeholder at others' expense.

## 3. Deviations from Plan

### Entropy did not show Goodhart
Plan expected genre entropy to decrease with N. Actual result: entropy is flat (3.1-3.3). With 1.8 genres per movie and top-10 selection, the entropy is constrained by the data, not by the scorer. The Goodhart signal manifests in WHICH genres are selected (reflected in utility), not in HOW MANY genres (reflected in entropy).

## 4. Implicit Assumptions Made Explicit

- **The Goodhart effect on MovieLens is a UTILITY transfer, not a SET-LEVEL effect**: The misspecified scorer changes which movies are selected, transferring utility from diversity to user. The set composition (number of genres, entropy) barely changes because genre features are too sparse. The effect is in the CONTENT of the selection, not its STRUCTURE.
- **User-diversity opposition drives the Goodhart signal**: The cos = -0.34 between user and diversity means any optimization toward user preferences is anti-optimization toward diversity. More training data makes this anti-optimization more precise.
- **N=50 is the practical breakpoint**: Below N=50, the BT model is too noisy to effectively Goodhart. Above N=50, additional data amplifies the misspecification. For a practitioner, this means ~50 preference pairs is the threshold where specification quality starts to matter more than data quantity.

## 5. Scope Changes for Paper

The Goodhart finding can now be reported honestly on both datasets:
- **Synthetic**: Hausdorff-based, min at N=100, 3× amplification to N=500
- **MovieLens**: Utility-based, diversity peaks at N=50, 42% degradation to N=2000
- **Mechanism**: identical — more data with misspecified weights amplifies error on hidden stakeholders
- **Difference**: the metric is different (Hausdorff vs utility gap), which is a methodological improvement

## 6. Metrics

| Metric | Value |
|--------|-------|
| Files modified | 2 (script rewritten, tests updated with entropy) |
| New tests | 2 (genre entropy) |
| Total tests | 284 |
| Runtime | 1656s (~28 min) |
| **Strategy 2 key numbers** | |
| User utility N=25→2000 | 18.68 → 21.33 (+14%) |
| Diversity utility peak | N=50, -6.34 |
| Diversity utility N=2000 | -8.98 (42% worse than peak) |
| Goodhart detected (diversity) | **Yes** |
| Goodhart detected (entropy) | No (flat signal) |
| Results artifact | `results/movielens_goodhart.json` (overwritten) |
