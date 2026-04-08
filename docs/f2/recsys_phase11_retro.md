# RecSys Phase 11: Functional Form Fitting — Retro

**Goal**: Test whether Gao et al.'s α√d − βd functional form fits our N-parameterized multi-stakeholder Goodhart data.

**Status**: Complete. Gao's form does NOT map cleanly. BT convergence channel produces a different pattern than RLHF policy optimization.

---

## 1. What Worked

### The convergence form fits improving stakeholders excellently
Form 2 (U = U_0 + α√(1 - e^{-N/τ})) achieves R² = 0.93-0.99 for all improving stakeholders (platform, synthetic society). The τ parameter (convergence rate) is interpretable: ~160-190 for MovieLens, ~250-900 for synthetic. BT convergence saturation is real and measurable.

### The analysis cleanly distinguished our regime from RLHF
The key finding is negative: Gao's form doesn't fit Goodhart degradation in our setting. This is not a failure — it's a structural difference between BT preference learning and RLHF policy optimization.

## 2. Surprises

### Gao's form fits degradation with NEGATIVE coefficients
Form 1 (α√N - βN) achieves R² = 0.96-0.98 for degrading stakeholders — but α and β are both negative. The fitter found a monotonic decline (U = U_0 - |α|√N + |β|N), not the rise-then-fall that Gao predicts. The "learning" peak (N=25→50) is too small relative to the overall decline for the positive √N term to capture.

### The "learning phase" in BT on finite pools is tiny
In RLHF, the policy gradually explores a vast output space — the learning phase is long and the rise is pronounced. In BT on a finite content pool, the model quickly identifies the target direction (a few dozen pairs suffice for the 19-dim genre space), and then top-K selection sharply concentrates. The "sweet spot" N* ≈ 25-50 is just 1-2 data points at the start of our sweep.

### N is not the right analog of d
N (data quantity, an input) is not the same as d_KL (distance traveled, an output). They're related through the BT convergence rate d(N) ≈ 1 - exp(-N/τ). Naively substituting N for d in Gao's form ignores this nonlinear mapping. The convergence form (Form 2) accounts for it, but even that can't capture the Goodhart dynamics for degrading stakeholders because the "learning" phase is too brief.

## 3. Implicit Assumptions Made Explicit

- **Gao's √d term assumes a prolonged learning phase**: The square-root learning rate reflects diminishing returns from exploring policy space. In BT on a finite pool, the "exploration" is over a 19-dim weight vector, which converges in O(50) pairs — much faster than RLHF policy exploration.
- **The Goodhart degradation in our setting is nearly monotonic**: After the first 25-50 pairs, every additional pair makes things worse for anti-correlated hidden stakeholders. There's no extended "safe zone" where more data helps. This is a stronger (and more concerning) form of Goodhart than Gao's gradual rise-then-fall.
- **7 data points is marginal for 3-4 parameter fits**: We should be honest about this. The R² values are high, but with 7 points and 3+ parameters, any smooth function will fit well. The interpretability of the coefficients is more valuable than the R² values.

## 4. What This Means for the Paper

The paper should NOT claim that Gao's functional form applies to BT preference learning. Instead:

- **The Goodhart direction condition (cos < 0, Phase 10) is the primary contribution** — it predicts WHEN Goodhart occurs.
- **The functional form is a secondary finding** — it characterizes HOW the degradation progresses (nearly monotonic after a brief sweet spot, with convergence saturation).
- **The difference from Gao is itself a finding**: BT preference learning on finite content pools exhibits "sharp Goodhart" (rapid onset, nearly monotonic) rather than Gao's "gradual Goodhart" (extended learning phase, slow decline). This is because BT convergence in low-dimensional weight space is much faster than RLHF policy optimization in high-dimensional output space.

## 5. Metrics

| Metric | Value |
|--------|-------|
| Degrading stakeholders fitted | 2 (ML-100K diversity, ML-1M diversity) |
| Improving stakeholders fitted | 4 (ML-100K/1M platform, synthetic platform/society) |
| Form 1 (Gao) R² on degrading | 0.961-0.978 (but negative coefficients) |
| Form 2 (convergence) R² on improving | 0.931-0.995 |
| BT convergence rate τ | 160-310 (MovieLens), 250-900 (synthetic) |
| Files created | 2 (analysis script, retro) |
| Results | `results/functional_form_fit.json` |
