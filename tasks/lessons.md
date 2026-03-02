# Lessons Learned

Self-improvement log. After any correction from the user, record the pattern and a rule to prevent the same mistake. Review at session start.

## Rules

1. **Don't sell the work short.** When summarizing project status, include the intellectual journey — the wrong hypotheses, course corrections, and key insights — not just the final metrics. A 4-phase project that discovered "it's the labels, not the loss" through 87 experiments is fundamentally different from "99% accuracy, done." *Source: F4 status recap, user correction.*

2. **Verify claims against actual code/docs before stating them.** Stated Phase 3 trained 3 models (it trained 1). Stated BT baseline was not considered effective (it was the winner). Both were wrong and contradicted what we'd just read. When uncertain, re-read the source rather than relying on memory. *Source: F4 recap discussion, user caught both errors.*

3. **Don't contradict your own earlier findings.** Called the diversity knob "a simple slider that gets you most of the way there" after previously establishing that its hardcoded coefficients (favorite + 0.8*repost + 0.5*follow) have no empirical basis. Recognize when a new framing contradicts an earlier one and reconcile them. *Source: Pareto frontier discussion, user flagged the contradiction.*

4. **Read source docs thoroughly, not just agent summaries.** Agent summaries can miss or mischaracterize details. When accuracy matters (project status, phase history), read the actual files. *Source: multiple corrections in F4 recap.*
