# Development Process

Every phase of work follows a four-step lifecycle.

## PLAN → TEST → IMPLEMENT → RETRO

### 1. PLAN

Write a plan document before writing any code. The plan should cover:

- **Context:** Why this change is needed, what prompted it
- **Approach:** What you'll build, key design decisions
- **Files:** Which files will be created or modified
- **Verification:** How to confirm the change works

Plans are written in Claude Code's plan mode or as docs in `docs/fN/` (grouped by feature).

### 2. TEST

Write or update tests for the planned changes. Tests come before implementation:

- Unit tests in `tests/` matching the module structure
- Use pytest fixtures for shared setup
- Test the behavior, not the implementation

### 3. IMPLEMENT

Write the code. All quality gates must pass:

```bash
make all    # runs test + lint + typecheck
```

- `make test` — pytest must pass
- `make lint` — ruff must pass (enhancements/, scripts/, tests/)
- `make typecheck` — mypy must pass (enhancements/, scripts/)

Do not merge code that fails any gate.

### 4. RETRO

Write `docs/fN/retro.md` using this 6-section format:

1. **What Worked** — techniques, tools, approaches that paid off
2. **Surprises** — unexpected findings or behaviors
3. **Deviations from Plan** — what changed and why
4. **Implicit Assumptions Made Explicit** — things we assumed that turned out to matter
5. **Scope Changes for Next Phase** — what the next phase should account for
6. **Metrics** — files changed, tests added, key quantitative results

Also:
- Update `CLAUDE.md` if the current state changed
- Add gotchas discovered during implementation
- Update `tasks/lessons.md` if any corrections were received
- Update `docs/results.md` with results and metrics

## Branch Conventions

- Phase branches: `fN-phaseM-description` (e.g., `f2-phase3-causal-verification`)
- Branch off main, merge via PR
- No force pushes to main

## Commit Conventions

- `feat(fN):` — new feature or phase
- `fix(fN):` — bug fix
- `refactor(fN):` — restructuring without behavior change
- `test(fN):` — test additions or changes
- `docs(fN):` — documentation only
