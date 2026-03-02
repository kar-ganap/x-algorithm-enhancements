.PHONY: test test-all lint typecheck format all

# Run reward modeling + analysis tests (active development)
test:
	uv run pytest tests/test_reward_modeling/ tests/test_analysis/ -v

# Run full test suite including optimization tests
test-all:
	uv run pytest tests/ -v

lint:
	uv run ruff check enhancements/ scripts/ tests/

typecheck:
	uv run mypy enhancements/ scripts/

format:
	uv run ruff format enhancements/ scripts/ tests/

# Quality gate: all three must pass before merging
all: test lint typecheck
