# Quality gate — one call, named in AGENTS.md and called by CI.
.PHONY: check check-fast lint format format-check test test-fast

check: lint format-check test          ## full gate (CI calls this)
check-fast: lint format-check test-fast ## same gate, slow tests skipped (local)

lint:
	uv run ruff check src/ tests/ scripts/

format-check:
	uv run ruff format --check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/

test:
	uv run pytest tests/ -q

test-fast:
	uv run pytest tests/ -q -m "not slow"
