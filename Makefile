# Entry points. Every recipe is a single command so that it behaves the same whether make
# hands it to sh or to cmd.exe, and CI calls these rather than repeating the flags.

.PHONY: sync lint format typecheck test up

sync:
	uv sync

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format .

typecheck:
	uv run pyright

test:
	uv run pytest

# Placeholder until the Compose file exists; this target fails until then, on purpose.
up:
	docker compose up --build
