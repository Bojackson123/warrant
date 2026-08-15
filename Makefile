# Entry points. Every recipe is a single command so that it behaves the same whether make
# hands it to sh or to cmd.exe, and CI calls these rather than repeating the flags.

.PHONY: sync lint format typecheck test db-up db-down migrate up

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

db-up:
	docker compose up -d db

# -v, so that "start over" means an empty database rather than one carrying whatever the last
# run left in it. Named separately from `up` for exactly that reason.
db-down:
	docker compose down -v

migrate:
	uv run python -m warrant.db

# Brings up everything the compose file defines. That is the database alone today; the API and
# the console join it once they exist.
up:
	docker compose up --build
