# Entry points. Every recipe is a single command so that it behaves the same whether make
# hands it to sh or to cmd.exe, and CI calls these rather than repeating the flags.

.PHONY: sync lint format typecheck test db-up db-down migrate catalog chunks up

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

# Checks the vendored catalog against its pin and prints what it contains. Reads one file and
# touches nothing else, so it answers "is the corpus the one this build was written for?"
# without a database or a network.
catalog:
	uv run python -m warrant.ingest.catalog_report

# Reports what the chunker makes of that catalog: how many chunks, against how many live records
# the pin declares, and how long they are. Needs neither a database nor an embedding model, so it
# answers "what is in the corpus?" before there is anywhere to put it.
chunks:
	uv run python -m warrant.ingest.chunk_report

# Brings up everything the compose file defines. That is the database alone today; the API and
# the console join it once they exist.
up:
	docker compose up --build
