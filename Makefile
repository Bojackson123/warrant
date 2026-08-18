# Entry points. Every recipe is a single command so that it behaves the same whether make
# hands it to sh or to cmd.exe, and CI calls these rather than repeating the flags.

.PHONY: sync lint format typecheck test db-up db-down migrate manifest manifest-write catalog chunks model tokenizer ingest ask record record-queries record-again serve up corpus-snapshot console-install console console-test

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

# Recomputes every pinned input -- catalog, resolver, chunker, embedding model, prompt template
# and tokenizer -- and exits non-zero if one has moved without being re-recorded, naming which and
# what the move invalidates. Needs no database, no embedding model and no network. It does need the
# tokenizer encoding cached, because the only honest way to check what an encoding counts is to
# count something with it; `make tokenizer` fetches it once.
manifest:
	uv run python -m warrant.manifest_check

# Records an intended change to those inputs, as a one-line diff per entry. Deliberately separate
# from the check above and never reached from it: a check that repaired what it found would be a
# record of whatever was on disk when it ran. The diff belongs in the pull request explaining why.
manifest-write:
	uv run python -m warrant.manifest_check --write

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

# The only target that needs a network. Fetches the pinned embedding weights into the local cache
# once, so that everything after it runs offline -- a model that downloads itself on first use is
# what would break the promise that this works with no key and no connection.
model:
	uv run python -m warrant.embedding

# The other target that needs a network, and the last one. Fetches the generation model's own
# tokenizer encoding into the local cache, then re-loads it in a second process with every socket
# refused -- so what this proves is that counting a prompt afterwards reaches nothing, rather than
# that a download succeeded. Run once per machine, like `make model`.
tokenizer:
	uv run python -m warrant.tokenizer

# Embeds the catalog and writes the corpus. Migrates first, so this works against an empty
# database in one command. Needs both one-off fetches to have run -- `make model` for the weights it
# embeds with, and `make tokenizer` for the encoding the manifest check counts a sample with -- and
# no network of its own.
ingest:
	uv run python -m warrant.ingest

# Asks the corpus a question and prints the chunks it retrieves, with the k it used:
# `make ask Q="how are inactive accounts disabled?"`. Needs `make ingest` to have run, and needs
# no API key and no network -- which is the property this target exists to make observable rather
# than only asserted in a test.
ask:
	uv run python -m warrant.retrieval "$(Q)"

# Runs the real pipeline over data/fixtures/questions.json and writes what came back. The only
# command here that spends money, and the only one that creates a recording. Needs the corpus, the
# embedding weights, the tokenizer encoding and WARRANT_MODEL_API_KEY; refuses before any of that if
# a pinned input has moved. A recording that already exists is left alone, so re-running is free and
# produces no diff -- `make record-again` is what a scheduled re-record uses.
record:
	uv run python -m warrant.fixtures

# The half of the above that needs no API key and no database: embeds each question and stores the
# vector replay will retrieve it with. Separated because it genuinely costs nothing, and folding it
# into a command that needs a credential would make it look as though it did.
record-queries:
	uv run python -m warrant.fixtures --queries

# Re-records every answer whether or not one is already stored. This is the monthly cadence and the
# response to a manifest change: the keys have not moved, what the model says may have, and the diff
# is the whole point. Its own target rather than a flag people have to remember, because the
# difference between this and `make record` is a provider bill.
record-again:
	uv run python -m warrant.fixtures --force

# Serves the one endpoint under uvicorn. Needs the corpus, the embedding weights, the tokenizer
# encoding and the recorded query vectors -- and no API key: with none set it comes up in replay
# mode, which is the path a reviewer takes. Verifies the corpus and the manifest at startup and
# refuses to serve if either has moved.
serve:
	uv run python -m warrant.api

# Brings up the whole stack: the database with the corpus already restored into it, then the API
# serving the console. Replay mode with no key and no network after the images build -- the path a
# reviewer takes. `make db-down` (down -v) is how to start over from an empty volume.
up:
	docker compose up --build

# Dumps the corpus from the running database into the snapshot the db image bakes. This is the
# regeneration step, run when a pinned input the corpus is a function of moves -- the embedding
# model, the chunker, parameter resolution. It only dumps: the database has to hold the current
# corpus first, which means a fresh ingest, and doing that against the baked db image is fine
# because ingest overwrites the restored rows in place. The full sequence is in
# data/corpus/README.md. `--no-owner --no-privileges` so the dump restores cleanly under whatever
# superuser the fresh container's init runs as, rather than one named after this machine.
corpus-snapshot:
	docker compose exec -T db pg_dump -U warrant -d warrant --no-owner --no-privileges | gzip > data/corpus/corpus.sql.gz

# Installs the console's dependencies from its lockfile. Run once, like `make sync` for Python.
console-install:
	npm --prefix console ci

# Serves the console on a Vite dev server, which proxies /answer to the API on port 8000. Needs
# `make serve` running in another terminal; with no API key that server is in replay mode, which is
# the path a reviewer takes -- question in, answer with citations, click a citation, read the clause.
console:
	npm --prefix console run dev

# The console's own tests: the answer parser and the citation rendering, run under vitest with no
# API and no network. What `make test` is for Python.
console-test:
	npm --prefix console test
