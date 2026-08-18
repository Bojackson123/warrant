# The image a reviewer runs: the API, the console it serves, and the two model caches baked in so
# the first run on a machine that has never seen this project needs no network and no key.
#
# Three things have to be true of the result, and each is a line below rather than a hope:
#   - the embedding weights and the tokenizer encoding are present before anything runs, fetched
#     here where a network is allowed rather than on first use where it is not;
#   - the console is built to static assets the API serves from its own origin;
#   - nothing dev-only -- no test runner, no type checker -- ships in it.

# ---- console: the React app, built to static assets ------------------------------------------
FROM node:22-bookworm-slim AS console

WORKDIR /console

# Dependencies first, from the lockfile, so the layer caches until the lockfile itself moves.
COPY console/package.json console/package-lock.json ./
RUN npm ci

COPY console/ ./
# `tsc --noEmit && vite build` -- the type check gates the build, the same as on a developer's
# machine, then Vite writes /console/dist.
RUN npm run build


# ---- app: python dependencies, the baked caches, and the source -----------------------------
# The uv image carries a pinned uv beside the interpreter; the tag fixes the Python minor to the
# one pyproject requires. Multi-arch, so this resolves the linux/arm64 or linux/amd64 wheels the
# lock already records for whichever engine builds it.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# UV_FROZEN: never touch the lock at build or run time -- an image that re-resolves is not the one
# the lock describes. UV_COMPILE_BYTECODE: pay the compile once here rather than on first import.
# UV_LINK_MODE=copy: the cache and the venv can be on different filesystems in a layered build, and
# uv's default hardlink warns and falls back across that boundary.
ENV UV_FROZEN=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1

# Baked into the image so both services -- ingest and the API -- inherit them without Compose
# repeating them, and so the build's own fetch below writes where the runtime will later read.
ENV WARRANT_MODEL_CACHE_DIR=/opt/warrant/model-cache \
    WARRANT_TOKENIZER_CACHE_DIR=/opt/warrant/tokenizer-cache \
    WARRANT_CONSOLE_DIST_PATH=/opt/warrant/console

WORKDIR /app

# Dependencies before source, so the heavy layer -- torch and the rest -- caches until the lock or
# the project metadata changes. `--no-install-project` stops uv from needing the source that has
# not been copied yet; the licence files are named in pyproject's metadata, so the build backend
# wants them present even for this metadata-only pass.
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
RUN uv sync --no-install-project

# The source and the data it reads: the catalog, the pins, the recorded fixtures replay serves
# from, the manifest ingest checks against.
COPY src ./src
COPY data ./data

# Install the project itself now that its source is here.
RUN uv sync

# The one place a network is used, and the reason the rest of the project can promise it does not:
# fetch the pinned embedding weights and the pinned tokenizer encoding into the baked cache dirs.
# Each command re-loads what it just fetched in a second process with the network refused, so a
# green build is proof the runtime will find these offline rather than proof a download succeeded.
RUN uv run python -m warrant.embedding \
 && uv run python -m warrant.tokenizer

# The built console, from the stage above.
COPY --from=console /console/dist /opt/warrant/console

# Documented, not published here: Compose decides what reaches the host, and binds this to
# loopback when it does.
EXPOSE 8000

# The interpreter from the venv directly, not `uv run`: uv would re-check the project against the
# lock and reinstall it on every container start, seconds of work for an image that is already
# exactly what the lock describes. Replay by default. The API reads its host, port and mode from the
# environment; Compose sets the host to 0.0.0.0 so the published port reaches it, and passes a key
# through only if one is set.
CMD [".venv/bin/python", "-m", "warrant.api"]
