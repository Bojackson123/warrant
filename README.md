# Warrant

Retrieval question answering over the NIST SP 800-53 catalog, where every claim in an
answer is tagged to the control that warrants it and the clause can be read in one click.

This README is a placeholder. Scope, non-goals, and the claims this project deliberately
does not make are written once the walking skeleton runs end to end.

## Development

Requires [`uv`](https://docs.astral.sh/uv/), GNU `make`, and Docker.

```
make sync       # create the environment from the lockfile
make lint       # ruff check and format check
make typecheck  # pyright
make test       # pytest — the integration tests start a real Postgres, so Docker must be running
make db-up      # start Postgres with pgvector
make migrate    # bring the database up to this build's schema
make db-down    # stop it and delete the volume

make model      # fetch the pinned embedding weights — the only step that needs a network
make ingest     # embed the catalog and build the corpus; migrates first, so an empty
                # database is fine. Runs offline, on CPU, and takes minutes.
```

`make model` runs once per machine. Everything after it — including embedding a question typed
into the console — runs from the local cache with no network and no API key, which is the
property the whole thing is arranged around. Re-running `make ingest` rebuilds the same corpus
rather than a second copy of it, and prints a fingerprint you can compare across runs to see that
it did.

## Licence

Source code is licensed under the [Apache License 2.0](LICENSE).

The NIST SP 800-53 catalog vendored under `data/catalog/` is not covered by that
licence: it is a U.S. Government work, not subject to domestic copyright, and is
reproduced unmodified. [`NOTICE`](NOTICE) records what came from where, and
[`data/catalog/PINNED.md`](data/catalog/PINNED.md) records exactly which release.
