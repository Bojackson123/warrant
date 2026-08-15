# Warrant

Retrieval question answering over the NIST SP 800-53 catalog, where every claim in an
answer is tagged to the control that warrants it and the clause can be read in one click.

This README is a placeholder. Scope, non-goals, and the claims this project deliberately
does not make are written once the walking skeleton runs end to end.

## Development

Requires [`uv`](https://docs.astral.sh/uv/) and GNU `make`.

```
make sync       # create the environment from the lockfile
make lint       # ruff check and format check
make typecheck  # pyright
make test       # pytest
```

## Licence

Source code is licensed under the [Apache License 2.0](LICENSE).

The NIST SP 800-53 catalog vendored under `data/catalog/` is not covered by that
licence: it is a U.S. Government work, not subject to domestic copyright, and is
reproduced unmodified. [`NOTICE`](NOTICE) records what came from where, and
[`data/catalog/PINNED.md`](data/catalog/PINNED.md) records exactly which release.
