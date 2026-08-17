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

make manifest   # check every pinned input against the record of what the stored corpus and
                # the recorded model calls were built from
make model      # fetch the pinned embedding weights — one of two steps that need a network
make tokenizer  # fetch the generation model's own tokenizer encoding — the other one
make ingest     # embed the catalog and build the corpus; migrates first, so an empty
                # database is fine. Runs offline, on CPU, and takes minutes.
make ask Q="how are inactive accounts disabled?"
                # ask the corpus a question and print the chunks it retrieves

make record-queries
                # embed each recorded question and store the vector replay retrieves it with.
                # Needs no API key and no database.
make record     # run the real pipeline over data/fixtures/questions.json and write down what
                # came back. The only command here that spends money.
make record-again
                # re-record every answer whether or not one is stored. The monthly cadence.
```

`make manifest` is the one that refuses. Everything stored here — the corpus vectors and the
recorded model answers — is a function of the catalog, the parameter resolver, the chunker,
the embedding model, the prompt template and the tokenizer, so changing one of those and
forgetting to rebuild what was made with it would otherwise be a green build over stale artefacts.
`data/manifest.json` records what each of those currently hashes to, and the check exits non-zero
naming the one that moved and what the move invalidates. Recording an intended change is
`make manifest-write`, a separate command on purpose: a check that repaired what it found would be
a record of nothing.

Not every entry costs the same to change, and the file says so per entry. Moving the prompt
template re-records the model answers and leaves the corpus alone. Moving the embedding model
invalidates everything, because different vectors retrieve different control identifiers and those
identifiers sit inside the prompt every recorded answer is keyed on. Moving the tokenizer is the
quiet one: it invalidates nothing at all and makes every prompt size ever reported wrong.

`make model` and `make tokenizer` run once per machine and are the only two commands here that
touch a network. Everything after them — embedding a question typed into the console, counting
what a prompt would cost — runs from the local cache with no network and no API key, which is the
property the whole thing is arranged around. `make ask` is where that is easiest to check: unplug
the machine, ask it something, and watch real retrieval answer. Both fetch commands prove the point
rather than assert it: each re-loads what it just fetched in a second process with every socket
refused, so a path that quietly reached out fails there instead of on somebody else's machine.

Re-running `make ingest` rebuilds the same corpus rather than a second copy of it, and prints a
fingerprint you can compare across runs to see that it did.

## Recorded answers

Asking a model costs money and returns something slightly different every time, so the answers to a
fixed list of questions are recorded once and replayed after that. That is what makes this runnable
with no API key and what makes a test over it deterministic. Everything is under `data/fixtures`,
in two directories that are kept apart on purpose: the answers are text, one file each, so a
re-record is a diff somebody can read; the question vectors are binary and never appear inline in
them.

A recording is filed under a SHA-256 of the **complete** request — the model, its pinned version,
the fully rendered prompt, and every sampling parameter. Never a subset. Editing one character of
the prompt template therefore produces a **miss**, not a stale answer served with stale token
counts, and **nothing is ever substituted**: a question that reads like a recorded one and is not
one has no recorded answer, and gets told so. Serving the nearest recorded answer would be an
answer whose warrant does not cover the question asked, which is exactly the failure this project
exists to detect.

The question vectors are recorded for a less obvious reason. Embedding is deterministic on a
machine and not across machines — floating-point results depend on the kernels torch picks for the
hardware — so two machines can embed one question into vectors that differ in the last bit, retrieve
two near-tied chunks the other way round, and produce a different prompt for the same question.
Recording the vectors confines that to questions typed freely, where there is nothing recorded to
miss. [`docs/decisions/fixtures.md`](docs/decisions/fixtures.md) works through the layout, the
storage arithmetic, and the **re-record cadence: monthly, and on any manifest change**, each one its
own pull request.

The question list in `data/fixtures/questions.json` is provisional and labelled as such. It groups
into the three classes a measured set will use — answerable, out of corpus, and questions whose
obvious answer the catalog contradicts — but it has not been sized or validated against anything.

## Licence

Source code is licensed under the [Apache License 2.0](LICENSE).

The NIST SP 800-53 catalog vendored under `data/catalog/` is not covered by that
licence: it is a U.S. Government work, not subject to domestic copyright, and is
reproduced unmodified. [`NOTICE`](NOTICE) records what came from where, and
[`data/catalog/PINNED.md`](data/catalog/PINNED.md) records exactly which release.
