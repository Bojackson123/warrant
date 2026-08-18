# Corpus snapshot

`corpus.sql.gz` is a `pg_dump` of the ingested corpus: the chunk rows, their embedding vectors,
and the provenance of what produced them. `Dockerfile.db` copies it into Postgres's
`/docker-entrypoint-initdb.d`, so `docker compose up` restores it on first init rather than
rebuilding it. This file is what makes the ninety-second, no-network bring-up possible, and it is
committed for the same reason the recorded query vectors and the recorded model answers are.

## Why a snapshot and not a rebuild

The vectors are deterministic — one pinned catalog, one pinned model, one chunker — so in principle
every machine could re-embed and arrive at the same corpus. Two things make that the wrong default:

- **Time.** Embedding roughly a thousand controls on CPU is minutes, and it is minutes on the
  reviewer's machine, spent before they see anything work. The whole property that survives from the
  prototype is that `docker compose up` reaches a working console quickly, with no account and no
  keys.

- **Determinism across machines, which is the load-bearing one.** Replay retrieves a recorded
  question through its recorded query vector, but against *these* corpus vectors. Float arithmetic
  differs in its last bits across CPUs and BLAS builds, so a corpus re-embedded on a different
  machine can rank two near-tied chunks the other way; the retrieved set then differs, the rendered
  prompt differs, and the recorded answer — keyed on that prompt — is not found. Replay would
  decline where it should answer. The snapshot is the exact corpus the recorded answers in
  `data/fixtures` were retrieved against, so every machine serves those same vectors.

## A stale snapshot fails loudly

The dump carries the provenance columns `corpus_ingest` records: the embedding model and its
revision, the vector width, the chunker version, and the parameter-resolution version. The API
compares those against the pins at startup (`warrant.retrieval.corpus_check`) and refuses to serve a
corpus built by anything other than what this build is pinned to. So forgetting to regenerate this
file after a pin moves is a refusal naming what moved, not a green build serving wrong citations.

## Regenerating it

Regenerate when a pinned input the corpus is a function of changes: the embedding model in
`data/embedder.json`, the chunker, or parameter resolution. The recorded query vectors and the
recorded answers are regenerated in the same breath — an embedder change invalidates all three, and
`data/manifest.json` is where that fan-out is written down.

From the running stack (the API image carries the weights and the tokenizer encoding, so this needs
no host caches and no network):

```
make up                                             # db restores the current snapshot; API comes up
docker compose run --rm api .venv/bin/python -m warrant.ingest
make corpus-snapshot                                # dump the freshly ingested corpus over this file
```

The ingest step re-embeds with the current pins and overwrites the restored rows in place, which is
why running it against the baked db image is fine: by the time `make corpus-snapshot` dumps, the
database holds the new corpus, not the one the image shipped. Rebuild the db image (`make up` builds
it) so the new snapshot is the one restored next time, and commit the changed `corpus.sql.gz`
alongside the regenerated fixtures.
