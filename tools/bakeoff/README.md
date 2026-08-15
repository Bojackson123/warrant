# Embedding model comparison inputs

**These are frozen inputs and evidence for one decision, not part of the running system.**
Nothing here is imported by the application, and the corpus produced here is superseded by
the real ingest pipeline. It is kept committed so the numbers in
[`docs/decisions/embedder.md`](../../docs/decisions/embedder.md) can be re-derived rather
than taken on trust.

The measurement itself is done by [`embed-bakeoff`](https://github.com/rashidalmarri/embed-bakeoff),
a separate project that ranks sentence-transformer models over an already-chunked labelled
corpus. It deliberately knows nothing about OSCAL or compliance, which is why the corpus, the
questions and the candidate configuration live here instead.

## Files

| File | |
|---|---|
| `prepare_corpus.py` | Catalog → `corpus.jsonl`. Standard library only. |
| `corpus.jsonl` | 1,014 documents, one per live control and enhancement. Generated. |
| `queries.jsonl` | Hand-written questions with their expected control ids. |
| `candidates.toml` | The models compared, with revisions pinned. |
| `report_lengths.py` | Token-length distribution and per-model truncation. |
| `lengths.json` | Its output. |
| `results.json`, `results-pinned.json` | Harness output, before and after pinning revisions. |

## Regenerating

```
python tools/bakeoff/prepare_corpus.py
```

Deterministic: same catalog in, byte-identical corpus out.

```
cd ../embed-bakeoff
uv sync --extra models

uv run embed-bakeoff validate \
  --corpus  ../warrant/tools/bakeoff/corpus.jsonl \
  --queries ../warrant/tools/bakeoff/queries.jsonl

uv run embed-bakeoff run \
  --config  ../warrant/tools/bakeoff/candidates.toml \
  --corpus  ../warrant/tools/bakeoff/corpus.jsonl \
  --queries ../warrant/tools/bakeoff/queries.jsonl \
  --output  ../warrant/tools/bakeoff/results-pinned.json

uv run --extra models python ../warrant/tools/bakeoff/report_lengths.py
```

## Choices made here, and why

**One document per control and per enhancement.** The document id is the control's own
catalog id, so `recall@k` means exactly "did retrieval find the right control" — the thing
the system is judged on — and the labels in `queries.jsonl` can be written and checked by
hand. Chunking is necessarily identical across candidates, which is the confound that would
otherwise make the comparison measure the chunker.

**Withdrawn controls are excluded** — 182 of the catalog's 1,196. Nearly all carry no prose,
only a link to the control that absorbed them. Embedding them would add documents that
cannot be the right answer to any question but can still be returned instead of one.

**Assessment objectives and methods are excluded.** The catalog carries SP 800-53A assessment
procedures inside every control, under `assessment-objective` and `assessment-method`. They
restate the control statement in assessment voice, roughly tripling the text while adding
little vocabulary a question would match on, and they would push most controls past a
512-token window. Whether the shipped chunker folds them in, splits them out, or drops them
is a separate decision; excluding them here keeps this comparison about the control text.

**Parameters are rendered as explicit slots.** Every one of the catalog's 1,600 parameters is
organisation-defined — none carries a concrete value — so `{{ insert: param, ac-02_odp.01 }}`
becomes `[organization-defined prerequisites and criteria]`. Leaving the braces in would put
template syntax into every vector and into any clause text a reader is shown.

## What `queries.jsonl` is not

Provisional scaffolding for choosing an embedding model. It has no provenance labelling, no
class balance, and no admission criterion for unanswerable questions. It is not an evaluation
set and must not be presented as one, or later reused as one.
