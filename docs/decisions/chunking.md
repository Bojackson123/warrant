# How the catalog is cut into chunks

*Status: decided. Superseding it means re-embedding the corpus, bumping `CHUNKER_VERSION`, and
re-running the embedding comparison, because the recall numbers in
[`embedder.md`](embedder.md) were measured under this chunking and describe no other.*

## The answer

**One chunk per live control and per live enhancement, following the catalog's own structure.**
Each chunk carries the control's title, its statement, and its guidance — nothing else.

```
AC-2(3) Disable Accounts          <- identifier and title, always first

Control:
Disable accounts within [organization-defined time period] when the accounts:
(a) Have expired;
(b) Are no longer associated with a user or individual;
...

Discussion:
Disabling expired, inactive, or otherwise anomalous accounts supports the concepts ...
```

| | |
|---|---|
| Chunks | **1,014** |
| Controls and enhancements in the catalog | 1,196 |
| Withdrawn, excluded | 182 |
| Assessment objectives and methods, excluded | 3,945 parts |
| Chunk length, median / p90 / p99 / max (characters) | 757 / 1,881 / 3,638 / 5,686 |
| Chunk length, median / p90 / p99 / max (tokens, `bge-base`) | 136 / 348 / 826 / 1,250 |

Implemented in [`src/warrant/ingest/chunker.py`](../../src/warrant/ingest/chunker.py). Run
`make chunks` to see the counts against the pin without a database or a model.

## Why granularity is a decision rather than a default

Chunking is where control identifiers are either preserved or lost.

Cut the catalog into character windows and every chunk needs its control identifier recovered
afterwards, by a heuristic reading the text. That heuristic is then load-bearing for the one
thing this project claims to do: a citation check can only be as trustworthy as the mapping from
chunk to control, and a mapping produced by pattern-matching prose is not falsifiable — when it
is wrong, the eval scores it as a retrieval failure or, worse, as a success.

Cutting on the boundaries the document already draws removes the problem instead of managing it.
`control_id` is *copied* from the source. A chunk cannot be about a control it is not labelled
with, and [`control-ids.md`](control-ids.md) settles which of the four published forms that
label is.

Enhancements are separate chunks rather than folded into their parent because they are separately
citable. AC-2 and AC-2(3) are different answers to different questions, and a corpus that could
only return the first would make every enhancement in the catalog uncitable. `base_control_id`
carries the parentage instead, taken from the document's own nesting.

## What is excluded, and why

### Assessment objectives and methods

Every control carries an `assessment-objective` part and, on average, three `assessment-method`
parts — SP 800-53A material shipped inside the same file. They are excluded: not folded into the
control's chunk, and not chunked separately.

They restate the control in assessment voice. Resolving every part of every live control produces
2,019,535 characters against the corpus's 1,007,525, so including them would double the text
while adding little vocabulary a question would match on — and it would push a much larger share
of the corpus past the embedding model's window than the 2.6% discussed below.

Chunking them separately was the other option and it is worse for the same reason enhancements
are chunked *together*: an assessment objective is not separately citable. Nobody asks a question
whose right answer is "the assessment objective of AC-2" rather than "AC-2", so separate chunks
would add 1,014 near-duplicates competing with the controls they restate.

### Withdrawn controls

182 of the 1,196 records. Nearly all are empty shells pointing at the control that absorbed them,
so a chunk for one is a document that cannot be a correct answer to any question but can still be
returned instead of one.

They remain **resolvable** — see [`control-ids.md`](control-ids.md) — which is deliberate and is
the other half of this decision. "You cited AC-2(10), which was withdrawn" and "you cited
something that does not exist" are different failures, and a corpus that excluded withdrawn
controls from retrieval *and* from resolution would make them indistinguishable.

## Long controls: the split rule

**There is no split.** A control is one chunk however long it is.

This is the explicit answer [`embedder.md`](embedder.md) asks the chunker for, and the reason it
is safe is the assembly order — which the chunker imposes rather than reads off the document, and
which the tests assert over the real corpus and over a control published the other way round:

1. the citation label and the control title,
2. `Control:` — the statement,
3. `Discussion:` — the guidance.

The pinned model's window is 512 tokens. 26 of the 1,014 chunks (2.6%) exceed it and are embedded
from their first 512 tokens; the longest chunk is 1,250. Because of the order above, what a cut
removes is always the tail of the discussion — never the identifier, never the title, never the
requirement. The loss is bounded, it falls on the least retrieval-bearing text in the chunk, and
the recall figures in `embedder.md` were measured *with* it, so they describe the shipped corpus
rather than an idealised one.

Two alternatives were considered and measured:

- **Split at the statement / guidance boundary.** Does not solve it. For the six longest
  controls — `pl-2`, `ca-2`, `ra-5`, `ac-16`, `pl-8`, `sa-8.24` — the discussion alone still
  exceeds the window. It would trade a 2.6% truncation for a smaller one, at the cost of a corpus
  that no longer matches what the embedder was chosen on.
- **Split recursively until every chunk fits.** Removes the truncation entirely and costs more
  than it is worth here. It needs a token budget at chunk time, which makes chunk text a function
  of the embedding model as well as the catalog — the chunker would have to load a tokenizer, and
  swapping the model would silently re-cut the corpus rather than merely re-embedding it.

Which chunks the pinned model truncates is reported by the ingest pipeline, because that is where
the tokenizer exists. The chunker does not guess at it.

Revisit this at the point where the corpus contains documents that are long *and* dense in
retrievable vocabulary throughout — a control set with substantive text past its first 512
tokens, rather than a discussion section that trails off.

## Whitespace

The chunker runs one document-wide whitespace pass over the assembled text: collapse runs of
spaces and tabs, remove a space before `;,.:`, strip trailing whitespace per line, then collapse
blank-line runs. The last two are in that order deliberately — a blank line holding a space is
still a blank line, and collapsing before the strip would not see one.

This lives here rather than in parameter resolution on purpose. The resolver tidies only the seam
around its own substitutions, because a module named for parameter resolution quietly rewriting
punctuation it did not produce is a diff nobody can explain two milestones later. A sweep over a
finished document is a chunking decision, it changes chunk text, and so it belongs under
`CHUNKER_VERSION`.

Measured on the pinned release, two of those rules do work: 63 controls carry a space before
punctuation the source itself wrote (`[AC-25](#ac-25) .`), and 5 carry a trailing space — one of
them inside a title, `SA-4(7) NIAP-approved Protection Profiles `. The rest fire on nothing and
are kept as cover for a release that varies.

The `control_label` and `title` columns go through the same pass as the text, so a row stores the
same strings the chunk's first line shows and a citation rendered from one does not inherit a
defect from the source that the chunk itself does not have.

## What `CHUNKER_VERSION` costs

`CHUNKER_VERSION` is `"1"`, it is stored on every row of `chunks`, and the fixture manifest reads
it.

Changing the chunker changes chunk text, which changes the vectors, which changes which control
ids come back from a search, and retrieved text is part of the prompt that recorded model calls
are keyed on. So a chunker change invalidates the corpus *and* every generation fixture
downstream of it. There is no partial re-record; the manifest detects the mismatch and cannot
repair it.

`chunker_fingerprint` is a SHA-256 over every field of every chunk, pinned in the tests, because a
version constant nobody bumps is a comment. An assembly change made without bumping the version
fails with two digests rather than shipping quietly.

## The claim this chunker can make, and how it is checked

The corpus the embedding comparison ranked models over was generated by
[`tools/bakeoff/prepare_corpus.py`](../../tools/bakeoff/prepare_corpus.py), which is
standard-library-only and frozen as evidence. The shipped chunker is a re-implementation of the
same rule against the typed objects, not an import of it, so the evidence stays fixed while the
chunker is free to change.

The two produce **byte-identical text across all 1,014 documents**, and a test asserts it. That
is what makes "the shipped corpus is the corpus the model was chosen on" a build failure rather
than a claim in a document.
