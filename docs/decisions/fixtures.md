# How recorded model calls are stored, and when they are re-recorded

*Status: decided. The layout can change; what cannot change quietly is the keying rule it stores
things under, because every recording on disk is filed by it.*

## The answer

Two directories under `data/fixtures`, and the separation between them is the whole design.

```
data/fixtures/
  questions.json              the list that gets recorded, grouped by class
  generation/<key>.json       one recorded answer per file — text, reviewable
  queries/index.json          which question owns which row
  queries/vectors.npy         the question vectors — binary, never inline in the above
```

`<key>` is the SHA-256 of the complete request: the model, its pinned version, the fully rendered
prompt, and every sampling parameter. Not a slug, not the question. A file is found by exactly that
number or it is not found, and there is no nearest match anywhere in the path.

| | |
|---|---|
| Recorded questions | 16, provisional |
| Committed query vectors | 16 × 768 × 4 bytes = **48 KiB** |
| Committed answers | ~2 KB each |
| Corpus vectors, for comparison | 1,014 × 768 × 4 bytes = **2.97 MiB**, in Postgres and not committed |

The corpus figure is worth writing down because an earlier estimate of it was wrong in an
instructive way. It was computed at 1,536 dimensions, which is a *hosted* embedder's width —
arithmetic left over from a model that had already been ruled out. At the width actually pinned it
is a third of that, and the committed part of it is nothing at all: the corpus lives in the
database, and what is committed here is sixteen question vectors.

---

## Why a re-record has to be readable, and what that costs

A recording is not a cache. It is the evidence that this system answered these questions this way,
on this date, with this model — and evidence nobody can read is not evidence. So the constraint
that decided the layout is: **what does `git diff` show when the answers are recorded again?**

Three consequences follow, and each of them is a choice that could easily have gone the other way.

**One file per recording rather than one file holding all of them.** A re-record then shows as
changed prose inside named files, and a question added or dropped shows as a file added or dropped.
Everything in one file makes those the same diff, and puts sixteen unrelated changes on adjacent
lines.

**The answer is stored as a list of lines.** This is the detail the whole requirement actually turns
on, and it is easy to miss. JSON escapes a newline into a two-character sequence, so an answer of
several paragraphs written as one string is *one enormous line* in the file — technically
satisfying "no vectors inline" while leaving the diff completely unreadable. Splitting on newlines
and joining on them is an exact inverse for every string, trailing newlines and embedded carriage
returns included, so this costs nothing in fidelity and buys a line-by-line diff.

Note what is *not* used: `str.splitlines`, which is the obvious reach and is not an inverse. It
drops the difference between text that ends in a newline and text that does not, and it treats a
lone carriage return as a break. Either would replay an answer the model did not give.

**Vectors are binary, in their own directory, and `.gitattributes` marks them.** Three kilobytes of
floats rendered as JSON numbers inside a file whose purpose is a readable diff would defeat the
purpose. The marking matters as much as the format: a line ending rewritten inside a float array is
a silently different vector, which retrieves a different control, which misses every answer
recorded against it.

---

## Why the question vectors are recorded at all

This is the least obvious part of the mechanism, because the embedding model runs locally, costs
nothing, and is pinned to a revision. Recording its output looks redundant.

It is not, and the reason is arithmetic rather than configuration. **Embedding is deterministic on
a machine and not across machines.** Floating-point results depend on the BLAS kernels torch selects
for the hardware it is on, so two machines embed one question into vectors that agree to every
decimal anybody cares about and differ in the last bit.

That difference is normally invisible. It stops being invisible when two chunks are nearly tied in
a ranking, because then it decides which comes first — and the retrieved chunks are inside the
rendered prompt, and the prompt is what a recording is keyed on. A fixture recorded on one machine
then misses on another, for a reason nothing in the failure points at: the question is the same,
the corpus is the same, the code is the same, and the key is different.

Recording the vectors closes it. A recorded question retrieves identically everywhere, and the
nondeterminism is confined to questions a reviewer types freely — where there is no recording to
miss and where it is genuinely harmless. Without this, the claim that replayed results have a noise
floor of zero is true only of the machine the recording was made on.

The delegation is the point rather than a fallback. `RecordedQueryEncoder` serves the stored vector
for a question it has, and asks the real model for everything else, so a reviewer typing their own
question still watches real retrieval run over the real corpus.

**The recorder retrieves through the vectors it has just written, not through the live model.** That
is what makes a recording self-consistent by construction rather than by coincidence of hardware:
the prompt the key was taken over is the prompt replay will render.

---

## Re-record cadence

**Monthly, and on any change to `data/manifest.json`.** Each re-record is its own pull request,
dated and model-versioned. `make record-again` is the command.

Stated as a rule rather than as "periodically", because a cadence nobody wrote down is a cadence
that becomes "whenever somebody noticed", and the age of a recording is exactly the thing a reader
needs to be able to judge. Every file carries the date it was written and the model version that
wrote it, so an eight-month-old recording is visibly eight months old.

`make record` records only what is missing, so an ordinary run costs nothing and produces no diff.
The distinction is a provider bill, which is why it is two commands and not a flag.

A re-record renews the answers and leaves the recorded question vectors alone. Those are rewritten
when what is stored no longer covers the current question list under the current embedder pin, and
otherwise only when `python -m warrant.fixtures --force-queries` asks for it outright. Folding them
into the monthly command would re-embed them on whichever machine happened to run it, moving their
last bits — which is the machine-to-machine wobble they exist to remove, and would orphan every
recording keyed through the ranking they produced.

### Not every manifest change costs the same

| Entry that moved | What has to be re-recorded |
|---|---|
| `prompt_template` | Every answer, and every grade over them. The corpus is untouched. |
| `judge_prompt` | Every grade. Answers are untouched — grading reads an answer that already exists. |
| `tokenizer` | Nothing. And that is why it needs an entry: it leaves every stored artefact valid and every reported prompt size wrong, with nothing left behind that looks wrong. |
| `catalog`, `parameter_resolution`, `chunker`, `embedder` | **Everything.** Corpus vectors, query vectors, every answer, every grade. |

The last row has no partial migration and is worth being blunt about. Different vectors retrieve
different control identifiers; those identifiers and their text sit inside the prompt every
recording is keyed on; so a changed embedder invalidates recordings that have nothing obviously to
do with embedding. The manifest detects the mismatch and cannot repair any of it.

---

## What is deliberately absent

**A manifest entry for `questions.json`.** Adding a question invalidates nothing. It simply has no
recording yet, which the recorder closes by recording it and which replay reports plainly in the
meantime. The manifest is for inputs whose change would otherwise go unnoticed, and this one cannot.

**Any deletion of a recording the recorder does not recognise.** A question removed from the list
leaves its recording behind, and the recorder names the file rather than removing it. Each one was
paid for, and dropping it belongs in the change that explains why the question went away.

**A measured question set.** What is committed is sixteen provisional questions across the three
classes the real set will use — answerable, out of corpus, and prior-conflict traps. The traps in
particular are reasoned guesses, and each one records its reasoning in a `because` field so that a
trap which turns out not to trap anything is a claim somebody can check. A trap admitted properly
needs a committed transcript of the ungrounded model answering it wrong, dated and model-versioned,
and that is not what is here.
