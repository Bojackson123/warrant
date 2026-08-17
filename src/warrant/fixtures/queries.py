"""Recorded question vectors: the part of replay that is not about model answers at all.

**Why these are recorded, when the model that makes them runs locally and for free.** Embedding is
deterministic per machine and not across them. Floating-point results depend on the BLAS kernels
torch selects for the hardware it is on, so two machines embed one question into vectors that agree
to every decimal anybody cares about and differ in the last bit. That difference is enough to swap
two near-equal chunks in a ranking; the swapped chunks are inside the rendered prompt; the prompt
is what a recorded answer is keyed on. So a fixture recorded here misses there, for a reason
nothing in the failure would point at.

Recording the vectors closes it. A recorded question retrieves the same chunks in the same order on
every machine, and the nondeterminism is confined to questions a reviewer types freely -- where
there is no recording to miss and where it is genuinely harmless.

**Binary, in their own directory, and never inline in a text fixture.** Three kilobytes per vector
of base-64 or JSON numbers inside a file whose whole purpose is a readable diff would make the diff
useless. `.npy` keeps them out, and `.gitattributes` marks the file so git neither converts nor
tries to diff it.

**float32, matching what is stored everywhere else.** The plan offers float16 as an option for
size; it is refused here. A narrowed vector is a different vector and would retrieve a different
order, which is the exact failure this module exists to prevent -- and the saving is fifty
kilobytes.

**The provenance is checked, not just recorded.** Vectors made by a different embedder are as stale
as a corpus made by one, and the manifest cannot see it: the manifest checks that the *pin* has not
moved, and these were written by whatever was loaded on the day. So the index names the model that
produced them and the read refuses a disagreement, the way `retrieval.corpus_check` refuses a
corpus built by something else.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError

from warrant.embedder_config import EmbedderConfig, get_embedder_config
from warrant.embedding import VECTOR_DTYPE, Encoder, ProgressCallback

_INDEX_NAME = "index.json"
_VECTORS_NAME = "vectors.npy"


class QueryVectorError(Exception):
    """The recorded query vectors cannot be used by this build."""


class RecordedQuery(BaseModel):
    """One question and where its vector sits in the array beside this file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str

    # The row this question's vector occupies. Explicit rather than implied by position in the
    # list, so that a hand edit which reorders the file is a loud failure rather than a silent
    # reassignment of every vector to the wrong question.
    row: int


class QueryIndex(BaseModel):
    """What the recorded vectors are, and what produced them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedder_name: str
    embedder_revision: str
    dimensions: int

    recorded_on: date

    queries: tuple[RecordedQuery, ...]


@dataclass(frozen=True, slots=True)
class RecordedQueries:
    """The vectors as they are used: by question, with the provenance they came with."""

    index: QueryIndex
    vectors: Mapping[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.vectors)


def write_query_vectors(
    root: Path,
    questions: Sequence[str],
    vectors: np.ndarray,
    config: EmbedderConfig,
    recorded_on: date,
) -> None:
    """Record one vector per question, sorted by question text.

    Sorted so that adding a question changes one row of the index and appends nothing anybody has
    to read, rather than renumbering every row below it. The rows follow the same order, so the
    array and the index move together.
    """
    if len(questions) != len(vectors):
        raise QueryVectorError(
            f"{len(questions)} questions and {len(vectors)} vectors were handed to the recorder. "
            "Writing these would file some vector against some other question's text, which is "
            "not a failure anything downstream could notice."
        )

    if len(set(questions)) != len(questions):
        raise QueryVectorError(
            "The same question appears twice in the list being recorded. A question is the "
            "identity of its vector here, so two of them is one recording and one silent loss."
        )

    if vectors.ndim != 2 or vectors.shape[1] != config.dimensions:
        raise QueryVectorError(
            f"The vectors are shaped {vectors.shape} and the pinned model {config.name} produces "
            f"{config.dimensions} dimensions. Recording these would store query vectors that "
            "cannot be compared against the corpus."
        )

    order = sorted(range(len(questions)), key=lambda row: questions[row])

    index = QueryIndex(
        embedder_name=config.name,
        embedder_revision=config.revision,
        dimensions=config.dimensions,
        recorded_on=recorded_on,
        queries=tuple(
            RecordedQuery(question=questions[source], row=row) for row, source in enumerate(order)
        ),
    )

    root.mkdir(parents=True, exist_ok=True)

    # Contiguous and explicitly typed. `np.save` records the dtype in the header, so a caller
    # handing this float64 would round-trip perfectly and store vectors twice the width the corpus
    # holds -- correct on the way out and wrong at every comparison.
    ordered = np.ascontiguousarray(vectors[order], dtype=VECTOR_DTYPE)

    with (root / _VECTORS_NAME).open("wb") as stream:
        np.save(stream, ordered, allow_pickle=False)

    # `ensure_ascii=False` for the reason a recording is written that way: the index holds question
    # text, and a question that quotes the catalog holds characters no reviewer should have to read
    # as escapes.
    (root / _INDEX_NAME).write_text(
        json.dumps(index.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_query_vectors(root: Path, config: EmbedderConfig | None = None) -> RecordedQueries:
    """Read the recorded vectors, refusing any that a different model produced.

    The config is a parameter for the reason `verify_corpus`'s is: it is how a test presents a
    deliberately mismatched pin without editing the file the whole project reads.
    """
    pinned = config if config is not None else get_embedder_config()

    index_path = root / _INDEX_NAME
    vectors_path = root / _VECTORS_NAME

    if not index_path.is_file() or not vectors_path.is_file():
        raise QueryVectorError(
            f"No recorded query vectors in {root}. Replay retrieves recorded questions through "
            "the vectors they were recorded with, so without these a recorded answer would be "
            "keyed on a prompt this machine might not reproduce. `make record-queries` writes "
            "them and needs no API key."
        )

    try:
        index = QueryIndex.model_validate(json.loads(index_path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError) as error:
        raise QueryVectorError(f"{index_path} is not a query vector index: {error}") from error

    _verify_pin(index, pinned)

    # `np.load` reports a truncated or non-`.npy` file as a bare `ValueError`. Carried here as the
    # module's own error so that a caller which treats an unreadable store as an answer of no --
    # the recorder does, before re-embedding -- catches it along with everything else this read
    # refuses, rather than only the failures that happen to be raised by hand.
    try:
        with vectors_path.open("rb") as stream:
            stored = np.load(stream, allow_pickle=False)
    except ValueError as error:
        raise QueryVectorError(
            f"{vectors_path} is not a readable array of vectors: {error}"
        ) from error

    if stored.dtype != VECTOR_DTYPE:
        raise QueryVectorError(
            f"{vectors_path} holds {stored.dtype} vectors and the corpus is stored at "
            f"{np.dtype(VECTOR_DTYPE)}. Comparing the two would rank the corpus against numbers "
            "that are not the ones the recording was made with."
        )

    if stored.shape != (len(index.queries), index.dimensions):
        raise QueryVectorError(
            f"{vectors_path} holds an array shaped {stored.shape} and {index_path} describes "
            f"{len(index.queries)} questions at {index.dimensions} dimensions. The two are "
            "written together, so one of them has been edited by hand."
        )

    rows = {query.row for query in index.queries}

    if rows != set(range(len(index.queries))):
        raise QueryVectorError(
            f"{index_path} does not assign each question a distinct row. Every vector would still "
            "load; some questions would simply be answered with another question's vector."
        )

    questions = [query.question for query in index.queries]

    if len(set(questions)) != len(questions):
        raise QueryVectorError(
            f"{index_path} lists the same question against more than one row. A question is the "
            "identity of its vector here, so the entries would collapse into one on the way in -- "
            "leaving a vector nothing can reach and a count that disagrees with the file."
        )

    return RecordedQueries(
        index=index,
        vectors={query.question: stored[query.row] for query in index.queries},
    )


@dataclass(frozen=True, slots=True)
class RecordedQueryEncoder:
    """Recorded vectors for recorded questions, and the real model for everything else.

    An `Encoder`, so `retrieve` is unchanged by its existence -- the seam was already there, and
    widening the search path to take an optional vector would have put the choice at every call
    site instead of in one object.

    **The delegation is the point, not a fallback.** A reviewer typing their own question has no
    recording and should watch real retrieval run over the real corpus; that is the console's
    second state and it is the honest one. What recording the golden questions buys is that the
    machine-to-machine wobble cannot reach anything a fixture is keyed on.
    """

    recorded: Mapping[str, np.ndarray]
    live: Encoder

    @property
    def dimensions(self) -> int:
        return self.live.dimensions

    def embed_documents(
        self,
        texts: Sequence[str],
        progress: ProgressCallback | None = None,
    ) -> np.ndarray:
        """Straight through. Nothing records corpus vectors; the corpus is in the database."""
        return self.live.embed_documents(texts, progress)

    def embed_query(self, text: str) -> np.ndarray:
        """The recorded vector for a recorded question, exactly; otherwise the model's own.

        Matched on the question as written, with no normalisation of whitespace or case. A lookup
        that tidied its input would make two visibly different questions share one recording,
        which is the nearest-match behaviour this project refuses everywhere else.
        """
        stored = self.recorded.get(text)

        return self.live.embed_query(text) if stored is None else stored


def _verify_pin(index: QueryIndex, pinned: EmbedderConfig) -> None:
    """Refuse vectors that the pinned model did not produce."""
    if index.embedder_name != pinned.name or index.embedder_revision != pinned.revision:
        raise QueryVectorError(
            f"The recorded query vectors were produced by {index.embedder_name} at revision "
            f"{index.embedder_revision}, and the pinned model is {pinned.name} at "
            f"{pinned.revision}. Different vectors retrieve different control ids, and those ids "
            "sit inside the prompt every recorded answer is keyed on, so these are stale rather "
            "than merely old. Re-record them."
        )

    if index.dimensions != pinned.dimensions:
        raise QueryVectorError(
            f"The recorded query vectors are {index.dimensions}-dimensional and the pinned model "
            f"{pinned.name} produces {pinned.dimensions}. They cannot be compared against the "
            "corpus at all."
        )
