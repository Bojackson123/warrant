"""Running the live pipeline over the question list and writing down what came back.

The one place in this project that spends money, and the only place a recording is created. Two
stages, deliberately separable, because they cost different things and fail for different reasons:
recording the question vectors needs a local model and no credential, and recording the answers
needs a provider.

**The generation stage retrieves through the vectors the query stage just wrote, not through the
live embedder.** That is the arrangement the whole scheme rests on and it is worth stating plainly.
A recorded answer is keyed on the rendered prompt, and the prompt contains the retrieved control
text; if this recorded the answer against a ranking produced by a freshly computed vector, then a
different machine -- reading the recorded vector back -- could rank two near-equal chunks the other
way round and miss a fixture that was recorded correctly. Recording through the same vectors replay
will read makes the fixture self-consistent by construction rather than by coincidence of hardware.

**Re-running records nothing.** A question whose key is already on disk is skipped, and the query
vectors are left alone when they already cover exactly this question list under this pin. A
recorder that rewrote its own output would put a date change into every file on every run, which
turns the diff that a re-record is supposed to be into noise, and would spend a provider call to
produce a file identical to the one already committed.

**A miss is never caught here.** The recorder calls the raising path, not the degrading one. A
recorder that shrugged at a failed call would report success having written nothing, which is the
one outcome that cannot be noticed later -- the missing fixture looks exactly like a question
nobody thought to add.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import psycopg

from warrant.embedder_config import EmbedderConfig
from warrant.embedding import VECTOR_DTYPE, Encoder
from warrant.fixtures.client import ModelClient
from warrant.fixtures.disk import DirectoryFixtureStore, RecordedCall, write_fixture
from warrant.fixtures.queries import (
    QueryVectorError,
    RecordedQueries,
    RecordedQueryEncoder,
    read_query_vectors,
    write_query_vectors,
)
from warrant.fixtures.questions import Question, QuestionSet
from warrant.fixtures.request import ModelRequest
from warrant.prompt import render_prompt
from warrant.retrieval.search import retrieve
from warrant.tokenizer import TokenCounter

# The two subdirectories of the fixture root. Named here rather than assembled at each call site,
# because the separation between them -- text one side, binary the other -- is the layout's whole
# reason for existing and is not something a caller should be able to get subtly wrong.
GENERATION_DIRECTORY = "generation"
QUERIES_DIRECTORY = "queries"

# Called with the question just recorded and how far through the list it is.
RecordProgress = Callable[[Question, int, int], None]


@dataclass(frozen=True, slots=True)
class QueryReport:
    """What the query stage did."""

    questions: int
    dimensions: int

    # Count times dimensions times four, by arithmetic. The figure the layout's storage note is
    # written against, reported so the note can be checked rather than believed.
    vector_bytes: int

    # Whether this run re-embedded anything. False means the recorded vectors already covered
    # exactly this question list under this pin, which is the idempotence property being claimed.
    rewritten: bool


@dataclass(frozen=True, slots=True)
class GenerationReport:
    """What the generation stage did, in the numbers worth checking afterwards."""

    recorded: tuple[str, ...]
    skipped: tuple[str, ...]

    # Recordings on disk that no current question keys to. Named rather than deleted: a recording
    # is a paid artefact, and removing one is a decision somebody makes in the change that explains
    # why the question went away.
    orphans: tuple[str, ...]

    prompt_tokens: int
    completion_tokens: int


def record_query_vectors(
    root: Path,
    questions: QuestionSet,
    encoder: Encoder,
    config: EmbedderConfig,
    today: date,
    force: bool = False,
) -> QueryReport:
    """Embed every question and store the vectors, unless they are already stored.

    Embedded one at a time through `embed_query`, which is what applies the pinned model's query
    prefix and what avoids padding a question against a longer neighbour. A question embedded here
    has to be the same vector as the same question embedded on the request path, and batching is
    the easy way for that to stop being true.
    """
    root = root / QUERIES_DIRECTORY
    texts = questions.texts

    if not force and _already_recorded(root, texts, config):
        return QueryReport(
            questions=len(texts),
            dimensions=config.dimensions,
            vector_bytes=_vector_bytes(len(texts), config.dimensions),
            rewritten=False,
        )

    vectors = np.vstack([encoder.embed_query(text) for text in texts])

    write_query_vectors(root, texts, vectors, config, today)

    return QueryReport(
        questions=len(texts),
        dimensions=config.dimensions,
        vector_bytes=_vector_bytes(len(texts), config.dimensions),
        rewritten=True,
    )


def record_generations(
    conn: psycopg.Connection,
    root: Path,
    questions: QuestionSet,
    recorded: RecordedQueries,
    encoder: Encoder,
    client: ModelClient,
    tokenizer: TokenCounter,
    today: date,
    config: EmbedderConfig | None = None,
    force: bool = False,
    progress: RecordProgress | None = None,
) -> GenerationReport:
    """Ask the provider each unrecorded question and write down what it said.

    `encoder` is the live model and is only reached for a question the query stage did not record,
    which under a consistent question list is none of them. It is required rather than optional so
    that this cannot silently fall back to embedding nothing.
    """
    store = DirectoryFixtureStore(root / GENERATION_DIRECTORY, "generation")
    replayed = RecordedQueryEncoder(recorded.vectors, encoder)

    existing = set(store.keys())
    keys: set[str] = set()

    written: list[str] = []
    skipped: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    for position, question in enumerate(questions.questions, start=1):
        retrieval = retrieve(conn, question.text, replayed, config=config)
        prompt = render_prompt(retrieval)
        request = ModelRequest.build("generation", prompt)

        keys.add(request.key)

        if request.key in existing and not force:
            skipped.append(question.id)
            continue

        # The raising path. See the module docstring: a recorder that degraded on a failure would
        # report success having written nothing.
        completion = client.complete(request)

        write_fixture(
            store.root,
            RecordedCall(
                key=request.key,
                purpose="generation",
                question=question.text,
                control_ids=retrieval.control_ids,
                model=completion.model,
                model_version=completion.model_version,
                recorded_on=today,
                usage_prompt_tokens=completion.usage.prompt_tokens,
                usage_completion_tokens=completion.usage.completion_tokens,
                counted_prompt_tokens=tokenizer.count(prompt),
                answer_lines=tuple(completion.answer.split("\n")),
            ),
        )

        written.append(question.id)
        prompt_tokens += completion.usage.prompt_tokens
        completion_tokens += completion.usage.completion_tokens

        if progress is not None:
            progress(question, position, len(questions.questions))

    return GenerationReport(
        recorded=tuple(written),
        skipped=tuple(skipped),
        orphans=tuple(sorted(existing - keys)),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _already_recorded(root: Path, texts: Sequence[str], config: EmbedderConfig) -> bool:
    """Whether the stored vectors already cover exactly these questions under this pin.

    A missing or unreadable store is not an error here, only an answer of no. The read is about to
    happen again for real, and refusing twice with two different messages helps nobody.
    """
    try:
        stored = read_query_vectors(root, config)
    except QueryVectorError:
        return False

    return set(stored.vectors) == set(texts)


def _vector_bytes(questions: int, dimensions: int) -> int:
    return questions * dimensions * np.dtype(VECTOR_DTYPE).itemsize
