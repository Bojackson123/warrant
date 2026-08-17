"""The linear path, assembled once: a question and a connection in, an answer and its citations out.

The steps are the recorder's, in the same order and through the same functions, and that is not a
coincidence to be tidied away -- it is the property the replay path depends on. A recorded answer
is keyed on the rendered prompt, and the prompt holds the retrieved control text, so the request
path can only reach a recorded answer if it retrieves, renders and builds the request exactly as
the recorder did. Retrieving through the recorded query vectors is the load-bearing part: a
recorded question embedded fresh on this machine could rank two near-tied chunks the other way and
render a prompt that never matches the recording. `RecordedQueryEncoder` returns the recorded
vector for a recorded question and the live model's own for anything else, so a freely typed
question still gets real retrieval and, having no recording, degrades.

**The heavy resources are built once and handed in, never constructed here.** Loading the embedding
weights, reading the query vectors, caching the encoding and building the catalog index are startup
work; doing any of it per question would make every question pay for it. `AppResources` is what
startup assembles and what a test substitutes stubs into, which is why the orchestration takes it
as an argument rather than reaching for globals.

**A way to borrow a connection is the parameter, not a connection.** The request path borrows one
from the pool for the retrieval and returns it before it generates: in live mode generation reaches
the provider and blocks for seconds, and a connection held across that would sit idle-in-transaction
the whole time, capping concurrency at the pool size for no query being run. Passing the borrow
rather than an open connection is also what lets a test drive this against a container database
without the pool in the way -- it hands in a source that yields the connection it owns.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from warrant.api.citations import check_citations
from warrant.api.schemas import AnswerResponse, ChunkView, CitationView, DeclineView
from warrant.db import ConnectionSource
from warrant.embedding import Encoder
from warrant.fixtures.client import Completion, ModelClient, complete_or_decline
from warrant.fixtures.queries import RecordedQueryEncoder
from warrant.fixtures.request import ModelRequest
from warrant.ingest.control_ids import ControlIndex
from warrant.prompt import render_prompt
from warrant.retrieval.search import RetrievedChunk, retrieve
from warrant.settings import Mode
from warrant.tokenizer import TokenCounter

# One line per answered question, as structured fields rather than prose. The names are chosen to
# survive the arrival of OpenTelemetry spans: each is a span attribute in waiting, so the logging
# shape here does not have to be rewritten when tracing lands.
_logger = logging.getLogger("warrant.api")


@dataclass(frozen=True, slots=True)
class AppResources:
    """Everything the request path needs that outlives a single request.

    Assembled once at startup and immutable thereafter. Held as a plain dataclass rather than read
    from module globals so that the pieces a test wants to fake -- the encoder, the client -- are
    substituted by construction rather than by patching.
    """

    encoder: Encoder
    # The recorded question vectors, by question text. What makes a recorded question retrieve the
    # same chunks on every machine.
    query_vectors: Mapping[str, np.ndarray]
    tokenizer: TokenCounter
    client: ModelClient
    index: ControlIndex

    # Whether this process reaches a provider or serves recordings. Equal to `client.mode`; carried
    # here so the response can report it without the response layer knowing which client it holds.
    mode: Mode


def answer_question(
    resources: AppResources,
    connect: ConnectionSource,
    question: str,
) -> AnswerResponse:
    """Retrieve, render, generate-or-decline, and check the citations of whatever came back.

    Returns a response in one of two states. An answered question carries its text and citations; a
    question nothing recorded carries the same retrieval and a plain statement that generation
    needs a key. Both carry the retrieved chunks and the locally counted prompt size, because both
    retrieved and both rendered a prompt.
    """
    started = time.monotonic()

    # The recorded vector for a recorded question, the live model's own for anything freely typed.
    replayed = RecordedQueryEncoder(resources.query_vectors, resources.encoder)

    # Borrowed for the retrieval and returned before anything else runs. Everything past this line
    # -- rendering, the token count, the generation call that in live mode blocks on the provider --
    # needs no database, so the connection goes back to the pool the moment the scan is done rather
    # than being pinned idle across a multi-second answer.
    with connect() as conn:
        retrieval = retrieve(conn, question, replayed)

    prompt = render_prompt(retrieval)
    request = ModelRequest.build("generation", prompt)

    # Counted whichever way the request ends: a miss still rendered this prompt to compute the key
    # that missed, so the number a token gate reads exists even when no answer does.
    prompt_token_count = resources.tokenizer.count(prompt)

    chunks = tuple(_chunk_view(chunk) for chunk in retrieval.chunks)

    # The degrading path: a miss becomes a declined response, not an exception. The console and this
    # endpoint are exactly the callers that should degrade; anything measuring quality calls the
    # raising path instead.
    result = complete_or_decline(resources.client, request)

    if isinstance(result, Completion):
        citations = check_citations(result.answer, retrieval, resources.index)
        response = AnswerResponse(
            question=question,
            mode=resources.mode,
            prompt_token_count=prompt_token_count,
            answered=True,
            answer=result.answer,
            recorded_on=result.recorded_on,
            citations=citations,
            chunks=chunks,
            decline=None,
        )
    else:
        citations = ()
        response = AnswerResponse(
            question=question,
            mode=resources.mode,
            prompt_token_count=prompt_token_count,
            answered=False,
            answer=None,
            recorded_on=None,
            citations=citations,
            chunks=chunks,
            decline=DeclineView(reason=result.reason, detail=result.detail),
        )

    _log_answer(
        question=question,
        response=response,
        citations=citations,
        duration_ms=(time.monotonic() - started) * 1000,
    )

    return response


def _chunk_view(chunk: RetrievedChunk) -> ChunkView:
    """A retrieved chunk as it crosses the boundary, clause text and all.

    `model_validate` over the dataclass rather than a field-by-field copy, so a field added to
    `RetrievedChunk` that `ChunkView` also names is carried without a second edit here. The
    protection this buys runs one way: a field `ChunkView` names that the chunk stops providing --
    a rename or a removal on the retrieval side -- raises here rather than silently reaching the
    console as null. A field added to `RetrievedChunk` that `ChunkView` does not name is dropped,
    not raised: `from_attributes` reads only the fields the model declares, so `extra="forbid"`
    cannot see it. Surfacing such a field is a deliberate edit to `ChunkView`, which is where the
    boundary's shape is decided.
    """
    return ChunkView.model_validate(chunk, from_attributes=True)


def _log_answer(
    question: str,
    response: AnswerResponse,
    citations: tuple[CitationView, ...],
    duration_ms: float,
) -> None:
    """Emit one structured record for a served question.

    Attached under a single key so the JSON formatter can spread it into the log object without
    guessing which of the record's attributes are ours. The counts rather than the citation objects
    themselves: a log line is for noticing that invalid citations happened, and the response is
    where the detail already lives.
    """
    invalid = sum(1 for citation in citations if not citation.valid)

    _logger.info(
        "answered_question",
        extra={
            "warrant": {
                "question": question,
                "mode": response.mode,
                "retrieved": len(response.chunks),
                "answered": response.answered,
                "decline_reason": response.decline.reason if response.decline is not None else None,
                "citations_total": len(citations),
                "citations_invalid": invalid,
                "prompt_tokens": response.prompt_token_count,
                "duration_ms": round(duration_ms, 1),
            }
        },
    )
