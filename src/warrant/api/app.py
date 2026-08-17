"""The FastAPI application: one endpoint, and the startup that makes it answerable.

**One endpoint, `POST /answer`.** A question in, an answer with its citations out -- or, when
nothing recorded matches and no key is set, the same retrieval with generation politely declined.
The second case is a `200`, because the retrieval ran and the clauses are real; only a malformed
request is an error status.

**Everything expensive is built once, at startup, and verified there.** The embedding weights, the
recorded query vectors, the tokenizer encoding and the catalog index are loaded into `AppResources`
before the first request, and the corpus is checked against the pins before any of it -- a database
built by a superseded model is caught here rather than by answers that cite the wrong control.
`get_model_client` picks the replay or live path from whether a key is present and verifies the
manifest as it does, so a machine with no key starts cleanly in replay mode. Loading the weights is
seconds of work that a per-request path would repeat on every question.

**The endpoint is a synchronous function on purpose.** Embedding a question is CPU-bound work in a
C extension, so Starlette running a sync path operation in a worker thread is exactly what the
request wants -- the same reason the connection pool underneath it is synchronous. An async endpoint
would either block the event loop on the encoder or have to hand the work back to a thread itself.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR

from warrant.api.pipeline import AppResources, answer_question
from warrant.api.schemas import (
    AnswerResponse,
    QuestionRequest,
    QuestionSetView,
    QuestionView,
)
from warrant.db import ConnectionSource, close_pool, connection, open_pool
from warrant.embedder_config import get_embedder_config
from warrant.embedding import load_embedder
from warrant.fixtures.client import get_model_client
from warrant.fixtures.disk import DirectoryFixtureStore
from warrant.fixtures.queries import read_query_vectors
from warrant.fixtures.questions import QuestionSet, QuestionSetError, get_question_set
from warrant.fixtures.recorder import GENERATION_DIRECTORY, QUERIES_DIRECTORY
from warrant.ingest.control_ids import get_control_index
from warrant.retrieval.corpus_check import verify_corpus
from warrant.retrieval.search import EmptyQuestionError, RetrievalError
from warrant.settings import get_settings
from warrant.tokenizer import load_tokenizer

_logger = logging.getLogger("warrant.api")


class JsonLogFormatter(logging.Formatter):
    """One log record as one JSON object, with our structured fields spread in.

    The fields a request attaches under `warrant` are lifted to the top level so a line reads as a
    flat object -- which is the shape an OpenTelemetry exporter would emit for the same event, so
    nothing consuming these logs has to change when spans arrive.
    """

    # The fields the record itself owns. A structured field that reuses one of these names is
    # re-homed rather than allowed to spread over the top -- a log call attaching `level` or `time`
    # must not overwrite the record's real level or timestamp in the emitted line and corrupt
    # whatever parses it.
    _RESERVED = frozenset({"time", "level", "logger", "event", "exception", "stack"})

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        fields = getattr(record, "warrant", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                payload[f"warrant_{key}" if key in self._RESERVED else key] = value

        # A record logged through `logger.exception(...)` or with `exc_info=True` carries its
        # traceback here; the base formatter appends it to the message text, but this one builds
        # the line itself and would drop it without this. `stack_info=True` is the same story for a
        # non-exception call site that asked to be located.
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload)


def _configure_logging() -> None:
    """Send this project's logs out as JSON lines, once.

    Idempotent, because the lifespan can run more than once in a process -- a test that opens the
    app twice should not stack a second handler and print every line twice. The handler is marked so
    a repeat call recognises its own work rather than counting handlers by type.
    """
    logger = logging.getLogger("warrant")

    if any(getattr(handler, "_warrant_json", False) for handler in logger.handlers):
        return

    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    handler._warrant_json = True  # type: ignore[attr-defined]

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # Not up to the root logger as well, which would print each line a second time under whatever
    # format happens to be configured there.
    logger.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the resources the endpoint answers from, and tear the pool down after.

    Synchronous work inside an async context manager, which is correct here: it runs once, before
    any request, and blocking the loop for the seconds it takes to load the weights is not blocking
    anything yet.
    """
    _configure_logging()

    settings = get_settings()
    config = get_embedder_config()

    open_pool()

    try:
        # Checked before the weights are loaded. A database built by a superseded model does not
        # become answerable by spending seconds loading an encoder, and the check names what moved.
        with connection() as conn:
            verify_corpus(conn, config)

        store = DirectoryFixtureStore(settings.fixtures_path / GENERATION_DIRECTORY, "generation")
        recorded = read_query_vectors(settings.fixtures_path / QUERIES_DIRECTORY, config)

        # Picks replay or live from whether a key is present, and verifies the manifest as it does.
        client = get_model_client(store)

        app.state.resources = AppResources(
            encoder=load_embedder(config),
            query_vectors=recorded.vectors,
            tokenizer=load_tokenizer(),
            client=client,
            index=get_control_index(),
            mode=client.mode,
        )

        yield
    finally:
        close_pool()


app = FastAPI(
    title="Warrant",
    summary="Retrieval question answering over NIST SP 800-53, with citations that can be checked.",
    lifespan=lifespan,
)


def get_connection_source() -> ConnectionSource:
    """The pool's borrow, handed to the request path.

    A dependency rather than a call inside the pipeline so a test can override it with a source that
    yields a database it controls. It returns the borrow itself, not an open connection: the
    pipeline decides how briefly to hold one, and it holds it for the retrieval alone rather than
    across the generation call, so a slow provider does not leave a pooled connection idle.
    """
    return connection


def get_resources(request: Request) -> AppResources:
    """The resources startup assembled. Overridden in tests to inject stubs."""
    return request.app.state.resources


def get_question_source() -> QuestionSet:
    """The recorded question list the picker offers.

    A dependency rather than a call in the route so a test can override it with a small list of its
    own, the way the connection and the resources are overridden. Reads from disk (cached), not
    from the database, so this endpoint answers before any corpus exists.

    A malformed list is the server's own state, not the caller's request -- a `500` whose detail is
    logged, like the retrieval failure the answer endpoint reports, rather than a traceback handed
    to a caller who cannot act on it. Caught here because the read happens here, before the route
    body runs.
    """
    try:
        return get_question_set()
    except QuestionSetError as error:
        _logger.exception("question_list_unavailable")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The question list is not readable. See the server logs.",
        ) from error


@app.post("/answer", response_model=AnswerResponse)
def answer(
    body: QuestionRequest,
    connect: Annotated[ConnectionSource, Depends(get_connection_source)],
    resources: Annotated[AppResources, Depends(get_resources)],
) -> AnswerResponse:
    """Answer one question, or return the retrieval and decline to generate.

    A fixture miss is not caught here: it has already become a declined response inside
    `answer_question`, which returns rather than raises. What is caught is retrieval refusing to
    run, and the two reasons it refuses are not the same fault. An empty or whitespace question is
    the client's to fix and is a `400`. An empty corpus, an encoder that disagrees with the pin, a
    `k` misconfigured below one are the server's own state -- a `500`, whose detail is logged for
    the operator rather than handed to a caller who cannot act on it.
    """
    try:
        return answer_question(resources, connect, body.question)
    except EmptyQuestionError as error:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except RetrievalError as error:
        _logger.exception("retrieval_unavailable")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The corpus is not answerable. See the server logs.",
        ) from error


@app.get("/questions", response_model=QuestionSetView)
def questions(
    question_set: Annotated[QuestionSet, Depends(get_question_source)],
) -> QuestionSetView:
    """The recorded question list, for a console to offer grouped by class.

    Provisional, and the response carries the `version` that says so. Reading the list -- and
    reporting a malformed one as a `500` -- is the dependency's job, so by here the set is valid and
    all that is left is to map it across the boundary.
    """
    return QuestionSetView(
        version=question_set.version,
        questions=tuple(
            QuestionView(
                id=question.id,
                question_class=question.question_class,
                text=question.text,
                because=question.because,
            )
            for question in question_set.questions
        ),
    )
