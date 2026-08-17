"""The endpoint, end to end, over a real corpus and real recordings.

What needs a database here is the same thing that needed one for the recorder: the retrieved
control text lives inside the rendered prompt, and the prompt is what a recording is keyed on, so
whether a question reaches its recorded answer is a question about a `SELECT ... ORDER BY distance`.
The rest -- citation validity, the two response states, the token count -- rides on top of that and
is asserted through the HTTP boundary rather than against the pipeline directly.

The encoder is the crude directional stub the record/replay tests use, not the pinned weights: what
is under test is the wiring and the citation check, not what the real model means by a question, and
a stub keeps the test off the half-gigabyte download. The catalog index, by contrast, is real -- the
citations have to resolve against the controls the catalog actually has.

The application's own lifespan is deliberately not run. `TestClient` is used without its context
manager, so startup -- which would open the pool against a database this test does not own and load
the real weights -- never fires; the resources and the connection are supplied through dependency
overrides instead.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient
from pgvector.psycopg import register_vector

from warrant.api.app import app, get_connection_source, get_resources
from warrant.api.pipeline import AppResources
from warrant.db import ConnectionSource
from warrant.db.migrator import apply_migrations
from warrant.embedder_config import EmbedderConfig, get_embedder_config
from warrant.embedding import VECTOR_DTYPE, ProgressCallback
from warrant.fixtures.client import Completion, ReplayClient
from warrant.fixtures.disk import DirectoryFixtureStore
from warrant.fixtures.queries import read_query_vectors
from warrant.fixtures.questions import QuestionSet
from warrant.fixtures.recorder import (
    GENERATION_DIRECTORY,
    QUERIES_DIRECTORY,
    record_generations,
    record_query_vectors,
)
from warrant.fixtures.store import Usage
from warrant.ingest.chunker import Chunk
from warrant.ingest.control_ids import get_control_index
from warrant.ingest.pipeline import ingest
from warrant.settings import Mode

pytestmark = pytest.mark.integration

TODAY = date(2026, 8, 17)

_CHUNKER_FINGERPRINT = "0" * 64
_CATALOG_SHA256 = "1" * 64

# One recorded question, answerable. Its recorded answer cites one control it retrieves and one it
# does not, so a single replay exercises both a valid citation and the invalid kind the retrieval
# check exists to catch.
QUESTIONS = QuestionSet.model_validate(
    {
        "version": "test-1",
        "questions": [
            {
                "id": "accounts",
                "class": "answerable",
                "text": "What happens to accounts nobody uses?",
                "because": "Exercises the answered state and both kinds of citation.",
            },
        ],
    }
)

# Real control ids, so the citation check resolves them against the catalog. `ac-2` is retrieved for
# an account question; `sc-28` is a genuine control that is not in this corpus at all.
CHUNKS = (
    Chunk(
        chunk_id="ac-2#a",
        control_id="ac-2",
        base_control_id="ac-2",
        control_label="AC-2",
        title="Account Management",
        part_path="a",
        text="Disable accounts when they have expired or are no longer required.",
    ),
    Chunk(
        chunk_id="ia-5#a",
        control_id="ia-5",
        base_control_id="ia-5",
        control_label="IA-5",
        title="Authenticator Management",
        part_path="a",
        text="Change password authenticators at an organization-defined frequency.",
    ),
    Chunk(
        chunk_id="au-11#a",
        control_id="au-11",
        base_control_id="au-11",
        control_label="AU-11",
        title="Audit Record Retention",
        part_path="a",
        text="Retain audit records for an organization-defined time period.",
    ),
)

# What the stub provider "said", recorded and then replayed. `[AC-2]` is retrieved and valid;
# `[SC-28]` exists in the catalog but was not retrieved, so it must come back flagged.
STUB_ANSWER = (
    "Accounts no longer required are disabled [AC-2].\n"
    "\n"
    "Stored records are protected separately [SC-28].\n"
)


class DirectionalEncoder:
    """Vectors pointing along axes chosen by a keyword. Crude on purpose; enough to rank."""

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.embedded: list[str] = []

    def embed_documents(
        self,
        texts: Sequence[str],
        progress: ProgressCallback | None = None,
    ) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        self.embedded.append(text)

        return self._vector(text)

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float64)
        lowered = text.lower()

        for axis, keyword in enumerate(("account", "password", "audit")):
            vector[axis] = 1.0 if keyword in lowered else 0.1

        return np.asarray(vector / np.linalg.norm(vector), dtype=VECTOR_DTYPE)


class StubClient:
    """A provider that answers every request with `STUB_ANSWER`."""

    @property
    def mode(self) -> Mode:
        return "live"

    def complete(self, request: object) -> Completion:
        return Completion(
            answer=STUB_ANSWER,
            model="stub",
            model_version="stub-1",
            usage=Usage(prompt_tokens=1200, completion_tokens=32),
            source="live",
        )


class StubTokenizer:
    """Counts words. The real encoding has its own tests; this only has to produce a number."""

    def count(self, text: str) -> int:
        return len(text.split())


@pytest.fixture
def config() -> EmbedderConfig:
    return get_embedder_config()


@pytest.fixture
def encoder(config: EmbedderConfig) -> DirectionalEncoder:
    return DirectionalEncoder(config.dimensions)


@pytest.fixture
def corpus(
    conn: psycopg.Connection, encoder: DirectionalEncoder, config: EmbedderConfig
) -> psycopg.Connection:
    apply_migrations(conn)
    register_vector(conn)
    ingest(conn, CHUNKS, encoder, config, _CHUNKER_FINGERPRINT, _CATALOG_SHA256)
    conn.commit()

    return conn


@pytest.fixture
def resources(
    tmp_path: Path,
    corpus: psycopg.Connection,
    encoder: DirectionalEncoder,
    config: EmbedderConfig,
) -> AppResources:
    """A full recording pass, then the resources the endpoint answers from.

    The query vectors and the generation are recorded here so the store the endpoint replays from is
    the real on-disk one, filled the way `make record` fills it -- not a mapping stood up by hand.
    """
    record_query_vectors(tmp_path, QUESTIONS, encoder, config, TODAY)
    vectors = read_query_vectors(tmp_path / QUERIES_DIRECTORY, config)

    record_generations(
        corpus,
        tmp_path,
        QUESTIONS,
        vectors,
        encoder,
        StubClient(),
        StubTokenizer(),
        TODAY,
        config=config,
    )

    store = DirectoryFixtureStore(tmp_path / GENERATION_DIRECTORY, "generation")

    return AppResources(
        encoder=encoder,
        query_vectors=vectors.vectors,
        tokenizer=StubTokenizer(),
        # `verify` stubbed: the manifest check re-chunks the real catalog and is exercised
        # elsewhere; this test is about what the client serves once it is past that gate.
        client=ReplayClient(store, verify=lambda: None),
        index=get_control_index(),
        mode="replay",
    )


@pytest.fixture
def client(corpus: psycopg.Connection, resources: AppResources) -> Iterator[TestClient]:
    """The app with its connection and resources overridden, and no lifespan run.

    Constructed without the context manager so startup never fires. The overrides are cleared after
    the test, because `dependency_overrides` lives on the shared app object.
    """

    @contextmanager
    def borrow() -> Iterator[psycopg.Connection]:
        # Lends the connection the `corpus` fixture owns; it does not close it, because the fixture
        # does, and the pipeline borrows one of these per question.
        yield corpus

    def use_corpus() -> ConnectionSource:
        return borrow

    app.dependency_overrides[get_connection_source] = use_corpus
    app.dependency_overrides[get_resources] = lambda: resources

    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_a_recorded_question_is_answered_with_checked_citations(client: TestClient) -> None:
    """The headline path: an answer, in replay mode, with citations and clause text, no key set."""
    response = client.post("/answer", json={"question": QUESTIONS.questions[0].text})

    assert response.status_code == 200

    body = response.json()
    assert body["answered"] is True
    assert body["mode"] == "replay"
    assert body["answer"] == STUB_ANSWER
    assert body["prompt_token_count"] > 0

    # The clause text is on the chunks, which is what a console needs to click a citation through
    # without a second request.
    assert body["chunks"]
    assert all(chunk["text"] for chunk in body["chunks"])

    citations = {citation["control_id"]: citation for citation in body["citations"]}
    assert citations["ac-2"]["valid"] is True


def test_a_cited_control_that_was_not_retrieved_is_flagged(client: TestClient) -> None:
    """The half that catches prior-knowledge citation, asserted through the endpoint.

    `SC-28` is a real control, so it exists; it is not in this corpus, so it was not retrieved --
    and the response has to say so rather than render it as a sound citation.
    """
    response = client.post("/answer", json={"question": QUESTIONS.questions[0].text})

    citations = {citation["control_id"]: citation for citation in response.json()["citations"]}

    assert citations["sc-28"]["exists"] is True
    assert citations["sc-28"]["retrieved"] is False
    assert citations["sc-28"]["valid"] is False


def test_an_unrecorded_question_degrades_rather_than_erroring(client: TestClient) -> None:
    """A freely typed question retrieves for real and declines to generate -- a 200, not a 500."""
    response = client.post("/answer", json={"question": "how long are audit records kept?"})

    assert response.status_code == 200

    body = response.json()
    assert body["answered"] is False
    assert body["answer"] is None
    assert body["decline"]["reason"] == "fixture_miss"

    # Retrieval still ran, and the prompt was still rendered and counted.
    assert body["chunks"]
    assert body["prompt_token_count"] > 0
    assert body["citations"] == []


def test_a_blank_question_is_a_client_error(client: TestClient) -> None:
    """Whitespace passes the length check and is refused by retrieval -- a 400, not a 500."""
    response = client.post("/answer", json={"question": "   "})

    assert response.status_code == 400


def test_an_empty_question_is_rejected_before_the_pipeline(client: TestClient) -> None:
    """An empty string fails request validation outright."""
    response = client.post("/answer", json={"question": ""})

    assert response.status_code == 422


def test_a_corpus_the_encoder_no_longer_matches_is_a_server_error(
    corpus: psycopg.Connection, resources: AppResources
) -> None:
    """A pin the encoder disagrees with is the server's own state, and the endpoint says so.

    Retrieval refuses a wrong-width encoder before it touches the database -- the same
    `RetrievalError` an un-ingested corpus raises. Neither is the caller's malformed request, so the
    endpoint answers `500`, not the `400` a blank question earns.
    """
    mismatched = replace(resources, encoder=DirectionalEncoder(resources.encoder.dimensions + 1))

    @contextmanager
    def borrow() -> Iterator[psycopg.Connection]:
        yield corpus

    app.dependency_overrides[get_connection_source] = lambda: borrow
    app.dependency_overrides[get_resources] = lambda: mismatched

    try:
        response = TestClient(app).post("/answer", json={"question": QUESTIONS.questions[0].text})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
