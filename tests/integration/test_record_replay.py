"""Record, then replay: the loop the whole fixture mechanism is for, over a real corpus.

Everything else about storage is checked without a database. What needs one is the part that
retrieves — because the retrieved control text is inside the rendered prompt, and the prompt is
what a recording is keyed on, so the question of whether a recording survives to be replayed is a
question about a `SELECT ... ORDER BY distance`.

The encoder here is a stub. What is under test is not what the pinned model means by a question;
it is that a recording made through one set of vectors is served back through the same ones, and
that a machine which would embed differently does not lose it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import psycopg
import pytest
from pgvector.psycopg import register_vector

from warrant.db.migrator import apply_migrations
from warrant.embedder_config import EmbedderConfig, get_embedder_config
from warrant.embedding import VECTOR_DTYPE, ProgressCallback
from warrant.fixtures.client import Completion, FixtureMissError, ReplayClient
from warrant.fixtures.disk import DirectoryFixtureStore
from warrant.fixtures.queries import RecordedQueryEncoder, read_query_vectors
from warrant.fixtures.questions import QuestionSet
from warrant.fixtures.recorder import (
    GENERATION_DIRECTORY,
    QUERIES_DIRECTORY,
    record_generations,
    record_query_vectors,
)
from warrant.fixtures.request import ModelRequest
from warrant.fixtures.store import Usage
from warrant.ingest.chunker import Chunk
from warrant.ingest.pipeline import ingest
from warrant.prompt import render_prompt
from warrant.retrieval.search import retrieve
from warrant.settings import Mode

pytestmark = pytest.mark.integration

TODAY = date(2026, 8, 17)

_CHUNKER_FINGERPRINT = "0" * 64
_CATALOG_SHA256 = "1" * 64

# Two questions, so that one recording can be shown not to answer for the other. Built through the
# validator rather than the constructor, so this goes in the way the committed file does.
QUESTIONS = QuestionSet.model_validate(
    {
        "version": "test-1",
        "questions": [
            {
                "id": "accounts",
                "class": "answerable",
                "text": "What happens to accounts nobody uses?",
                "because": "Exercises the answerable path.",
            },
            {
                "id": "passwords",
                "class": "prior_conflict_trap",
                "text": "How often must passwords be changed?",
                "because": "Exercises a second key over the same corpus.",
            },
        ],
    }
)


class DirectionalEncoder:
    """Vectors pointing along axes chosen by a keyword, with an optional rotation.

    Deliberately crude, and crude in the one way that matters: two texts sharing a keyword get
    vectors that are near each other, so a ranking over them is a ranking rather than an
    enumeration. The rotation is what stands in for a second machine.

    Its size is exaggerated -- the real difference between two machines is in the last bit of a
    float, and it matters only when two chunks are all but tied. Exaggerating it is what lets a
    test observe in one run what would otherwise show up as one flipped fixture a month later.
    """

    def __init__(self, dimensions: int, rotation: float = 0.0) -> None:
        self.dimensions = dimensions
        self._rotation = rotation
        self.embedded: list[str] = []

    def embed_documents(
        self,
        texts: Sequence[str],
        progress: ProgressCallback | None = None,
    ) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        self.embedded.append(text)

        return self._vector(text, rotate=True)

    def _vector(self, text: str, rotate: bool = False) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float64)
        lowered = text.lower()

        for axis, keyword in enumerate(("account", "password", "audit")):
            vector[axis] = 1.0 if keyword in lowered else 0.1

        if rotate and self._rotation:
            # Tips the balance between two chunks that are otherwise all but tied.
            vector[0], vector[1] = vector[1] + self._rotation, vector[0] + self._rotation

        return np.asarray(vector / np.linalg.norm(vector), dtype=VECTOR_DTYPE)


class StubClient:
    """A provider that answers every request the same way, and counts being asked."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    @property
    def mode(self) -> Mode:
        return "live"

    def complete(self, request: ModelRequest) -> Completion:
        self.requests.append(request)

        return Completion(
            answer=(
                "The catalog leaves the period to the organization [AC-2].\n"
                "\n"
                "It does not state a fixed number of days.\n"
            ),
            model=request.model,
            model_version=request.model_version,
            usage=Usage(prompt_tokens=1200, completion_tokens=32),
            source="live",
        )


class StubTokenizer:
    """Counts words. The real encoding has its own tests; this one only has to produce a number."""

    def count(self, text: str) -> int:
        return len(text.split())


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


@pytest.fixture
def config() -> EmbedderConfig:
    return get_embedder_config()


@pytest.fixture
def encoder(config: EmbedderConfig) -> DirectionalEncoder:
    return DirectionalEncoder(config.dimensions)


@pytest.fixture
def corpus(
    conn: psycopg.Connection, encoder: DirectionalEncoder, config: EmbedderConfig
) -> Iterator[psycopg.Connection]:
    apply_migrations(conn)
    register_vector(conn)
    ingest(conn, CHUNKS, encoder, config, _CHUNKER_FINGERPRINT, _CATALOG_SHA256)
    conn.commit()

    yield conn


@pytest.fixture
def recorded(
    tmp_path: Path,
    corpus: psycopg.Connection,
    encoder: DirectionalEncoder,
    config: EmbedderConfig,
) -> tuple[StubClient, DirectoryFixtureStore]:
    """One full recording pass: question vectors, then answers."""
    record_query_vectors(tmp_path, QUESTIONS, encoder, config, TODAY)

    client = StubClient()
    vectors = read_query_vectors(tmp_path / QUERIES_DIRECTORY, config)

    record_generations(
        corpus,
        tmp_path,
        QUESTIONS,
        vectors,
        encoder,
        client,
        StubTokenizer(),
        TODAY,
        config=config,
    )

    return client, DirectoryFixtureStore(tmp_path / GENERATION_DIRECTORY, "generation")


def test_every_question_is_recorded_once(
    recorded: tuple[StubClient, DirectoryFixtureStore],
) -> None:
    """One provider call per question, and one file per call."""
    client, store = recorded

    assert len(client.requests) == len(QUESTIONS.questions)
    assert len(store.keys()) == len(QUESTIONS.questions)


def test_a_recorded_answer_replays_byte_for_byte(
    recorded: tuple[StubClient, DirectoryFixtureStore],
) -> None:
    """The loop closed: what the provider said is what replay serves, through the filesystem."""
    client, store = recorded

    replay = ReplayClient(store, verify=lambda: None)
    served = replay.complete(client.requests[0])

    assert served.answer == client.complete(client.requests[0]).answer
    assert served.source == "replay"
    assert served.recorded_on == TODAY


def test_a_paraphrased_question_does_not_reach_a_recorded_answer(
    corpus: psycopg.Connection,
    recorded: tuple[StubClient, DirectoryFixtureStore],
    encoder: DirectionalEncoder,
    config: EmbedderConfig,
) -> None:
    """The failure this whole project exists to detect, asserted through the real path.

    A question that reads like a recorded one retrieves real controls and has no recorded answer.
    Serving the neighbouring answer would be an answer whose warrant does not cover what was asked,
    rendered indistinguishably from one that does.
    """
    _, store = recorded

    retrieval = retrieve(corpus, "What becomes of unused accounts?", encoder, config=config)
    request = ModelRequest.build("generation", render_prompt(retrieval))

    with pytest.raises(FixtureMissError):
        ReplayClient(store, verify=lambda: None).complete(request)


def test_re_running_the_recorder_asks_for_nothing(
    tmp_path: Path,
    corpus: psycopg.Connection,
    recorded: tuple[StubClient, DirectoryFixtureStore],
    encoder: DirectionalEncoder,
    config: EmbedderConfig,
) -> None:
    """A recording that exists is left exactly as it is, so an ordinary run costs nothing."""
    _, store = recorded

    again = StubClient()
    report = record_generations(
        corpus,
        tmp_path,
        QUESTIONS,
        read_query_vectors(tmp_path / QUERIES_DIRECTORY, config),
        encoder,
        again,
        StubTokenizer(),
        TODAY,
        config=config,
    )

    assert again.requests == []
    assert report.recorded == ()
    assert len(report.skipped) == len(QUESTIONS.questions)
    assert report.orphans == ()


def test_forcing_re_records_every_answer_under_the_same_keys(
    tmp_path: Path,
    corpus: psycopg.Connection,
    recorded: tuple[StubClient, DirectoryFixtureStore],
    encoder: DirectionalEncoder,
    config: EmbedderConfig,
) -> None:
    """A scheduled re-record: the keys hold, the answers are renewed, and the diff is prose."""
    _, store = recorded
    before = store.keys()

    again = StubClient()
    report = record_generations(
        corpus,
        tmp_path,
        QUESTIONS,
        read_query_vectors(tmp_path / QUERIES_DIRECTORY, config),
        encoder,
        again,
        StubTokenizer(),
        date(2026, 9, 17),
        config=config,
        force=True,
    )

    assert len(again.requests) == len(QUESTIONS.questions)
    assert len(report.recorded) == len(QUESTIONS.questions)
    assert store.keys() == before

    served = store.get(before[0])
    assert served is not None
    assert served.recorded_on == date(2026, 9, 17)


def test_a_recording_no_question_keys_to_is_named_and_kept(
    tmp_path: Path,
    corpus: psycopg.Connection,
    recorded: tuple[StubClient, DirectoryFixtureStore],
    encoder: DirectionalEncoder,
    config: EmbedderConfig,
) -> None:
    """A dropped question leaves a paid artefact behind; removing it is somebody's decision."""
    _, store = recorded
    kept = QuestionSet(version=QUESTIONS.version, questions=QUESTIONS.questions[:1])

    report = record_generations(
        corpus,
        tmp_path,
        kept,
        read_query_vectors(tmp_path / QUERIES_DIRECTORY, config),
        encoder,
        StubClient(),
        StubTokenizer(),
        TODAY,
        config=config,
    )

    assert len(report.orphans) == 1
    assert len(store.keys()) == len(QUESTIONS.questions), "an orphan was deleted rather than named"


def test_a_recorded_question_retrieves_identically_on_a_machine_that_embeds_differently(
    tmp_path: Path,
    corpus: psycopg.Connection,
    recorded: tuple[StubClient, DirectoryFixtureStore],
    encoder: DirectionalEncoder,
    config: EmbedderConfig,
) -> None:
    """The criterion the recorded query vectors exist for, asserted through retrieval itself.

    Two things have to hold together, and either alone would be misleading. The perturbed encoder
    has to actually change what comes back — otherwise the test passes against a search that
    ignores its input. And the recorded question has to be untouched by it, which is what makes a
    fixture recorded on one machine replayable on another.
    """
    _, store = recorded
    question = QUESTIONS.questions[0].text

    here = retrieve(corpus, question, encoder, config=config)

    elsewhere_encoder = DirectionalEncoder(config.dimensions, rotation=0.05)
    elsewhere = retrieve(corpus, question, elsewhere_encoder, config=config)

    assert elsewhere.control_ids != here.control_ids, (
        "the perturbation changed nothing, so this test cannot show that the recording did"
    )

    replayed = RecordedQueryEncoder(
        read_query_vectors(tmp_path / QUERIES_DIRECTORY, config).vectors,
        elsewhere_encoder,
    )
    on_the_other_machine = retrieve(corpus, question, replayed, config=config)

    assert on_the_other_machine.control_ids == here.control_ids

    # And therefore the same prompt, the same key, and the recording still hits.
    request = ModelRequest.build("generation", render_prompt(on_the_other_machine))

    assert store.get(request.key) is not None


def test_a_freely_typed_question_still_reaches_the_live_model(
    tmp_path: Path,
    corpus: psycopg.Connection,
    recorded: tuple[StubClient, DirectoryFixtureStore],
    encoder: DirectionalEncoder,
    config: EmbedderConfig,
) -> None:
    """Delegation is the design: an unrecorded question gets real retrieval, not a recorded one."""
    live = DirectionalEncoder(config.dimensions)
    replayed = RecordedQueryEncoder(
        read_query_vectors(tmp_path / QUERIES_DIRECTORY, config).vectors,
        live,
    )

    result = retrieve(corpus, "how long are audit records kept?", replayed, config=config)

    assert live.embedded == ["how long are audit records kept?"]
    assert result.chunks
