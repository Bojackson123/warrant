"""Recorded question vectors: stored bit for bit, and read back in place of the model.

The property this file exists for is the one nothing else in the project can establish. Embedding
is deterministic on a machine and not across them, so the only way a recorded answer survives a
different CPU is if the vector its retrieval ran on came out of a file rather than out of the
model. That is asserted here directly, by perturbing the model and watching the recorded question
not move.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from warrant.embedder_config import EmbedderConfig, get_embedder_config
from warrant.embedding import VECTOR_DTYPE, ProgressCallback
from warrant.fixtures.queries import (
    QueryVectorError,
    RecordedQueryEncoder,
    read_query_vectors,
    write_query_vectors,
)

QUESTIONS = (
    "What happens to accounts nobody uses?",
    "How often must passwords be changed?",
    "Who approves a new account?",
)

TODAY = date(2026, 8, 17)


class StubEncoder:
    """An encoder whose vectors are a known function of the text, plus an optional nudge.

    The nudge is what stands in for a different machine. It is tiny -- far smaller than any
    difference that means anything about similarity -- and that is the point: the failure this
    guards against is not a wrong vector, it is a vector right to every decimal anyone cares about
    and different in the last bit.
    """

    def __init__(self, dimensions: int, nudge: float = 0.0) -> None:
        self.dimensions = dimensions
        self._nudge = nudge
        self.embedded: list[str] = []

    def embed_documents(
        self,
        texts: Sequence[str],
        progress: ProgressCallback | None = None,
    ) -> np.ndarray:
        return np.vstack([self.embed_query(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        self.embedded.append(text)

        generator = np.random.default_rng(abs(hash(text)) % (2**32))
        vector = generator.random(self.dimensions, dtype=np.float32)

        return np.asarray(vector + self._nudge, dtype=VECTOR_DTYPE)


@pytest.fixture
def config() -> EmbedderConfig:
    return get_embedder_config()


@pytest.fixture
def encoder(config: EmbedderConfig) -> StubEncoder:
    return StubEncoder(config.dimensions)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "queries"


def _record(root: Path, encoder: StubEncoder, config: EmbedderConfig) -> np.ndarray:
    vectors = np.vstack([encoder.embed_query(text) for text in QUESTIONS])
    write_query_vectors(root, QUESTIONS, vectors, config, TODAY)

    return vectors


def test_vectors_round_trip_bit_for_bit(
    root: Path, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """Compared as bytes, not as approximately equal numbers.

    `allclose` would pass on a store that silently narrowed to float16, which is precisely the
    saving this layout refuses -- a narrowed vector is a different vector and retrieves a different
    order.
    """
    written = _record(root, encoder, config)
    read = read_query_vectors(root, config)

    for row, question in enumerate(QUESTIONS):
        assert read.vectors[question].tobytes() == written[row].tobytes()


def test_the_stored_dtype_is_what_the_corpus_holds(
    root: Path, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """Stored at the width the corpus is stored at, so the two can be compared at all."""
    _record(root, encoder, config)

    assert read_query_vectors(root, config).vectors[QUESTIONS[0]].dtype == VECTOR_DTYPE


def test_a_wider_vector_is_narrowed_to_the_pinned_width(root: Path, config: EmbedderConfig) -> None:
    """A caller handing float64 gets float32 stored, rather than a file the corpus cannot match."""
    wide = np.ones((len(QUESTIONS), config.dimensions), dtype=np.float64)

    write_query_vectors(root, QUESTIONS, wide, config, TODAY)

    assert read_query_vectors(root, config).vectors[QUESTIONS[0]].dtype == VECTOR_DTYPE


def test_the_index_is_sorted_so_a_new_question_moves_little(
    root: Path, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """Rows follow sorted question text, so adding one does not renumber everything below it."""
    _record(root, encoder, config)

    index = read_query_vectors(root, config).index

    assert [query.question for query in index.queries] == sorted(QUESTIONS)
    assert [query.row for query in index.queries] == list(range(len(QUESTIONS)))


def test_the_recorded_vectors_carry_what_produced_them(
    root: Path, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """Provenance beside the numbers, so a reader can date them and a check can refuse them."""
    _record(root, encoder, config)

    index = read_query_vectors(root, config).index

    assert index.embedder_name == config.name
    assert index.embedder_revision == config.revision
    assert index.recorded_on == TODAY


def test_vectors_from_another_model_are_refused(
    root: Path, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """As stale as a corpus built by another model, and the manifest cannot see it.

    The manifest checks that the pin has not moved. These were written by whatever was loaded on
    the day, so the file has to name it and the read has to compare.
    """
    _record(root, encoder, config)

    other = config.model_copy(update={"name": "something/else"})

    with pytest.raises(QueryVectorError) as raised:
        read_query_vectors(root, other)

    assert "stale rather than merely old" in str(raised.value)


def test_vectors_of_another_width_are_refused(
    root: Path, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """Different dimensionality cannot be compared against the corpus at all."""
    _record(root, encoder, config)

    narrower = config.model_copy(update={"dimensions": 384})

    with pytest.raises(QueryVectorError):
        read_query_vectors(root, narrower)


def test_a_missing_store_says_what_writes_it(root: Path, config: EmbedderConfig) -> None:
    """The message names the command, and names that it needs no key."""
    with pytest.raises(QueryVectorError) as raised:
        read_query_vectors(root, config)

    assert "make record-queries" in str(raised.value)


def test_an_index_that_does_not_match_the_array_is_refused(
    root: Path, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """The two files are written together, so a disagreement is a hand edit of one of them."""
    _record(root, encoder, config)

    index_path = root / "index.json"
    document = index_path.read_text(encoding="utf-8")
    edited = document.replace(f'"question": "{QUESTIONS[0]}"', '"question": "something else"')

    index_path.write_text(edited.replace('"row": 0', '"row": 1'), encoding="utf-8")

    with pytest.raises(QueryVectorError) as raised:
        read_query_vectors(root, config)

    assert "distinct row" in str(raised.value)


def test_recording_the_same_question_twice_is_refused(
    root: Path, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """A question is the identity of its vector, so two of them is one silent loss."""
    repeated = (*QUESTIONS, QUESTIONS[0])
    vectors = np.vstack([encoder.embed_query(text) for text in repeated])

    with pytest.raises(QueryVectorError) as raised:
        write_query_vectors(root, repeated, vectors, config, TODAY)

    assert "appears twice" in str(raised.value)


def test_a_count_mismatch_is_refused(
    root: Path, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """Fewer vectors than questions would file some vector against another question's text."""
    vectors = np.vstack([encoder.embed_query(text) for text in QUESTIONS[:2]])

    with pytest.raises(QueryVectorError):
        write_query_vectors(root, QUESTIONS, vectors, config, TODAY)


def test_a_recorded_question_is_read_and_not_embedded(
    root: Path, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """The recorded vector is served without the model being asked, which is the whole point."""
    _record(root, encoder, config)

    live = StubEncoder(config.dimensions)
    replayed = RecordedQueryEncoder(read_query_vectors(root, config).vectors, live)

    replayed.embed_query(QUESTIONS[0])

    assert live.embedded == []


def test_an_unrecorded_question_goes_to_the_model(
    root: Path, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """Delegation is the design, not a fallback: a typed question gets real retrieval."""
    _record(root, encoder, config)

    live = StubEncoder(config.dimensions)
    replayed = RecordedQueryEncoder(read_query_vectors(root, config).vectors, live)

    replayed.embed_query("something nobody recorded")

    assert live.embedded == ["something nobody recorded"]


def test_a_machine_that_embeds_differently_gets_the_same_recorded_vector(
    root: Path, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """The criterion this module exists for, simulated by perturbing the model.

    A recorded question must not move when the embedder does, or every answer recorded on one
    machine misses on the next -- and it misses for a reason nothing in the failure would point
    at, because the question, the corpus and the code are all identical.
    """
    written = _record(root, encoder, config)

    elsewhere = StubEncoder(config.dimensions, nudge=1e-7)
    replayed = RecordedQueryEncoder(read_query_vectors(root, config).vectors, elsewhere)

    assert elsewhere.embed_query(QUESTIONS[0]).tobytes() != written[0].tobytes()
    assert replayed.embed_query(QUESTIONS[0]).tobytes() == written[0].tobytes()


def test_a_question_is_matched_exactly(
    root: Path, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """No trimming, no case folding. A tidied lookup is a nearest match wearing a different hat."""
    _record(root, encoder, config)

    live = StubEncoder(config.dimensions)
    replayed = RecordedQueryEncoder(read_query_vectors(root, config).vectors, live)

    replayed.embed_query(f" {QUESTIONS[0]}")

    assert live.embedded == [f" {QUESTIONS[0]}"]
