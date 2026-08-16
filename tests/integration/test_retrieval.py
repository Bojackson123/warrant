"""The query path against a real database, with a stub encoder rather than the pinned model.

What is under test here is the mechanism between a vector and a ranked list: that `k` chunks come
back in distance order, that ties resolve the same way twice, that `k` comes from one place and is
reported with the result, and that a corpus built by something other than this build's pins is
refused rather than searched. None of that is a property of the model, and requiring half a
gigabyte of weights to check it would mean it went unchecked in CI.

The stub is the ingest tests' own, imported rather than restated. Two deterministic encoders in
one test suite are two things that can drift apart, and the second one to drift is the one nobody
is looking at. Its vectors are deliberately not unit-normalised, which makes them useful here in
their own right: cosine distance ranks them by direction alone, and an operator that ranked partly
by magnitude would order them differently.

Whether an English question about accounts actually retrieves AC-2 is a different claim needing a
real model, and it lives beside this file.
"""

from __future__ import annotations

import psycopg
import pytest
from pgvector.psycopg import register_vector

from warrant.db.migrator import apply_migrations
from warrant.embedder_config import EmbedderConfig, get_embedder_config
from warrant.ingest.chunker import CHUNKER_VERSION
from warrant.ingest.parameters import RESOLUTION_VERSION
from warrant.retrieval import (
    CorpusMismatchError,
    CorpusMissingError,
    RetrievalError,
    retrieve,
    verify_corpus,
)
from warrant.settings import get_settings

from .test_ingest import StubEncoder, chunk, run

pytestmark = pytest.mark.integration


@pytest.fixture
def config() -> EmbedderConfig:
    return get_embedder_config()


@pytest.fixture
def encoder(config: EmbedderConfig) -> StubEncoder:
    return StubEncoder(config.dimensions)


@pytest.fixture
def migrated(conn: psycopg.Connection) -> psycopg.Connection:
    apply_migrations(conn)
    register_vector(conn)
    return conn


@pytest.fixture
def corpus(
    migrated: psycopg.Connection,
    encoder: StubEncoder,
    config: EmbedderConfig,
) -> psycopg.Connection:
    """A migrated database with a small corpus in it, written by the pipeline itself."""
    chunks = [chunk(f"ac-{n}", f"Control text number {n}.") for n in range(1, 7)]
    run(migrated, chunks, encoder, config)

    return migrated


def test_returns_k_chunks_ranked_nearest_first(
    corpus: psycopg.Connection,
    encoder: StubEncoder,
) -> None:
    result = retrieve(corpus, "a question", encoder, k=3)

    assert len(result.chunks) == 3
    assert [retrieved.rank for retrieved in result.chunks] == [1, 2, 3]

    scores = [retrieved.score for retrieved in result.chunks]
    assert scores == sorted(scores, reverse=True)


def test_a_chunk_carries_the_identifiers_a_citation_needs(
    corpus: psycopg.Connection,
    encoder: StubEncoder,
) -> None:
    """A control id and its clause text, which is what the whole path exists to produce."""
    top = retrieve(corpus, "a question", encoder, k=1).chunks[0]

    assert top.control_id.startswith("ac-")
    assert top.control_label == top.control_id.upper()
    assert top.base_control_id == top.control_id
    assert top.part_path == "smt+gdn"
    assert top.chunk_id == f"{top.control_id}:smt+gdn"
    assert top.text.startswith("Control text number ")
    assert top.title == f"Title for {top.control_id}"


def test_the_result_reports_the_k_it_used(
    corpus: psycopg.Connection,
    encoder: StubEncoder,
) -> None:
    """A number measured at one k cannot be compared against one measured at another."""
    assert retrieve(corpus, "a question", encoder, k=2).k == 2
    assert retrieve(corpus, "a question", encoder, k=4).k == 4


def test_k_defaults_to_the_configured_value(
    corpus: psycopg.Connection,
    encoder: StubEncoder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One configuration source, reached by a caller that simply does not pass a k."""
    monkeypatch.setenv("WARRANT_RETRIEVAL_K", "2")
    get_settings.cache_clear()

    try:
        result = retrieve(corpus, "a question", encoder)
    finally:
        get_settings.cache_clear()

    assert result.k == 2
    assert len(result.chunks) == 2


def test_a_k_beyond_the_corpus_returns_all_of_it(
    corpus: psycopg.Connection,
    encoder: StubEncoder,
) -> None:
    """And still reports the k asked for, so the shortfall is visible rather than inferred."""
    result = retrieve(corpus, "a question", encoder, k=50)

    assert result.k == 50
    assert len(result.chunks) == 6


def test_control_ids_come_back_in_rank_order(
    corpus: psycopg.Connection,
    encoder: StubEncoder,
) -> None:
    """Duplicates kept: two chunks of one control being near is not one chunk being near."""
    result = retrieve(corpus, "a question", encoder, k=4)

    assert result.control_ids == tuple(retrieved.control_id for retrieved in result.chunks)


def test_ties_resolve_the_same_way_twice(
    migrated: psycopg.Connection,
    encoder: StubEncoder,
    config: EmbedderConfig,
) -> None:
    """Retrieved text ends up inside what recorded model calls are keyed on.

    Identical text means identical vectors from this encoder, so every distance here is equal and
    the order is the tie-break's alone. Without one Postgres may return them in whatever order the
    scan produced, and a fixture would miss later for no visible reason.
    """
    identical = [chunk(f"ac-{n}", "The same text in every chunk.") for n in range(1, 6)]
    run(migrated, identical, encoder, config)

    first = retrieve(migrated, "a question", encoder, k=5)
    second = retrieve(migrated, "a question", encoder, k=5)

    ids = [retrieved.chunk_id for retrieved in first.chunks]

    assert ids == sorted(ids)
    assert ids == [retrieved.chunk_id for retrieved in second.chunks]


def test_an_empty_corpus_is_refused_rather_than_returning_nothing(
    migrated: psycopg.Connection,
    encoder: StubEncoder,
) -> None:
    """Exact search returns rows whenever there are rows, so none of them means no corpus."""
    with pytest.raises(RetrievalError, match="corpus is empty"):
        retrieve(migrated, "a question", encoder, k=5)


def test_a_wrong_width_encoder_is_refused_before_the_database_sees_it(
    corpus: psycopg.Connection,
    config: EmbedderConfig,
) -> None:
    """A sentence naming both numbers, rather than a type error several frames away."""
    with pytest.raises(RetrievalError, match="dimensional"):
        retrieve(corpus, "a question", StubEncoder(config.dimensions // 2), k=5)


def test_a_k_below_one_is_refused(
    corpus: psycopg.Connection,
    encoder: StubEncoder,
) -> None:
    """An answer with nothing to cite is a thing to decide, not to arrive at."""
    with pytest.raises(RetrievalError, match="nearest 0"):
        retrieve(corpus, "a question", encoder, k=0)


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_a_blank_question_is_refused(
    corpus: psycopg.Connection,
    encoder: StubEncoder,
    blank: str,
) -> None:
    """A blank question ranks the corpus against the model's own prefix and looks like a result."""
    with pytest.raises(RetrievalError, match="question is empty"):
        retrieve(corpus, blank, encoder, k=3)


def test_a_matching_corpus_is_accepted(
    corpus: psycopg.Connection,
    config: EmbedderConfig,
) -> None:
    provenance = verify_corpus(corpus, config)

    assert provenance.embedder_name == config.name
    assert provenance.embedder_revision == config.revision
    assert provenance.chunk_count == 6


def test_a_database_with_no_corpus_says_so(
    migrated: psycopg.Connection,
    config: EmbedderConfig,
) -> None:
    """Distinguishable from a corpus built by something else, and a subclass of it."""
    with pytest.raises(CorpusMissingError, match="no corpus"):
        verify_corpus(migrated, config)


@pytest.mark.parametrize(
    ("moved", "named"),
    [
        ({"name": "some-other/model"}, "embedding model"),
        ({"revision": "0" * 40}, "revision"),
        ({"dimensions": 384}, "vector width"),
    ],
)
def test_a_corpus_built_by_another_model_is_refused(
    corpus: psycopg.Connection,
    config: EmbedderConfig,
    moved: dict[str, object],
    named: str,
) -> None:
    """The failure a vector column cannot catch: same width, different weights, wrong answers.

    The pin moves rather than the stored row, because that is the direction it happens in — a
    build carrying a new pin against a database nobody re-ingested.
    """
    with pytest.raises(CorpusMismatchError, match=named):
        verify_corpus(corpus, config.model_copy(update=moved))


def test_a_corpus_built_by_another_chunker_is_refused(
    corpus: psycopg.Connection,
    config: EmbedderConfig,
) -> None:
    """Chunk text decides the vectors, and the vectors decide which control ids come back."""
    with corpus.cursor() as cursor:
        cursor.execute(
            "UPDATE corpus_ingest SET chunker_version = %s WHERE id = 1",
            (f"{CHUNKER_VERSION}-superseded",),
        )

    with pytest.raises(CorpusMismatchError, match="chunker"):
        verify_corpus(corpus, config)


def test_a_corpus_built_under_other_parameter_resolution_is_refused(
    corpus: psycopg.Connection,
    config: EmbedderConfig,
) -> None:
    """Resolution changes the prose that was embedded and leaves no other trace in a row."""
    with corpus.cursor() as cursor:
        cursor.execute(
            "UPDATE corpus_ingest SET resolution_version = %s WHERE id = 1",
            (f"{RESOLUTION_VERSION}-superseded",),
        )

    with pytest.raises(CorpusMismatchError, match="parameter resolution"):
        verify_corpus(corpus, config)


def test_retrieval_reads_the_corpus_that_is_there_now(
    migrated: psycopg.Connection,
    encoder: StubEncoder,
    config: EmbedderConfig,
) -> None:
    """A re-ingest between two searches changes the second one's answer."""

    def top() -> str:
        return retrieve(migrated, "a question", encoder, k=1).chunks[0].text

    run(migrated, [chunk("ac-1", "The original statement.")], encoder, config)
    assert top() == "The original statement."

    run(migrated, [chunk("ac-1", "The rewritten statement.")], encoder, config)
    assert top() == "The rewritten statement."
