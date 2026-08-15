"""The ingest pipeline against a real database.

Run with a stub encoder rather than the pinned model, deliberately. What is under test here is
storage: that a second run collides with the rows already present, that a chunk the chunker has
stopped producing is removed rather than left to be retrieved instead of a right answer, and that
the corpus is either wholly replaced or wholly untouched. None of that is a property of the model,
and requiring half a gigabyte of weights to check it would mean it went unchecked in CI.

The stub is deterministic and derives each vector from the chunk's text, so "the vectors did not
move" and "the text did not move" are the same statement — which is what lets these tests exercise
the fingerprint without embedding anything.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import psycopg
import pytest
from pgvector.psycopg import register_vector

from warrant.db.migrator import apply_migrations
from warrant.embedder_config import EmbedderConfig, get_embedder_config
from warrant.embedding import VECTOR_DTYPE, ProgressCallback
from warrant.ingest.chunker import CHUNKER_VERSION, Chunk
from warrant.ingest.parameters import RESOLUTION_VERSION
from warrant.ingest.pipeline import (
    CorpusProvenance,
    IngestError,
    IngestReport,
    ingest,
    read_provenance,
)

pytestmark = pytest.mark.integration

# Stand-ins for the values the command computes from the catalog. Their content does not matter to
# the pipeline, which stores them; that they are stored is what the provenance test checks.
CHUNKER_FINGERPRINT = "0" * 64
CATALOG_SHA256 = "1" * 64


class StubEncoder:
    """A deterministic encoder: same text in, same vector out, no weights involved."""

    def __init__(self, dimensions: int) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(
        self,
        texts: Sequence[str],
        progress: ProgressCallback | None = None,
    ) -> np.ndarray:
        vectors = np.vstack([self._vector(text) for text in texts])

        if progress is not None:
            progress(len(texts), len(texts))

        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def _vector(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        generator = np.random.default_rng(seed)

        return generator.standard_normal(self._dimensions).astype(VECTOR_DTYPE)


def chunk(control_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{control_id}:smt+gdn",
        control_id=control_id,
        base_control_id=control_id.split("(")[0],
        control_label=control_id.upper(),
        title=f"Title for {control_id}",
        part_path="smt+gdn",
        text=text,
    )


@pytest.fixture
def config() -> EmbedderConfig:
    return get_embedder_config()


@pytest.fixture
def encoder(config: EmbedderConfig) -> StubEncoder:
    return StubEncoder(config.dimensions)


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        chunk("ac-2", "Account management. The organization manages accounts."),
        chunk("ac-2.3", "Disable accounts that have been inactive for a defined period."),
        chunk("au-6", "Review and analyse audit records for indications of inappropriate use."),
    ]


@pytest.fixture
def migrated(conn: psycopg.Connection) -> psycopg.Connection:
    apply_migrations(conn)
    register_vector(conn)
    return conn


def run(
    conn: psycopg.Connection,
    chunks: list[Chunk],
    encoder: StubEncoder,
    config: EmbedderConfig,
) -> IngestReport:
    return ingest(conn, chunks, encoder, config, CHUNKER_FINGERPRINT, CATALOG_SHA256)


def stored(conn: psycopg.Connection) -> dict[str, tuple[str, np.ndarray]]:
    with conn.cursor() as cursor:
        cursor.execute("SELECT chunk_id, text, embedding FROM chunks ORDER BY chunk_id")
        return {row[0]: (row[1], np.asarray(row[2])) for row in cursor.fetchall()}


def test_writes_every_chunk(
    migrated: psycopg.Connection,
    chunks: list[Chunk],
    encoder: StubEncoder,
    config: EmbedderConfig,
) -> None:
    report = run(migrated, chunks, encoder, config)
    rows = stored(migrated)

    assert report.chunks_written == len(chunks)
    assert set(rows) == {chunk.chunk_id for chunk in chunks}
    assert report.previous_fingerprint is None
    assert report.vector_bytes == len(chunks) * config.dimensions * 4


def test_running_twice_leaves_one_corpus(
    migrated: psycopg.Connection,
    chunks: list[Chunk],
    encoder: StubEncoder,
    config: EmbedderConfig,
) -> None:
    """The acceptance criterion, as two digests rather than as a claim."""
    first = run(migrated, chunks, encoder, config)
    before = stored(migrated)

    second = run(migrated, chunks, encoder, config)
    after = stored(migrated)

    assert second.chunks_written == first.chunks_written
    assert second.rows_deleted == 0
    assert second.corpus_fingerprint == first.corpus_fingerprint
    assert second.unchanged

    assert set(after) == set(before)
    for chunk_id, (text, vector) in after.items():
        assert text == before[chunk_id][0]
        assert np.array_equal(vector, before[chunk_id][1])


def test_a_chunk_the_chunker_stopped_producing_is_deleted(
    migrated: psycopg.Connection,
    chunks: list[Chunk],
    encoder: StubEncoder,
    config: EmbedderConfig,
) -> None:
    """An orphan cannot be a right answer to anything, and can still be returned instead of one."""
    run(migrated, chunks, encoder, config)

    remaining = chunks[:-1]
    report = run(migrated, remaining, encoder, config)

    assert report.rows_deleted == 1
    assert set(stored(migrated)) == {chunk.chunk_id for chunk in remaining}
    assert not report.unchanged


def test_changed_text_updates_in_place(
    migrated: psycopg.Connection,
    chunks: list[Chunk],
    encoder: StubEncoder,
    config: EmbedderConfig,
) -> None:
    run(migrated, chunks, encoder, config)
    before = stored(migrated)

    edited = [replace(chunks[0], text="Rewritten statement."), *chunks[1:]]
    report = run(migrated, edited, encoder, config)
    after = stored(migrated)

    assert report.rows_deleted == 0
    assert len(after) == len(before)
    assert after[chunks[0].chunk_id][0] == "Rewritten statement."
    assert not np.array_equal(after[chunks[0].chunk_id][1], before[chunks[0].chunk_id][1])

    # Everything else is untouched, so an edit to one control does not silently re-embed the rest
    # into different numbers.
    for other in chunks[1:]:
        assert np.array_equal(after[other.chunk_id][1], before[other.chunk_id][1])


def test_records_what_produced_the_corpus(
    migrated: psycopg.Connection,
    chunks: list[Chunk],
    encoder: StubEncoder,
    config: EmbedderConfig,
) -> None:
    """The width check cannot tell two models of the same width apart. This is what can."""
    assert read_provenance(migrated) is None

    report = run(migrated, chunks, encoder, config)

    assert read_provenance(migrated) == CorpusProvenance(
        embedder_name=config.name,
        embedder_revision=config.revision,
        dimensions=config.dimensions,
        chunker_version=CHUNKER_VERSION,
        chunker_fingerprint=CHUNKER_FINGERPRINT,
        resolution_version=RESOLUTION_VERSION,
        catalog_sha256=CATALOG_SHA256,
        chunk_count=len(chunks),
        corpus_fingerprint=report.corpus_fingerprint,
    )


def test_provenance_stays_one_row(
    migrated: psycopg.Connection,
    chunks: list[Chunk],
    encoder: StubEncoder,
    config: EmbedderConfig,
) -> None:
    """It describes the corpus that is here now, not a history of the runs that built one."""
    run(migrated, chunks, encoder, config)
    run(migrated, chunks[:-1], encoder, config)

    with migrated.cursor() as cursor:
        cursor.execute("SELECT count(*), max(chunk_count) FROM corpus_ingest")
        assert cursor.fetchone() == (1, len(chunks) - 1)


def test_a_failed_write_leaves_the_previous_corpus(
    migrated: psycopg.Connection,
    chunks: list[Chunk],
    encoder: StubEncoder,
    config: EmbedderConfig,
) -> None:
    """Atomic is the property; a chunk too long for its column is a convenient way to break one."""
    first = run(migrated, chunks, encoder, config)
    migrated.commit()

    # control_label is text and has no length limit, so the failure has to come from somewhere the
    # schema does constrain: a null in a NOT NULL column.
    broken = [replace(chunks[0], title=None), *chunks[1:]]  # type: ignore[arg-type]

    with pytest.raises(psycopg.errors.NotNullViolation):
        run(migrated, broken, encoder, config)

    migrated.rollback()

    with migrated.cursor() as cursor:
        cursor.execute("SELECT corpus_fingerprint FROM corpus_ingest WHERE id = 1")
        assert cursor.fetchone() == (first.corpus_fingerprint,)

    assert set(stored(migrated)) == {chunk.chunk_id for chunk in chunks}


def test_wrong_width_vectors_are_refused_before_the_database_sees_them(
    migrated: psycopg.Connection,
    chunks: list[Chunk],
    config: EmbedderConfig,
) -> None:
    """The column would reject them anyway; this says so in a sentence naming both numbers."""
    with pytest.raises(IngestError, match="dimensional"):
        run(migrated, chunks, StubEncoder(config.dimensions // 2), config)


def test_ingesting_nothing_is_refused(
    migrated: psycopg.Connection,
    encoder: StubEncoder,
    config: EmbedderConfig,
) -> None:
    """Emptying the corpus is a thing to ask for, not to arrive at."""
    with pytest.raises(IngestError, match="No chunks"):
        run(migrated, [], encoder, config)
