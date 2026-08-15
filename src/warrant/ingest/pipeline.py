"""Embed the chunks and store them, so that running it twice leaves one corpus rather than two.

Two properties this module exists to hold, both of which are easier to claim than to have:

**Re-running changes nothing.** Rows are keyed on the chunk id the chunker derives from the
catalog's own identifiers, so a second run collides with the rows already there instead of
doubling the corpus. Chunks the current chunker no longer produces are deleted rather than left
behind — an orphan from a superseded chunker is a document that cannot be a right answer to
anything but can still be returned instead of one.

**The write is atomic.** Embedding runs first, holding no transaction, and then every row lands in
one. An ingest interrupted halfway through the ten minutes of CPU leaves the previous corpus
exactly as it was, rather than a corpus half of which came from one chunker and half from another.
That costs holding the vectors in memory — three megabytes at the size this catalog produces — and
the trade stops being obvious somewhere around a hundred thousand chunks, which is also roughly
where the migration that created `chunks` says to revisit its lack of a vector index.

The provenance row written at the end is what lets anything downstream refuse a corpus built by a
model or a chunker this build no longer carries. The embedding column's declared width already
catches a change of dimensionality; it cannot catch a change between two models of the same width,
which is the change that produces confidently wrong control ids rather than an error.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from warrant.embedder_config import EmbedderConfig
from warrant.embedding import VECTOR_DTYPE, Encoder, ProgressCallback
from warrant.ingest.chunker import CHUNKER_VERSION, Chunk
from warrant.ingest.parameters import RESOLUTION_VERSION

# Every column the chunker and the pipeline own. `created_at` is the database's and is left to its
# default, so it keeps meaning "when this chunk first entered the corpus" across a re-ingest.
_UPSERT_CHUNK = """
    INSERT INTO chunks (
        chunk_id, control_id, base_control_id, control_label, title, part_path, text,
        embedding, chunker_version
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (chunk_id) DO UPDATE SET
        control_id      = EXCLUDED.control_id,
        base_control_id = EXCLUDED.base_control_id,
        control_label   = EXCLUDED.control_label,
        title           = EXCLUDED.title,
        part_path       = EXCLUDED.part_path,
        text            = EXCLUDED.text,
        embedding       = EXCLUDED.embedding,
        chunker_version = EXCLUDED.chunker_version
"""

# `<> ALL` rather than NOT IN, which answers NULL rather than true if the array ever contained
# one, and would then delete nothing while looking like it had worked.
_DELETE_STALE = "DELETE FROM chunks WHERE chunk_id <> ALL(%s)"

_READ_PROVENANCE = """
    SELECT embedder_name, embedder_revision, dimensions, chunker_version, chunker_fingerprint,
           resolution_version, catalog_sha256, chunk_count, corpus_fingerprint
    FROM corpus_ingest
    WHERE id = 1
"""

_UPSERT_PROVENANCE = """
    INSERT INTO corpus_ingest (
        id, embedder_name, embedder_revision, dimensions, chunker_version, chunker_fingerprint,
        resolution_version, catalog_sha256, chunk_count, corpus_fingerprint, embed_seconds
    )
    VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        embedder_name       = EXCLUDED.embedder_name,
        embedder_revision   = EXCLUDED.embedder_revision,
        dimensions          = EXCLUDED.dimensions,
        chunker_version     = EXCLUDED.chunker_version,
        chunker_fingerprint = EXCLUDED.chunker_fingerprint,
        resolution_version  = EXCLUDED.resolution_version,
        catalog_sha256      = EXCLUDED.catalog_sha256,
        chunk_count         = EXCLUDED.chunk_count,
        corpus_fingerprint  = EXCLUDED.corpus_fingerprint,
        embed_seconds       = EXCLUDED.embed_seconds,
        ingested_at         = now()
"""

_TABLE_BYTES = "SELECT pg_total_relation_size('chunks')"


class IngestError(Exception):
    """The corpus cannot be built from what was handed to the pipeline."""


@dataclass(frozen=True, slots=True)
class CorpusProvenance:
    """What produced the corpus currently stored in a database.

    Read rather than only written: this is what lets a caller refuse to search a corpus built by a
    model or a chunker it no longer carries, which is the reason for recording it. The comparison
    itself belongs to whoever is refusing — the integrity check has its own vocabulary for saying
    which input moved and what that costs — so this returns the values and draws no conclusion.
    """

    embedder_name: str
    embedder_revision: str
    dimensions: int
    chunker_version: str
    chunker_fingerprint: str
    resolution_version: str
    catalog_sha256: str
    chunk_count: int
    corpus_fingerprint: str


def read_provenance(conn: psycopg.Connection) -> CorpusProvenance | None:
    """What built the corpus in this database, or None if nothing has ingested one yet."""
    with conn.cursor() as cursor:
        cursor.execute(_READ_PROVENANCE)
        row = cursor.fetchone()

    return None if row is None else CorpusProvenance(*row)


@dataclass(frozen=True, slots=True)
class IngestReport:
    """What one ingest did, in the numbers worth checking afterwards."""

    chunks_written: int
    rows_deleted: int
    dimensions: int
    embed_seconds: float
    write_seconds: float

    # The vectors alone, by arithmetic: count x dimensions x 4. This is the figure
    # `data/embedder.json` records an expectation for.
    vector_bytes: int

    # What the table actually occupies, which is several times the above: it also carries the
    # chunk text, its TOAST, two indexes, and the row versions a re-ingest superseded but
    # autovacuum has not reclaimed yet -- so this roughly doubles on a second run that changed
    # nothing at all. Reported rather than compared, because there is no pinned number for it
    # that would not be a pin on Postgres's storage layout.
    table_bytes: int

    corpus_fingerprint: str

    # What the previous ingest left, if there was one. Equal fingerprints are the idempotence
    # check; a difference is the visible form of the corpus having changed.
    previous_fingerprint: str | None

    @property
    def unchanged(self) -> bool:
        """Whether this ingest reproduced the corpus that was already there."""
        return self.previous_fingerprint == self.corpus_fingerprint


def ingest(
    conn: psycopg.Connection,
    chunks: Sequence[Chunk],
    encoder: Encoder,
    config: EmbedderConfig,
    chunker_fingerprint: str,
    catalog_sha256: str,
    progress: ProgressCallback | None = None,
) -> IngestReport:
    """Embed every chunk and replace the stored corpus with it, in one transaction.

    The connection is used directly rather than taken from the pool: this holds a transaction for
    the length of the write, and a pooled connection could be handed back in the middle of it.
    """
    if not chunks:
        raise IngestError(
            "No chunks were handed to the pipeline. Ingesting nothing would empty the corpus, "
            "which is a thing to ask for deliberately rather than to arrive at by handing this "
            "an empty list."
        )

    # Checked against the pin rather than against the encoder's own claim about itself. An
    # encoder consistent with itself and inconsistent with the pin is the case that matters: the
    # column would reject its vectors, but several minutes later and as a type error rather than
    # as a sentence naming the two numbers that disagree.
    if encoder.dimensions != config.dimensions:
        raise IngestError(
            f"The encoder produces {encoder.dimensions}-dimensional vectors and the pinned model "
            f"{config.name} produces {config.dimensions}. The corpus is stored at the pinned "
            "width, so these are not vectors this database can hold."
        )

    register_vector(conn)

    started = time.perf_counter()
    vectors = encoder.embed_documents([chunk.text for chunk in chunks], progress)
    embed_seconds = time.perf_counter() - started

    if vectors.shape != (len(chunks), config.dimensions):
        raise IngestError(
            f"The encoder returned vectors shaped {vectors.shape} for {len(chunks)} chunks at "
            f"{config.dimensions} dimensions. Storing them would either fail against the "
            "column's declared width or, worse, store a vector against the wrong chunk."
        )

    started = time.perf_counter()

    with conn.transaction(), conn.cursor() as cursor:
        was_there = read_provenance(conn)

        cursor.executemany(
            _UPSERT_CHUNK,
            [
                (
                    chunk.chunk_id,
                    chunk.control_id,
                    chunk.base_control_id,
                    chunk.control_label,
                    chunk.title,
                    chunk.part_path,
                    chunk.text,
                    vector,
                    CHUNKER_VERSION,
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )

        cursor.execute(_DELETE_STALE, ([chunk.chunk_id for chunk in chunks],))
        deleted = cursor.rowcount

        fingerprint = corpus_fingerprint([chunk.chunk_id for chunk in chunks], vectors)

        cursor.execute(
            _UPSERT_PROVENANCE,
            (
                config.name,
                config.revision,
                config.dimensions,
                CHUNKER_VERSION,
                chunker_fingerprint,
                RESOLUTION_VERSION,
                catalog_sha256,
                len(chunks),
                fingerprint,
                embed_seconds,
            ),
        )

        cursor.execute(_TABLE_BYTES)
        table_bytes = _scalar(cursor)

    write_seconds = time.perf_counter() - started

    return IngestReport(
        chunks_written=len(chunks),
        rows_deleted=max(deleted, 0),
        dimensions=encoder.dimensions,
        embed_seconds=embed_seconds,
        write_seconds=write_seconds,
        vector_bytes=len(chunks) * encoder.dimensions * np.dtype(VECTOR_DTYPE).itemsize,
        table_bytes=table_bytes,
        corpus_fingerprint=fingerprint,
        previous_fingerprint=None if was_there is None else was_there.corpus_fingerprint,
    )


def corpus_fingerprint(chunk_ids: Sequence[str], vectors: np.ndarray) -> str:
    """SHA-256 over every stored vector and the id it belongs to, in order.

    The counterpart to `chunker_fingerprint`, one layer down: that one says the chunks did not
    change, this one says embedding them produced the same numbers. Together they turn "ingest is
    idempotent" into two digests to compare rather than a claim about a pipeline.

    **This is not a cross-machine identity.** Floating-point results depend on the BLAS kernels
    torch selects for the hardware it is on, so two machines can produce vectors that are equal to
    every decimal anyone cares about and differ in the last bit. What it does establish is that on
    one machine, with the pinned versions, re-running changed nothing — which is the property
    being claimed.
    """
    digest = hashlib.sha256()

    for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(np.ascontiguousarray(vector, dtype=VECTOR_DTYPE).tobytes())
        digest.update(b"\x1e")

    return digest.hexdigest()


def _scalar(cursor: psycopg.Cursor) -> int:
    row = cursor.fetchone()

    if row is None:
        raise IngestError("A query that always returns one row returned none.")

    return int(row[0])
