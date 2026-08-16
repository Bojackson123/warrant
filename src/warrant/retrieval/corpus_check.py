"""Checks that the corpus in the database was built by the model and chunker this build carries.

The counterpart to `db.schema_check`, one layer up. That module compares the embedding column's
declared width against the pin, which catches a database built for a model of a different
dimensionality. This one catches what a width cannot: two 768-dimensional models are
indistinguishable to the column, so a corpus embedded by a superseded pin passes every existing
check and returns confidently wrong control ids. That is not a crash, it is a plausible answer
citing the wrong control, which is the worst failure this project has and the reason
`corpus_ingest` records what produced the rows rather than only that they exist.

The chunker matters here for the same reason the model does. Chunk text decides the vectors, the
vectors decide which control ids come back, and a corpus assembled under a superseded chunker is
a corpus this build's citations were not measured against.

**What is deliberately not checked here:** the catalog digest and the chunker fingerprint. Both
mean re-reading a ten megabyte file and re-chunking it, which is file-level integrity work rather
than a startup question, and it belongs to the check that owns the pinned inputs as a set. What
this compares is what is already in memory, so it is cheap enough to run before serving and
cheap enough to run again.

Called once, at the point something is about to search — not per query. The corpus does not
change under a running process.
"""

from __future__ import annotations

import psycopg

from warrant.embedder_config import EmbedderConfig, get_embedder_config
from warrant.ingest.chunker import CHUNKER_VERSION
from warrant.ingest.parameters import RESOLUTION_VERSION
from warrant.ingest.pipeline import CorpusProvenance, read_provenance


class CorpusMismatchError(Exception):
    """The stored corpus was not built by the inputs this build is pinned to."""


class CorpusMissingError(CorpusMismatchError):
    """There is no corpus in this database at all.

    A subclass, so a caller refusing to search on a mismatch refuses on an empty database
    without learning a second exception, while a caller that wants to tell "never ingested"
    from "ingested by something else" still can.
    """


def verify_corpus(
    conn: psycopg.Connection,
    config: EmbedderConfig | None = None,
) -> CorpusProvenance:
    """Compare what built the stored corpus against this build's pins; return what it found.

    Raises rather than reporting, because there is nothing useful to do with a corpus built by
    something else. Every difference below invalidates the vectors wholesale, and searching one
    anyway means answers that cite the wrong control while looking exactly like answers that do
    not.

    The config is a parameter so that a test can hand this a deliberately mismatched pin without
    rewriting the file the whole project reads.
    """
    pinned = config if config is not None else get_embedder_config()
    provenance = read_provenance(conn)

    if provenance is None:
        raise CorpusMissingError(
            "This database has no corpus: the schema is in place but nothing has been ingested "
            "into it. `make ingest` embeds the catalog and builds one, and needs no network of "
            "its own once `make model` has cached the weights."
        )

    # Ordered widest to narrowest. An embedder change invalidates everything downstream of the
    # vectors as well as the vectors themselves, so it is the difference to name first if more
    # than one has moved.
    if provenance.embedder_name != pinned.name:
        raise CorpusMismatchError(
            _moved(
                "the embedding model",
                provenance.embedder_name,
                pinned.name,
                "Vectors from two different models are not comparable at all, so every "
                "distance this corpus would return is meaningless. Re-ingest.",
            )
        )

    if provenance.embedder_revision != pinned.revision:
        raise CorpusMismatchError(
            _moved(
                f"the revision of {pinned.name}",
                provenance.embedder_revision,
                pinned.revision,
                "Published weights move under a tag, and the column's declared width cannot "
                "tell two revisions apart -- so this corpus would be searched successfully and "
                "would cite the wrong controls. Re-ingest.",
            )
        )

    if provenance.dimensions != pinned.dimensions:
        raise CorpusMismatchError(
            _moved(
                "the vector width",
                str(provenance.dimensions),
                str(pinned.dimensions),
                "The schema check compares the column against the pin and this compares the "
                "stored corpus against it; the two disagreeing means the table was migrated "
                "without a re-ingest. Re-ingest.",
            )
        )

    if provenance.chunker_version != CHUNKER_VERSION:
        raise CorpusMismatchError(
            _moved(
                "the chunker",
                provenance.chunker_version,
                CHUNKER_VERSION,
                "Chunk text decides the vectors and the vectors decide which control ids come "
                "back, so a corpus assembled by a different chunker is not the corpus this "
                "build's retrieval was written against. Re-ingest.",
            )
        )

    if provenance.resolution_version != RESOLUTION_VERSION:
        raise CorpusMismatchError(
            _moved(
                "parameter resolution",
                provenance.resolution_version,
                RESOLUTION_VERSION,
                "Resolution runs before chunking and changes the prose that was embedded, so it "
                "leaves no trace in a stored row beyond the text itself. Re-ingest.",
            )
        )

    return provenance


def _moved(what: str, stored: str, expected: str, cost: str) -> str:
    """One difference, as a sentence naming both sides of it and what it costs.

    A shared shape rather than five hand-written messages, because the useful content is always
    the same three things and a message missing one of them is the one somebody reads at the
    point they are already confused.
    """
    return (
        f"The stored corpus was built with {what} {stored!r}, and this build is pinned to "
        f"{expected!r}. {cost}"
    )
