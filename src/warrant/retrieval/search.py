"""Turn a question into ranked chunks, each carrying the control id that would be cited.

The whole path, in order: embed the question with the pinned model, scan every stored vector,
rank by distance, return the nearest `k` with their identifiers and their text. There is no
graph, no agent loop, no reranker and no query reformulation, and their absence is a decision
rather than an omission — each of them is a thing to add once there is a measurement saying it
helps, and adding one now would mean the first such measurement was taken against it.

**`k` is configuration and it travels with the result.** It is read from one place, `Settings`,
and `Retrieval` carries the value the search actually used. That is structural on purpose: a
caller cannot hold ranked chunks apart from the `k` that produced them, so a number measured at
one `k` cannot quietly be compared against a number measured at another. The two are not
commensurable, and the failure is silent in exactly the way that gets a wrong figure quoted.

**Nothing here needs a key or a network.** The weights come from the local cache and the scan
runs against local vectors, which is what makes an arbitrary question answerable on a machine
that has neither — the property the replay path rests on. A retrieval path that reached a
provider would make replay a recording of canned questions instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from warrant.embedder_config import EmbedderConfig, get_embedder_config
from warrant.embedding import Encoder
from warrant.settings import get_settings

# Cosine distance, `<=>`, and the choice is worth stating because three operators would rank the
# corpus identically today and only one keeps doing so.
#
# The pinned model normalises its output, so on unit vectors cosine distance and negative inner
# product are the same ordering. `<=>` is what survives that stopping being true: a pin with
# `normalize: false` would leave `<#>` ranking partly by magnitude, which is not a similarity and
# would not announce itself. It also gives a score a person can read, `1 - distance`, rather than
# a negative dot product.
#
# Exact, over every row. The migration that created this table argues the case at length: an
# approximate index would make recall a property of the index's search parameters rather than of
# the embedding model, so every retrieval number this project reports would be measuring the two
# together without saying so. A thousand chunks is a few milliseconds of scan.
#
# The vector is bound once, in the select list, and the ordering is by that output column. Writing
# the expression twice would be two copies of the thing that decides both the score reported and
# the order returned, free to drift apart.
#
# `chunk_id` breaks ties. Equal distances are otherwise returned in whatever order the scan
# produced them, and retrieved text goes on to be part of what recorded model calls are keyed on:
# a tie that resolves differently between two runs would surface much later, as a fixture that
# misses for no visible reason.
_SEARCH = """
    SELECT chunk_id, control_id, base_control_id, control_label, title, part_path, text,
           embedding <=> %s AS distance
    FROM chunks
    ORDER BY distance, chunk_id
    LIMIT %s
"""


class RetrievalError(Exception):
    """The corpus cannot be searched, or cannot be searched with the encoder supplied."""


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One stored chunk, with where it placed and how near it was.

    The `chunks` columns, plus the two values that only exist relative to a query. Rank is
    1-based because it is read by people, in a console and in a printed list.
    """

    rank: int
    # Cosine similarity: 1 for an identical direction, 0 for an unrelated one. Derived from the
    # distance the scan ordered on rather than computed separately, so what is displayed is what
    # the ranking used.
    score: float
    chunk_id: str
    # Canonical form. This is what a citation is checked against, and what a click-through
    # resolves.
    control_id: str
    base_control_id: str
    # The form a reader sees. Rendered, never matched on.
    control_label: str
    title: str
    part_path: str
    text: str


@dataclass(frozen=True, slots=True)
class Retrieval:
    """What one question retrieved, and the `k` it was retrieved at.

    `k` is a field rather than something the caller is trusted to remember. It is the value this
    search used, which is not necessarily what is configured now — a process whose configuration
    changed between two searches has two results that say so.
    """

    question: str
    k: int
    chunks: tuple[RetrievedChunk, ...]

    @property
    def control_ids(self) -> tuple[str, ...]:
        """The control ids retrieved, in rank order, with duplicates kept.

        Duplicates are meaningful: two chunks of the same control both being near is a different
        result from one being near, and collapsing them here would hide it from anything counting.
        """
        return tuple(chunk.control_id for chunk in self.chunks)


def retrieve(
    conn: psycopg.Connection,
    question: str,
    encoder: Encoder,
    k: int | None = None,
    config: EmbedderConfig | None = None,
) -> Retrieval:
    """Embed the question and return the `k` nearest stored chunks, nearest first.

    `k` defaults to the configured value, resolved here rather than at each call site, so that
    "one configuration source" stays true of a caller that simply does not pass it.

    The connection must already be able to send a `vector`. `db.connection()` guarantees that;
    registering the adapter here instead would cost a catalog round trip on every question, for
    setup that belongs to the connection rather than to the query.

    The config is a parameter so a test can hand this a deliberately wrong pin without editing
    the file the whole project reads.
    """
    pinned = config if config is not None else get_embedder_config()
    width = k if k is not None else get_settings().retrieval_k

    # Checked before the database is touched, and against the pin rather than against the stored
    # column. The column would reject the vector anyway, but as a type error several frames from
    # the encoder that produced it rather than as a sentence naming the two numbers.
    if encoder.dimensions != pinned.dimensions:
        raise RetrievalError(
            f"The encoder produces {encoder.dimensions}-dimensional vectors and the pinned model "
            f"{pinned.name} produces {pinned.dimensions}. The corpus is stored at the pinned "
            "width, so a question embedded by this encoder cannot be compared against it."
        )

    if width < 1:
        raise RetrievalError(
            f"Asked for the nearest {width} chunks. Retrieving none is not a cheaper search, it "
            "is an answer with nothing to cite, which is a thing to decide deliberately rather "
            "than to arrive at through a configuration value."
        )

    # Refused rather than embedded. An asymmetric model turns a blank question into the
    # instruction prefix alone, which is a perfectly good vector and gives a ranking that looks
    # like every other ranking — so the emptiness would survive all the way to an answer that
    # cites controls nobody asked about.
    if not question.strip():
        raise RetrievalError(
            "The question is empty. Embedding it would produce a vector for the model's own "
            "instruction prefix and rank the corpus against that, which returns chunks that look "
            "like an answer to something."
        )

    # The query side of an asymmetric model. `embed_query` applies the instruction prefix the
    # model card specifies and `embed_documents` does not; embedding a question through the
    # document path costs retrieval quality with nothing to notice.
    vector = encoder.embed_query(question)

    with conn.cursor() as cursor:
        cursor.execute(_SEARCH, (vector, width))
        rows = cursor.fetchall()

    # An exact scan over a non-empty table always returns min(k, count) rows, because every row
    # has a distance. No rows therefore means no corpus, not a question nothing matched — which
    # is worth separating, since the second would be a retrieval result and the first is a
    # database nobody has ingested into.
    if not rows:
        raise RetrievalError(
            "The corpus is empty, so there is nothing to retrieve. Exact search returns the "
            "nearest rows whatever the question, so this is a database that has not been "
            "ingested into rather than a question that matched nothing. `make ingest` builds "
            "the corpus."
        )

    return Retrieval(
        question=question,
        k=width,
        chunks=tuple(
            RetrievedChunk(
                rank=rank,
                score=1.0 - float(distance),
                chunk_id=chunk_id,
                control_id=control_id,
                base_control_id=base_control_id,
                control_label=control_label,
                title=title,
                part_path=part_path,
                text=text,
            )
            for rank, (
                chunk_id,
                control_id,
                base_control_id,
                control_label,
                title,
                part_path,
                text,
                distance,
            ) in enumerate(rows, start=1)
        ),
    )
