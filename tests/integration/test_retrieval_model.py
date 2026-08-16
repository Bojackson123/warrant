"""Asking about account management retrieves AC-2, with the real model and the real catalog.

The one test here that a stub encoder cannot stand in for. Everything between a question and a
ranked list is checked without weights next door; what is left is whether the pieces are wired the
way the model needs them — the query prefix applied on the query side and not the document side,
cosine over vectors the model actually produced, the ordering not inverted. Each of those failures
returns a confident ranking of the wrong controls rather than an error, and a stub encoder cannot
see any of them because it has no notion of meaning to get wrong.

**A subset of the catalog, not all of it.** Embedding all 1,014 chunks is roughly ten minutes of
CPU, and a test that costs ten minutes is a test that gets marked slow and then stops running.
The ninety-odd below cost about a minute, once, for the whole module. They are chosen to be a
hard neighbourhood rather than an easy one: AC-2 sits among its own thirteen enhancements and
among the other access-control and identification families, which is where a question about
accounts can plausibly land on the wrong row. Searching them is a tenth of a second, which is the
half of the cost that a request would actually pay.

**This is not a recall measurement and must not be quoted as one.** Recall over the whole corpus
is what the golden set exists to measure, and the figure the embedding comparison already records
for this model is a floor rather than a boast. What this asserts is that the path works, on a
neighbourhood small enough to run every time.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from pgvector.psycopg import register_vector

from warrant.db.migrator import apply_migrations
from warrant.embedder_config import EmbedderConfig, get_embedder_config
from warrant.embedding import Embedder, EmbedderError, load_embedder
from warrant.ingest.catalog import load_catalog
from warrant.ingest.chunker import Chunk, chunk_catalog
from warrant.ingest.pipeline import ingest
from warrant.retrieval import retrieve
from warrant.settings import get_settings

pytestmark = [pytest.mark.integration, pytest.mark.model]

# The neighbourhood AC-2 has to be picked out of: its own enhancements, the rest of access
# control, and the identification family whose vocabulary overlaps it most. Near misses rather
# than a random sample, because a random sample of a 1,014-way choice makes the question easy.
_NEIGHBOURHOOD = ("ac-2", "ac-3", "ac-5", "ac-6", "ac-17", "ia-2", "ia-4", "ia-5", "au-6", "cm-5")

# Stand-ins for the values the ingest command computes from the catalog. This test does not read
# them back; that they are stored is the pipeline's own test.
_CHUNKER_FINGERPRINT = "0" * 64
_CATALOG_SHA256 = "1" * 64


@pytest.fixture(scope="module")
def config() -> EmbedderConfig:
    return get_embedder_config()


@pytest.fixture(scope="module")
def embedder(config: EmbedderConfig) -> Embedder:
    try:
        return load_embedder(config)
    except EmbedderError as error:
        pytest.skip(f"the pinned weights are not cached: {error}")


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    """Real chunks, produced by the real chunker from the vendored catalog."""
    catalog = load_catalog(get_settings().catalog_path)

    return [chunk for chunk in chunk_catalog(catalog) if chunk.base_control_id in _NEIGHBOURHOOD]


@pytest.fixture(scope="module")
def corpus(
    module_dsn: str,
    chunks: list[Chunk],
    embedder: Embedder,
    config: EmbedderConfig,
) -> Iterator[psycopg.Connection]:
    """One embedding pass for the whole module.

    Every test below reads this corpus and none of them changes it, which is what makes sharing
    it safe. Per test it would be the same ninety-odd controls embedded five times over, for five
    identical databases.
    """
    with psycopg.connect(module_dsn) as conn:
        apply_migrations(conn)
        register_vector(conn)
        ingest(conn, chunks, embedder, config, _CHUNKER_FINGERPRINT, _CATALOG_SHA256)

        yield conn


def test_the_neighbourhood_is_a_real_one(chunks: list[Chunk]) -> None:
    """A guard on the fixture rather than on the code: an empty subset would pass everything."""
    assert len(chunks) > 50
    assert "ac-2" in {chunk.control_id for chunk in chunks}
    assert any(chunk.control_id.startswith("ac-2.") for chunk in chunks)


def test_asking_about_account_management_retrieves_ac_2(
    corpus: psycopg.Connection,
    embedder: Embedder,
) -> None:
    """The claim the walking skeleton is demonstrated on, as an assertion."""
    result = retrieve(
        corpus,
        "How does an organization create, review and disable system accounts?",
        embedder,
        k=5,
    )

    assert "ac-2" in result.control_ids


def test_asking_about_disabling_inactive_accounts_reaches_the_enhancement(
    corpus: psycopg.Connection,
    embedder: Embedder,
) -> None:
    """An enhancement is retrievable in its own right, which is why it gets its own chunk."""
    result = retrieve(corpus, "disabling accounts that have been inactive", embedder, k=5)

    assert any(control_id.startswith("ac-2") for control_id in result.control_ids)


def test_a_question_from_another_family_does_not_land_on_ac_2_first(
    corpus: psycopg.Connection,
    embedder: Embedder,
) -> None:
    """Ranking rather than a constant: a different question moves the top of the list.

    Without this, every assertion above would also pass against a search that ignored the query
    and returned the same rows in the same order every time.
    """
    accounts = retrieve(corpus, "creating and disabling system accounts", embedder, k=3)
    auditing = retrieve(corpus, "reviewing and analysing audit records", embedder, k=3)

    assert accounts.chunks[0].control_id != auditing.chunks[0].control_id


def test_scores_are_similarities_in_rank_order(
    corpus: psycopg.Connection,
    embedder: Embedder,
) -> None:
    """Normalised vectors put a real cosine similarity in [-1, 1], and the top one well above 0."""
    result = retrieve(corpus, "account management", embedder, k=5)
    scores = [chunk.score for chunk in result.chunks]

    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= score <= 1.0 for score in scores)
    assert scores[0] > 0.5
