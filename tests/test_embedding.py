"""The pinned embedding model, loaded the way ingest and the query path load it.

Marked `model` and skipped when the weights are not cached, so a checkout that has never run
`make model` still has a green test suite. What is checked here is not embedding quality — that
was measured before the model was pinned, and the record of it is `docs/decisions/embedder.md` —
but the three properties the rest of the system assumes and would not notice losing: that the
model matches its pin, that it loads with no network, and that embedding the same text twice gives
the same numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from warrant.embedder_config import EmbedderConfig, get_embedder_config
from warrant.embedding import Embedder, EmbedderError, load_embedder

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def config() -> EmbedderConfig:
    return get_embedder_config()


@pytest.fixture(scope="module")
def embedder(config: EmbedderConfig) -> Embedder:
    try:
        return load_embedder(config)
    except EmbedderError as error:
        pytest.skip(f"the pinned weights are not cached: {error}")


def test_matches_the_pin(embedder: Embedder, config: EmbedderConfig) -> None:
    """`load_embedder` refuses a model that disagrees, so reaching here is most of the assertion."""
    vector = embedder.embed_query("How often must accounts be reviewed?")

    assert embedder.dimensions == config.dimensions
    assert vector.shape == (config.dimensions,)
    assert vector.dtype == np.float32


def test_embedding_the_same_text_twice_gives_the_same_vector(embedder: Embedder) -> None:
    """Idempotent ingest rests on this; without it the corpus fingerprint means nothing."""
    text = "The organization reviews and analyzes audit records for inappropriate activity."

    assert np.array_equal(embedder.embed_query(text), embedder.embed_query(text))
    assert np.array_equal(
        embedder.embed_documents([text, text])[0],
        embedder.embed_documents([text])[0],
    )


def test_vectors_are_normalised(embedder: Embedder, config: EmbedderConfig) -> None:
    """The pin says they are, and cosine ranking over unnormalised vectors is quietly wrong."""
    if not config.normalize:
        pytest.skip("the pinned model is not configured to normalise")

    vectors = embedder.embed_documents(["Account management.", "Audit record review."])

    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_the_query_prefix_is_applied(embedder: Embedder, config: EmbedderConfig) -> None:
    """The model is asymmetric. A question embedded through the document path is a silent loss."""
    if not config.query_prefix:
        pytest.skip("the pinned model is symmetric")

    question = "Who approves a new account?"

    assert not np.array_equal(
        embedder.embed_query(question),
        embedder.embed_documents([question])[0],
    )


def test_an_uncached_revision_is_an_error_rather_than_a_download(config: EmbedderConfig) -> None:
    """The no-network property, asserted rather than described.

    A revision nobody has ever fetched cannot be in any cache, so the only way this could succeed
    is by reaching the network — which is exactly what must not happen outside `make model`.
    """
    with pytest.raises(EmbedderError, match="not in the local cache"):
        load_embedder(config.model_copy(update={"revision": "0" * 40}))
