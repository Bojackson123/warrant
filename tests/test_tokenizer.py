"""Counting a prompt: locally, with the model's own encoding, and against nothing.

The number this produces is what a token gate later fires on, so two things have to hold before it
is worth reporting at all. It has to come from the encoding the provider bills against rather than
an approximation of it, and getting it has to cost no network and no API call -- otherwise the
check it exists for is behind a credential and cannot run on a pull request.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from warrant import manifest as manifest_module
from warrant.manifest import ManifestMismatchError, observe, verify_manifest
from warrant.model_config import ModelConfig, TokenizerPin, get_model_config
from warrant.prompt import render_prompt
from warrant.retrieval.search import Retrieval, RetrievedChunk
from warrant.tokenizer import Tokenizer, TokenizerError, load_tokenizer, tokenizer_fingerprint

pytestmark = pytest.mark.tokenizer


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    """The pinned encoding, or a skip naming the command that fetches it.

    The same shape `test_embedding` uses for the embedding weights, and for the same reason: an
    absent cache is a machine that has not run the one-off fetch, not a broken build.
    """
    try:
        return load_tokenizer()
    except TokenizerError as error:
        pytest.skip(f"the pinned encoding is not cached: {error}")


@pytest.fixture
def retrieval() -> Retrieval:
    return Retrieval(
        question="How are inactive accounts disabled?",
        k=1,
        chunks=(
            RetrievedChunk(
                rank=1,
                score=0.8,
                chunk_id="ac-2#a",
                control_id="ac-2",
                base_control_id="ac-2",
                control_label="AC-2",
                title="Account Management",
                part_path="a",
                text="Disable accounts when they have expired or are no longer required.",
            ),
        ),
    )


def test_the_pinned_encoding_is_the_one_that_loads(tokenizer: Tokenizer) -> None:
    """What loads is what `data/model.json` names, not what the library maps the model to."""
    assert tokenizer.name == get_model_config().tokenizer.encoding


def test_counting_reaches_no_network(
    monkeypatch: pytest.MonkeyPatch, tokenizer: Tokenizer, retrieval: Retrieval
) -> None:
    """The property the whole local-counting claim rests on, proved rather than asserted.

    A path that quietly fetched something would otherwise pass on the strength of the number
    coming back, on a machine that happens to have a connection.
    """

    def refuse(*arguments: Any, **keywords: Any) -> socket.socket:
        raise AssertionError("counting a prompt opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)

    assert tokenizer.count(render_prompt(retrieval)) > 0


def test_adding_text_to_a_prompt_raises_the_count(
    tokenizer: Tokenizer, retrieval: Retrieval
) -> None:
    """The mechanism a token gate fires on: more prompt is a larger number, monotonically."""
    smaller = tokenizer.count(render_prompt(retrieval))

    longer = Retrieval(
        question=retrieval.question,
        k=retrieval.k,
        chunks=(
            *retrieval.chunks,
            RetrievedChunk(
                rank=2,
                score=0.7,
                chunk_id="ac-2.3#a",
                control_id="ac-2.3",
                base_control_id="ac-2",
                control_label="AC-2(3)",
                title="Disable Accounts",
                part_path="a",
                text="Disable accounts within an organization-defined time period.",
            ),
        ),
    )

    assert tokenizer.count(render_prompt(longer)) > smaller


def test_counting_is_deterministic(tokenizer: Tokenizer, retrieval: Retrieval) -> None:
    """Two counts of one prompt agree, so a reported size describes the prompt and not the run."""
    rendered = render_prompt(retrieval)

    assert tokenizer.count(rendered) == tokenizer.count(rendered)


def test_an_empty_string_counts_as_nothing(tokenizer: Tokenizer) -> None:
    """No hidden preamble is added to what it is handed; the count covers the string given."""
    assert tokenizer.count("") == 0


def test_the_manifest_records_the_encoding_and_not_the_model(tokenizer: Tokenizer) -> None:
    """The identity is the encoding alone.

    Naming the model here too would fire this entry on a model swap that changed no count — and
    the model's identity is already hashed into every recorded call's key, so it cannot move
    unannounced anyway.
    """
    observed = observe()["tokenizer"]
    model = get_model_config()

    assert observed.identity == model.tokenizer.encoding
    assert model.version not in (observed.identity or "")


def test_a_repointed_encoding_fails_on_its_digest(
    monkeypatch: pytest.MonkeyPatch, tokenizer: Tokenizer
) -> None:
    """The failure with no wrong artefact left behind.

    An encoding that keeps its name and moves its token boundaries invalidates nothing stored and
    makes every size ever reported wrong. The digest is over what it counts, so it notices.
    """
    monkeypatch.setattr(manifest_module, "tokenizer_fingerprint", lambda model: "0" * 64)

    with pytest.raises(ManifestMismatchError) as raised:
        verify_manifest()

    message = str(raised.value)
    assert "`tokenizer` entry" in message
    assert "is unchanged" in message


def test_the_recorded_cost_of_a_tokenizer_change_is_counts_alone() -> None:
    """It re-records nothing, which is exactly why it needs an entry."""
    entry = manifest_module.get_manifest().entries["tokenizer"]

    assert set(entry.invalidates) == {"counts"}


def test_an_encoding_the_library_does_not_know_is_refused() -> None:
    """A typo in the pin fails naming the pin, rather than falling back to a default."""
    pinned = get_model_config()
    wrong = ModelConfig(
        name=pinned.name,
        version=pinned.version,
        sampling=pinned.sampling,
        tokenizer=TokenizerPin(encoding="not-an-encoding"),
    )

    with pytest.raises(TokenizerError) as raised:
        tokenizer_fingerprint(wrong)

    assert "not-an-encoding" in str(raised.value)
