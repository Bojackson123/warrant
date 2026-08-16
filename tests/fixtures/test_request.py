"""The keying rule, tested from the direction it fails.

Every test below changes one thing about a request and asserts the key moves. That is the whole
ticket: a key computed over a subset of the request would hold still for at least one of these,
and holding still means a recorded answer is served for a request that would produce a different
one.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from warrant.fixtures.request import ModelRequest
from warrant.model_config import ModelConfig, Sampling, get_model_config

PROMPT = (
    "Answer only from the controls below, citing their identifiers.\n\n"
    "AC-2 Account Management: the organization manages system accounts.\n\n"
    "Question: how are inactive accounts disabled?"
)


@pytest.fixture
def config() -> ModelConfig:
    return get_model_config()


@pytest.fixture
def request_(config: ModelConfig) -> ModelRequest:
    return ModelRequest.build("generation", PROMPT, config)


def test_the_key_is_stable_for_an_unchanged_request(request_: ModelRequest) -> None:
    """Twice over the same request is the same number, or nothing else here means anything."""
    assert request_.key == request_.key
    assert request_.key == ModelRequest.build("generation", PROMPT, get_model_config()).key


def test_one_changed_character_of_the_prompt_moves_the_key(request_: ModelRequest) -> None:
    """The criterion this whole module exists for.

    A prompt-template edit must produce a miss rather than a hit against the recorded answer for
    the old template. One character is the smallest edit anybody would make by accident, so it is
    the one worth pinning.
    """
    edited = request_.model_copy(update={"prompt": PROMPT.replace("Answer", "answer", 1)})

    assert edited.prompt != request_.prompt
    assert edited.key != request_.key


def test_trailing_whitespace_moves_the_key(request_: ModelRequest) -> None:
    """A change nobody sees in a diff still changes what is sent, so it still misses."""
    assert request_.model_copy(update={"prompt": PROMPT + "\n"}).key != request_.key


def test_a_changed_sampling_parameter_moves_the_key(request_: ModelRequest) -> None:
    warmer = request_.sampling.model_copy(update={"temperature": 0.7})

    assert request_.model_copy(update={"sampling": warmer}).key != request_.key


def test_every_sampling_parameter_is_in_the_key(request_: ModelRequest) -> None:
    """Field by field, so a parameter left out of the digest is caught by name.

    Checking one representative parameter would pass while three others sat outside the key, and
    a parameter outside the key is a value that changes the answer without changing which
    recording is served.
    """
    moved = {
        "temperature": 0.5,
        "top_p": 0.9,
        "seed": 1,
        "max_output_tokens": 2048,
    }

    for field, value in moved.items():
        sampling = request_.sampling.model_copy(update={field: value})

        assert sampling != request_.sampling, field
        assert request_.model_copy(update={"sampling": sampling}).key != request_.key, field


def test_the_purpose_is_in_the_key(config: ModelConfig) -> None:
    """A generation fixture must never be able to serve a judge call.

    The judge prompt would have to coincide exactly with a generation prompt for this to matter,
    which is unlikely -- and unlikely is not the same as impossible, and the cost of it happening
    is a grade produced by something that was not the grader.
    """
    generation = ModelRequest.build("generation", PROMPT, config)
    judge = ModelRequest.build("judge", PROMPT, config)

    assert generation.key != judge.key


def test_the_model_version_is_in_the_key(request_: ModelRequest) -> None:
    """A model swap invalidates every recorded answer, and this is what enforces it."""
    later = request_.model_copy(update={"model_version": request_.model_version + "-later"})

    assert later.key != request_.key


def test_the_key_does_not_move_when_fields_are_written_in_another_order(
    request_: ModelRequest,
) -> None:
    """Two requests equal in content hash the same however they were constructed.

    Otherwise the key would record the order somebody happened to type the arguments in, and a
    refactor would look exactly like a changed prompt.
    """
    reordered = ModelRequest(
        sampling=request_.sampling,
        prompt=request_.prompt,
        model_version=request_.model_version,
        model=request_.model,
        purpose=request_.purpose,
    )

    assert reordered.key == request_.key


def test_the_canonical_form_shows_what_was_hashed(request_: ModelRequest) -> None:
    """The key answers "did this match"; this answers "how did it differ".

    JSON-encoded, so the prompt's newlines are escaped rather than literal -- which is what makes
    two of these diffable line by line at all.
    """
    canonical = request_.canonical()

    assert json.dumps(request_.prompt)[1:-1] in canonical
    assert request_.model_version in canonical
    assert '"purpose":"generation"' in canonical
    assert '"temperature":0.0' in canonical


def test_an_unknown_field_is_refused(config: ModelConfig) -> None:
    """A misspelt field would be dropped silently and left out of the key.

    That is the same failure as keying on a subset, arriving through a typo rather than through a
    decision, so it is refused rather than ignored.
    """
    with pytest.raises(ValidationError):
        ModelRequest(
            purpose="generation",
            model=config.name,
            model_version=config.version,
            prompt=PROMPT,
            sampling=config.sampling,
            temperture=0.0,  # type: ignore[call-arg]
        )


def test_build_takes_the_model_from_the_pin(config: ModelConfig) -> None:
    """So no caller restates the pin, and none can restate it wrongly."""
    built = ModelRequest.build("generation", PROMPT, config)

    assert built.model == config.name
    assert built.model_version == config.version
    assert built.sampling == config.sampling


def test_a_request_cannot_be_mutated_after_its_key_is_taken(request_: ModelRequest) -> None:
    with pytest.raises(ValidationError):
        request_.prompt = "something else"  # type: ignore[misc]


def test_sampling_refuses_values_that_are_not_calls() -> None:
    """Zero output tokens is not a cheaper call, it is a call that cannot answer."""
    with pytest.raises(ValidationError):
        Sampling(temperature=0.0, top_p=1.0, seed=0, max_output_tokens=0)
