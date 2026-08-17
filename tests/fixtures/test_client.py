"""Replay, and the two opposite things a missing recording means.

The property under test throughout is that nothing is ever substituted. A request either matches a
recording exactly or it does not match at all, and when it does not, what happens next depends
entirely on who asked -- which is the distinction this module exists to make impossible to blur.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from warrant.fixtures.client import (
    Completion,
    FixtureMissError,
    GenerationDeclined,
    ModelError,
    ReplayClient,
    complete_or_decline,
    get_model_client,
)
from warrant.fixtures.request import ModelRequest
from warrant.fixtures.store import Fixture, MappingFixtureStore, Usage
from warrant.manifest import (
    Manifest,
    ManifestMismatchError,
    Observation,
    check,
    get_manifest,
)
from warrant.model_config import ModelConfig, get_model_config
from warrant.settings import get_settings

PROMPT = (
    "Answer only from the controls below, citing their identifiers.\n\n"
    "AC-2 Account Management: the organization manages system accounts.\n\n"
    "Question: how are inactive accounts disabled?"
)

ANSWER = "AC-2 requires the organization to disable accounts after a defined inactivity period."


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Run each test as if on a machine that has never configured this project.

    The same shape `test_settings` uses, and load-bearing here rather than tidy: a developer with
    a key exported for live-mode work would otherwise select the live client in the tests that
    assert an absent key selects replay.
    """
    monkeypatch.delenv("WARRANT_MODEL_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    yield

    get_settings.cache_clear()


@pytest.fixture
def config() -> ModelConfig:
    return get_model_config()


@pytest.fixture
def request_(config: ModelConfig) -> ModelRequest:
    return ModelRequest.build("generation", PROMPT, config)


@pytest.fixture
def fixture(request_: ModelRequest, config: ModelConfig) -> Fixture:
    return Fixture(
        key=request_.key,
        answer=ANSWER,
        model=config.name,
        model_version=config.version,
        recorded_on=date(2026, 8, 16),
        usage=Usage(prompt_tokens=812, completion_tokens=64),
    )


@pytest.fixture
def store(fixture: Fixture) -> MappingFixtureStore:
    return MappingFixtureStore({fixture.key: fixture})


@pytest.fixture
def client(store: MappingFixtureStore) -> ReplayClient:
    """A replay client whose manifest check is stubbed out.

    Stubbed because the real check re-reads and re-chunks a ten-megabyte catalog, and a file that
    paid that per test would spend its whole runtime there. The real check is wired in
    `test_the_default_check_is_the_real_one` and refused in `test_a_moved_input_is_not_served`,
    which are the two things worth spending the time on.
    """
    return ReplayClient(store, verify=lambda: None)


def test_replay_serves_a_recorded_answer(client: ReplayClient, request_: ModelRequest) -> None:
    completion = client.complete(request_)

    assert completion.answer == ANSWER
    assert completion.source == "replay"
    assert completion.recorded_on == date(2026, 8, 16)
    assert completion.usage.prompt_tokens == 812


def test_replay_needs_no_key_and_no_network(
    client: ReplayClient,
    request_: ModelRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property `docker compose up` on a clean machine rests on.

    Asserted by making the network unusable rather than by trusting that nothing calls it. A
    replay path that quietly reached a provider would still pass a test that only checked the
    answer came back.
    """

    def _refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("replay opened a socket")

    monkeypatch.setattr(socket, "socket", _refuse)

    assert get_settings().mode == "replay"
    assert client.complete(request_).answer == ANSWER


def test_the_same_request_replays_byte_identically(
    client: ReplayClient,
    request_: ModelRequest,
) -> None:
    """Nothing is reformatted on the way out, so two runs cannot differ in whitespace."""
    first = client.complete(request_)
    second = client.complete(request_)

    assert first.answer == second.answer
    assert first == second


def test_a_changed_prompt_template_misses(client: ReplayClient, request_: ModelRequest) -> None:
    """End to end, through the client, the failure the ticket exists for.

    `test_request` proves the key moves. This proves the client acts on that rather than falling
    back to something near it.
    """
    edited = request_.model_copy(update={"prompt": PROMPT.replace("Answer", "answer", 1)})

    with pytest.raises(FixtureMissError):
        client.complete(edited)


def test_a_changed_sampling_parameter_misses(
    client: ReplayClient,
    request_: ModelRequest,
) -> None:
    warmer = request_.sampling.model_copy(update={"temperature": 0.7})

    with pytest.raises(FixtureMissError):
        client.complete(request_.model_copy(update={"sampling": warmer}))


def test_a_paraphrased_question_is_not_answered_from_the_nearest_recording(
    client: ReplayClient,
    config: ModelConfig,
) -> None:
    """The single most important line in the replay story.

    Serving the nearest recorded answer to a question nobody recorded produces an answer whose
    warrant does not cover what was asked -- which is exactly the failure this project exists to
    detect, arriving from inside the project.
    """
    paraphrase = PROMPT.replace(
        "how are inactive accounts disabled?",
        "what happens to dormant accounts?",
    )

    with pytest.raises(FixtureMissError):
        client.complete(ModelRequest.build("generation", paraphrase, config))


def test_a_miss_names_its_key_and_carries_the_request(
    client: ReplayClient,
    request_: ModelRequest,
) -> None:
    """The only useful question on a miss is what differed, so the error carries the material."""
    with pytest.raises(FixtureMissError) as raised:
        client.complete(request_.model_copy(update={"prompt": "unrecorded"}))

    assert raised.value.key in str(raised.value)
    assert raised.value.request.prompt == "unrecorded"


def test_a_miss_degrades_for_the_caller_that_should_degrade(
    client: ReplayClient,
    request_: ModelRequest,
) -> None:
    """The console and the API path: retrieval shown, generation declined, not an error."""
    outcome = complete_or_decline(client, request_.model_copy(update={"prompt": "unrecorded"}))

    assert isinstance(outcome, GenerationDeclined)
    assert outcome.reason == "fixture_miss"


def test_a_hit_is_returned_unchanged_by_the_degrading_path(
    client: ReplayClient,
    request_: ModelRequest,
) -> None:
    outcome = complete_or_decline(client, request_)

    assert isinstance(outcome, Completion)
    assert outcome.answer == ANSWER


def test_a_miss_is_never_representable_as_a_refusal() -> None:
    """Asserted on the types, so nothing measuring refusal rates can conflate the two.

    A refusal is a `Completion`: the model was reached, it answered, and its answer declines. A
    miss is a `GenerationDeclined`, and there is no field on it that could hold model text -- so
    counting misses as refusals would require inventing a type, which is a change somebody
    reviews rather than a line somebody writes.
    """
    declined = GenerationDeclined(reason="fixture_miss", detail="")

    assert not isinstance(declined, Completion)
    assert {field for field in Completion.__slots__} & {"answer"}
    assert "answer" not in GenerationDeclined.__slots__

    # And the error itself is its own type rather than something a broad `except` would swallow
    # alongside a provider outage, which calls for the opposite response.
    assert issubclass(FixtureMissError, ModelError)
    assert not issubclass(ModelError, FixtureMissError)


def test_the_harness_path_raises_rather_than_returning_anything(
    client: ReplayClient,
    request_: ModelRequest,
) -> None:
    """`complete` is what anything measuring quality calls, and it has no degrading branch.

    A miss silently skipped shrinks the denominator of every rate computed over the run, which
    flatters the result without changing anything a reader could see.
    """
    with pytest.raises(FixtureMissError):
        client.complete(request_.model_copy(update={"prompt": "unrecorded"}))


def test_a_moved_input_is_not_served(store: MappingFixtureStore, request_: ModelRequest) -> None:
    """A recording made from inputs this build no longer carries is refused before it is read.

    Driven through the real comparison with hand-built observations, the way `test_manifest`
    does: recomputing them would mean parsing the catalog, and the thing under test is that the
    client refuses on a mismatch rather than how a mismatch is detected.
    """
    manifest = get_manifest()
    observed = _as_observed(manifest)
    observed["chunker"] = Observation("chunker 2", "0" * 64)

    with pytest.raises(ManifestMismatchError):
        ReplayClient(store, verify=lambda: check(manifest, observed))


def test_matching_inputs_are_served(store: MappingFixtureStore, request_: ModelRequest) -> None:
    """The other half of the check: it refuses a mismatch and does not refuse a match."""
    manifest = get_manifest()
    client = ReplayClient(store, verify=lambda: check(manifest, _as_observed(manifest)))

    assert client.complete(request_).answer == ANSWER


@pytest.mark.tokenizer
def test_the_default_check_is_the_real_one(
    cached_encoding: None, store: MappingFixtureStore
) -> None:
    """Nothing else proves the injectable check is wired to `verify_manifest` by default.

    Parses the catalog and counts a sample with the pinned encoding, which is why it is one test
    rather than a fixture every test uses -- and why it is the only one here that needs a fetch to
    have happened.
    """
    assert ReplayClient(store).mode == "replay"


def test_no_key_selects_replay(store: MappingFixtureStore, request_: ModelRequest) -> None:
    """Absence of credentials is the selector, and it is not a crash deferred to first request."""
    client = get_model_client(store, verify=lambda: None)

    assert isinstance(client, ReplayClient)
    assert client.complete(request_).answer == ANSWER


def test_a_key_selects_live(store: MappingFixtureStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructed, not called. A live call would need a real key and a network."""
    monkeypatch.setenv("WARRANT_MODEL_API_KEY", "sk-not-a-real-key")
    get_settings.cache_clear()

    client = get_model_client(store, verify=lambda: None)

    assert client.mode == "live"
    assert not isinstance(client, ReplayClient)


def _as_observed(manifest: Manifest) -> dict[str, Observation]:
    """The manifest's own recorded values, as though they had just been observed."""
    return {
        name: Observation(entry.identity, entry.digest) for name, entry in manifest.entries.items()
    }
