"""What the live path refuses to record.

No network here: the provider's own response objects are constructed directly and handed to the
client, which is the only way to exercise the results that matter -- a truncated answer, a
refusal, a substituted model -- since none of them can be provoked on demand from a real call.

Every case below is a response that *looks usable*. That is the whole reason these checks exist:
this client's output becomes a recording, a recording is replayed indefinitely, and nothing
downstream re-examines it. Anything wrong with a completion has to be caught here or it is
preserved.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from warrant.fixtures.client import LiveClient, ProviderError, get_model_client
from warrant.fixtures.request import ModelRequest
from warrant.fixtures.store import MappingFixtureStore
from warrant.model_config import ModelConfig, get_model_config
from warrant.settings import get_settings

PROMPT = "AC-2: the organization manages system accounts.\n\nQ: inactive accounts?"

ANSWER = "AC-2 requires disabling inactive accounts."


@pytest.fixture(autouse=True)
def _live_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    monkeypatch.setenv("WARRANT_MODEL_API_KEY", "sk-not-a-real-key")
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


USAGE = CompletionUsage(prompt_tokens=41, completion_tokens=9, total_tokens=50)


def _response(
    model: str,
    content: str | None = ANSWER,
    finish_reason: str = "stop",
    refusal: str | None = None,
    choices: list[Choice] | None = None,
    usage: CompletionUsage | None = USAGE,
) -> ChatCompletion:
    """A provider response, built from the SDK's own types rather than a stand-in.

    A hand-rolled stub would let a test pass against a shape the provider never sends, which is
    the one way a test of provider handling can be worse than no test.
    """
    if choices is None:
        choices = [
            Choice(
                finish_reason=finish_reason,  # type: ignore[arg-type]
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content, refusal=refusal),
            )
        ]

    return ChatCompletion(
        id="chatcmpl-test",
        choices=choices,
        created=0,
        model=model,
        object="chat.completion",
        usage=usage,
    )


class Provider:
    """Stands in for the SDK's completions endpoint, serving one prepared response.

    Held per test rather than in a module-level variable, so nothing a test prepares can be read
    by the next one.
    """

    def __init__(self) -> None:
        self.response: ChatCompletion | None = None
        self.calls: list[dict[str, object]] = []

    def serves(self, response: ChatCompletion) -> None:
        self.response = response

    def create(self, **kwargs: object) -> ChatCompletion:
        assert self.response is not None, "the test prepared no provider response"

        self.calls.append(kwargs)

        return self.response


@pytest.fixture
def provider() -> Provider:
    return Provider()


@pytest.fixture
def client(provider: Provider, monkeypatch: pytest.MonkeyPatch) -> LiveClient:
    """A live client whose provider call returns whatever the test prepared."""
    built = LiveClient("sk-not-a-real-key")

    monkeypatch.setattr(built._client.chat.completions, "create", provider.create)

    return built


def test_a_finished_answer_is_returned(
    client: LiveClient,
    provider: Provider,
    request_: ModelRequest,
) -> None:
    provider.serves(_response(request_.model_version))

    completion = client.complete(request_)

    assert completion.answer == ANSWER
    assert completion.source == "live"
    assert completion.recorded_on is None
    assert completion.usage.prompt_tokens == 41


def test_every_pinned_sampling_parameter_is_sent(
    client: LiveClient,
    provider: Provider,
    request_: ModelRequest,
) -> None:
    """Nothing is left to a provider default.

    A parameter the client does not send is decided by a default that sits outside the key and is
    free to move when the provider moves it -- so the answer changes while the key does not.
    """
    provider.serves(_response(request_.model_version))
    client.complete(request_)

    sent = provider.calls[0]

    assert sent["model"] == request_.model_version
    assert sent["temperature"] == request_.sampling.temperature
    assert sent["top_p"] == request_.sampling.top_p
    assert sent["seed"] == request_.sampling.seed
    assert sent["max_completion_tokens"] == request_.sampling.max_output_tokens


def test_the_completion_describes_the_request_that_was_keyed(
    client: LiveClient,
    provider: Provider,
    request_: ModelRequest,
) -> None:
    """Provenance comes from the request, not from whatever pin the client happens to hold.

    `build` takes a config override, so the two can differ -- and a fixture pairing one pin's
    family name with another's version would be a provenance record that describes no model at
    all, in the fields whose only job is telling a reader which model produced this.
    """
    provider.serves(_response(request_.model_version))

    completion = client.complete(request_)

    assert completion.model == request_.model
    assert completion.model_version == request_.model_version


def test_a_substituted_model_is_refused(
    client: LiveClient,
    provider: Provider,
    request_: ModelRequest,
) -> None:
    """A dated snapshot was pinned so this cannot drift; checking is what makes that true."""
    provider.serves(_response("gpt-4.1-mini-2099-01-01"))

    with pytest.raises(ProviderError, match="2099"):
        client.complete(request_)


def test_a_truncated_answer_is_refused(
    client: LiveClient,
    provider: Provider,
    request_: ModelRequest,
) -> None:
    """The one that arrives silently.

    An answer citing several controls is exactly the shape that reaches the output cap, and it
    comes back as prose ending mid-sentence with nothing in the payload marking it incomplete.
    Recorded, it is replayed forever as though the model chose to stop there.
    """
    provider.serves(
        _response(
            request_.model_version,
            content="AC-2 requires the organization to disa",
            finish_reason="length",
        )
    )

    with pytest.raises(ProviderError, match="length"):
        client.complete(request_)


def test_a_filtered_answer_is_refused(
    client: LiveClient,
    provider: Provider,
    request_: ModelRequest,
) -> None:
    provider.serves(_response(request_.model_version, finish_reason="content_filter"))

    with pytest.raises(ProviderError, match="content_filter"):
        client.complete(request_)


def test_a_provider_refusal_is_refused_and_quoted(
    client: LiveClient,
    provider: Provider,
    request_: ModelRequest,
) -> None:
    """Not turned into an answer, and not thrown away either.

    A model declining because the retrieved text does not support a claim is ordinary content and
    becomes a completion like any other -- that is what the prompt asks for. This is the
    provider's own layer declining to run the request, which is a fact about the request rather
    than an answer to it. It is refused, and the error quotes it, because an error that cannot say
    why is read at the moment somebody is already confused.
    """
    provider.serves(
        _response(request_.model_version, content=None, refusal="I can't help with that.")
    )

    with pytest.raises(ProviderError, match="I can't help with that"):
        client.complete(request_)


def test_no_choices_is_refused_rather_than_raising_an_index_error(
    client: LiveClient,
    provider: Provider,
    request_: ModelRequest,
) -> None:
    """Every unusable provider result leaves here as one type.

    An `IndexError` escaping would slip past the `ModelError` a caller catches, which is the
    contract this module publishes.
    """
    provider.serves(_response(request_.model_version, choices=[]))

    with pytest.raises(ProviderError, match="no choices"):
        client.complete(request_)


def test_missing_usage_is_refused(
    client: LiveClient,
    provider: Provider,
    request_: ModelRequest,
) -> None:
    """A recording without counts cannot report what the call cost."""
    provider.serves(_response(request_.model_version, usage=None))

    with pytest.raises(ProviderError, match="usage"):
        client.complete(request_)


def test_the_live_path_checks_the_manifest_too() -> None:
    """And for a stronger reason than the replay path does.

    Replay serves recordings made from inputs this build may no longer carry; live creates them,
    over prompts holding text retrieved from a corpus the manifest says has moved. Skipping the
    check here would mean the same command hard-failed without a key and ran cheerfully with one.
    """
    checked: list[str] = []

    def _refuse() -> None:
        checked.append("ran")
        raise AssertionError("the manifest was checked")

    with pytest.raises(AssertionError, match="the manifest was checked"):
        get_model_client(MappingFixtureStore({}), verify=_refuse)

    assert checked == ["ran"]
