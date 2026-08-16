"""Two ways to get an answer -- ask the provider, or read a recorded one -- behind one interface.

A thin `Protocol` over the provider SDK, with no framework underneath it. There is nothing here to
orchestrate: one prompt goes out, one answer comes back. A graph or an agent loop would be a thing
to add once a measurement says it helps, and adding one now would mean the first such measurement
was taken against it.

**Mode is derived, never configured.** `Settings.mode` reports replay when no key is present, so
the absence of credentials selects the recorded path rather than crashing on the first request.
That is what makes `docker compose up` work on a machine that has never had an API key.

**What a miss costs depends on who asked, and the two answers are opposite.**

- The console and the API degrade. A reviewer typing their own question has no recorded answer for
  it, and the honest response is the retrieval they can check plus a plain statement that
  generating an answer needs a key. `complete_or_decline` is that path.
- Anything measuring quality must fail hard. A miss counted as a refusal would fail the
  false-refusal gate on every retrieval improvement -- because improving retrieval changes the
  prompt, which changes the key, which misses -- and a miss quietly skipped shrinks the
  denominator, which flatters every rate computed from it. So `complete` raises, and the harness
  does not catch it.

The harness itself does not exist yet. The error type and the two paths are defined here so that
when it arrives it cannot quietly adopt a softer rule.

**A miss can never be represented as a refusal, structurally rather than by convention.** A
refusal is a `Completion`: the model was asked, it answered, and its answer declines. A miss is a
`GenerationDeclined`, which has no field that could hold model text, because no model was reached.
Conflating them would require inventing a new type, which is a change somebody reviews.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Literal, Protocol

from warrant.fixtures.request import ModelRequest
from warrant.fixtures.store import Fixture, FixtureStore, Usage
from warrant.manifest import verify_manifest
from warrant.settings import Mode, get_settings

# Why no answer was generated. One value today, and a closed set rather than free text so that a
# caller can branch on it and so that a second reason has to be added deliberately.
DeclineReason = Literal["fixture_miss"]


class ModelError(Exception):
    """A model call could not be completed."""


class FixtureMissError(ModelError):
    """No recorded call matches this request.

    Its own type, and never raised for anything else, because two very different callers have to
    tell it apart from every other failure: one degrades on it, and one must fail the run on it.
    A generic error would collapse "nobody recorded this" into "the provider is down", and those
    call for opposite responses.

    Carries the key and the canonical request, because the only useful question on seeing this is
    "how does this differ from what was recorded?", and a bare hex digest cannot answer it.
    """

    def __init__(self, request: ModelRequest) -> None:
        self.request = request
        self.key = request.key

        super().__init__(
            f"No recorded {request.purpose} call is stored under {self.key}. The key covers the "
            f"whole request -- model {request.model_version}, the rendered prompt, and every "
            "sampling parameter -- so a miss means one of those differs from anything recorded, "
            "not that the question is unfamiliar. Re-record to close it; there is deliberately no "
            "nearest match."
        )


class ProviderError(ModelError):
    """The provider returned something this build cannot record.

    Separate from a miss for the reason above: it is not degradable, and it is not the harness's
    signal to demand a re-record either. It means the call happened and its result is unusable.
    """


@dataclass(frozen=True, slots=True)
class Completion:
    """An answer that exists, whatever it says.

    A model that declines to answer produces one of these, carrying the declining text. That is
    the deliberate asymmetry with `GenerationDeclined` below: a refusal is something the model
    said, and this is the type for things the model said.
    """

    answer: str

    model: str
    model_version: str
    usage: Usage

    # Which path produced it. Carried on the result rather than looked up later, so a caller
    # cannot hold an answer apart from the knowledge of whether a provider was reached.
    source: Mode

    # `None` for a live call, which is happening now. A replayed answer carries the day it was
    # recorded, so anything displaying it can say how old it is.
    recorded_on: date | None = None


@dataclass(frozen=True, slots=True)
class GenerationDeclined:
    """No answer was generated, and there is no text to show.

    **There is no field here that could hold model output, and that is the design.** This is what
    the console renders as its second state: real retrieval, real clause text, and a plain
    sentence saying generation needs a key. A caller cannot accidentally treat it as an answer,
    because there is nothing on it to treat as one.
    """

    reason: DeclineReason
    detail: str


class ModelClient(Protocol):
    """Somewhere a rendered request can be turned into an answer."""

    @property
    def mode(self) -> Mode:
        """Whether this client reaches a provider or reads recordings."""
        ...

    def complete(self, request: ModelRequest) -> Completion:
        """Answer the request, or raise.

        Raises `FixtureMissError` when nothing recorded matches, on any implementation that serves
        recordings. Callers that must degrade use `complete_or_decline`; callers that must fail
        call this and let it propagate.
        """
        ...


class ReplayClient:
    """Serves recorded answers. No key, no network, no nearest match.

    Checks the manifest once, at construction, before it will serve anything. A recording made
    from inputs this build no longer carries is not a slightly stale answer -- the retrieved text
    inside its prompt came from a different corpus -- and the check is the difference between
    noticing that and serving it.

    Once, rather than per call, because verification re-reads and re-chunks a ten-megabyte
    catalog. That is startup work; doing it per question would make every question pay for it.
    """

    def __init__(self, store: FixtureStore, verify: Callable[[], object] | None = None) -> None:
        """Take a store, and the check that must pass before it is read.

        `verify` is injectable for the reason `manifest.check` is split out of `verify_manifest`:
        a test needs to present a mismatch that no code currently produces, and it needs the other
        tests in the file not to spend their runtime re-parsing the catalog. Left alone it is the
        real check.
        """
        _verified(verify)

        self._store = store

    @property
    def mode(self) -> Mode:
        return "replay"

    def complete(self, request: ModelRequest) -> Completion:
        fixture = self._store.get(request.key)

        if fixture is None:
            raise FixtureMissError(request)

        return _replayed(fixture)


class LiveClient:
    """Calls the provider. Requires a key.

    Every sampling parameter is sent explicitly, including ones whose value equals the provider's
    current default, because anything not sent is decided by a default that sits outside the key
    and is free to move when the provider moves it.
    """

    def __init__(self, api_key: str) -> None:
        # Imported here rather than at module scope so that the replay path -- the one a reviewer
        # with no key takes -- does not depend on the provider SDK being importable at all.
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)

    @property
    def mode(self) -> Mode:
        return "live"

    def complete(self, request: ModelRequest) -> Completion:
        sampling = request.sampling

        response = self._client.chat.completions.create(
            model=request.model_version,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            seed=sampling.seed,
            max_completion_tokens=sampling.max_output_tokens,
        )

        # The provider echoes which model actually served the call. A dated snapshot was requested
        # precisely so that this cannot drift, and checking it is what turns a silent substitution
        # into a failure -- otherwise a recording would be filed under a key naming a model that
        # did not produce it, and every later replay would be confidently mislabelled.
        if response.model != request.model_version:
            raise ProviderError(
                f"The request asked for {request.model_version!r} and the provider served "
                f"{response.model!r}. A recording of this call would be filed under a key naming "
                "the model that was asked for, so nothing downstream would ever notice the "
                "substitution."
            )

        # Every remaining check below refuses a result that is *usable-looking but unrecordable*.
        # This client exists to produce recordings, and a recording is forever: anything wrong
        # with a completion has to be caught here or it is replayed indefinitely by a path that
        # cannot re-examine it.
        if not response.choices:
            raise ProviderError(
                f"The provider returned no choices for a {request.purpose} call to "
                f"{request.model_version}. There is no completion here to record or to answer "
                "with."
            )

        choice = response.choices[0]

        # Anything but a natural stop means the text is not the answer the model would have given.
        # `length` is the one that arrives silently and matters most: an answer citing several
        # controls is exactly the shape that reaches the output cap, and it comes back as prose
        # that ends mid-sentence with nothing in the payload marking it as incomplete. Recording
        # that would file a truncated answer as a finished one.
        if choice.finish_reason != "stop":
            raise ProviderError(
                f"The provider stopped a {request.purpose} call to {request.model_version} with "
                f"{choice.finish_reason!r} rather than finishing it. Whatever text came back is "
                "not a completed answer, and recording it would preserve the truncation as though "
                "the model had chosen to stop there."
            )

        # A safety refusal arrives here rather than in `content`, and it is deliberately not turned
        # into an answer. A model declining because the retrieved text does not support a claim is
        # ordinary content and becomes a `Completion` like any other -- that is the behaviour the
        # prompt asks for. This is the provider's own layer declining to run the request at all,
        # which is a fact about the request rather than an answer to it, and recording it would
        # enshrine one moment of a classifier as this question's permanent answer.
        if choice.message.refusal is not None:
            raise ProviderError(
                f"The provider refused a {request.purpose} call to {request.model_version}: "
                f"{choice.message.refusal!r}. That is the provider declining to answer, not the "
                "model answering that it cannot -- so there is nothing here to record."
            )

        answer = choice.message.content

        if answer is None:
            raise ProviderError(
                f"The provider returned no content for a {request.purpose} call to "
                f"{request.model_version}. There is nothing to record and nothing to answer with."
            )

        usage = response.usage

        if usage is None:
            raise ProviderError(
                "The provider returned no usage figures. Those are the counts a recorded call "
                "carries, and a fixture without them cannot report what the call cost."
            )

        # Both taken from the request rather than from the client's own pin. `build` accepts a
        # config override, so the two can differ -- and the request is the one the key was
        # computed over, which makes it the only pair that describes what was actually recorded.
        return Completion(
            answer=answer,
            model=request.model,
            model_version=request.model_version,
            usage=Usage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            ),
            source="live",
        )


def complete_or_decline(
    client: ModelClient,
    request: ModelRequest,
) -> Completion | GenerationDeclined:
    """Answer the request, or say plainly that it was not answered.

    **The degrading path, and only for callers that should degrade** -- the console and the API,
    where a reviewer typing an unrecorded question should see the retrieval work rather than an
    error. Anything measuring quality calls `client.complete` directly, so a miss ends the run
    naming what has to be re-recorded.

    Only a miss is caught. A provider failure or a manifest mismatch is not something to render a
    polite sentence about; those propagate.
    """
    try:
        return client.complete(request)
    except FixtureMissError as miss:
        return GenerationDeclined(
            reason="fixture_miss",
            detail=(
                "No recorded answer matches this question, so none is shown. Retrieval ran and "
                "the clauses above are the real result; generating an answer over them needs an "
                f"API key. Recorded calls are keyed on the whole request ({miss.key[:12]}), so "
                "nothing similar is substituted."
            ),
        )


def get_model_client(
    store: FixtureStore,
    verify: Callable[[], object] | None = None,
) -> ModelClient:
    """The client this process should use, decided by whether a key is present.

    No `WARRANT_MODE` variable, for the reason `Settings.mode` gives: a mode set alongside a key
    can disagree with it, and "live with no credentials" is not a configuration, it is a crash
    deferred to the first request.
    """
    settings = get_settings()

    if settings.mode == "replay":
        return ReplayClient(store, verify)

    # The live path checks the manifest too, and for a stronger reason than the replay path does.
    # Replay serves recordings made from inputs this build may no longer carry; live *creates*
    # them, over prompts holding text retrieved from a corpus the manifest says has moved. A
    # recording made against disowned inputs is a fresh artefact that is already stale, and
    # skipping the check here would mean the same command hard-failed without a key and ran
    # cheerfully with one.
    _verified(verify)

    # Narrowing for the type checker. `mode` is `live` only when a key is present, so this cannot
    # be reached with `None` -- but the two live one property apart, and an assertion here is
    # cheaper than the two drifting silently.
    assert settings.model_api_key is not None

    return LiveClient(settings.model_api_key)


def _verified(verify: Callable[[], object] | None) -> None:
    """Run the manifest check, or whatever a test substituted for it.

    One helper rather than the conditional written twice, so that a path added later cannot get
    the default wrong -- and so it is visible that both paths run the same check rather than one
    of them running a weaker version of it.
    """
    (verify if verify is not None else verify_manifest)()


def _replayed(fixture: Fixture) -> Completion:
    """A stored fixture as a completion, byte for byte.

    Nothing is recomputed, reformatted or re-wrapped on the way out. Replaying the same request
    twice returns the same bytes because the bytes are what was stored.
    """
    return Completion(
        answer=fixture.answer,
        model=fixture.model,
        model_version=fixture.model_version,
        usage=fixture.usage,
        source="replay",
        recorded_on=fixture.recorded_on,
    )
