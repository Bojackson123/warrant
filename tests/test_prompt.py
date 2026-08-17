"""The one renderer, and what has to be true of the string it produces.

Two properties, and the second is the one worth spending a file on. The prompt has to carry every
retrieved control in the order the search ranked them, because an answer is only checkable against
text the reader can see. And the rendering has to be pinned by something other than a version
constant, because a version constant is a claim and the failure it cannot catch -- somebody changes
how the sections go together and does not say so -- silently invalidates every recorded answer.
"""

from __future__ import annotations

import pytest

from warrant import manifest as manifest_module
from warrant.manifest import ManifestMismatchError, observe, verify_manifest
from warrant.prompt import PROMPT_TEMPLATE_VERSION, prompt_fingerprint, render_prompt
from warrant.retrieval.search import Retrieval, RetrievedChunk


def _chunk(rank: int, label: str, title: str, text: str) -> RetrievedChunk:
    identifier = label.lower()

    return RetrievedChunk(
        rank=rank,
        score=1.0 / rank,
        chunk_id=f"{identifier}#a",
        control_id=identifier,
        base_control_id=identifier,
        control_label=label,
        title=title,
        part_path="a",
        text=text,
    )


@pytest.fixture
def retrieval() -> Retrieval:
    return Retrieval(
        question="How are inactive accounts disabled?",
        k=2,
        chunks=(
            _chunk(1, "AC-2", "Account Management", "Disable accounts when they have expired."),
            _chunk(2, "AC-2(3)", "Disable Accounts", "Disable accounts after an inactive period."),
        ),
    )


def test_every_retrieved_control_reaches_the_prompt(retrieval: Retrieval) -> None:
    """A control that was retrieved and not rendered is a citation nobody can check."""
    rendered = render_prompt(retrieval)

    for chunk in retrieval.chunks:
        assert chunk.control_label in rendered
        assert chunk.title in rendered
        assert chunk.text in rendered


def test_the_excerpts_keep_the_order_the_search_ranked_them_in(retrieval: Retrieval) -> None:
    """Rank order is part of the rendered string, so it is part of every key taken over it."""
    rendered = render_prompt(retrieval)

    positions = [rendered.index(chunk.text) for chunk in retrieval.chunks]

    assert positions == sorted(positions)


def test_the_question_is_rendered_after_the_retrieved_text(retrieval: Retrieval) -> None:
    """The question comes last, so nothing typed into it can be read as a retrieved control."""
    rendered = render_prompt(retrieval)

    assert rendered.index(retrieval.question) > rendered.index(retrieval.chunks[-1].text)


def test_the_template_asks_the_model_to_decline(retrieval: Retrieval) -> None:
    """Version 1 carries the refusal instruction rather than gaining it later.

    Adding it afterwards would invalidate every answer recorded before it, so this is asserted
    from the day the template exists rather than from the day something measures refusals.
    """
    rendered = render_prompt(retrieval).lower()

    assert "does not support an answer" in rendered
    assert "declining is the correct response" in rendered


def test_rendering_is_deterministic(retrieval: Retrieval) -> None:
    """The same retrieval renders to the same bytes, which is what the key over it rests on."""
    assert render_prompt(retrieval) == render_prompt(retrieval)


def test_a_different_question_over_the_same_chunks_renders_differently(
    retrieval: Retrieval,
) -> None:
    """The question is inside the rendered string, so two questions cannot share one recording."""
    paraphrased = Retrieval(
        question="What happens to accounts nobody has used?",
        k=retrieval.k,
        chunks=retrieval.chunks,
    )

    assert render_prompt(paraphrased) != render_prompt(retrieval)


def test_the_fingerprint_is_stable_across_calls() -> None:
    """It is a digest of this build's template, not of anything that varies between runs."""
    assert prompt_fingerprint() == prompt_fingerprint()


def test_the_manifest_records_the_template_this_build_carries() -> None:
    """The version and the fingerprint reach the manifest as one entry.

    The entry is what makes a template change announce itself. Observing it here is the check that
    the two halves are wired to the module rather than to a literal somebody typed in.
    """
    observed = observe()["prompt_template"]

    assert observed.identity == f"template {PROMPT_TEMPLATE_VERSION}"
    assert observed.digest is not None
    assert observed.digest != prompt_fingerprint()


def test_an_announced_template_change_fails_on_its_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bumped version with no re-record: the change was declared and nothing was rebuilt."""
    monkeypatch.setattr(manifest_module, "PROMPT_TEMPLATE_VERSION", "2")

    with pytest.raises(ManifestMismatchError) as raised:
        verify_manifest()

    message = str(raised.value)
    assert "`prompt_template` entry" in message
    assert "'template 1'" in message and "'template 2'" in message


def test_an_unannounced_template_change_fails_on_its_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure a version string cannot catch, from the template's side.

    Somebody reorders a section or drops a field from an excerpt, leaves the version alone, and
    every recorded answer is now keyed on a prompt this build no longer produces. Nothing about the
    template announces that; the digest is what notices.
    """
    monkeypatch.setattr(manifest_module, "prompt_fingerprint", lambda: "0" * 64)

    with pytest.raises(ManifestMismatchError) as raised:
        verify_manifest()

    message = str(raised.value)
    assert "`prompt_template` entry" in message
    assert "'template 1' is unchanged" in message


def test_the_recorded_cost_of_a_template_change_is_answers_and_grades() -> None:
    """Not the corpus. Changing the template re-records what a model said, not what was embedded.

    Recorded here rather than left to the file, because getting it wrong in the reassuring
    direction — declaring a cheaper cost than the change really has — is the one mistake the
    manifest cannot catch on its own.
    """
    entry = manifest_module.get_manifest().entries["prompt_template"]

    assert set(entry.invalidates) == {"generation", "judge"}
