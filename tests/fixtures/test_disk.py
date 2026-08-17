"""What a recording looks like on disk, and the two things that has to be true of it.

It has to come back exactly as it went in, because replay is a claim about bytes and not about
meaning. And it has to be readable in a diff, because a re-record is reviewed by a person opening
one -- a requirement that a file can satisfy in the letter and defeat in practice, which is why it
is asserted here against the bytes rather than described in prose somewhere.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from warrant.fixtures.disk import (
    DirectoryFixtureStore,
    FixtureFileError,
    RecordedCall,
    read_fixture,
    write_fixture,
)
from warrant.fixtures.store import Fixture

KEY = "a" * 64

ANSWER = (
    "The catalog requires accounts to be disabled when they are no longer needed [AC-2].\n"
    "\n"
    "It does not state how long an account may sit unused before that applies; the period is\n"
    "left to the organization [AC-2(3)].\n"
)


def _call(key: str = KEY, answer: str = ANSWER) -> RecordedCall:
    return RecordedCall(
        key=key,
        purpose="generation",
        question="What happens to accounts nobody uses?",
        control_ids=("ac-2", "ac-2.3"),
        model="gpt-4.1-mini",
        model_version="gpt-4.1-mini-2025-04-14",
        recorded_on=date(2026, 8, 17),
        usage_prompt_tokens=1812,
        usage_completion_tokens=64,
        counted_prompt_tokens=1805,
        answer_lines=tuple(answer.split("\n")),
    )


@pytest.fixture
def store(tmp_path: Path) -> DirectoryFixtureStore:
    return DirectoryFixtureStore(tmp_path / "generation", "generation")


@pytest.mark.parametrize(
    "answer",
    [
        ANSWER,
        "one line with no newline at all",
        "trailing newline\n",
        "two trailing newlines\n\n",
        "\nleading newline",
        "a carriage return \r inside a line",
        "windows line ending\r\n",
        "",
    ],
    ids=[
        "several paragraphs",
        "no newline",
        "trailing newline",
        "two trailing newlines",
        "leading newline",
        "carriage return",
        "windows line ending",
        "empty",
    ],
)
def test_an_answer_comes_back_exactly_as_it_was_recorded(
    store: DirectoryFixtureStore, answer: str
) -> None:
    """Splitting on newlines and joining on them is an exact inverse, and this is the proof.

    Every case here is one somebody would reach for `str.splitlines` to handle, which is not an
    inverse: it drops the distinction between text that ends in a newline and text that does not,
    and it treats a lone carriage return as a line break. Either would replay an answer the model
    did not give.
    """
    write_fixture(store.root, _call(answer=answer))

    served = store.get(KEY)

    assert served is not None
    assert served.answer == answer


def test_the_answer_is_written_as_separate_lines(store: DirectoryFixtureStore) -> None:
    """The readable-diff requirement, checked against the bytes rather than asserted in prose.

    An answer stored as one JSON string is one enormous line carrying escaped newlines, which
    satisfies "no vectors inline" and still leaves a re-record unreviewable. What makes the diff
    readable is that each line of prose is a line of the file.
    """
    path = write_fixture(store.root, _call())
    written = path.read_text(encoding="utf-8")

    assert "\\n" not in written, "the answer is escaped into one line rather than split across them"

    for line in ANSWER.split("\n"):
        if line:
            assert line in written


def test_the_provenance_is_written_before_the_answer(store: DirectoryFixtureStore) -> None:
    """The part that changes on a re-record sits at the end, not in the middle of the metadata."""
    written = write_fixture(store.root, _call()).read_text(encoding="utf-8")

    assert written.index('"answer_lines"') > written.index('"recorded_on"')


def test_a_recording_carries_its_date_and_its_model(store: DirectoryFixtureStore) -> None:
    """A key proves a match and cannot be read; these are what date an old recording."""
    write_fixture(store.root, _call())

    served = store.get(KEY)

    assert served is not None
    assert served.recorded_on == date(2026, 8, 17)
    assert served.model_version == "gpt-4.1-mini-2025-04-14"
    assert served.usage.prompt_tokens == 1812


def test_the_question_is_stored_for_the_reader(store: DirectoryFixtureStore) -> None:
    """A directory of hex filenames is unreviewable without it, and nothing matches on it."""
    path = write_fixture(store.root, _call())

    assert "What happens to accounts nobody uses?" in path.read_text(encoding="utf-8")
    assert read_fixture(path).control_ids == ("ac-2", "ac-2.3")


def test_a_key_with_no_file_is_a_miss(store: DirectoryFixtureStore) -> None:
    """`None`, per the store contract: what a miss costs is decided by whoever asked."""
    write_fixture(store.root, _call())

    assert store.get("b" * 64) is None


def test_a_file_filed_under_the_wrong_name_is_refused(store: DirectoryFixtureStore) -> None:
    """A recording copied or renamed by hand would answer a request it was not made for.

    That is the substitution keying on the whole request exists to prevent, arriving through the
    filesystem where the key cannot see it. Nothing else in the system would notice.
    """
    write_fixture(store.root, _call())
    (store.root / f"{KEY}.json").rename(store.root / f"{'b' * 64}.json")

    with pytest.raises(FixtureFileError) as raised:
        store.get("b" * 64)

    assert "not its own" in str(raised.value)


def test_a_recording_of_another_purpose_is_refused(tmp_path: Path) -> None:
    """A grade sitting where answers are kept cannot be served as an answer."""
    store = DirectoryFixtureStore(tmp_path / "generation", "judge")

    write_fixture(store.root, _call())

    with pytest.raises(FixtureFileError) as raised:
        store.get(KEY)

    assert "generation" in str(raised.value) and "judge" in str(raised.value)


def test_a_broken_file_raises_rather_than_missing(store: DirectoryFixtureStore) -> None:
    """A corrupt recording is not a missing one, and the difference is what gets said out loud.

    Returning `None` here would have the console report that nobody recorded an answer to a
    question somebody did record -- the one sentence this project must never say untruthfully.
    """
    store.root.mkdir(parents=True)
    (store.root / f"{KEY}.json").write_text("{ not json", encoding="utf-8")

    with pytest.raises(FixtureFileError) as raised:
        store.get(KEY)

    assert "broken recording rather than a missing one" in str(raised.value)


def test_a_file_missing_a_field_is_refused(store: DirectoryFixtureStore) -> None:
    """Half a recording is not a recording; there is no default worth inventing for any of it."""
    path = write_fixture(store.root, _call())
    document = path.read_text(encoding="utf-8").replace('"model_version"', '"model_verison"')
    path.write_text(document, encoding="utf-8")

    with pytest.raises(FixtureFileError):
        store.get(KEY)


def test_the_store_lists_what_it_holds(store: DirectoryFixtureStore) -> None:
    """The recorder needs to know what it would duplicate and what it has left behind."""
    write_fixture(store.root, _call())
    write_fixture(store.root, _call(key="c" * 64))

    assert store.keys() == ("a" * 64, "c" * 64)


def test_an_absent_directory_holds_nothing(tmp_path: Path) -> None:
    """A first run has no directory yet, and that is a count of zero rather than a failure."""
    store = DirectoryFixtureStore(tmp_path / "nothing-here", "generation")

    assert store.keys() == ()
    assert store.get(KEY) is None


def test_a_served_fixture_is_the_narrow_type(store: DirectoryFixtureStore) -> None:
    """What the client receives carries no field it could match on other than the key."""
    write_fixture(store.root, _call())

    served = store.get(KEY)

    assert isinstance(served, Fixture)
    assert served.key == KEY
