"""How the question list fails when somebody edits it.

The list is hand-written, which makes a typo in it the most likely way this file is ever wrong. A
typo is not an unexpected condition here; it is the expected one, and it has to arrive as a message
naming the file rather than as a traceback out of the JSON parser.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from warrant.fixtures.questions import QuestionSetError, load_question_set

QUESTION = {
    "id": "unused-accounts",
    "class": "answerable",
    "text": "What happens to accounts nobody uses?",
    "because": "The catalog addresses this directly, so a correct answer is a citable one.",
}


def _write(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    return path


def test_a_list_is_read_with_its_classes_and_its_version(tmp_path: Path) -> None:
    """`class` in the file, `question_class` in code, and the version comes back as written."""
    path = _write(tmp_path / "questions.json", {"version": "3", "questions": [QUESTION]})

    questions = load_question_set(path)

    assert questions.version == "3"
    assert questions.texts == (QUESTION["text"],)
    assert questions.of_class("answerable")[0].question_class == "answerable"


def test_a_file_that_is_not_json_names_the_file(tmp_path: Path) -> None:
    """A trailing comma is a mistake somebody makes; a traceback is not a report of it."""
    path = tmp_path / "questions.json"
    path.write_text('{"version": "1", "questions": [],}', encoding="utf-8")

    with pytest.raises(QuestionSetError) as raised:
        load_question_set(path)

    assert str(path) in str(raised.value)


def test_two_questions_with_one_id_are_refused(tmp_path: Path) -> None:
    """An id is what a recording and a picker link to, so two of them addresses neither."""
    other = {**QUESTION, "text": "Who approves a new account?"}
    path = _write(tmp_path / "questions.json", {"version": "1", "questions": [QUESTION, other]})

    with pytest.raises(QuestionSetError) as raised:
        load_question_set(path)

    assert "same id" in str(raised.value)


def test_the_same_question_twice_is_refused(tmp_path: Path) -> None:
    """A question is the identity of its recorded vector, so the second silently replaces it."""
    other = {**QUESTION, "id": "unused-accounts-again"}
    path = _write(tmp_path / "questions.json", {"version": "1", "questions": [QUESTION, other]})

    with pytest.raises(QuestionSetError) as raised:
        load_question_set(path)

    assert "same question twice" in str(raised.value)
