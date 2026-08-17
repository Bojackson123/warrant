"""The question-list endpoint, over a small injected list rather than the committed one.

No database and no lifespan: the list is read from disk, so the endpoint answers before any corpus
exists, and a dependency override stands a short list in place of the real file the way the other
api tests override the connection and the resources. What is asserted is the shape a console groups
by -- the `version` it shows, and each question's `class` on the wire -- not the contents of the
provisional list, which is free to change without breaking this.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from warrant.api.app import app, get_question_source
from warrant.fixtures.questions import QuestionSet

# One of each class, built through the validator so it goes in the way the committed file does --
# `class` on the wire, aliased to `question_class` in the model.
QUESTIONS = QuestionSet.model_validate(
    {
        "version": "test-7",
        "questions": [
            {
                "id": "accounts",
                "class": "answerable",
                "text": "What happens to accounts nobody uses?",
                "because": "The catalog states it directly.",
            },
            {
                "id": "passwords",
                "class": "prior_conflict_trap",
                "text": "How often must passwords be changed?",
                "because": "Ninety days is remembered and not in the catalog.",
            },
            {
                "id": "budget",
                "class": "out_of_corpus",
                "text": "What should a SIEM cost?",
                "because": "A control catalog is not about cost.",
            },
        ],
    }
)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """The app with the question source overridden, and no lifespan run."""
    app.dependency_overrides[get_question_source] = lambda: QUESTIONS

    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_the_question_list_is_served_with_its_version(client: TestClient) -> None:
    """The whole list, and the version a console shows to say which list it is looking at."""
    response = client.get("/questions")

    assert response.status_code == 200

    body = response.json()
    assert body["version"] == "test-7"
    assert len(body["questions"]) == len(QUESTIONS.questions)


def test_each_question_carries_its_class_on_the_wire(client: TestClient) -> None:
    """`class`, not `question_class` -- the vocabulary the picker groups by and the file uses."""
    body = client.get("/questions").json()

    by_id = {question["id"]: question for question in body["questions"]}

    assert by_id["passwords"]["class"] == "prior_conflict_trap"
    assert by_id["passwords"]["because"]
    assert set(by_id["accounts"]) == {"id", "class", "text", "because"}
