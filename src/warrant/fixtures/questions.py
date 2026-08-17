"""The questions this build records answers for, and the three kinds it separates them into.

Provisional, and the file it reads says so in its own words. A measured question set is a later
piece of work with an admission criterion attached; what exists here is a short list that exercises
the path end to end and gives a reviewer something to click. The classes are the ones the measured
set will use, so growing into it is a longer file rather than a different shape.

**Why the classes are in the data and not inferred.** Two of them exist to be visibly different
from the first. A question the catalog cannot answer and a question whose obvious answer the
catalog contradicts both look exactly like ordinary questions, and the interesting thing about
either is what the system does *not* say. Labelling them is what lets a picker group them and what
stops the second class quietly reading as a set of failures.

**Nothing here has a manifest entry, deliberately.** Adding a question invalidates nothing: it has
no recording yet, which the recorder handles by recording it and which replay handles by declining
to generate. That is the same argument `data/model.json` makes for its own absence -- the manifest
is for inputs whose change would otherwise go unnoticed, and this one cannot.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from warrant.settings import get_settings

# What a question is for. `prior_conflict_trap` is the one worth naming carefully: it is not a
# question the catalog fails to answer, it is one whose answer most people -- and most models --
# already believe they know, and where the catalog says something else.
QuestionClass = Literal["answerable", "out_of_corpus", "prior_conflict_trap"]

_QUESTIONS_NAME = "questions.json"


class QuestionSetError(Exception):
    """The recorded question list cannot be read."""


class Question(BaseModel):
    """One question, and what it is there to exercise."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    # Stable across re-records and across rewordings of the text. What a picker links to and what
    # a later measured set carries forward when a question is rephrased.
    id: str

    # `class` in the file, because that is what it is called everywhere else in this project and in
    # the plan it comes from; `question_class` in code, because the other spelling is a keyword.
    question_class: QuestionClass = Field(alias="class")

    text: str

    # Why this question is in the class it is in. Prose, and required rather than optional: a trap
    # with no statement of what prior it conflicts with is an untested guess about model behaviour,
    # which is exactly what the later admission criterion exists to stop this list becoming.
    because: str


class QuestionSet(BaseModel):
    """Every question this build records, in the order the file lists them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Bumped by hand when the list changes. Not hashed into anything and not in the manifest --
    # it is here so a console can say which list a reviewer is looking at.
    version: str

    questions: tuple[Question, ...]

    @property
    def texts(self) -> tuple[str, ...]:
        """Just the question strings, in file order."""
        return tuple(question.text for question in self.questions)

    def of_class(self, kind: QuestionClass) -> tuple[Question, ...]:
        """Every question of one kind, for a picker that groups them."""
        return tuple(question for question in self.questions if question.question_class == kind)


@lru_cache(maxsize=1)
def get_question_set() -> QuestionSet:
    """Read the recorded question list once per process."""
    return load_question_set(get_settings().fixtures_path / _QUESTIONS_NAME)


def load_question_set(path: Path) -> QuestionSet:
    """Read a question list from a specific file.

    A parameter rather than a global, for the reason every other loader here takes one: it is how
    a test presents a different list without editing the file the whole project reads.
    """
    document = json.loads(path.read_text(encoding="utf-8"))

    # The file carries a `_comment` block saying what the list is and is not. Removed by name, the
    # way the other hand-edited files here have theirs removed.
    if isinstance(document, dict):
        document.pop("_comment", None)

    try:
        questions = QuestionSet.model_validate(document)
    except ValidationError as error:
        raise QuestionSetError(f"{path} is not a question list: {error}") from error

    identifiers = [question.id for question in questions.questions]

    if len(set(identifiers)) != len(identifiers):
        raise QuestionSetError(
            f"{path} uses the same id for more than one question. An id is what a recording and a "
            "picker link to, so two of them is one question nobody can address."
        )

    texts = [question.text for question in questions.questions]

    if len(set(texts)) != len(texts):
        raise QuestionSetError(
            f"{path} lists the same question twice. A question is the identity of its recorded "
            "vector, so the second one would silently replace the first."
        )

    return questions
