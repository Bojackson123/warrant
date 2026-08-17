"""What crosses the HTTP boundary: one question in, an answer and its citations out.

**One response shape covers both states a request can end in**, and the difference between them is
a field rather than a status code. A question with a recorded answer comes back with
`answered=true`, the text, and its citations; a question nothing recorded comes back with
`answered=false`, a `decline` block, and the same retrieval -- as a `200`, because the retrieval
ran and the clauses are real. A miss is not an error, and giving it an error status would tell the
console to show a failure where it should show the second state.

**Every field the console needs to render either state is here, and nothing it would have to fetch
separately.** The retrieved chunks carry their full clause text so a citation can be clicked
through without a second request, and each citation carries its own validity so the console never
has to recompute the check the server already ran.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from warrant.fixtures.questions import QuestionClass
from warrant.settings import Mode


class QuestionRequest(BaseModel):
    """A question to answer. The whole of the request body."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)


class CitationView(BaseModel):
    """One control identifier an answer cited, and whether it stands up.

    The two halves are reported separately as well as combined, because they fail for different
    reasons and a reader distinguishing them is the point. `exists=false` is an invented
    identifier; `exists=true, retrieved=false` is a real control the model cited without it having
    been retrieved -- the case that catches an answer drawing on prior knowledge rather than on the
    text it was given.
    """

    model_config = ConfigDict(frozen=True)

    # The identifier exactly as it was written in the answer, e.g. "AC-2" or "AC-2(3)". Rendered,
    # never matched on -- the canonical form below is what the checks were run against.
    cited_as: str

    # The canonical control id it resolves to, or `None` when it names no control in the catalog.
    control_id: str | None
    # The published label and title of the resolved control, for a console that renders the
    # citation. `None` together with `control_id` when nothing resolved.
    control_label: str | None
    title: str | None

    # The cited id names a control in the catalog.
    exists: bool
    # That control was among the chunks retrieved for this question.
    retrieved: bool
    # Both of the above. The single value a gate reads.
    valid: bool


class ChunkView(BaseModel):
    """One retrieved chunk, carrying the clause text a click-through needs.

    The full text rather than a preview: this is what the citation panel opens to, and a truncated
    field would mean a second request to read the clause a citation rests on.
    """

    model_config = ConfigDict(frozen=True)

    rank: int
    score: float
    chunk_id: str
    control_id: str
    base_control_id: str
    control_label: str
    title: str
    part_path: str
    text: str


class DeclineView(BaseModel):
    """Why no answer was generated, in words a console can show as its second state."""

    model_config = ConfigDict(frozen=True)

    reason: str
    detail: str


class AnswerResponse(BaseModel):
    """The complete result of one question, in either state.

    `prompt_token_count` is present whichever state this is: a miss still rendered the prompt to
    compute the key that missed, so the number the token gate reads exists even when no answer
    does. `answer`, `recorded_on` and `citations` describe the answered state; `decline` describes
    the second. `chunks` and `mode` are common to both.
    """

    model_config = ConfigDict(frozen=True)

    question: str
    mode: Mode

    # Counted locally with the pinned encoding, over the rendered prompt. Always present.
    prompt_token_count: int

    answered: bool
    answer: str | None
    # The day a replayed answer was recorded, so a console can say how old it is. `None` for a live
    # answer, which is happening now, and `None` in the declined state.
    recorded_on: date | None

    citations: tuple[CitationView, ...]
    chunks: tuple[ChunkView, ...]

    decline: DeclineView | None


class QuestionView(BaseModel):
    """One recorded question, as the console's picker needs it.

    `class` on the wire, `question_class` in code -- the wire name is the vocabulary the picker
    groups by and the one the source file uses, and the code name avoids the keyword. `because` is
    carried so a trap can show what prior it conflicts with rather than being an unexplained tag.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    question_class: QuestionClass = Field(serialization_alias="class")
    text: str
    because: str


class QuestionSetView(BaseModel):
    """The recorded question list a console offers, and which list it is.

    `version` is here for the console to show: the list is provisional, and a reviewer looking at
    it should be able to see which one it is rather than mistake it for a measured set.
    """

    model_config = ConfigDict(frozen=True)

    version: str
    questions: tuple[QuestionView, ...]
