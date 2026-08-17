"""The one renderer: retrieved chunks and a question in, the string a model is sent out.

**One function, one string, and that is the whole discipline here.** A recorded model call is
keyed on a SHA-256 of the fully rendered request, so the moment two code paths can both produce
"the prompt" the keying rule stops meaning anything: one path records, the other replays, and a
difference between them is a permanent miss nobody can explain from the key alone. Every caller
that needs a prompt -- the recorder, the request path, whatever counts tokens -- calls
`render_prompt` and holds the string it returns.

**It takes a `Retrieval`, not a question and a list of chunks.** That type exists so a caller
cannot hold ranked chunks apart from the question and the `k` that produced them, and taking it
whole extends the same guarantee here: rendering one question against another question's chunks
is not a mistake this signature permits.

**What the template instructs, and why each part is load-bearing.**

- *Answer only from the text below.* The claim this project makes is that every statement in an
  answer is warranted by a clause a reader can open. A model answering partly from what it already
  knows about NIST SP 800-53 produces text that looks identical and is not checkable.
- *Cite the identifier shown.* Citations are parsed back out and checked twice -- the identifier
  exists in the catalog, and it was actually retrieved for this question. Asking for the label form
  printed beside each control is what makes that parse a lookup rather than a guess.
- *Decline when the retrieved text does not support an answer.* Part of version 1 rather than an
  addition later, because the prior-conflict traps depend on the system declining to invent facts
  the catalog does not state, and a refusal instruction added after answers were recorded would
  invalidate every one of them.

**Why the version is a constant and the fingerprint is a computation.** `PROMPT_TEMPLATE_VERSION`
is a claim a person maintains; `prompt_fingerprint` is the number that notices when the claim went
stale. The two are recorded together in the manifest for the reason the chunker's are: a version
string cannot catch somebody changing how the pieces are assembled and not saying so, and that
change moves every key downstream of it.
"""

from __future__ import annotations

import hashlib

from warrant.retrieval.search import Retrieval, RetrievedChunk

# Bumped by hand when the template below changes on purpose. It is the half of the manifest entry
# a person reads; `prompt_fingerprint` is the half that cannot be forgotten.
PROMPT_TEMPLATE_VERSION = "1"

_INSTRUCTIONS = """\
You are answering a question about the NIST SP 800-53 Revision 5 security control catalog.

Below are the control excerpts retrieved for this question, followed by the question itself.

Answer using only the retrieved text. Do not add requirements, figures, intervals or
definitions from anything you know about NIST SP 800-53 that is not written below, even if you
are confident it is correct.

Cite the control identifier shown in the heading of each excerpt you rely on, in square
brackets, immediately after the statement it supports -- for example: accounts are reviewed at
an organisation-defined frequency [AC-2].

If the retrieved text does not support an answer to the question, say so plainly and say what
it does cover instead. Do not answer from memory, and do not offer a partial answer as though
it were a complete one. Declining is the correct response to a question this catalog does not
address.\
"""

# Written between the sections so that a control's text cannot run into the next heading, and so
# that the question is visibly not part of the retrieved material.
_SECTION_RULE = "-" * 72

# What one excerpt looks like. `control_label` is the published form a reader recognises and the
# form a citation is written in; `part_path` says which parts of the control the text came from,
# which is what makes an excerpt identifiable as an excerpt rather than as the whole control.
_EXCERPT = "[{label}] {title}\n({part_path})\n\n{text}"


def render_prompt(retrieval: Retrieval) -> str:
    """The complete string a model is sent for this retrieval.

    Deterministic in every part: the excerpts appear in rank order, which the search fixed with a
    tie-break, and nothing here iterates a set or a dictionary. The same retrieval renders to the
    same bytes, which is what makes the key over it mean anything.
    """
    excerpts = "\n\n".join(_excerpt(chunk) for chunk in retrieval.chunks)

    return (
        f"{_INSTRUCTIONS}\n\n"
        f"{_SECTION_RULE}\nRETRIEVED CONTROL TEXT\n{_SECTION_RULE}\n\n"
        f"{excerpts}\n\n"
        f"{_SECTION_RULE}\nQUESTION\n{_SECTION_RULE}\n\n"
        f"{retrieval.question}\n"
    )


def prompt_fingerprint() -> str:
    """SHA-256 of this template's rendering of a fixed retrieval.

    The counterpart to `chunker_fingerprint` one layer up, and it exists for the same reason:
    `PROMPT_TEMPLATE_VERSION` on its own is a comment, and a comment cannot notice that somebody
    reordered the sections, changed a heading or dropped a field from an excerpt and did not bump
    it. Pin this beside the version and an unannounced change fails with two digests -- rather
    than reaching the recorder, where it would silently invalidate every recorded answer.

    Computed over a synthetic retrieval held in this module rather than over the real corpus, so
    that it moves when the template moves and stays still when the catalog does. The catalog has
    an entry of its own.
    """
    return hashlib.sha256(render_prompt(_FINGERPRINT_RETRIEVAL).encode("utf-8")).hexdigest()


def _excerpt(chunk: RetrievedChunk) -> str:
    """One retrieved chunk as it appears in the prompt.

    The rank and the similarity score are deliberately absent. Both are real numbers about the
    search and neither is evidence about the control: printing them invites a model to treat the
    top result as the most authoritative rather than the most similar, and a score is a
    calibration nothing has established.
    """
    return _EXCERPT.format(
        label=chunk.control_label,
        title=chunk.title,
        part_path=chunk.part_path,
        text=chunk.text,
    )


# Two excerpts rather than one, so that the separator between them is inside the fingerprint, and
# short text rather than real control prose, so that reading this file shows what is being hashed.
_FINGERPRINT_RETRIEVAL = Retrieval(
    question="What does this fingerprint cover?",
    k=2,
    chunks=(
        RetrievedChunk(
            rank=1,
            score=0.5,
            chunk_id="xx-1#a",
            control_id="xx-1",
            base_control_id="xx-1",
            control_label="XX-1",
            title="First",
            part_path="a",
            text="The first excerpt.",
        ),
        RetrievedChunk(
            rank=2,
            score=0.25,
            chunk_id="xx-2#b",
            control_id="xx-2",
            base_control_id="xx-2",
            control_label="XX-2",
            title="Second",
            part_path="b",
            text="The second excerpt.",
        ),
    ),
)
