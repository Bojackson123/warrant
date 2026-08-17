"""Pull the cited control ids out of an answer and check each one two ways.

This is where citation *validity* becomes a computed value, and validity here is deliberately two
separate questions rather than one:

1. **Does the cited control exist in the catalog?** An answer can cite `AC-99`, which is a
   perfectly well-formed identifier that names nothing. Resolving it through the same index the
   ingest path uses is what tells an invented identifier from a real one.
2. **Was the cited control actually retrieved for this question?** This is the half that matters
   most and the half a lone "does it exist" check misses entirely. A model that cites a real
   control it was never shown is answering from what it already knows about NIST SP 800-53, and
   the whole claim this project makes is that every citation traces to a clause the reader was
   given. A citation to a control outside the retrieved set is flagged, not rendered as sound.

Neither check spends anything -- no database, no network, no model. The catalog index is built
once per process and the retrieved ids are already in hand, so this runs on the request thread
without widening the request path.

**What counts as a citation is decided by syntax first, existence second.** A bracketed span is
treated as a citation candidate only if it *parses* as a control id, so an answer's ordinary
prose in brackets -- `[see below]` -- is not mistaken for a citation and then flagged as an
invented one. `AC-99` parses and so is kept, and then fails the existence check where it should:
the distinction the parse draws is "shaped like an identifier" versus "names a control", and only
the first belongs here.

**Matched strictly against the identifier that was shown.** The prompt instructs the model to cite
the identifier in each excerpt's heading, which is that chunk's own control id. So a citation is
"retrieved" when its canonical form is among the retrieved chunks' canonical ids exactly --
citing a base control for a retrieved enhancement, or the reverse, is not a match. Loosening this
to accept a parent or child would let a citation the retrieval does not support read as though it
did, which is the failure the second check exists to make visible.
"""

from __future__ import annotations

import re

from warrant.api.schemas import CitationView
from warrant.ingest.control_ids import ControlIndex, canonical_id
from warrant.retrieval.search import Retrieval

# One bracketed span. The prompt asks for citations in square brackets, so this is where a control
# id would be; whether the contents are actually one is decided by `canonical_id`, not by this.
_BRACKETED = re.compile(r"\[([^\[\]]+)\]")


def parse_citations(answer: str) -> tuple[str, ...]:
    """The control ids an answer cites, in order, deduplicated by the control they name.

    A bracketed span is kept only when it parses as a control identifier, which drops incidental
    brackets without dropping a well-formed identifier that happens to name nothing -- `AC-99`
    survives here and fails the existence check, which is where "does not exist" belongs.

    Deduplicated by canonical id rather than by spelling, so `AC-2` and `ac-02` cited in the same
    answer are one citation; the first spelling seen is the one kept, because that is the form the
    reader wrote and the form a console echoes back.
    """
    seen: set[str] = set()
    cited: list[str] = []

    for match in _BRACKETED.finditer(answer):
        token = match.group(1).strip()
        canonical = canonical_id(token)

        # Not shaped like an identifier at all -- ordinary prose in brackets. Not a citation.
        if canonical is None:
            continue

        if canonical in seen:
            continue

        seen.add(canonical)
        cited.append(token)

    return tuple(cited)


def check_citations(
    answer: str,
    retrieval: Retrieval,
    index: ControlIndex,
) -> tuple[CitationView, ...]:
    """Every citation in the answer, each with both halves of its validity resolved.

    The retrieved set is taken from the retrieval's own control ids, so "was it retrieved" is asked
    against exactly what this question returned rather than against the corpus at large.
    """
    retrieved = set(retrieval.control_ids)
    views: list[CitationView] = []

    for token in parse_citations(answer):
        identity = index.resolve(token)

        exists = identity is not None
        was_retrieved = identity is not None and identity.id in retrieved

        views.append(
            CitationView(
                cited_as=token,
                control_id=identity.id if identity is not None else None,
                control_label=identity.label if identity is not None else None,
                title=identity.title if identity is not None else None,
                exists=exists,
                retrieved=was_retrieved,
                valid=exists and was_retrieved,
            )
        )

    return tuple(views)
