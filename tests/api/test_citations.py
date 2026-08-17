"""The two-half citation check, exercised without a database or a network.

Both halves are asserted apart from each other, because they catch different failures and only one
of them is obvious. That a cited identifier exists is the easy half; that a cited identifier was
actually retrieved is the half that catches an answer citing a real control it was never shown --
the model answering from what it already knows rather than from the text it was given, which is the
thing the whole project claims not to do.

The catalog index is the real one: `AC-2` and `SC-28` are controls Revision 5 actually has, and
`AC-99` is a well-formed identifier it does not. Nothing here needs the corpus in a database -- the
retrieved set is supplied directly, which is what lets the "exists but not retrieved" case be built
at all.
"""

from __future__ import annotations

import pytest

from warrant.api.citations import check_citations, parse_citations
from warrant.ingest.control_ids import ControlIndex, get_control_index
from warrant.retrieval.search import Retrieval, RetrievedChunk


@pytest.fixture(scope="module")
def index() -> ControlIndex:
    return get_control_index()


def _retrieval(*control_ids: str) -> Retrieval:
    """A retrieval that returned exactly these controls, one chunk each.

    Only the control id is load-bearing for the citation check; the rest is filled with plausible
    values so the chunk is a real `RetrievedChunk` rather than a mock the check might read
    differently from the real thing.
    """
    chunks = tuple(
        RetrievedChunk(
            rank=rank,
            score=1.0 - rank * 0.1,
            chunk_id=f"{control_id}#a",
            control_id=control_id,
            base_control_id=control_id.split(".")[0],
            control_label=control_id.upper(),
            title="A control",
            part_path="a",
            text="Some clause text.",
        )
        for rank, control_id in enumerate(control_ids, start=1)
    )

    return Retrieval(question="a question", k=len(chunks), chunks=chunks)


def test_a_retrieved_control_that_exists_is_valid(index: ControlIndex) -> None:
    """The ordinary case: cited, real, and among what was retrieved."""
    (citation,) = check_citations("Accounts are disabled [AC-2].", _retrieval("ac-2"), index)

    assert citation.cited_as == "AC-2"
    assert citation.control_id == "ac-2"
    assert citation.exists
    assert citation.retrieved
    assert citation.valid


def test_a_cited_id_absent_from_the_catalog_is_flagged(index: ControlIndex) -> None:
    """The existence half: a well-formed identifier that names no control.

    `AC-99` parses as an identifier, so it is treated as a citation and then fails where it should
    -- on existence -- rather than being quietly ignored as though it had never been written.
    """
    (citation,) = check_citations("As required [AC-99].", _retrieval("ac-2"), index)

    assert citation.cited_as == "AC-99"
    assert citation.control_id is None
    assert not citation.exists
    assert not citation.retrieved
    assert not citation.valid


def test_a_real_control_that_was_not_retrieved_is_flagged(index: ControlIndex) -> None:
    """The half that matters: a real control the retrieval did not return.

    `SC-28` is a genuine control, so the existence check passes -- and it was not among what this
    question retrieved, so the citation is still invalid. This is the shape of an answer citing from
    prior knowledge, and flagging it is the point of checking retrieval and not only existence.
    """
    (citation,) = check_citations(
        "Stored data is encrypted [SC-28].", _retrieval("ac-2", "ia-5"), index
    )

    assert citation.control_id == "sc-28"
    assert citation.exists
    assert not citation.retrieved
    assert not citation.valid


def test_prose_in_brackets_is_not_a_citation(index: ControlIndex) -> None:
    """A bracketed phrase that is not shaped like an identifier is left alone.

    Otherwise every `[see below]` would be reported as an invented citation, burying the real ones
    under noise the check itself created.
    """
    assert parse_citations("See the discussion [see below] and the table [Table 1].") == ()
    assert check_citations("See [see below].", _retrieval("ac-2"), index) == ()


def test_the_same_control_cited_twice_is_one_citation(index: ControlIndex) -> None:
    """Two spellings of one control collapse, keeping the first as written.

    Deduplication is by the control named, not by the string, so `AC-2` and `ac-02` are one
    citation -- and the form the reader wrote is the one echoed back.
    """
    citations = check_citations(
        "Accounts are disabled [AC-2], and again [ac-02].", _retrieval("ac-2"), index
    )

    assert len(citations) == 1
    assert citations[0].cited_as == "AC-2"


def test_an_enhancement_is_matched_against_the_enhancement(index: ControlIndex) -> None:
    """Citing an enhancement is retrieved only when that enhancement was, not its parent.

    The prompt asks the model to cite the identifier shown in each excerpt heading, so a citation to
    `AC-2(3)` is a claim that the enhancement was retrieved. Its parent `AC-2` being retrieved does
    not make that true, and matching loosely would let a citation the retrieval does not support
    read as though it did.
    """
    (citation,) = check_citations("Disabled on expiry [AC-2(3)].", _retrieval("ac-2"), index)

    assert citation.control_id == "ac-2.3"
    assert citation.exists
    assert not citation.retrieved

    (enhancement,) = check_citations("Disabled on expiry [AC-2(3)].", _retrieval("ac-2.3"), index)

    assert enhancement.retrieved
    assert enhancement.valid
