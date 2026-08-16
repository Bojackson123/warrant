"""Vector search over stored chunks, and the control ids a question retrieves.

Two modules, and the split is between asking and being allowed to ask. `search` embeds a
question and ranks the corpus against it; `corpus_check` compares what built that corpus against
what this build is pinned to, and refuses when they differ. The second exists because the first
cannot detect its own worst failure: a corpus embedded by a superseded model of the same width
is searched successfully and cites the wrong controls.
"""

from warrant.retrieval.corpus_check import (
    CorpusMismatchError,
    CorpusMissingError,
    verify_corpus,
)
from warrant.retrieval.search import Retrieval, RetrievalError, RetrievedChunk, retrieve

__all__ = [
    "CorpusMismatchError",
    "CorpusMissingError",
    "Retrieval",
    "RetrievalError",
    "RetrievedChunk",
    "retrieve",
    "verify_corpus",
]
