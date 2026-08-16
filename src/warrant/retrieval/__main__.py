"""Ask the corpus a question and print what it retrieves.

    python -m warrant.retrieval "how are inactive accounts disabled?"

Needs a database with a corpus in it and the weights in the local cache. Needs no API key and no
network, and that is the reason this command exists rather than waiting for the console: the
claim that a reviewer can type an arbitrary question and watch real retrieval work is worth being
able to check directly, on a machine with neither credential nor connection, without a browser in
the way.

Every listing opens with the `k` it used. A retrieval output that does not say what `k` produced
it is a number that can be compared against one measured at a different `k`, which is the
specific accident the configuration is arranged to prevent — so the command that prints results
prints the configuration with them.
"""

from __future__ import annotations

import sys
import textwrap

import psycopg

from warrant.db import close_pool, connection
from warrant.embedder_config import get_embedder_config
from warrant.embedding import EmbedderError, load_embedder
from warrant.ingest.pipeline import CorpusProvenance
from warrant.retrieval.corpus_check import CorpusMismatchError, verify_corpus
from warrant.retrieval.search import Retrieval, RetrievalError, retrieve
from warrant.settings import get_settings

# Where a retrieved chunk's text is cut for display. Enough to recognise the clause and see that
# it is the right one; the whole thing is what goes into a prompt, and printing it for ten chunks
# would bury the ranking this command exists to show.
_PREVIEW_CHARACTERS = 240

# Indent for everything under a result's first line, so the identifier and the score stay in a
# column a reader can scan down.
_DETAIL_INDENT = " " * 13


def main() -> int:
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        print(
            'usage: python -m warrant.retrieval "your question"\n   or: make ask Q="your question"',
            file=sys.stderr,
        )
        return 1

    settings = get_settings()

    try:
        config = get_embedder_config()

        # Checked before the model is loaded. Loading the weights is seconds of work, and a
        # database with no corpus or the wrong one does not become searchable by spending them.
        #
        # Its own connection, released before that load. The pool is small and this holds nothing
        # across the slow part, which is what a request path would also want.
        with connection() as conn:
            provenance = verify_corpus(conn, config)

        embedder = load_embedder(config)

        with connection() as conn:
            result = retrieve(conn, question, embedder, config=config)
    except (CorpusMismatchError, EmbedderError, RetrievalError) as error:
        # Each of these names what disagreed with what, and none is helped by a traceback.
        print(f"error: {error}", file=sys.stderr)
        return 1
    except psycopg.errors.UndefinedTable:
        print(
            "error: this database has no corpus table, so no migration has been applied to it. "
            "`make migrate` installs the schema and `make ingest` fills it.",
            file=sys.stderr,
        )
        return 1
    except psycopg.OperationalError as error:
        print(
            f"error: could not connect to the database at {settings.database_url}: {error}\n"
            "Is it running? `docker compose up -d db` starts it.",
            file=sys.stderr,
        )
        return 1
    except OSError as error:
        # The message carries its own filename, and that is the point: the block above reads both
        # the embedder pin and the weights out of the local cache, so naming either one here would
        # send whoever is reading to a file that is perfectly fine.
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        # The pool starts worker threads, and a command that leaves them running exits only when
        # they happen to notice.
        close_pool()

    _report(result, provenance)
    return 0


def _report(result: Retrieval, provenance: CorpusProvenance) -> None:
    """Print the configuration that produced this result, then the result."""
    print()
    print(
        f"  k = {result.k}   {provenance.embedder_name} @ {provenance.embedder_revision[:8]}"
        f"   {provenance.chunk_count:,} chunks"
    )
    print(f"  {result.question}")
    print()

    for chunk in result.chunks:
        print(f"  {chunk.rank:>2}  {chunk.score:.3f}  {chunk.control_label}  {chunk.title}")
        print(f"{_DETAIL_INDENT}{chunk.chunk_id}")

        for line in _preview(chunk.text):
            print(f"{_DETAIL_INDENT}{line}")

        print()

    # Said out loud rather than left to be inferred from the count, because "fewer than k came
    # back" is a fact about the corpus and reads like a fact about the question.
    if len(result.chunks) < result.k:
        print(
            f"  The corpus holds {len(result.chunks)} chunks, fewer than the {result.k} asked "
            "for, so this is all of it rather than the nearest part of it."
        )
        print()


def _preview(text: str) -> list[str]:
    """The opening of a chunk's text, collapsed onto wrapped lines.

    Collapsed because a chunk carries its own headings and blank lines, and reproducing that
    layout inside an indented list turns a ranking into a wall. What this has to show is enough
    of the clause to recognise it.
    """
    collapsed = " ".join(text.split())

    return textwrap.wrap(
        textwrap.shorten(collapsed, width=_PREVIEW_CHARACTERS, placeholder=" ..."),
        width=100 - len(_DETAIL_INDENT),
    )


if __name__ == "__main__":
    raise SystemExit(main())
