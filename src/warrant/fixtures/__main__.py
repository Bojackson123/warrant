"""Record the fixtures: the question list in, reviewable recordings out.

    python -m warrant.fixtures [--queries] [--force]

The only command here that spends money, and the only one that creates a recording. It runs the
real pipeline -- retrieve, render, ask -- so what it writes is what the request path would have
produced, rather than an approximation assembled for storage.

`--queries` stops after the question vectors, which need a local model and no credential. Without a
key that is what happens anyway, and the command says so rather than failing: half of this work is
genuinely doable on a machine with no account, and refusing it outright would make the recorded
vectors look like they cost something they do not.

`--force` re-records what is already on disk. That is the monthly cadence: the keys have not moved,
the answers may have, and the diff is the point. Without it a recording that exists is left exactly
as it is, so an ordinary run costs nothing and produces no diff.

**The manifest is checked before anything else happens**, and the ordering is the reason this
command has a check at all. A recording made from inputs this build no longer carries is not
slightly stale; the control text inside its prompt came from a corpus that has been superseded. It
would be a brand new artefact, born wrong, and committed in a pull request whose whole subject is
that the fixtures are current.
"""

from __future__ import annotations

import sys
from datetime import date

import psycopg

from warrant.catalog_pin import CatalogPinError
from warrant.db import close_pool, connection
from warrant.embedder_config import get_embedder_config
from warrant.embedding import EmbedderError, load_embedder
from warrant.fixtures.client import ModelError, get_model_client
from warrant.fixtures.disk import DirectoryFixtureStore, FixtureFileError
from warrant.fixtures.queries import (
    QueryVectorError,
    read_query_vectors,
)
from warrant.fixtures.questions import Question, QuestionSetError, get_question_set
from warrant.fixtures.recorder import (
    GENERATION_DIRECTORY,
    QUERIES_DIRECTORY,
    GenerationReport,
    QueryReport,
    RecordProgress,
    record_generations,
    record_query_vectors,
)
from warrant.manifest import ManifestError, verify_manifest
from warrant.retrieval.corpus_check import CorpusMismatchError, verify_corpus
from warrant.retrieval.search import RetrievalError
from warrant.settings import get_settings
from warrant.tokenizer import TokenizerError, load_tokenizer

_USAGE = "usage: python -m warrant.fixtures [--queries] [--force]"

_FLAGS = {"--queries", "--force"}


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv

    if set(arguments) - _FLAGS:
        print(_USAGE, file=sys.stderr)
        return 1

    queries_only = "--queries" in arguments
    force = "--force" in arguments

    settings = get_settings()
    root = settings.fixtures_path
    today = date.today()

    generations: GenerationReport | None = None

    try:
        config = get_embedder_config()
        questions = get_question_set()

        # First, and before a key is read or a model is loaded. This is the refusal the command
        # exists to make: everything after it writes something, and everything it writes is a
        # function of the inputs this checks.
        comparisons = verify_manifest()

        print(f"{len(questions.questions)} questions, list {questions.version}.", flush=True)

        with connection() as conn:
            verify_corpus(conn, config)

        embedder = load_embedder(config)

        vectors = record_query_vectors(root, questions, embedder, config, today, force)
        _report_queries(vectors, questions.version)

        if queries_only:
            return 0

        # Read back rather than kept from the write. What replay will use is what is on disk, so
        # recording against anything else would leave the two agreeing only by construction of this
        # process -- which is exactly the assumption a different machine breaks.
        recorded = read_query_vectors(root / QUERIES_DIRECTORY, config)

        store = DirectoryFixtureStore(root / GENERATION_DIRECTORY, "generation")

        # Verified once, above. The client would otherwise check again, and verification re-reads
        # and re-chunks a ten-megabyte catalog; what it is handed here is the result of the check
        # that already ran rather than a check being skipped.
        client = get_model_client(store, verify=lambda: comparisons)

        if client.mode == "replay":
            print(
                "\nNo API key is set, so the answers were not recorded. The question vectors "
                "above are written and cost nothing; recording answers needs "
                "WARRANT_MODEL_API_KEY.",
            )
            return 0

        tokenizer = load_tokenizer()

        with connection() as conn:
            generations = record_generations(
                conn,
                root,
                questions,
                recorded,
                embedder,
                client,
                tokenizer,
                today,
                config=config,
                force=force,
                progress=_progress(),
            )
    except (
        CatalogPinError,
        CorpusMismatchError,
        EmbedderError,
        FixtureFileError,
        ManifestError,
        ModelError,
        QuestionSetError,
        QueryVectorError,
        RetrievalError,
        TokenizerError,
    ) as error:
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
        # The message carries its own filename, and that is the point: the block above reads the
        # question list, the pins, the weights and the recorded vectors, so naming any one of them
        # here would send whoever is reading to a file that is perfectly fine.
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        close_pool()

    _report_generations(generations)
    return 0


def _progress() -> RecordProgress:
    """Print a line as each answer comes back.

    A provider call takes seconds and there are a dozen of them, so a command that printed nothing
    until the end would be indistinguishable from one that had stopped. Flushed for the reason the
    ingest's progress is: piped to a log, Python buffers this and the whole run arrives at once.
    """

    def report(question: Question, position: int, total: int) -> None:
        print(f"  {position:>3}/{total}  {question.id}", flush=True)

    return report


def _report_queries(report: QueryReport, version: str) -> None:
    """Print what the query stage left behind, and whether it changed anything."""
    print()
    print(
        f"  {report.questions} question vectors at {report.dimensions} dimensions, "
        f"{report.vector_bytes:,} bytes"
    )

    if report.rewritten:
        print(f"  written under list {version}")
    else:
        print("  unchanged: the recorded vectors already cover exactly these questions")

    print()


def _report_generations(report: GenerationReport | None) -> None:
    """Print what was recorded, what was already there, and what is left over."""
    if report is None:
        return

    print()
    print(f"  recorded  {len(report.recorded):>3}")
    print(f"  skipped   {len(report.skipped):>3}  already recorded under an unchanged key")

    if report.recorded:
        print(
            f"  tokens    {report.prompt_tokens:,} in, {report.completion_tokens:,} out, "
            "as the provider counted them"
        )

    if report.orphans:
        print()
        print(
            f"  {len(report.orphans)} recordings key to no current question. They are left in "
            "place rather than deleted -- each one was paid for, and removing it belongs in the "
            "change that explains why the question went away:"
        )

        for key in report.orphans:
            print(f"    {key}")

    print()

    if report.recorded:
        print(
            "Recorded. The diff on data/fixtures/generation is the review: read the answers, and "
            "check that each control cited is one the retrieval above actually returned."
        )
    else:
        print("Nothing to record. Every question already has an answer under its current key.")


if __name__ == "__main__":
    raise SystemExit(main())
