"""Build the corpus: catalog on disk in, embedded rows in Postgres out.

    python -m warrant.ingest

One command from an empty database, so it migrates before it writes rather than requiring
`make migrate` first. No API key and no network: the weights come from the local cache that
`make model` fills once, and the manifest check below counts a sample with the pinned tokenizer
encoding, which comes from the cache `make tokenizer` fills once. Both are one-off fetches, and
this command needs them both to have happened.

Everything that can fail cheaply is checked before anything expensive happens. Embedding the
catalog is minutes of CPU, and discovering afterwards that the file on disk is not the pinned one,
or that the schema was built for a different model, would be minutes spent to produce a corpus
that has to be thrown away.
"""

from __future__ import annotations

import sys
import time

import psycopg

from warrant.catalog_pin import CatalogPinError, get_catalog_pin, verify_catalog
from warrant.db.migrator import MigrationStateError, SchemaDriftError, apply_migrations
from warrant.db.schema_check import SchemaDimensionError, verify_embedding_dimensions
from warrant.db.scripts import MigrationSetError
from warrant.embedder_config import EmbedderConfig, get_embedder_config
from warrant.embedding import EmbedderError, ProgressCallback, load_embedder
from warrant.ingest.catalog import CatalogError, load_catalog
from warrant.ingest.chunker import Chunk, chunk_catalog, chunker_fingerprint
from warrant.ingest.pipeline import IngestError, IngestReport, ingest
from warrant.manifest import ManifestError, verify_manifest
from warrant.settings import get_settings
from warrant.tokenizer import TokenizerError

# Progress is printed at most this often. A line per batch is thirty-odd lines of noise; a line
# per tenth is enough to tell a run that is working from one that is stuck, and reads the same in
# a terminal as in a log — which is why this writes lines rather than redrawing one.
_PROGRESS_STEP = 0.1


def main() -> int:
    settings = get_settings()

    try:
        pin = get_catalog_pin()
        config = get_embedder_config()

        # Before anything else, because it is the check that covers the inputs as a set. The
        # catalog hash and the chunker fingerprint below are computed a second time by it; that is
        # seconds against the ten minutes this command would otherwise spend embedding a corpus out
        # of a chunker or a resolver nobody has re-recorded against.
        verify_manifest()

        catalog_sha256 = verify_catalog(settings.catalog_path, pin)
        catalog = load_catalog(settings.catalog_path)
        chunks = chunk_catalog(catalog)
        fingerprint = chunker_fingerprint(catalog)

        if (failure := _count_disagreement(chunks, pin.expected_counts.live)) is not None:
            print(f"error: {failure}", file=sys.stderr)
            return 1

        # Flushed, here and in the progress callback below. Python buffers stdout when it is not
        # a terminal, so a run piped to a file or a log would print nothing for ten minutes and
        # then everything at once — which is indistinguishable from a run that is stuck.
        print(f"{len(chunks)} chunks from {pin.release}.", flush=True)
        print(
            f"Loading {config.name} at revision {config.revision} from the local cache.",
            flush=True,
        )

        embedder = load_embedder(config)

        # Its own connection rather than one from the pool, for the reason the migrator's is: the
        # advisory lock lives in the session, and the write below holds a transaction open.
        with psycopg.connect(settings.database_url) as conn:
            apply_migrations(conn)
            verify_embedding_dimensions(conn, config)

            # That check ran a query, which opened an implicit transaction. Closing it here is
            # what keeps the pipeline's transaction the outermost one: nested inside another it
            # would be a savepoint, and the write would durably land at connection close rather
            # than at the point the pipeline says it does.
            conn.commit()

            report = ingest(
                conn,
                chunks,
                embedder,
                config,
                fingerprint,
                catalog_sha256,
                _progress(),
            )
    except (
        CatalogError,
        CatalogPinError,
        EmbedderError,
        IngestError,
        ManifestError,
        MigrationSetError,
        MigrationStateError,
        SchemaDimensionError,
        SchemaDriftError,
        TokenizerError,
    ) as error:
        # Each of these names what disagreed with what, and none is helped by a traceback. The
        # tokenizer's arrives from the manifest check rather than from anything here: it counts a
        # sample with the pinned encoding, so this command needs that one-off fetch as much as it
        # needs the weights.
        print(f"error: {error}", file=sys.stderr)
        return 1
    except psycopg.OperationalError as error:
        print(
            f"error: could not connect to the database at {settings.database_url}: {error}\n"
            "Is it running? `docker compose up -d db` starts it.",
            file=sys.stderr,
        )
        return 1
    except OSError as error:
        # The message carries its own filename, and that is the point: the block above also reads
        # the catalog pin, the embedder pin and the migration scripts, so naming the catalog here
        # would send whoever is reading to a file that is perfectly fine.
        print(f"error: {error}", file=sys.stderr)
        return 1

    _report(report, config)
    return 0


def _count_disagreement(chunks: tuple[Chunk, ...], expected: int) -> str | None:
    """The message for a chunk count that does not match the pin, or None if it does.

    The same check `make chunks` makes, repeated here rather than deferred to it, because this is
    the command that would otherwise spend ten minutes embedding a corpus of the wrong size.
    """
    if len(chunks) == expected:
        return None

    return (
        f"the chunker produced {len(chunks)} chunks from a catalog pinned at {expected} live "
        "records. One chunk per live control and per live enhancement is the rule the embedding "
        "comparison was measured under, so the count is not loose: either the chunker changed, "
        "or the catalog is not the pinned one. Run `make chunks`, which checks this without "
        "needing a database or a model."
    )


def _progress() -> ProgressCallback:
    """A progress callback that prints a line every tenth of the corpus, and one at the end."""
    started = time.perf_counter()
    printed = 0.0

    def report(done: int, total: int) -> None:
        nonlocal printed

        fraction = done / total

        if fraction < printed + _PROGRESS_STEP and done < total:
            return

        printed = fraction
        elapsed = time.perf_counter() - started
        remaining = elapsed / fraction - elapsed

        print(
            f"  embedded {done:>5}/{total}  {fraction:>4.0%}  {_duration(remaining)} remaining",
            flush=True,
        )

    return report


def _report(report: IngestReport, config: EmbedderConfig) -> None:
    """Print what was written, next to what the pin says should have been."""
    model = f"{config.name} @ {config.revision[:8]}"

    rows = [
        ("chunks", f"{report.chunks_written:,}", _against(config.expected_chunks)),
        ("deleted", f"{report.rows_deleted:,}", "no longer produced by this chunker"),
        ("dimensions", f"{report.dimensions:,}", model),
        ("vector bytes", f"{report.vector_bytes:,}", _against(config.expected_corpus_bytes)),
        ("table bytes", f"{report.table_bytes:,}", "with text, indexes and unvacuumed rows"),
        ("embedded in", _duration(report.embed_seconds), ""),
        ("written in", _duration(report.write_seconds), ""),
    ]

    print()
    for name, value, note in rows:
        print(f"  {name:>14}  {value:>12}  {note}".rstrip())
    print()
    print(f"  fingerprint  {report.corpus_fingerprint}")

    if report.previous_fingerprint is None:
        print("\nCorpus built. Run this again and the fingerprint should not move.")
    elif report.unchanged:
        print("\nCorpus unchanged: same chunks, same vectors as the previous ingest.")
    else:
        print(
            f"\nCorpus changed. The previous ingest left {report.previous_fingerprint[:16]}…, "
            "so the vectors stored before this run are not the ones stored now — every recorded "
            "model call downstream of them describes a corpus that no longer exists."
        )


def _against(expected: int | None) -> str:
    """The pinned figure alongside an observed one, or nothing if the pin does not carry it."""
    return "" if expected is None else f"pinned {expected:,}"


def _duration(seconds: float) -> str:
    """Wall-clock as a person would say it: `9m41s`, `0.4s`."""
    if seconds < 10:
        return f"{seconds:.1f}s"

    minutes, remainder = divmod(int(seconds), 60)

    return f"{minutes}m{remainder:02d}s" if minutes else f"{remainder}s"


if __name__ == "__main__":
    raise SystemExit(main())
