"""Report what the chunker produces from the catalog on disk.

    python -m warrant.ingest.chunk_report

Reads two files and touches nothing else — no database, no embedding model, no network — so it
answers "what does the corpus look like?" on a machine that has neither, and before the ingest
pipeline that would otherwise be the only way to find out.

It does not hash the catalog. `make catalog` reports the pin and the integrity check enforces it;
a second command computing the same digest would give the number a second home to drift from.
What this compares against the pin is the count, because the number of live records is exactly
the number of chunks the corpus should contain.
"""

from __future__ import annotations

import sys

from warrant.catalog_pin import get_catalog_pin
from warrant.ingest.catalog import CatalogError, load_catalog
from warrant.ingest.chunker import CHUNKER_VERSION, Chunk, chunk_catalog, chunker_fingerprint
from warrant.settings import get_settings


def main() -> int:
    settings = get_settings()

    try:
        pin = get_catalog_pin()
        catalog = load_catalog(settings.catalog_path)
        chunks = chunk_catalog(catalog)
        fingerprint = chunker_fingerprint(catalog)
    except CatalogError as error:
        # Names what disagreed with what, and is not helped by a traceback.
        print(f"error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"error: {settings.catalog_path}: {error}", file=sys.stderr)
        return 1

    counts = catalog.counts()

    print(f"chunker {CHUNKER_VERSION}")
    print(f"  fingerprint  {fingerprint}")
    print()
    print(f"  {'chunks':>14}  {len(chunks):>6}")
    print(f"  {'pinned live':>14}  {pin.expected_counts.live:>6}")
    print(f"  {'withdrawn':>14}  {counts.withdrawn:>6}  excluded from the corpus")
    print()

    # A corpus of nothing is reachable without a malformed catalog — a release that withdrew
    # every control would produce one — and there is no distribution to describe. Reported by
    # the count check below, which says something useful, rather than by a traceback here.
    if chunks:
        lengths = sorted(len(chunk.text) for chunk in chunks)
        print("  text length in characters")
        for name, value in _distribution(lengths).items():
            print(f"  {name:>14}  {value:>6}")

        print()
        print(f"  {'part paths':>14}  {', '.join(_part_paths(chunks))}")

    if len(chunks) != pin.expected_counts.live:
        print(
            f"\nerror: the chunker produced {len(chunks)} chunks from a catalog pinned at "
            f"{pin.expected_counts.live} live records. One chunk per live control and per live "
            "enhancement is the rule the embedding comparison was measured under, so the count "
            "is not loose: either the chunker changed, or the file at "
            f"{settings.catalog_path} is not the pinned catalog. Run `make catalog`, which "
            "checks the file itself against the pin, to tell the two apart.",
            file=sys.stderr,
        )
        return 1

    print(f"\nOne chunk per live record, matching {pin.release}.")
    return 0


def _distribution(lengths: list[int]) -> dict[str, int]:
    """Median, p90, p99 and max of an already-sorted, non-empty list.

    Nearest-rank rather than interpolated, and clamped at the end, so the numbers reported are
    lengths of chunks that exist rather than averages of two that do.
    """
    last = len(lengths) - 1

    return {
        "median": lengths[min(last, len(lengths) // 2)],
        "p90": lengths[min(last, int(len(lengths) * 0.9))],
        "p99": lengths[min(last, int(len(lengths) * 0.99))],
        "max": lengths[last],
    }


def _part_paths(chunks: tuple[Chunk, ...]) -> list[str]:
    """The distinct part paths in the corpus, most common first.

    One value today, because every live control has exactly one statement part and one guidance
    part. Reported rather than assumed: a second value appearing here is the visible form of the
    catalog gaining a shape the chunker was not written against.
    """
    order: dict[str, int] = {}

    for chunk in chunks:
        order[chunk.part_path] = order.get(chunk.part_path, 0) + 1

    return [f"{path} ({count})" for path, count in sorted(order.items(), key=lambda p: -p[1])]


if __name__ == "__main__":
    raise SystemExit(main())
