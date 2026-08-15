"""Report which catalog is on disk and what it contains.

    python -m warrant.ingest.catalog_report

Reads one file and touches nothing else — no database, no network — so it answers "is the
corpus the one this build was written for?" on a machine that has configured neither.

A module of its own rather than the ingest package's `__main__`, which belongs to the pipeline
that writes chunks, and rather than a `__main__` guard inside the loader, which the package
imports at startup and would therefore execute twice.
"""

from __future__ import annotations

import sys

from warrant.catalog_pin import CatalogPinError, get_catalog_pin, verify_catalog
from warrant.ingest.catalog import CatalogError, load_catalog
from warrant.settings import get_settings


def main() -> int:
    settings = get_settings()

    try:
        pin = get_catalog_pin()
        observed_hash = verify_catalog(settings.catalog_path, pin)
        counts = load_catalog(settings.catalog_path).counts()
    except (CatalogError, CatalogPinError) as error:
        # Both name what disagreed with what. Neither is helped by a traceback.
        print(f"error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"error: {settings.catalog_path}: {error}", file=sys.stderr)
        return 1

    print(pin.release)
    print(f"  file    {settings.catalog_path}")
    print(f"  sha256  {observed_hash} (matches the pin)")
    print()

    for name, value in counts.model_dump().items():
        print(f"  {name:>14}  {value:>6}")

    differences = counts.differences(pin.expected_counts)

    if differences:
        # Reachable only by editing the pin, since the hash matched a line earlier. It is
        # still worth reporting rather than asserting: the counts are how a person checks that
        # a deliberate catalog change was recorded correctly.
        print(
            "\nerror: the file matches the pinned hash but not the pinned counts, so the "
            "counts were recorded from something else:",
            file=sys.stderr,
        )
        for name, (found, expected) in differences.items():
            print(f"  {name}: found {found}, pinned {expected}", file=sys.stderr)
        return 1

    print("\nCounts match the pin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
