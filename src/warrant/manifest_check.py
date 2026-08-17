"""Check the pinned inputs against the manifest, or deliberately record a change to them.

    python -m warrant.manifest_check
    python -m warrant.manifest_check --write

Reads the manifest, the two pins, the catalog and the cached tokenizer encoding, and touches
nothing else — no database, no embedding model, no network — so the question "are the inputs this
build carries the ones everything stored was made from?" is answerable on a machine that has none
of those.

The encoding is the one input here that has to have been fetched first. What an encoding counts can
only be checked by counting something with it, so `make tokenizer` is a prerequisite of this check
rather than only of the commands that report a prompt size.

The two modes are separate rather than one command with a repair path, and that is the mechanism
rather than a preference about interfaces. A check that fixes what it finds records whatever
happens to be on disk at the moment it ran, which is not a record of anything. Writing is asked
for, its diff goes into a pull request, and the reason goes in beside it.

Its own module rather than the manifest's `__main__`, for the reason the catalog report is not the
catalog's: `warrant.manifest` is imported by everything that validates before it does expensive
work, and running a package module as a script imports it a second time under another name.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import ValidationError

from warrant.catalog_pin import CatalogPinError
from warrant.ingest.catalog import CatalogError
from warrant.manifest import (
    Comparison,
    Manifest,
    ManifestError,
    compare,
    get_manifest,
    observe,
    verify_manifest,
    write_manifest,
)
from warrant.settings import get_settings
from warrant.tokenizer import TokenizerError

_USAGE = "usage: python -m warrant.manifest_check [--write]"

# Column width for the entry name. `parameter_resolution` is the longest today; a longer one added
# later spills its column rather than misaligning the ones above it, which is the failure mode
# worth having.
_NAME_WIDTH = 21


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv

    if set(arguments) - {"--write"}:
        print(_USAGE, file=sys.stderr)
        return 1

    write = "--write" in arguments
    path = get_settings().manifest_path

    try:
        manifest = get_manifest()

        if write:
            _write(path, manifest)
            return 0

        comparisons = verify_manifest(manifest)
    except (CatalogError, CatalogPinError, ManifestError, TokenizerError) as error:
        # Each names what disagreed with what, and none is helped by a traceback. The tokenizer's
        # is the one that is not a disagreement: observing that entry counts a sample with the
        # pinned encoding, so a machine that has not run `make tokenizer` cannot answer the
        # question this command asks. Its message names the cache and the command that fills it,
        # which is what somebody in that position needs.
        print(f"error: {error}", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, ValidationError) as error:
        # Every file this reads is edited by hand, so a malformed one is an ordinary failure rather
        # than a bug. `load_manifest` already turns its own two into `ManifestError` above; the
        # catalog and embedder pins are read through loaders this module does not own, and a
        # half-saved edit to either arrives here as the parser's own exception.
        print(f"error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        # Carries its own filename, which is the point: the block above reads the manifest, the
        # catalog pin, the embedder pin and the catalog itself, so naming any one of them here
        # would send whoever is reading to a file that is perfectly fine.
        print(f"error: {error}", file=sys.stderr)
        return 1

    _report(comparisons)
    print(f"\nEvery pinned input matches {path}.")
    return 0


def _write(path: Path, manifest: Manifest) -> None:
    """Record what the inputs currently are, printing what that changes first.

    Printed before rather than after, and printed whether or not anything moved, because the useful
    output of a regeneration that changed nothing is the sentence saying so — that is how somebody
    who ran this expecting a diff finds out they edited something that is not a manifest input.
    """
    observed = observe()
    comparisons = compare(manifest, observed)
    moved = [comparison for comparison in comparisons if not comparison.matches]

    _report(comparisons)

    if not moved:
        print(f"\nNothing moved. {path} already records these inputs; it is rewritten unchanged.")
    else:
        print()
        for comparison in moved:
            print(f"  {comparison.name}")
            print(f"    was  {_shown(comparison.entry.identity, comparison.entry.digest)}")
            print(f"    now  {_shown(comparison.observed.identity, comparison.observed.digest)}")
            print(f"    {comparison.entry.cost()}")

    write_manifest(path, observed)

    if moved:
        print(
            f"\n{path} now records the inputs above. Everything named as invalidated was made "
            "with the inputs it no longer records, so it is stale until it is rebuilt — this "
            "command records the change and cannot repair it."
        )


def _report(comparisons: tuple[Comparison, ...]) -> None:
    """Print every entry with what it currently is, whether or not anything moved."""
    print()

    for comparison in comparisons:
        state = "ok" if comparison.matches else "moved"
        identity = comparison.observed.identity or "not built"

        print(f"  {state:>5}  {comparison.name:<{_NAME_WIDTH}}  {identity}")

        if comparison.observed.digest is not None:
            print(f"  {'':>5}  {'':<{_NAME_WIDTH}}  {comparison.observed.digest}")


def _shown(identity: str | None, digest: str | None) -> str:
    """An entry's recorded pair on one line, or what it says when the input is not built."""
    return "not built" if identity is None and digest is None else f"{identity}  {digest}"


if __name__ == "__main__":
    raise SystemExit(main())
