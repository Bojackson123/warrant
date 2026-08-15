"""The pinned catalog release, read from `data/catalog/pinned.json`.

That file is the single source of truth for which document the system was built against: its
release, its content hash, and the counts a correct copy produces when parsed. Everything
downstream — chunk text, vectors, retrieved ids, the prompts those ids appear in — is a
function of that one file, so a change to it invalidates all of them at once.

Kept out of the ingest package for the same reason the embedder pin is: the ingest pipeline,
the retrieval path and the integrity check all need to know what was pinned, and none of them
should have to import a parser to find out.

Nothing here reads the catalog itself. Verification is offered, not performed — the integrity
check owns the decision to refuse to start, and hashing ten megabytes on every load to
duplicate a check that happens once at startup would be waste.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from warrant.settings import get_settings

# Read in blocks rather than whole. The catalog is ten megabytes today and the number is not
# guaranteed to stay small; a hash function that needs the file to fit in memory is a
# constraint nothing here gains from.
_READ_BLOCK_BYTES = 1024 * 1024


class CatalogPinError(Exception):
    """The catalog on disk is not the one that was pinned.

    Carries the expectation and the observation, because the useful question on seeing this is
    always "which file is this, then" and a bare "checksum mismatch" does not answer it.
    """


class CatalogCounts(BaseModel):
    """What a correct copy of the catalog contains.

    Lives here rather than beside the parser because both sides of the comparison need it: the
    pin declares these, and the loader produces them. Equality on the model is the check.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    groups: int
    base_controls: int
    enhancements: int
    withdrawn: int
    live: int
    parameters: int
    parts: int

    def differences(self, other: CatalogCounts) -> dict[str, tuple[int, int]]:
        """Fields where two count sets disagree, as `field -> (self, other)`.

        A caller reporting a mismatch wants to name the field that moved. Returning the pairs
        rather than a formatted string keeps that decision with whoever is printing.
        """
        return {
            field: (getattr(self, field), getattr(other, field))
            for field in type(self).model_fields
            if getattr(self, field) != getattr(other, field)
        }


class CatalogPin(BaseModel):
    """The identity of the pinned catalog file."""

    # Frozen for the same reason the embedder pin is: this describes a pin. Changing the
    # corpus means editing the file and re-ingesting, never mutating an object in memory.
    model_config = ConfigDict(frozen=True, extra="ignore")

    file: str
    source_repository: str
    source_commit: str
    source_path: str
    release_tag: str
    catalog_version: str
    last_modified: str
    oscal_version: str
    size_bytes: int
    sha256: str
    expected_counts: CatalogCounts

    @property
    def release(self) -> str:
        """One string naming the release, for logs and for the integrity check's report.

        The catalog's own version and the repository release that published it are different
        numbers — `5.2.0` shipped in `v1.5.0` — and quoting either alone has repeatedly been
        enough to make two people think they were talking about the same file.
        """
        return f"SP 800-53 Rev 5 catalog {self.catalog_version} ({self.release_tag})"


@lru_cache(maxsize=1)
def get_catalog_pin() -> CatalogPin:
    """Read the pinned catalog record once per process."""
    return load_catalog_pin(get_settings().catalog_pin_path)


def load_catalog_pin(path: Path) -> CatalogPin:
    """Read a pinned catalog record from a specific file."""
    document = json.loads(path.read_text(encoding="utf-8"))

    # The file carries `_comment` keys addressed to whoever opens it. `extra="ignore"` drops
    # them, here and in the nested counts.
    return CatalogPin.model_validate(document)


def content_hash(path: Path) -> str:
    """SHA-256 of a file's bytes, exactly as they sit on disk.

    Deliberately unlike the migration checksum, which normalises line endings before hashing.
    There the subject is a script someone edits, and a carriage return is not an edit. Here the
    subject is a file published by someone else, whose hash is quoted in this repository and
    computable by anyone who downloads it; normalising would produce a number that agrees with
    nothing upstream. `.gitattributes` marks the catalog binary so a checkout cannot rewrite it.
    """
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while block := handle.read(_READ_BLOCK_BYTES):
            digest.update(block)

    return digest.hexdigest()


def verify_catalog(path: Path, pin: CatalogPin | None = None) -> str:
    """Check a catalog file against the pin, returning its hash.

    Raises `CatalogPinError` naming both hashes on a mismatch. Callers that want a boolean can
    compare `content_hash` themselves; this exists so that the failure, when it happens, reads
    as a sentence about which document is present.
    """
    pin = pin or get_catalog_pin()
    observed = content_hash(path)

    if observed != pin.sha256:
        raise CatalogPinError(
            f"{path} is not the pinned catalog. Expected {pin.release} with SHA-256 "
            f"{pin.sha256}, found {observed}. Every stored vector and every recorded model "
            "call was produced from the pinned file, so a different catalog means they no "
            "longer describe it. Restore the file, or change the pin and re-ingest."
        )

    return observed
