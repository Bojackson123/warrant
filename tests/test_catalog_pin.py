"""The pinned catalog record, and the hash the integrity check will be built on."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from warrant.catalog_pin import (
    CatalogCounts,
    CatalogPin,
    CatalogPinError,
    content_hash,
    get_catalog_pin,
    load_catalog_pin,
    verify_catalog,
)
from warrant.settings import get_settings


@pytest.fixture
def pin() -> CatalogPin:
    return get_catalog_pin()


@pytest.fixture
def catalog_path() -> Path:
    return get_settings().catalog_path


def test_the_vendored_catalog_is_the_pinned_one(pin: CatalogPin, catalog_path: Path) -> None:
    """The claim the whole record exists to support, checked against the file on disk."""
    assert content_hash(catalog_path) == pin.sha256
    assert catalog_path.stat().st_size == pin.size_bytes
    assert catalog_path.name == pin.file


def test_release_names_both_version_numbers(pin: CatalogPin) -> None:
    """The catalog's version and the release that published it are different numbers."""
    assert pin.catalog_version in pin.release
    assert pin.release_tag in pin.release


def test_a_changed_file_hashes_differently(tmp_path: Path, catalog_path: Path) -> None:
    """One appended byte, which is the smallest change a swapped release could amount to."""
    copy = tmp_path / catalog_path.name
    shutil.copyfile(catalog_path, copy)

    before = content_hash(copy)
    copy.write_bytes(copy.read_bytes() + b" ")

    assert content_hash(copy) != before


def test_verify_refuses_a_different_catalog(
    tmp_path: Path, catalog_path: Path, pin: CatalogPin
) -> None:
    """The failure the fixture manifest will hard-fail on, and what it has to say."""
    copy = tmp_path / catalog_path.name
    copy.write_bytes(catalog_path.read_bytes() + b" ")

    with pytest.raises(CatalogPinError) as raised:
        verify_catalog(copy, pin)

    message = str(raised.value)
    assert pin.sha256 in message, "the expected hash has to be in the message"
    assert content_hash(copy) in message, "so does the one actually found"


def test_verify_returns_the_hash_of_a_good_file(catalog_path: Path, pin: CatalogPin) -> None:
    assert verify_catalog(catalog_path, pin) == pin.sha256


def test_pin_is_read_once(catalog_path: Path) -> None:
    assert get_catalog_pin() is get_catalog_pin()


def test_pin_cannot_be_edited_in_memory(pin: CatalogPin) -> None:
    """A pin that a caller can change is not a pin."""
    with pytest.raises(ValidationError, match="frozen"):
        pin.sha256 = "0" * 64  # type: ignore[misc]


def test_load_from_an_explicit_path(tmp_path: Path) -> None:
    """Comment keys are dropped; everything else survives the round trip."""
    document = """
    {
      "_comment": "ignored",
      "file": "c.json",
      "source_repository": "example.invalid/repo",
      "source_commit": "0123456789abcdef",
      "source_path": "path/to/c.json",
      "release_tag": "v9.9.9",
      "catalog_version": "9.9.9",
      "last_modified": "2026-01-01T00:00:00.00000-00:00",
      "oscal_version": "1.2.2",
      "size_bytes": 12,
      "sha256": "abc",
      "expected_counts": {
        "_comment": "ignored too",
        "groups": 1, "base_controls": 2, "enhancements": 3, "withdrawn": 4,
        "live": 1, "parameters": 5, "parts": 6
      }
    }
    """
    path = tmp_path / "pinned.json"
    path.write_text(document, encoding="utf-8")

    loaded = load_catalog_pin(path)

    assert loaded.release_tag == "v9.9.9"
    assert loaded.expected_counts.parameters == 5


def test_differences_names_only_the_fields_that_moved() -> None:
    """What a mismatch report needs: which number changed, not that something did."""
    counts = CatalogCounts(
        groups=20,
        base_controls=324,
        enhancements=872,
        withdrawn=182,
        live=1014,
        parameters=1600,
        parts=12729,
    )
    other = counts.model_copy(update={"parameters": 1601})

    assert counts.differences(other) == {"parameters": (1600, 1601)}
    assert counts.differences(counts) == {}
