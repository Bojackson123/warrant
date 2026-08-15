"""Discovery, ordering and checksumming of the numbered SQL files.

No database here. Everything below is about the set of files being coherent before anything is
applied, which is the half of the mechanism that can be wrong on a developer's machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from warrant.db.scripts import (
    MigrationSetError,
    checksum_of,
    load_all,
    load_from,
    target_version,
)


def write(directory: Path, *names: str) -> Path:
    """Create a migration set with one trivial statement per named file."""
    for name in names:
        (directory / name).write_text(f"SELECT '{name}';\n", encoding="utf-8")

    return directory


def test_shipped_migrations_load() -> None:
    """The files this build actually ships are a valid set."""
    migrations = load_all()

    assert [migration.version for migration in migrations] == list(range(1, len(migrations) + 1))
    assert target_version() == migrations[-1].version


def test_shipped_migrations_start_with_the_ledger() -> None:
    """The ledger has to exist before anything can be recorded in it."""
    assert "schema_version" in load_all()[0].sql


def test_files_are_ordered_by_number_not_by_name(tmp_path: Path) -> None:
    """Lexical order would put 10 before 2, which is the whole reason to parse the number."""
    write(tmp_path, "0002_second.sql", "0010_tenth.sql", "0001_first.sql")
    for version in range(3, 10):
        (tmp_path / f"{version:04d}_filler.sql").write_text("SELECT 1;\n", encoding="utf-8")

    versions = [migration.version for migration in load_from(tmp_path)]

    assert versions == sorted(versions)
    assert versions[-1] == 10


def test_a_gap_is_refused(tmp_path: Path) -> None:
    """A file that was never committed shows up here, not in whatever the gap contained."""
    write(tmp_path, "0001_first.sql", "0003_third.sql")

    with pytest.raises(MigrationSetError, match="found 3 where 2 was expected"):
        load_from(tmp_path)


def test_a_duplicate_number_is_refused(tmp_path: Path) -> None:
    """Two files claiming one version means one of them would never be applied."""
    write(tmp_path, "0001_first.sql", "0001_also_first.sql", "0002_second.sql")

    with pytest.raises(MigrationSetError):
        load_from(tmp_path)


def test_an_unnumbered_file_is_refused(tmp_path: Path) -> None:
    write(tmp_path, "0001_first.sql", "cleanup.sql")

    with pytest.raises(MigrationSetError, match="not named"):
        load_from(tmp_path)


def test_an_empty_set_is_refused(tmp_path: Path) -> None:
    """No migrations means the files did not make it into the build."""
    with pytest.raises(MigrationSetError, match="No migrations"):
        load_from(tmp_path)


def test_non_sql_files_are_ignored(tmp_path: Path) -> None:
    write(tmp_path, "0001_first.sql")
    (tmp_path / "README.md").write_text("notes\n", encoding="utf-8")

    assert len(load_from(tmp_path)) == 1


def test_checksum_ignores_line_endings() -> None:
    """The same file checked out CRLF and LF has to hash the same, or drift is a coin toss."""
    assert checksum_of("SELECT 1;\r\nSELECT 2;\r\n") == checksum_of("SELECT 1;\nSELECT 2;\n")


def test_checksum_changes_when_a_statement_changes() -> None:
    """The check exists to notice edits, so it has to notice this one."""
    assert checksum_of("SELECT 1;\n") != checksum_of("SELECT 2;\n")


def test_checksum_notices_whitespace_that_is_not_a_line_ending() -> None:
    """Only carriage returns are normalised; anything else is a change to the file."""
    assert checksum_of("SELECT 1;\n") != checksum_of("SELECT  1;\n")


def test_file_name_round_trips(tmp_path: Path) -> None:
    """Messages name the file, so the reconstructed name has to match the one on disk."""
    write(tmp_path, "0001_schema_version.sql")

    assert load_from(tmp_path)[0].file_name == "0001_schema_version.sql"
