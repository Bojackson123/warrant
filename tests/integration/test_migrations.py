"""The migration runner against a real database.

These cover the properties that only exist once Postgres is involved: that concurrent starts
serialise on the advisory lock, that an edited migration is refused, and that the schema the
files produce is the one the pinned model needs.
"""

from __future__ import annotations

import threading
from dataclasses import replace

import psycopg
import pytest

from warrant.db.migrator import MigrationStateError, apply_migrations
from warrant.db.schema_check import SchemaDimensionError, verify_embedding_dimensions
from warrant.db.scripts import Migration, checksum_of, load_all, target_version
from warrant.embedder_config import get_embedder_config

pytestmark = pytest.mark.integration


def ledger(conn: psycopg.Connection) -> list[tuple[int, str]]:
    with conn.cursor() as cursor:
        cursor.execute("SELECT version, name FROM schema_version ORDER BY version")
        return [(version, name) for version, name in cursor.fetchall()]


def test_applies_from_empty(conn: psycopg.Connection) -> None:
    assert apply_migrations(conn) == target_version()
    assert [version for version, _ in ledger(conn)] == [
        migration.version for migration in load_all()
    ]


def test_reapplying_changes_nothing(conn: psycopg.Connection) -> None:
    """Safe to run unconditionally on every start is the whole point."""
    apply_migrations(conn)
    before = ledger(conn)
    conn.commit()

    assert apply_migrations(conn) == target_version()
    assert ledger(conn) == before


def test_a_connection_mid_transaction_is_refused(conn: psycopg.Connection) -> None:
    """Each migration commits on its own, so an open transaction is not a state to start from."""
    conn.execute("SELECT 1")

    with pytest.raises(MigrationStateError, match="no transaction in progress"):
        apply_migrations(conn)


def test_two_sessions_starting_together_apply_each_migration_once(dsn: str) -> None:
    """Two processes starting at once must not both apply the same file.

    Two threads rather than two processes: the advisory lock is held by the *session*, so what
    has to be tested is two connections racing, and threads give that with a barrier that
    releases them into the lock at the same moment. Two processes would exercise the same lock
    with more moving parts and a worse failure message.
    """
    ready = threading.Barrier(2)
    failures: list[BaseException] = []
    versions: list[int] = []
    lock = threading.Lock()

    def start() -> None:
        try:
            with psycopg.connect(dsn) as conn:
                ready.wait(timeout=30)
                version = apply_migrations(conn)

            with lock:
                versions.append(version)
        except BaseException as error:  # noqa: BLE001 — re-raised below, in the main thread
            with lock:
                failures.append(error)

    threads = [threading.Thread(target=start) for _ in range(2)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    if failures:
        raise failures[0]

    assert versions == [target_version(), target_version()]

    with psycopg.connect(dsn) as conn:
        applied = ledger(conn)

    # One row per file and no more: a duplicate here would mean the lock did not hold, and a
    # missing one would mean a start returned before its work was recorded.
    assert applied == [(migration.version, migration.name) for migration in load_all()]


def test_an_edited_migration_is_refused(
    conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure has to name the file and both checksums, or it cannot be acted on."""
    apply_migrations(conn)

    original = load_all()
    edited_sql = original[0].sql + "\n-- a line added after this file shipped\n"
    edited = replace(original[0], sql=edited_sql, checksum=checksum_of(edited_sql))
    monkeypatch.setattr("warrant.db.migrator.load_all", lambda: (edited, *original[1:]))

    with pytest.raises(Exception) as caught:
        apply_migrations(conn)

    message = str(caught.value)

    assert edited.file_name in message
    assert original[0].checksum in message
    assert edited.checksum in message


def test_a_database_ahead_of_this_build_is_tolerated(conn: psycopg.Connection) -> None:
    """An older build over a newer schema should still start, so a rollback needs no DBA."""
    apply_migrations(conn)

    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO schema_version (version, name, checksum) VALUES (%s, %s, %s)",
            (target_version() + 1, "from_a_newer_build", "0" * 64),
        )
    conn.commit()

    assert apply_migrations(conn) == target_version()


def test_a_failing_migration_leaves_nothing_behind(
    conn: psycopg.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DDL is transactional, so a half-applied file is not a state the next start can meet."""
    broken = Migration(
        version=target_version() + 1,
        name="broken",
        sql="CREATE TABLE half_a_table (id integer); SELECT this_function_does_not_exist();",
        checksum="irrelevant",
    )
    monkeypatch.setattr("warrant.db.migrator.load_all", lambda: (*load_all(), broken))

    with pytest.raises(psycopg.Error):
        apply_migrations(conn)

    conn.rollback()

    with conn.cursor() as cursor:
        cursor.execute("SELECT to_regclass('public.half_a_table')")
        row = cursor.fetchone()

    assert row is not None
    assert row[0] is None


class TestSchema:
    """What the migrations actually produce."""

    @pytest.fixture(autouse=True)
    def _migrated(self, conn: psycopg.Connection) -> None:
        apply_migrations(conn)

    def test_the_vector_extension_is_installed(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            assert cursor.fetchone() is not None

    def test_control_id_is_a_column_and_not_a_json_key(self, conn: psycopg.Connection) -> None:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = 'chunks' AND column_name = 'control_id'"
            )
            row = cursor.fetchone()

        assert row is not None
        assert row[0] == "text"

    def test_the_embedding_width_matches_the_pinned_model(self, conn: psycopg.Connection) -> None:
        assert verify_embedding_dimensions(conn) == get_embedder_config().dimensions

    def test_a_pin_the_schema_was_not_built_for_is_refused(self, conn: psycopg.Connection) -> None:
        """The check exists so this disagreement is a sentence, not a puzzling insert error."""
        pinned = get_embedder_config()
        other = pinned.model_copy(update={"dimensions": pinned.dimensions + 1})

        with pytest.raises(SchemaDimensionError, match=str(pinned.dimensions)):
            verify_embedding_dimensions(conn, other)

    def test_a_vector_of_the_wrong_width_is_rejected_by_the_column(
        self, conn: psycopg.Connection
    ) -> None:
        """The column type is what makes a mismatched ingest fail rather than silently rank
        against nonsense."""
        dimensions = get_embedder_config().dimensions

        def vector(width: int) -> str:
            return "[" + ",".join("0" for _ in range(width)) + "]"

        row = (
            "ac-02",
            "ac-02",
            "ac-02",
            "AC-2",
            "Account Management",
            "a.",
            "some clause text",
            "test",
        )
        insert = (
            "INSERT INTO chunks (chunk_id, control_id, base_control_id, control_label, title, "
            "part_path, text, chunker_version, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, "
            "%s, %s)"
        )

        with conn.cursor() as cursor:
            cursor.execute(insert, (*row, vector(dimensions)))

            with pytest.raises(psycopg.Error, match=f"{dimensions}"):
                cursor.execute(insert, ("ac-02-wrong", *row[1:], vector(dimensions - 1)))

        conn.rollback()
