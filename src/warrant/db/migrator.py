"""Brings a database up to the schema this build was written against.

**Forty lines of SQL rather than a migrations framework.** There are two tables and no domain
model to map, so a framework would be the largest dependency here by some distance, and the
whole mechanism has to be explicable at a whiteboard. When the schema starts changing shape
often enough that ordering and rollback need real machinery, the ledger below is already the
table that machinery would want.

**Serialised across processes by a session advisory lock.** Two processes starting together
would otherwise both apply the same file, and the loser would fail on a duplicate object at the
least convenient moment. There is one process today; the lock is two statements and closes the
case permanently.

**One transaction per file.** Postgres makes DDL transactional, so a script that fails halfway
leaves nothing behind and no ledger row, and the next start retries it from a known state. A
single transaction spanning every file would also be atomic, but it makes a five-migration
deployment fail as one opaque unit instead of naming the file that broke.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import LiteralString

import psycopg

from warrant.db.scripts import Migration, load_all, target_version

_log = logging.getLogger(__name__)

# The advisory lock every process contends on: the ASCII bytes of `WARRANT`. Advisory locks
# share one namespace per database, so the value is arbitrary but must be stable and unique to
# this use. Derived from text rather than picked at random so that a pg_locks row is
# identifiable by someone who has never read this file.
ADVISORY_LOCK_KEY = 0x57415252414E54

# to_regclass returns NULL rather than raising for an unknown relation, which is what makes
# "has this database ever been migrated?" a query rather than a caught UndefinedTable. The
# ledger is created by the first migration itself.
_LEDGER_EXISTS = "SELECT to_regclass('public.schema_version') IS NOT NULL"

_READ_LEDGER = "SELECT version, name, checksum FROM schema_version ORDER BY version"

_RECORD = "INSERT INTO schema_version (version, name, checksum) VALUES (%s, %s, %s)"


class MigrationStateError(Exception):
    """The connection handed to the migrator is not in a state it can safely work from."""


class SchemaDriftError(Exception):
    """A migration this database has already applied no longer matches the file in this build.

    Fatal rather than advisory. The ledger says version N was applied and the code believes
    version N is something else, so nothing downstream can be trusted to be looking at the
    schema it was written against. The fix is a new numbered migration, never an edit to one
    that has shipped.
    """

    def __init__(self, file_name: str, recorded: str, current: str) -> None:
        super().__init__(
            f"Migration '{file_name}' was applied to this database as checksum {recorded}, but "
            f"the file in this build hashes to {current}. A migration is immutable once it has "
            "shipped: put the change in a new numbered file, and if the two have genuinely "
            "diverged, reconcile them deliberately rather than by starting the application."
        )
        self.file_name = file_name
        self.recorded_checksum = recorded
        self.current_checksum = current


def apply_migrations(conn: psycopg.Connection) -> int:
    """Apply every migration this database has not recorded; return the version it is left at.

    Idempotent: with nothing pending this is two queries and no writes, which is what makes it
    safe to run unconditionally on every start.

    The connection is used directly rather than taken from a pool, because the advisory lock is
    held by the session and the caller must not be able to hand that session back mid-apply.
    """
    expected = target_version()

    # Without autocommit the first statement below would open an implicit transaction that
    # stays open for the whole apply, and each `conn.transaction()` inside it would be a
    # savepoint rather than a commit — so no migration would durably land until the last one
    # had. Autocommit is what makes "one transaction per file" true.
    with _autocommit(conn):
        _execute(conn, "SELECT pg_advisory_lock(%s)", ADVISORY_LOCK_KEY)

        try:
            applied = _read_ledger(conn)
            _verify_nothing_has_drifted(applied, expected)

            pending = 0

            for migration in load_all():
                if migration.version in applied:
                    continue

                _apply_one(conn, migration)
                pending += 1

            _log.info(
                "Schema is at version %d (%d migration(s) applied this start).",
                expected,
                pending,
            )

            return expected
        finally:
            # Not strictly required — closing the connection drops the lock too. Explicit
            # anyway, because "the lock is released as a side effect of the connection being
            # garbage collected" is not a sentence anyone should have to reconstruct while a
            # deployment is stuck.
            _execute(conn, "SELECT pg_advisory_unlock(%s)", ADVISORY_LOCK_KEY)


@contextmanager
def _autocommit(conn: psycopg.Connection) -> Iterator[None]:
    """Put the connection in autocommit for the apply, and restore what it was afterwards.

    Refuses a connection with a transaction already open. Committing it would commit work this
    code knows nothing about, and rolling it back would discard it; both are decisions for
    whoever started it. The callers that matter hand over a connection they have just opened.
    """
    if conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
        raise MigrationStateError(
            "Migrations need a connection with no transaction in progress, because each "
            "migration has to commit on its own. Commit or roll back first, or hand this a "
            "freshly opened connection."
        )

    previous = conn.autocommit
    conn.autocommit = True

    try:
        yield
    finally:
        conn.autocommit = previous


def _apply_one(conn: psycopg.Connection, migration: Migration) -> None:
    _log.info("Applying migration %s.", migration.file_name)

    # The statements and the ledger row commit together, so there is no interval in which a
    # migration has run without being recorded — which the next start would replay.
    with conn.transaction(), conn.cursor() as cursor:
        # No parameters, which is what allows a file to contain several statements: psycopg
        # sends a parameterless query whole and lets the server split it.
        #
        # Encoded rather than passed as text because psycopg types a str query as LiteralString
        # to catch queries assembled out of user input. This one is the whole of a .sql file
        # shipped inside the package — the one case that guard cannot recognise — and handing
        # over the bytes says that deliberately instead of suppressing the check.
        cursor.execute(migration.sql.encode("utf-8"))
        cursor.execute(_RECORD, (migration.version, migration.name, migration.checksum))


def _read_ledger(conn: psycopg.Connection) -> dict[int, tuple[str, str]]:
    """Read the applied migrations as `version -> (name, checksum)`."""
    with conn.cursor() as cursor:
        cursor.execute(_LEDGER_EXISTS)
        row = cursor.fetchone()

        if row is None or row[0] is not True:
            _log.info("No schema_version table: treating this as an unmigrated database.")
            return {}

        cursor.execute(_READ_LEDGER)

        return {version: (name, checksum) for version, name, checksum in cursor.fetchall()}


def _verify_nothing_has_drifted(applied: dict[int, tuple[str, str]], expected: int) -> None:
    for migration in load_all():
        record = applied.get(migration.version)

        if record is not None and record[1] != migration.checksum:
            raise SchemaDriftError(migration.file_name, record[1], migration.checksum)

    # A version in the ledger this build has no file for means an older build has been deployed
    # over a newer schema. Logged, not fatal: what that migration added is not something this
    # build references, and refusing to start would make rolling a bad release back require a
    # database operation first — exactly when nobody wants one. The fatal case is the one
    # above, where the same version means two different things.
    for version, (name, _) in applied.items():
        if version > expected:
            _log.warning(
                "Database is at schema version %d, ahead of this build's %d. Continuing — this "
                "build does not use anything %s added — but it is running against a schema it "
                "was not written for.",
                version,
                expected,
                name,
            )


def _execute(conn: psycopg.Connection, sql: LiteralString, *params: object) -> None:
    with conn.cursor() as cursor:
        cursor.execute(sql, params)
