"""Readiness: is this database the one this build expects?

A real round trip, and a meaningful one. `SELECT 1` would prove a socket is open; checking that
a connection object is non-null would prove nothing at all. Reading the ledger exercises the
pool, the credentials, the database and the table the rest of the system depends on, and it
answers the question a deployment actually has rather than the question a ping answers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import psycopg
from psycopg import sql

from warrant.db.pool import connection
from warrant.db.scripts import target_version

_log = logging.getLogger(__name__)

# An aggregate rather than a row count, so an unexpected extra row cannot be read as the current
# version by accident. NULL comes back for a ledger that exists but has recorded nothing.
_READ_VERSION = "SELECT max(version) FROM schema_version"

# Well inside any sensible probe interval. A readiness check that hangs is indistinguishable
# from a failing one to whatever is polling it, but takes far longer to say so.
_TIMEOUT_SECONDS = 5


@dataclass(frozen=True, slots=True)
class SchemaHealth:
    """What the schema check found, in a shape an endpoint can serialise."""

    healthy: bool
    expected_version: int
    applied_version: int | None
    detail: str


def check_schema() -> SchemaHealth:
    """Read the applied schema version and compare it with the version this build expects.

    Read on every call rather than cached from startup: a cached answer would report the
    database healthy for as long as the process lived, whatever happened to it afterwards.
    """
    expected = target_version()

    try:
        with connection() as conn, conn.cursor() as cursor:
            # SET takes no bound parameters, so the value is composed rather than passed.
            cursor.execute(
                sql.SQL("SET LOCAL statement_timeout = {}").format(_TIMEOUT_SECONDS * 1000)
            )
            cursor.execute(_READ_VERSION)
            row = cursor.fetchone()
    except psycopg.Error:
        # Logged here and deliberately not returned. An unreachable database names hosts, ports
        # and sometimes usernames in its message, and a health endpoint is reachable by anything
        # that can see the port.
        _log.exception("The schema version could not be read.")

        return SchemaHealth(
            healthy=False,
            expected_version=expected,
            applied_version=None,
            detail="the schema version could not be read — see the application log",
        )

    applied = row[0] if row is not None else None

    if applied is None:
        return SchemaHealth(
            healthy=False,
            expected_version=expected,
            applied_version=None,
            detail="the schema_version table is present but empty",
        )

    if applied != expected:
        return SchemaHealth(
            healthy=False,
            expected_version=expected,
            applied_version=applied,
            detail=f"the database is at schema version {applied} and this build expects {expected}",
        )

    return SchemaHealth(
        healthy=True,
        expected_version=expected,
        applied_version=applied,
        detail=f"schema version {applied}",
    )
