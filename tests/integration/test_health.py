"""The readiness check, against a real database.

The question it exists to answer is "is this database the one this build expects?", so the
cases worth testing are the ones where the honest answer is no.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest

from warrant.db import pool
from warrant.db.health import check_schema
from warrant.db.migrator import apply_migrations
from warrant.db.scripts import target_version
from warrant.settings import get_settings

pytestmark = pytest.mark.integration


@pytest.fixture
def pooled(dsn: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point the process-wide pool at this test's database, and put it back afterwards."""
    monkeypatch.setenv("WARRANT_DATABASE_URL", dsn)
    get_settings.cache_clear()
    pool.close_pool()

    try:
        yield
    finally:
        pool.close_pool()
        get_settings.cache_clear()


def test_a_migrated_database_is_healthy(conn: psycopg.Connection, pooled: None) -> None:
    apply_migrations(conn)

    health = check_schema()

    assert health.healthy
    assert health.applied_version == target_version()
    assert health.expected_version == target_version()


def test_an_unmigrated_database_is_not_healthy(pooled: None) -> None:
    """No ledger at all: the check reports rather than raising, because a probe has to answer."""
    health = check_schema()

    assert not health.healthy
    assert health.applied_version is None


def test_an_empty_ledger_is_not_healthy(conn: psycopg.Connection, pooled: None) -> None:
    """A table present but recording nothing is somebody's hand-rolled restore, not health."""
    apply_migrations(conn)

    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM schema_version")
    conn.commit()

    health = check_schema()

    assert not health.healthy
    assert health.applied_version is None
    assert "empty" in health.detail


def test_a_schema_behind_this_build_is_not_healthy(conn: psycopg.Connection, pooled: None) -> None:
    """The case a deployment actually hits: new code, database not yet migrated to match."""
    apply_migrations(conn)

    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM schema_version WHERE version = %s", (target_version(),))
    conn.commit()

    health = check_schema()

    assert not health.healthy
    assert health.applied_version == target_version() - 1
    assert str(target_version()) in health.detail


def test_an_unreachable_database_reports_without_leaking_the_connection_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A health endpoint is reachable by anything that can see the port, so the detail it
    returns must not name hosts, ports or usernames."""
    monkeypatch.setenv(
        "WARRANT_DATABASE_URL",
        "postgresql://nobody:hunter2@127.0.0.1:1/nothing?connect_timeout=1",
    )
    # Nothing is listening, so the only thing waiting the full startup budget would buy is a
    # slower test.
    monkeypatch.setenv("WARRANT_DB_CONNECT_TIMEOUT_SECONDS", "2")
    get_settings.cache_clear()
    pool.close_pool()

    try:
        health = check_schema()
    finally:
        pool.close_pool()
        get_settings.cache_clear()

    assert not health.healthy
    assert health.applied_version is None

    for leak in ("hunter2", "nobody", "127.0.0.1", "nothing"):
        assert leak not in health.detail
