"""The connection pool, against a real database in both of the states it has to survive.

The interesting state is the one nobody designs for: a database that is running, reachable and
completely empty. The pool has to open against it, because the readiness check exists to say so
and cannot say anything if it cannot get a connection — and it has to keep working once the
migrations that were missing have been applied.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import psycopg
import pytest

from warrant.db import pool
from warrant.db.migrator import apply_migrations
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


def test_the_pool_opens_against_a_database_with_no_extension(pooled: None) -> None:
    """Vector adaptation cannot be registered before the migration that creates the type.

    Failing to open here would be a slow failure rather than a loud one: the wait would run to
    the connect timeout and report a database that is running as unreachable.
    """
    with pool.connection() as conn, conn.cursor() as cursor:
        cursor.execute("SELECT 1")

        assert cursor.fetchone() == (1,)


def test_a_pool_opened_before_the_migrations_still_sends_vectors_after_them(
    conn: psycopg.Connection, pooled: None
) -> None:
    """The state a first deployment is in between `docker compose up` and the first migration.

    Connections created while the extension was missing stay in the pool afterwards, so if
    adaptation were only ever attempted when a connection was created, this would fail on a
    pool that had been open for seconds and succeed on one opened a moment later.
    """
    # Opened first, and used, so the pool is holding a connection from before the migration.
    with pool.connection() as pooled_conn, pooled_conn.cursor() as cursor:
        cursor.execute("SELECT 1")
        cursor.fetchone()

    apply_migrations(conn)

    with pool.connection() as pooled_conn, pooled_conn.cursor() as cursor:
        cursor.execute("SELECT %s::vector", (np.array([1.0, 2.0, 3.0], dtype=np.float32),))

        assert cursor.fetchone()[0].to_list() == [1.0, 2.0, 3.0]


def test_a_pool_that_could_not_open_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed start must not outlive the thing that caused it.

    `wait()` closes the pool on timeout, so a pool cached before it opened would leave a closed
    pool in the module global and turn one unreachable database into every later call raising
    `PoolClosed` — including the calls made after the database came back.
    """
    monkeypatch.setenv(
        "WARRANT_DATABASE_URL", "postgresql://nobody:hunter2@127.0.0.1:1/nothing?connect_timeout=1"
    )
    monkeypatch.setenv("WARRANT_DB_CONNECT_TIMEOUT_SECONDS", "2")
    get_settings.cache_clear()
    pool.close_pool()

    try:
        with pytest.raises(psycopg.Error):
            pool.open_pool()

        assert pool._pool is None
    finally:
        pool.close_pool()
        get_settings.cache_clear()
