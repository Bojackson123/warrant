"""The process-wide connection pool.

Synchronous, deliberately. Embedding a question is CPU-bound work in a C extension, so the
request path has to run in a worker thread whatever the database driver does; an async pool
would buy nothing there and would force the ingest command to either drive an event loop or
open a second pool of its own. One pool, one kind of connection, both callers.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg_pool import ConnectionPool

from warrant.settings import get_settings

_pool: ConnectionPool | None = None


def open_pool() -> ConnectionPool:
    """Create the pool if it does not exist yet, and return it.

    Called once on startup. `open=False` then an explicit `wait()` so that a database which is
    not reachable is reported here, by the code that was trying to reach it, rather than by the
    first request that happens to need a connection.
    """
    global _pool

    if _pool is None:
        settings = get_settings()

        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            open=False,
        )
        _pool.open(wait=True, timeout=settings.db_connect_timeout_seconds)

    return _pool


def close_pool() -> None:
    """Close the pool and forget it. Called on shutdown; safe if it was never opened."""
    global _pool

    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Borrow a connection from the pool for the duration of the block."""
    with open_pool().connection() as conn:
        yield conn
