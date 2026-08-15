"""The process-wide connection pool.

Synchronous, deliberately. Embedding a question is CPU-bound work in a C extension, so the
request path has to run in a worker thread whatever the database driver does; an async pool
would buy nothing there and would force the ingest command to either drive an event loop or
open a second pool of its own. One pool, one kind of connection, both callers.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from warrant.settings import get_settings

_pool: ConnectionPool | None = None

# Connections whose vector adaptation is already in place. Weak, so that a connection the pool
# has discarded is forgotten along with it rather than kept alive by this bookkeeping.
_adapted: weakref.WeakSet[psycopg.Connection] = weakref.WeakSet()


def open_pool() -> ConnectionPool:
    """Create the pool if it does not exist yet, and return it.

    Called once on startup. `open=False` then an explicit `wait()` so that a database which is
    not reachable is reported here, by the code that was trying to reach it, rather than by the
    first request that happens to need a connection.
    """
    global _pool

    if _pool is None:
        settings = get_settings()

        pool: ConnectionPool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            configure=_adapt,
            open=False,
        )

        # Cached only once it is open, and closed if it is not. `wait()` closes the pool itself
        # on timeout, so assigning first would leave a permanently closed pool in the global:
        # one slow start would become every later call raising `PoolClosed`, with no way back
        # short of restarting the process.
        try:
            pool.open(wait=True, timeout=settings.db_connect_timeout_seconds)
        except BaseException:
            pool.close()
            raise

        _pool = pool

    return _pool


def _adapt(conn: psycopg.Connection) -> None:
    """Teach one connection to send and receive `vector` values.

    Done per connection rather than at each call site because a query that forgets it fails at
    the point of sending a parameter, several frames away from the omission.

    A database with no migrations applied has no `vector` type to look up, and pgvector raises
    rather than returning. That is tolerated, because the pool has to open against an unmigrated
    database: reporting *which* schema version is applied is the whole job of the readiness
    check, and refusing every connection would make it wait out the connect timeout and then
    answer "unreachable" for a database that is running and merely empty. `connection()` retries
    the registration, so a connection opened before the extension existed still adapts once it
    does.
    """
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        return

    _adapted.add(conn)


def close_pool() -> None:
    """Close the pool and forget it. Called on shutdown; safe if it was never opened."""
    global _pool

    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Borrow a connection from the pool for the duration of the block.

    The membership test is the retry `_adapt` describes. It costs nothing on a migrated database,
    where every connection is adapted as the pool fills it, and it is what stops a pool opened
    before the migrations ran from handing out connections that cannot send a vector afterwards.
    """
    with open_pool().connection() as conn:
        if conn not in _adapted:
            _adapt(conn)

        yield conn
