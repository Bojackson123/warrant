"""A real Postgres for the tests that need one.

Real, not a double. What is under test here is pgvector's behaviour, Postgres's transactional
DDL and its advisory locks — a fake database would assert only that the test's own assumptions
are self-consistent, which is the least useful thing it could tell anyone.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql
from testcontainers.community.postgres import PostgresContainer

# The same image the compose file names. Three Postgres versions across three environments is a
# bad afternoon waiting to happen, and the extension version matters as much as the server's.
IMAGE = "pgvector/pgvector:0.8.6-pg17-trixie"


@pytest.fixture(scope="session")
def postgres() -> Iterator[PostgresContainer]:
    """One container for the whole session; each test gets a clean database inside it."""
    with PostgresContainer(IMAGE, driver=None) as container:
        yield container


@pytest.fixture
def dsn(postgres: PostgresContainer, request: pytest.FixtureRequest) -> Iterator[str]:
    """A connection string to a database created fresh for this test and dropped after it.

    A per-test database rather than a per-test container: starting Postgres costs seconds and
    CREATE DATABASE costs milliseconds, and the isolation is the same. It has to be genuine
    isolation rather than a truncate, because most of these tests are about what happens to an
    unmigrated database.
    """
    digest = hashlib.sha256(request.node.nodeid.encode("utf-8")).hexdigest()[:16]
    name = f"test_{digest}"
    admin = postgres.get_connection_url(driver=None)

    # CREATE DATABASE cannot run inside a transaction block, hence autocommit.
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))

    try:
        yield admin.rsplit("/", 1)[0] + f"/{name}"
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            # FORCE, because a test that left a connection open should not also leave a
            # database behind for the next run to collide with.
            conn.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )


@pytest.fixture
def conn(dsn: str) -> Iterator[psycopg.Connection]:
    """An open connection to this test's own database."""
    with psycopg.connect(dsn) as connection:
        yield connection
