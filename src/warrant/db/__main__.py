"""Bring a database up to this build's schema, from the command line.

    python -m warrant.db

The same two calls the API makes on startup, exposed as a command so that a developer can
migrate a database without starting a server, and so that the step has a name in the Makefile.
"""

from __future__ import annotations

import logging
import sys

import psycopg

from warrant.db.migrator import MigrationStateError, SchemaDriftError, apply_migrations
from warrant.db.schema_check import SchemaDimensionError, verify_embedding_dimensions
from warrant.db.scripts import MigrationSetError
from warrant.settings import get_settings


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    settings = get_settings()

    try:
        # Its own connection rather than one from the pool: the advisory lock lives in the
        # session, and a pooled connection could be handed back mid-apply.
        with psycopg.connect(settings.database_url) as conn:
            version = apply_migrations(conn)
            dimensions = verify_embedding_dimensions(conn)
    except (
        SchemaDriftError,
        SchemaDimensionError,
        MigrationSetError,
        MigrationStateError,
    ) as error:
        # These are the failures worth reading rather than a traceback: each one names what
        # disagreed with what, and none of them is fixed by looking at a stack.
        print(f"error: {error}", file=sys.stderr)
        return 1
    except psycopg.OperationalError as error:
        print(
            f"error: could not connect to the database at {settings.database_url}: {error}\n"
            "Is it running? `docker compose up -d db` starts it.",
            file=sys.stderr,
        )
        return 1

    print(f"Schema is at version {version}; embeddings are {dimensions}-dimensional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
