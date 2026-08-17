"""Storage: the schema, the migration runner that installs it, and the connection pool.

Plain SQL over psycopg, no ORM. There are two tables and no domain model to map, so an ORM
would be the largest dependency here and would earn nothing in return.
"""

from warrant.db.health import SchemaHealth, check_schema
from warrant.db.migrator import MigrationStateError, SchemaDriftError, apply_migrations
from warrant.db.pool import ConnectionSource, close_pool, connection, open_pool
from warrant.db.schema_check import SchemaDimensionError, verify_embedding_dimensions
from warrant.db.scripts import MigrationSetError, target_version

__all__ = [
    "ConnectionSource",
    "MigrationSetError",
    "MigrationStateError",
    "SchemaDimensionError",
    "SchemaDriftError",
    "SchemaHealth",
    "apply_migrations",
    "check_schema",
    "close_pool",
    "connection",
    "open_pool",
    "target_version",
    "verify_embedding_dimensions",
]
