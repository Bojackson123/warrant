"""Checks that the schema in the database agrees with the model this build was pinned to.

A `vector(n)` column type cannot be parameterised, so the width appears as a literal in the
migration that creates the table. That is a second place the dimensionality is written down,
and this module is what stops the two from disagreeing quietly: the declared width is read back
out of the catalog at startup and compared against `data/embedder.json`.

Getting this wrong is not a crash, which is why it is worth a check. Vectors of the wrong width
are rejected by the column type, so an ingest against a mismatched schema fails — but a schema
built for a model nobody is using any more would otherwise be discovered by a confusing insert
error rather than by a sentence naming both numbers.
"""

from __future__ import annotations

import psycopg

from warrant.embedder_config import EmbedderConfig, get_embedder_config

# format_type renders the type as it was declared, so a vector column comes back as
# 'vector(768)' rather than as the bare type name plus a modifier nobody can read. Restricted
# to a live, non-dropped column of the named table.
_DECLARED_TYPE = """
    SELECT format_type(attribute.atttypid, attribute.atttypmod)
    FROM pg_attribute AS attribute
    JOIN pg_class AS relation ON relation.oid = attribute.attrelid
    JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
    WHERE namespace.nspname = 'public'
      AND relation.relname = %s
      AND attribute.attname = %s
      AND attribute.attnum > 0
      AND NOT attribute.attisdropped
"""

_TABLE = "chunks"
_COLUMN = "embedding"


class SchemaDimensionError(Exception):
    """The embedding column's width does not match the pinned model's dimensionality."""

    def __init__(self, declared: str, expected: int, model: str) -> None:
        super().__init__(
            f"The {_TABLE}.{_COLUMN} column is declared {declared}, but the pinned model "
            f"{model} produces {expected}-dimensional vectors. The schema and the model pin "
            "have diverged: either the database was built for an earlier model and needs a new "
            "migration and a full re-ingest, or the pin was changed without one."
        )
        self.declared_type = declared
        self.expected_dimensions = expected


def verify_embedding_dimensions(
    conn: psycopg.Connection,
    config: EmbedderConfig | None = None,
) -> int:
    """Compare the embedding column's declared width against the pin; return the width.

    The config is a parameter so that a test can hand this a deliberately mismatched pin
    without rewriting the file the whole project reads.
    """
    pinned = config if config is not None else get_embedder_config()

    with conn.cursor() as cursor:
        cursor.execute(_DECLARED_TYPE, (_TABLE, _COLUMN))
        row = cursor.fetchone()

    if row is None:
        raise SchemaDimensionError("absent", pinned.dimensions, pinned.name)

    declared: str = row[0]

    if declared != f"vector({pinned.dimensions})":
        raise SchemaDimensionError(declared, pinned.dimensions, pinned.name)

    return pinned.dimensions
