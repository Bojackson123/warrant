-- The ledger every later migration records itself in.
--
-- Applied at startup, in one transaction, under an advisory lock. The runner discovers this
-- file by name: the leading number is both its order and its identity.
--
-- The ledger is created by a migration rather than by a bootstrap string inside the runner,
-- so that its definition sits in SQL next to every other table instead of somewhere nobody
-- would think to look. The runner asks to_regclass whether this table exists yet, which is
-- what makes "has this database ever been migrated?" a query rather than a caught error.
--
-- This file is immutable now that it has shipped. The runner stores a checksum of what it
-- applied and refuses to start against a database whose recorded checksum no longer matches,
-- because a schema that has quietly drifted from the code is a system confidently wrong about
-- its own state. Schema changes go in a new numbered file.

CREATE TABLE IF NOT EXISTS schema_version (
    version    integer     NOT NULL PRIMARY KEY,
    name       text        NOT NULL,
    checksum   text        NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE schema_version IS
    'One row per applied migration file. Written at startup, read by the schema health check.';

COMMENT ON COLUMN schema_version.checksum IS
    'SHA-256 of the migration file as applied, carriage returns stripped. A mismatch means the '
    'file changed after it shipped, which is a startup failure rather than a warning.';
