"""The numbered SQL files that define the schema, and the checksum recorded for each.

The files live inside the package rather than beside it. A migration is part of the build that
ships it, so it should not be separately deletable, mountable over, or missing from a container
because someone forgot a line in a Dockerfile — the failure that produces is an application
that starts happily against a database it has not migrated.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from importlib.resources.abc import Traversable

# `0002_chunks.sql`: the number orders the file and is its identity in the ledger, the name is
# for the ledger and the logs.
_FILE_NAME = re.compile(r"^(?P<version>\d+)_(?P<name>[a-z0-9_]+)\.sql$")

_MIGRATIONS_PACKAGE = f"{__package__}.migrations"


class MigrationSetError(Exception):
    """The migrations shipped in this build are not a coherent set.

    Raised for an unparseable filename, a duplicated or skipped version, or no migrations at
    all. All of those mean the build is wrong, and all of them are worth refusing to start
    over: the alternative is an application running against half a schema.
    """


@dataclass(frozen=True, slots=True)
class Migration:
    """One numbered `.sql` file, with the checksum the ledger records for it."""

    version: int
    name: str
    sql: str
    checksum: str

    @property
    def file_name(self) -> str:
        """The name this script was loaded from, for messages and the logs."""
        return f"{self.version:04d}_{self.name}.sql"

    def __str__(self) -> str:
        return self.file_name


def checksum_of(sql: str) -> str:
    """Hash a script's text, ignoring carriage returns.

    Line endings are normalised because the same file is checked out CRLF on a Windows
    development machine and LF in a Linux container, and a migration that hashed differently
    depending on where it ran would turn the drift check into a coin toss. What is being
    detected is an edit to the statements, and a carriage return is not one.
    """
    return hashlib.sha256(sql.replace("\r", "").encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def load_all() -> tuple[Migration, ...]:
    """Load every shipped migration, ordered by version.

    Cached: the files are part of the build and cannot change while the process is running,
    and both the migrator and the health check ask for them.
    """
    return load_from(resources.files(_MIGRATIONS_PACKAGE))


def load_from(directory: Traversable) -> tuple[Migration, ...]:
    """Load an ordered, validated migration set from a directory.

    Separate from `load_all` so the validation below can be exercised against deliberately
    broken sets without shipping any.
    """
    migrations: list[Migration] = []

    for entry in directory.iterdir():
        if not entry.name.endswith(".sql"):
            continue

        match = _FILE_NAME.match(entry.name)

        if match is None:
            raise MigrationSetError(
                f"Migration '{entry.name}' is not named <number>_<lowercase_name>.sql. The "
                "number is what orders it and what identifies it in schema_version, so a file "
                "without one cannot be applied safely."
            )

        sql = entry.read_text(encoding="utf-8")

        migrations.append(
            Migration(
                version=int(match.group("version")),
                name=match.group("name"),
                sql=sql,
                checksum=checksum_of(sql),
            )
        )

    if not migrations:
        raise MigrationSetError(
            f"No migrations were found in {directory}. The package ships its .sql files as "
            "package data; an empty set means they did not make it into the build."
        )

    migrations.sort(key=lambda migration: migration.version)

    # Contiguous from 1, so a file that was never committed — or one that landed on a number
    # someone else had already used — is caught here rather than by whatever the gap turns out
    # to have contained.
    for index, migration in enumerate(migrations, start=1):
        if migration.version != index:
            present = ", ".join(str(candidate) for candidate in migrations)
            raise MigrationSetError(
                f"Migration versions must run 1..{len(migrations)} with no gap and no "
                f"duplicate; found {migration.version} where {index} was expected. "
                f"Migrations present: {present}."
            )

    return tuple(migrations)


def target_version() -> int:
    """The schema version this build expects to be running against."""
    return load_all()[-1].version
