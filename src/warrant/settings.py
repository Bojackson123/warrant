"""Runtime configuration, read from the environment with defaults that need no setup.

The load-bearing property here is that importing this module and constructing `Settings`
with nothing set at all succeeds. A reviewer with no API key, no database and no network
gets a working object reporting replay mode, because that is the path `docker compose up`
takes and it must not depend on anyone having read the documentation first.

Nothing in here touches the filesystem. The paths below are held, not opened; the code that
needs a file is the code that validates it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Mode = Literal["live", "replay"]

# Anchored to this file rather than the working directory, so a default path means the same
# thing whether the process was started from the repository root or from inside a container.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configuration for one process, with every field defaulted."""

    model_config = SettingsConfigDict(
        env_prefix="WARRANT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_api_key` would otherwise collide with pydantic's reserved `model_` prefix
        # and warn on every import. The field is named for the generation model, and
        # `WARRANT_MODEL_API_KEY` is the clearer variable to document.
        protected_namespaces=(),
    )

    # Absence of this key is what selects replay mode. See `mode` below.
    model_api_key: str | None = None

    # Matches the credentials the local Compose database is brought up with.
    database_url: str = "postgresql://warrant:warrant@localhost:5432/warrant"

    # One process serves one console. A minimum above zero keeps the first request off the
    # connection-establishment path; the maximum is a ceiling nothing at this size approaches,
    # present so that a leak shows up as a pool timeout naming the pool rather than as Postgres
    # refusing connections to everything on the machine.
    db_pool_min_size: int = 1
    db_pool_max_size: int = 8

    # How long startup waits for the database to accept connections. Compose starts the
    # application once the database reports healthy, so this covers a restart racing recovery
    # rather than a cold initdb.
    db_connect_timeout_seconds: float = 30.0

    catalog_path: Path = REPO_ROOT / "data" / "catalog" / "NIST_SP-800-53_rev5_catalog.json"
    catalog_pin_path: Path = REPO_ROOT / "data" / "catalog" / "pinned.json"
    embedder_config_path: Path = REPO_ROOT / "data" / "embedder.json"

    # The record of the two above and of the code that reads them, checked as a set. Beside the
    # pins it covers rather than beside the recorded model calls it governs, because the things
    # that read it -- ingest, and anything validating before it serves or measures -- do not read a
    # recorded call.
    manifest_path: Path = REPO_ROOT / "data" / "manifest.json"

    # Where the embedding weights are cached. `None` means the machine's ordinary Hugging Face
    # cache, which is where a developer who has used these models before already has them; a
    # container sets this to a path inside the image, populated at build time. Either way the
    # weights are present before anything runs, because a first-run download is exactly what
    # breaks the promise that this works with no network.
    model_cache_dir: Path | None = None

    # Chunks embedded per forward pass. Trades peak memory against throughput and nothing else:
    # the model is deterministic per input, so this does not change a stored vector. Bounded
    # because zero and negatives are not slow or wasteful but meaningless, and they surface as an
    # exception out of `range` or `numpy` with nothing in it naming the variable that caused it.
    embed_batch_size: int = Field(default=32, gt=0)

    # How many chunks a question retrieves. Not a literal buried in a query, because it is the
    # number every retrieval measurement is relative to: a recall@5 baseline and a recall@10
    # measurement are not comparable, so moving this re-baselines rather than compares. It is
    # reported alongside every retrieval for that reason.
    #
    # Ten is a measured point rather than a round one. The embedding comparison swept k over
    # 1, 3, 5 and 10 on this corpus, and recall climbs from 0.679 at five to 0.774 at ten --
    # see docs/decisions/embedder.md, which says outright that five is not obviously right.
    # What the extra five chunks cost is prompt: roughly twice the retrieved text in every
    # request, which is counted locally and visible rather than silent.
    #
    # Bounded below for the reason the batch size is: zero retrieves nothing and negatives are
    # meaningless, and both surface as an empty result rather than as an error naming the
    # variable that caused it.
    retrieval_k: int = Field(default=10, gt=0)

    @property
    def mode(self) -> Mode:
        """Whether model calls go to the provider or come from recorded fixtures.

        Derived from the key rather than configured separately. A `WARRANT_MODE` variable
        alongside a key would allow the two to disagree, and there is no useful meaning for
        "live mode with no credentials" — it is just a crash deferred to the first request.
        """
        return "live" if self.model_api_key else "replay"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, read from the environment once."""
    return Settings()
