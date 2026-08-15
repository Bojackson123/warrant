"""The pinned embedding model, read from `data/embedder.json`.

That file is the single source of truth for the model, its revision and its dimensionality.
Changing anything in it invalidates the stored corpus vectors and every recorded generation
downstream of them, so the values are read from one place and never restated in code.

Kept out of the database package on purpose: the ingest pipeline and the retrieval path need
the same object, and none of them should have to import a storage module to learn which model
was pinned.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from warrant.settings import get_settings


class EmbedderConfig(BaseModel):
    """The pinned model's identity and the properties that have to match it everywhere."""

    # Frozen because this describes a pin. Code that wants a different model changes the file
    # and re-ingests; nothing should be able to change it in memory and carry on.
    model_config = ConfigDict(frozen=True, extra="ignore", protected_namespaces=())

    name: str
    revision: str
    dimensions: int
    max_sequence_length: int
    normalize: bool

    # Asymmetric models want the query and the document embedded differently. Defaulted to
    # empty so a symmetric model can simply omit them.
    query_prefix: str = ""
    document_prefix: str = ""

    expected_chunks: int | None = Field(default=None, exclude=True)


@lru_cache(maxsize=1)
def get_embedder_config() -> EmbedderConfig:
    """Read the pinned model configuration once per process."""
    return load_embedder_config(get_settings().embedder_config_path)


def load_embedder_config(path: Path) -> EmbedderConfig:
    """Read a pinned model configuration from a specific file."""
    document = json.loads(path.read_text(encoding="utf-8"))

    # The file carries `_comment` keys explaining what a change to it costs. They are for the
    # reader, and `extra="ignore"` drops them along with anything else added later.
    expected = document.get("expected_storage", {})

    return EmbedderConfig.model_validate(
        {**document, "expected_chunks": expected.get("chunks")},
    )
