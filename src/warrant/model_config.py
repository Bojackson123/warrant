"""The pinned generation model and its sampling parameters, read from `data/model.json`.

The sibling of `embedder_config`, and pinned for the same reason with one difference worth
naming. A moved embedding pin is caught by comparing it against what the corpus records; a moved
generation pin needs no such comparison, because the model's identity and every sampling
parameter are hashed into the key a recorded call is stored under. Changing either produces a
miss, which is a loud, local failure rather than a stale answer served confidently.

That is why this file has no entry in the manifest. The manifest exists for inputs whose change
would otherwise go unnoticed; this one cannot change unnoticed.

**Sampling lives here rather than in `Settings`.** A value that decides what a model returns is a
pin, not configuration: two machines reading different temperatures would key their fixtures
differently and neither would be wrong about it locally. Settings holds what may legitimately
differ between machines -- the key, the database, the cache directory -- and nothing that changes
an answer.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from warrant.settings import get_settings


class Sampling(BaseModel):
    """Every parameter this client sets on a request, with nothing left to a provider default.

    Exhaustive on purpose. A parameter the client does not send takes whatever value the provider
    currently defaults to, which puts something that decides the output outside the key and free
    to move when the provider moves it. Everything here is sent on every call, including values
    that happen to equal the current default.

    Adding a field changes every key, and therefore misses every recorded answer. That is correct:
    a request made with a new parameter is a different request.
    """

    # `extra="forbid"`, unlike the other pins in this project, and the difference is the whole
    # point of the class. A parameter written into the file that no field matches would otherwise
    # be accepted, dropped, never sent and never hashed -- leaving whoever added it believing it
    # was in effect and in the key when it was in neither. That is exactly the failure described
    # above, arriving through a hand edit rather than through a decision. The file's comment keys
    # are stripped in the loader instead, which is what the other pins use `extra="ignore"` for.
    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    seed: int

    # Bounded below for the reason `retrieval_k` is: zero is not a cheaper call, it is a call that
    # cannot return an answer, and it surfaces as an empty completion rather than as an error
    # naming the value that caused it.
    max_output_tokens: int = Field(gt=0)


class ModelConfig(BaseModel):
    """Which generation model this build talks to, and how."""

    # Frozen because this describes a pin. Code that wants a different model edits the file and
    # re-records; nothing should be able to change it in memory and carry on against fixtures that
    # were made with something else. Extras forbidden for the reason `Sampling` forbids them: this
    # file is hand-edited, and a key nothing reads is a value somebody believes is in effect.
    model_config = ConfigDict(frozen=True, extra="forbid")

    # The family, for a reader. Never sent.
    name: str

    # The dated snapshot, and the exact string sent to the provider. A floating alias would be
    # repointed at a new model without anything in this repository changing, which is the silent
    # substitution the pin exists to prevent.
    version: str

    sampling: Sampling


@lru_cache(maxsize=1)
def get_model_config() -> ModelConfig:
    """Read the pinned generation model once per process."""
    return load_model_config(get_settings().model_config_path)


def load_model_config(path: Path) -> ModelConfig:
    """Read a pinned generation model from a specific file.

    A parameter rather than a global for the reason `verify_corpus` takes an overridable config:
    it is how a test hands the key a deliberately different pin without editing the file the whole
    project reads.
    """
    document = json.loads(path.read_text(encoding="utf-8"))

    # The file carries `_comment` keys explaining what a change to it costs, at the top level and
    # inside `sampling`. They are for the reader and are removed by name here, rather than by
    # letting the models ignore everything they do not recognise -- because a sampling parameter
    # the models ignored would be dropped just as quietly, and that one decides the answer.
    document.pop("_comment", None)

    sampling = document.get("sampling")

    if isinstance(sampling, dict):
        sampling.pop("_comment", None)

    return ModelConfig.model_validate(document)
