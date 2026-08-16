"""One model call, and the number a recorded copy of it is filed under.

**The key is a SHA-256 over the complete request. Never a subset.** That single rule is what this
module exists to enforce, and it exists because the obvious alternative fails in the worst
possible direction. Key a recorded call on the question, or on the question and the model, and
editing the prompt template produces a *hit*: the recorded answer comes back, its recorded token
count comes back, every test passes, and none of it describes the request the code would now
send. A green build over a wrong answer is worse than a red build, because nobody investigates it.

Hashing everything turns that into a miss, which is loud, local, and fixed by re-recording.

**What "everything" means.** The purpose, the model and its pinned version, the fully rendered
prompt -- template, retrieved chunks and question, already assembled into the one string that
would go over the wire -- and every sampling parameter. The prompt arrives here as a string rather
than as pieces this module assembles, because there must be exactly one renderer; two code paths
producing "the prompt" would reintroduce the stale-hit bug by another route.

**Why there is no prompt-template version field.** The rendered prompt is already hashed, so a
version string beside it would be a second record of the same fact, free to disagree with it. The
template's version belongs in the manifest, where a person reads it; the key does not need to be
told what it can compute.

**Why `purpose` is a field.** The plan requires judge calls to be recorded too -- keyed on the
answer, the cited control ids and the judge prompt version -- so that an automated grade is not a
live API call inside a job advertised as costing nothing. All three of those sit inside a rendered
judge prompt, so hashing the whole request already covers them; what `purpose` adds is that a
generation fixture can never serve a judge call even if the two prompts somehow coincided. The
judge is therefore a value of an existing field rather than a second mechanism bolted on later.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from warrant.model_config import ModelConfig, Sampling, get_model_config

# What a call is for. Two values today; the point of the field is that adding a third is an
# addition to this list rather than a parallel store with its own keying rule.
Purpose = Literal["generation", "judge"]


class ModelRequest(BaseModel):
    """Everything a model call is made of, and nothing that is not part of it."""

    # Frozen so a request cannot be mutated after its key has been taken, and `extra="forbid"`
    # rather than the `extra="ignore"` the pins use: a pin file is hand-edited and may carry
    # comments, but a misspelt field here would be silently dropped and quietly left out of the
    # key, which is the exact failure the module exists to prevent.
    #
    # `protected_namespaces` cleared because `model_version` would otherwise collide with
    # pydantic's reserved `model_` prefix and warn on every import.
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    purpose: Purpose

    model: str
    model_version: str

    # The whole string that would be sent, rendered by the one renderer. Not the question, not the
    # chunks, not the template -- those are its inputs, and hashing inputs instead of the result
    # would miss any change in how they are assembled.
    prompt: str

    sampling: Sampling

    @classmethod
    def build(
        cls,
        purpose: Purpose,
        prompt: str,
        config: ModelConfig | None = None,
    ) -> ModelRequest:
        """A request against the pinned model, so no caller restates the pin.

        The config is a parameter for the reason `verify_corpus` takes one: it is how a test keys
        a request against a deliberately different model without editing the file the whole
        project reads.
        """
        pinned = config if config is not None else get_model_config()

        return cls(
            purpose=purpose,
            model=pinned.name,
            model_version=pinned.version,
            prompt=prompt,
            sampling=pinned.sampling,
        )

    @property
    def key(self) -> str:
        """The SHA-256 this request's recorded answer is filed under.

        Computed from the whole model rather than from named fields, so that a field added later
        is hashed without anybody remembering to add it here. The consequence is that adding a
        field misses every existing fixture -- which is correct, because a request carrying a new
        parameter is a different request, and serving the old answer for it is the stale hit this
        module exists to prevent.
        """
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    def canonical(self) -> str:
        """The request as one string that does not move when fields are reordered.

        Separate from `key` so a failure can show what was hashed. Debugging "why did this miss?"
        by staring at two hex digests is not debugging; diffing two of these is.
        """
        return json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
