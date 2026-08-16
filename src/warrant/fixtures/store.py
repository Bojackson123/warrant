"""What a recorded model call holds, and the narrow interface the client reads it through.

Deliberately only an interface and an in-memory implementation. Where fixtures live on disk, how
a re-record arrives as a readable diff, and how vectors are kept out of the text files are all
storage questions, and answering them here would mean the client depended on a layout before
there was one. The client depends on `FixtureStore`; a directory-backed implementation slots in
underneath it without the client changing.

**A store answers one question and does not answer it approximately.** `get` takes a key and
returns the fixture recorded under exactly that key, or nothing. There is no nearest match, no
similarity threshold, no fallback to a related question -- and the absence is the point rather
than an unfinished feature. Serving the nearest recorded answer to a question nobody recorded is
precisely the failure this project exists to detect: an answer whose warrant does not cover what
was asked, rendered indistinguishably from one that does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Usage:
    """What the provider billed for the recorded call.

    Recorded rather than recomputed, because it is the provider's own count of the request that
    was actually sent. A locally computed number is an estimate of the same thing and belongs
    beside the prompt that produced it, not here.
    """

    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class Fixture:
    """One recorded model call: the answer, and enough provenance to distrust it later.

    The provenance fields duplicate what is already inside the key, and that is their whole
    purpose. The key proves a fixture matches a request; it cannot be read. These say, in a diff a
    person opens, which model produced this and when -- which is what makes a re-record reviewable
    and what makes an eight-month-old recording visibly eight months old.
    """

    key: str
    answer: str

    model: str
    model_version: str

    recorded_on: date
    usage: Usage


class FixtureStore(Protocol):
    """Somewhere recorded calls can be looked up by key."""

    def get(self, key: str) -> Fixture | None:
        """The fixture recorded under exactly this key, or `None`.

        `None` rather than an exception, because a miss is not an error at this layer -- what a
        miss costs depends entirely on who asked, and that judgement belongs to the client and its
        callers rather than to storage.
        """
        ...


@dataclass(frozen=True, slots=True)
class MappingFixtureStore:
    """A store over an in-memory mapping.

    What a test uses to present a known set of recordings without touching a filesystem, and what
    keeps this module honest about the interface: if the client can be driven by a dictionary,
    nothing about the on-disk layout has leaked into it.
    """

    fixtures: Mapping[str, Fixture]

    def get(self, key: str) -> Fixture | None:
        return self.fixtures.get(key)
