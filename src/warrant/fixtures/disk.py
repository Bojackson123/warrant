"""Where a recorded model call lives on disk, in a form somebody can review.

`store.py` deferred this deliberately, so that the client depended on an interface rather than on
a layout. This is the layout, and every choice in it answers the same question: what does a
re-record look like in a pull request?

**One file per key, not one file holding every recording.** A monthly re-record then arrives as
changed prose inside named files, and a fixture that was added or dropped arrives as a file that
was added or dropped. Everything in one file makes both of those the same diff.

**The answer is stored as a list of lines, and this is the detail the whole thing turns on.** JSON
escapes a newline into a two-character sequence, so an answer of several paragraphs written as one
string is one enormous line in the file -- a diff nobody can read, satisfying the letter of "no
vectors inline" and defeating the point of it. `str.split("\\n")` and `"\\n".join(...)` are exact
inverses for every string, trailing newlines and embedded carriage returns included, so this costs
nothing in fidelity and buys a line-by-line diff. The round trip is asserted, in both directions.

**A file that exists and cannot be read raises; only a file that is absent is a miss.** The
`FixtureStore` contract says a miss is `None` because a miss is not an error at that layer -- but a
corrupt recording is not a miss. Degrading on it would render a broken file as "nobody recorded an
answer to this question", which is the one sentence this project must never say untruthfully.

**The filename is the key, and the file says so too.** They are checked against each other on
every read. A recording copied or renamed by hand is a file that would answer a request it was not
made for, which is precisely what keying on the whole request exists to prevent -- and the keying
rule cannot defend itself against a filesystem.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from warrant.fixtures.request import Purpose
from warrant.fixtures.store import Fixture, Usage

# What a recording is written with. Two spaces, sorted nowhere -- the field order below is the
# order it is read in, and `answer` is last so that the part which changes on a re-record is at the
# end of the diff rather than in the middle of the provenance.
_INDENT = 2

_SUFFIX = ".json"


class FixtureFileError(Exception):
    """A file in the fixture directory is not a recording this build can serve."""


class RecordedCall(BaseModel):
    """One recorded model call as it is written down.

    Wider than the `Fixture` a client is served, and the extra fields are documentation rather
    than mechanism. A key proves that a recording matches a request and cannot be read; these are
    what let somebody opening the diff see which question this was, which controls its prompt was
    built from, and whether the local token count agrees with what the provider billed. Nothing
    matches on them.
    """

    # `extra="forbid"`, so a field renamed in code does not silently leave stale data in every
    # committed file with nothing reading it.
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    purpose: Purpose

    # For the reader. A directory of hex filenames is unreviewable without them.
    question: str
    control_ids: tuple[str, ...]

    model: str
    model_version: str

    recorded_on: date

    # What the provider billed, and what this build counted locally for the same prompt. They do
    # not match exactly and are not meant to: the provider counts the message envelope around the
    # prompt and this counts the string. What the pair establishes is that the pinned encoding
    # tracks the bill, which is the assumption a token gate rests on.
    usage_prompt_tokens: int
    usage_completion_tokens: int
    counted_prompt_tokens: int

    # Last, and split on newlines. See the module docstring: this is the field the readable-diff
    # requirement is about.
    answer_lines: tuple[str, ...]

    @property
    def answer(self) -> str:
        """The recorded text, rebuilt exactly as it was recorded."""
        return "\n".join(self.answer_lines)

    def as_fixture(self) -> Fixture:
        """The narrower thing a client is served."""
        return Fixture(
            key=self.key,
            answer=self.answer,
            model=self.model,
            model_version=self.model_version,
            recorded_on=self.recorded_on,
            usage=Usage(
                prompt_tokens=self.usage_prompt_tokens,
                completion_tokens=self.usage_completion_tokens,
            ),
        )


def write_fixture(root: Path, call: RecordedCall) -> Path:
    """Write one recording under its own key; return where it went.

    Written with an explicit `\\n`, because a recording is compared byte for byte across machines
    and letting the platform decide its line endings would make a Windows re-record a whole-file
    diff of a file nobody changed.

    Written as UTF-8 rather than as escapes, for the same reason the answer is split into lines: a
    model quoting the catalog returns section marks and curly quotes, and `\\u00a7` in place of `§`
    is a readable diff given up for nothing -- the file is already declared UTF-8 on the way in.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{call.key}{_SUFFIX}"

    document = call.model_dump(mode="json")
    path.write_text(
        json.dumps(document, indent=_INDENT, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return path


def read_fixture(path: Path, purpose: Purpose | None = None) -> RecordedCall:
    """Read one recording, refusing anything that is not one.

    `purpose` is checked when given. A judge grade sitting in the generation directory is a file
    that would be served for a request it was not made for, and the directory it is in is not
    something the key can defend.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FixtureFileError(
            f"{path} is in the fixture directory and is not valid JSON: {error}. This is a broken "
            "recording rather than a missing one, and serving nothing for it would report that "
            "nobody recorded an answer to a question somebody did record."
        ) from error

    try:
        call = RecordedCall.model_validate(document)
    except ValidationError as error:
        raise FixtureFileError(f"{path} is not a recorded model call: {error}") from error

    if call.key != path.stem:
        raise FixtureFileError(
            f"{path} records the key {call.key}, so it is filed under a name that is not its own. "
            "A recording is looked up by filename, so this one would be served for a request it "
            "was not made for -- which is the substitution that keying on the whole request "
            "exists to prevent, arriving through the filesystem where the key cannot see it."
        )

    if purpose is not None and call.purpose != purpose:
        raise FixtureFileError(
            f"{path} records a {call.purpose} call and is stored where {purpose} calls are kept. "
            "Recordings are separated by what they are for so that one kind can never answer for "
            "another."
        )

    return call


@dataclass(frozen=True, slots=True)
class DirectoryFixtureStore:
    """Recorded calls in a directory, looked up by exact key.

    The on-disk counterpart to `MappingFixtureStore`, satisfying the same `FixtureStore` protocol,
    so the client is unchanged by the existence of a layout. No index, no cache: a lookup is a
    filename, which means a fixture added by a re-record is visible to a running process without
    anything having to be invalidated.
    """

    root: Path
    purpose: Purpose

    def get(self, key: str) -> Fixture | None:
        path = self.root / f"{key}{_SUFFIX}"

        if not path.is_file():
            return None

        return read_fixture(path, self.purpose).as_fixture()

    def keys(self) -> tuple[str, ...]:
        """Every key this directory holds, sorted.

        For the recorder, which needs to know what it would be duplicating and what it has left
        behind. Not part of the `FixtureStore` protocol, which is deliberately one question wide.
        """
        if not self.root.is_dir():
            return ()

        return tuple(sorted(path.stem for path in self.root.glob(f"*{_SUFFIX}")))
