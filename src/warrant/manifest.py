"""The pinned inputs as a set, and the check that refuses to continue when one has moved.

Everything this project stores is a function of a handful of inputs: the vendored catalog, the
parameter resolver, the chunker, the embedding model, and the prompt and tokenizer that turn
retrieved text into a request. Each of those already has a version or a pin of its own. What none
of them has on its own is a committed record that *this* build's inputs are the ones the stored
corpus and the recorded model calls were produced from.

That is the gap this closes, and it is the specific failure it exists for: change how a chunk is
assembled, forget to bump the version, forget to re-embed, and every test passes over a corpus and
a set of recorded answers that no longer describe the code that produced them. Nothing else
notices, because nothing else compares the inputs as a set.

**Each entry records a number that lives nowhere else.** The catalog's own SHA-256 is in
`data/catalog/pinned.json` and the model's revision is in `data/embedder.json`, and neither is
copied here — two records of one value is one record and one stale copy, and for the catalog the
stale copy would be the number people quote, because that hash is computable by anyone who
downloads the file. What an entry holds instead is a digest over that input's canonical identity,
derived at check time from whatever already owns it.

**Identity and digest are compared separately** because they fail differently. A moved identity is
an announced change that nobody re-recorded. A moved digest under an unchanged identity is the
case `chunker.py` and `parameters.py` both warn about in their own docstrings: somebody changed
what the module produces and did not say so. The second is the one a version string alone cannot
catch, and it is the reason this file exists rather than a table of version numbers.

**Callers, and the boundary between them.** `python -m warrant.ingest` checks before it loads the
embedder, because that is the command that would otherwise spend ten minutes building a corpus out
of inputs nobody verified. An application serving requests checks once at startup, for the same
reason and at the same cost. Anything measuring quality checks before it measures anything at all,
and treats a mismatch as a hard error rather than as a result — a measurement taken against inputs
that do not match the recorded ones is not a worse measurement, it is not a measurement.

What is deliberately *not* a caller is the per-question path. This re-reads a ten-megabyte file and
re-chunks it; `retrieval.corpus_check` covers that path against the pins already in memory, which
is cheap enough to run before every search.

Regeneration is a separate command and is never reached from the check. A check that repairs what
it finds is a record of whatever happens to be on disk, which is not a record at all.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from warrant.catalog_pin import CatalogPin, CatalogPinError, get_catalog_pin, verify_catalog
from warrant.embedder_config import EmbedderConfig, get_embedder_config
from warrant.ingest.catalog import load_catalog
from warrant.ingest.chunker import CHUNKER_VERSION, chunker_fingerprint
from warrant.ingest.parameters import RESOLUTION_VERSION, resolution_fingerprint
from warrant.model_config import ModelConfig, get_model_config
from warrant.prompt import PROMPT_TEMPLATE_VERSION, prompt_fingerprint
from warrant.settings import get_settings
from warrant.tokenizer import tokenizer_fingerprint

# Every input this build can account for. The file lists the same names, and the two sets are
# checked against each other rather than assumed equal: a name in the file that no code computes
# is a record of nothing, and a name computed here that the file omits is an input going unchecked.
#
# The three at the end are not built yet. They are named here, with their cost recorded in the
# file, so that adding one is a regeneration of an existing entry rather than the introduction of a
# new mechanism — and so that the cost of changing a prompt is written down before there is a
# prompt to change.
ENTRY_NAMES: tuple[str, ...] = (
    "catalog",
    "parameter_resolution",
    "chunker",
    "embedder",
    "prompt_template",
    "tokenizer",
    "judge_prompt",
)

# What each kind of invalidation costs, as the clause it contributes to a failure message. The
# vocabulary is small on purpose: an entry declares which of these its change destroys, and the
# sentence is assembled from that rather than written out per entry, so no entry can be given a
# reassuring description of a change that is not reassuring. An entry naming a kind that is not
# here is refused rather than ignored, for the same reason.
#
# Written costliest first, and that order is read rather than decorative: it is what ranks two
# entries against each other when both have moved.
_COSTS = {
    "corpus": "the stored corpus vectors",
    "queries": "the recorded query vectors",
    "generation": "every recorded model answer",
    "judge": "every recorded grade",
    "counts": "every locally counted prompt size",
}

_COST_RANK = {kind: rank for rank, kind in enumerate(_COSTS)}

# Written between the pieces of a digest, so that two fields concatenating into the same bytes
# cannot produce the same number. The same separator the chunker and the resolver use for the same
# reason.
_RECORD_SEPARATOR = b"\x1e"


class ManifestError(Exception):
    """The manifest cannot be used as a record of what this build was made from."""


class ManifestMismatchError(ManifestError):
    """An input is not the one the manifest records.

    Hard-fails its caller rather than warning. Every difference this can report means that
    something stored — vectors, recorded answers, recorded grades — was produced by inputs this
    build no longer carries, and continuing produces results that look exactly like results
    produced correctly.
    """


class ManifestEntry(BaseModel):
    """One pinned input: what it is, what it currently hashes to, and what changing it destroys."""

    # Frozen, and `extra="ignore"` so the file's `_comment` keys reach a reader and not the model.
    # The same shape as the catalog and embedder pins, for the same reason: this describes a pin.
    model_config = ConfigDict(frozen=True, extra="ignore")

    # Prose, authored rather than generated. Regeneration rewrites the two values below and leaves
    # this alone, because what an input covers is a judgement and a digest is not.
    covers: str

    # Both null together, and null means this build does not have this input at all. An entry that
    # gains a computed value while the file still records null is a mismatch like any other, which
    # is how a newly built prompt template announces itself instead of arriving silently.
    identity: str | None = None
    digest: str | None = None

    invalidates: tuple[str, ...] = ()

    @field_validator("invalidates")
    @classmethod
    def _costs_are_known(cls, kinds: tuple[str, ...]) -> tuple[str, ...]:
        """Refuse a kind of invalidation nothing knows the cost of.

        Without this the failure is silent and reassuring in exactly the wrong direction: `cost()`
        assembles its sentence from the kinds it recognises, so a misspelt `corpus` produces the
        sentence for an entry that invalidates nothing at all. Adding an input is a hand edit of
        the file, which makes a typo the ordinary failure rather than an exotic one.
        """
        unknown = sorted(kind for kind in kinds if kind not in _COSTS)

        if unknown:
            raise ValueError(
                f"{unknown} names nothing this build knows the cost of; the kinds it knows are "
                f"{sorted(_COSTS)}. An unrecognised kind would be left out of the failure message, "
                "which would describe a change that invalidates the corpus as invalidating nothing."
            )

        return kinds

    @property
    def built(self) -> bool:
        """Whether this input exists in the build that recorded the manifest."""
        return self.identity is not None or self.digest is not None

    def cost(self) -> str:
        """What a change to this input destroys, as a sentence.

        Assembled from `invalidates` rather than stored as prose, so that the file cannot carry a
        description of a cost that disagrees with the cost it declares. Every kind is known to
        `_COSTS`, because an entry naming one that is not was refused at load.
        """
        named = [_COSTS[kind] for kind in self.invalidates]

        if not named:
            return "Nothing recorded depends on this yet."

        listed = named[0] if len(named) == 1 else f"{', '.join(named[:-1])} and {named[-1]}"

        if "corpus" in self.invalidates:
            return (
                f"That invalidates {listed}. There is no partial re-record: the vectors decide "
                "which control ids come back, and those ids and their text sit inside the prompt "
                "every recorded call is keyed on, so everything downstream has to be rebuilt."
            )

        return f"That invalidates {listed}, which have to be re-recorded."


class Manifest(BaseModel):
    """Every pinned input this build is checked against."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    entries: dict[str, ManifestEntry]


@dataclass(frozen=True, slots=True)
class Observation:
    """What an input actually is right now, as against what the manifest says it was."""

    identity: str | None
    digest: str | None


@dataclass(frozen=True, slots=True)
class Comparison:
    """One entry, checked. Returned on success so a caller can print what it looked at."""

    name: str
    entry: ManifestEntry
    observed: Observation

    @property
    def identity_moved(self) -> bool:
        return self.entry.identity != self.observed.identity

    @property
    def digest_moved(self) -> bool:
        return self.entry.digest != self.observed.digest

    @property
    def matches(self) -> bool:
        return not (self.identity_moved or self.digest_moved)


@lru_cache(maxsize=1)
def get_manifest() -> Manifest:
    """Read the manifest once per process."""
    return load_manifest(get_settings().manifest_path)


def load_manifest(path: Path) -> Manifest:
    """Read a manifest from a specific file.

    Every way the file itself can be wrong leaves here as `ManifestError`, including the two that
    `json` and pydantic raise on their own. This file is edited by hand — adding an input is a hand
    edit and then `make manifest-write` — so a trailing comma or an entry missing its prose is the
    ordinary failure, and what should meet it is one line naming the file rather than a traceback
    through a parser. A failure to read the file at all stays an `OSError`, which already names it.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        manifest = Manifest.model_validate(document)
    except json.JSONDecodeError as error:
        raise ManifestError(f"{path} is not readable as JSON: {error}") from error
    except ValidationError as error:
        raise ManifestError(f"{path} is not usable as a manifest: {error}") from error

    recorded = set(manifest.entries)
    known = set(ENTRY_NAMES)

    if recorded != known:
        raise ManifestError(
            f"{path} does not list the inputs this build knows how to check. "
            f"It omits {sorted(known - recorded) or 'nothing'} and adds "
            f"{sorted(recorded - known) or 'nothing'}. An input the manifest omits goes "
            "unchecked, and one it names that nothing computes is a record of nothing; both are "
            "fixed by hand, because what an entry covers and what it invalidates are judgements "
            "rather than values."
        )

    return manifest


def observe(
    catalog_path: Path | None = None,
    pin: CatalogPin | None = None,
    config: EmbedderConfig | None = None,
    model: ModelConfig | None = None,
) -> dict[str, Observation]:
    """Compute what every pinned input currently is.

    The inputs are parameters rather than globals for the reason `verify_corpus` takes an
    overridable config: this is how a test hands the check a deliberately mismatched input without
    rewriting the files the whole project reads.

    Reads and parses the catalog once, then fingerprints it twice. Both fingerprints walk the whole
    document, which is the cost of the guarantee — a version string is a claim and these are the
    numbers that check it.
    """
    settings = get_settings()
    catalog_path = catalog_path if catalog_path is not None else settings.catalog_path
    pin = pin if pin is not None else get_catalog_pin()
    config = config if config is not None else get_embedder_config()
    model = model if model is not None else get_model_config()

    try:
        verify_catalog(catalog_path, pin)
    except CatalogPinError as error:
        # Re-raised rather than left to escape, so that every failure out of this module is one
        # exception type naming one entry. The original message is better than anything written
        # here would be — it names both hashes — so it is carried rather than replaced.
        raise ManifestMismatchError(f"The `catalog` entry does not match: {error}") from error

    catalog = load_catalog(catalog_path)

    return {
        "catalog": Observation(pin.release, _digest(_canonical(pin.model_dump()))),
        "parameter_resolution": Observation(
            f"resolution {RESOLUTION_VERSION}",
            _digest(RESOLUTION_VERSION, resolution_fingerprint(catalog)),
        ),
        "chunker": Observation(
            f"chunker {CHUNKER_VERSION}",
            _digest(CHUNKER_VERSION, chunker_fingerprint(catalog)),
        ),
        "embedder": Observation(
            f"{config.name} @ {config.revision[:8]}",
            _digest(_canonical(config.model_dump())),
        ),
        "prompt_template": Observation(
            f"template {PROMPT_TEMPLATE_VERSION}",
            _digest(PROMPT_TEMPLATE_VERSION, prompt_fingerprint()),
        ),
        # The encoding alone, not the model it is pinned beside. The model's identity is hashed
        # into every recorded call's key, so restating it here would be a second announcement of a
        # change that already announces itself — and would fire this entry on a model swap that
        # left the counting untouched.
        "tokenizer": Observation(
            model.tokenizer.encoding,
            _digest(model.tokenizer.encoding, tokenizer_fingerprint(model)),
        ),
        # Not built. It becomes a real computation at the point the thing it names exists, and the
        # manifest is regenerated in the same change — which is the announcement.
        "judge_prompt": Observation(None, None),
    }


def compare(manifest: Manifest, observed: Mapping[str, Observation]) -> tuple[Comparison, ...]:
    """Pair every recorded entry with what was observed, costliest difference first.

    Ordered by what a change destroys rather than by the order the file happens to list them in, so
    that when several inputs have moved the one that invalidates everything is the one reported. A
    file reordered by hand cannot quietly change which failure a reader sees first: entries that
    destroy the same things are ordered by name, and a name is not something the file's layout
    decides.

    An entry the caller observed nothing for is refused rather than skipped, which is the same
    requirement `load_manifest` puts on the file from the other side. Between them, a name is
    either checked or it is a loud failure; what it cannot be is recorded and quietly unexamined.
    """
    unobserved = sorted(name for name in manifest.entries if name not in observed)

    if unobserved:
        raise ManifestError(
            f"Nothing computed a current value for {unobserved}, so the manifest records an input "
            "that no check looks at. An entry is added by hand in two places — the file, and the "
            "code that observes it — and this is the half that was left out."
        )

    comparisons = [
        Comparison(name, manifest.entries[name], observed[name]) for name in manifest.entries
    ]

    return tuple(sorted(comparisons, key=lambda item: (_severity(item.entry), item.name)))


def check(manifest: Manifest, observed: Mapping[str, Observation]) -> tuple[Comparison, ...]:
    """Refuse the first difference; return the comparisons if there is none.

    Split from `verify_manifest` so that the comparison can be driven with observations assembled
    by hand — a test needs to present an input that no code currently produces, and recomputing is
    exactly what stops it doing that.
    """
    comparisons = compare(manifest, observed)

    for comparison in comparisons:
        if not comparison.matches:
            raise ManifestMismatchError(_mismatch(comparison))

    return comparisons


def verify_manifest(
    manifest: Manifest | None = None,
    catalog_path: Path | None = None,
    pin: CatalogPin | None = None,
    config: EmbedderConfig | None = None,
    model: ModelConfig | None = None,
) -> tuple[Comparison, ...]:
    """Recompute every pinned input and refuse to continue if one has moved.

    Returns the comparisons on success, so the command that runs this can print what it checked.
    **Never writes.** Repairing a mismatch is `write_manifest`, reached only by asking for it.
    """
    manifest = manifest if manifest is not None else get_manifest()

    return check(manifest, observe(catalog_path, pin, config, model))


def write_manifest(path: Path, observed: Mapping[str, Observation]) -> None:
    """Rewrite the recorded values in place, leaving everything a person wrote alone.

    Surgical rather than regenerating the document: `covers`, `invalidates`, the comment blocks and
    the order of the entries are all authored, and a command that re-emitted them would either
    lose them or need to hold a second copy of them in code. What this rewrites is two values per
    entry, which is what makes an intended change a diff somebody can read in the pull request that
    explains it.

    Adding an input is therefore a hand edit followed by this command. That is deliberate: what a
    new entry covers and what changing it costs are judgements, and this can only supply numbers.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    entries = document.get("entries")

    if not isinstance(entries, dict):
        raise ManifestError(f"{path} has no `entries` object to rewrite.")

    for name, observation in observed.items():
        entry = entries.get(name)

        if not isinstance(entry, dict):
            raise ManifestError(
                f"{path} has no entry named `{name}`. Add it by hand, with what it covers and "
                "what changing it invalidates, and then run this again to fill in its digest — "
                "those two are judgements this command has no way to make."
            )

        entry["identity"] = observation.identity
        entry["digest"] = observation.digest

    # `newline` set explicitly, because the default on Windows translates every line ending and
    # would turn a one-line regeneration into a whole-file diff on half the machines that run it.
    # `.gitattributes` checks this file out with line feeds everywhere, so this writes them back.
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _severity(entry: ManifestEntry) -> tuple[int, int]:
    """What a change to this entry destroys, as a key that sorts the worst of it first.

    Ranked by the costliest single thing an entry invalidates before how many things it
    invalidates, because those two disagree: an entry that destroys only the corpus destroys
    everything downstream of the corpus, and outranks one naming two of the cheaper kinds. Counting
    the kinds would put it second.
    """
    ranks = sorted(_COST_RANK[kind] for kind in entry.invalidates)

    return (ranks[0] if ranks else len(_COST_RANK), -len(ranks))


def _mismatch(comparison: Comparison) -> str:
    """One difference, as a sentence naming the entry, both sides of it, and what it costs.

    A shared shape rather than one message per entry, for the reason `corpus_check._moved` is: the
    useful content is always the same three things, and the message missing one of them is the one
    somebody reads at the point they are already confused.
    """
    entry = comparison.entry
    observed = comparison.observed

    if not entry.built and observed.digest is not None:
        detail = (
            f"The manifest records it as not built, and this build produces "
            f"{observed.identity!r}. It exists now, so its cost applies now."
        )
    elif entry.built and observed.digest is None:
        detail = (
            f"The manifest records {entry.identity!r} and this build does not produce it at all. "
            "An input that has gone away invalidates what was made with it exactly as a changed "
            "one does."
        )
    elif comparison.identity_moved:
        detail = (
            f"The manifest records {entry.identity!r} and this build carries "
            f"{observed.identity!r}. The change was announced and the manifest was not "
            "regenerated, so nothing downstream of it has been rebuilt."
        )
    else:
        detail = (
            f"{entry.identity!r} is unchanged and what it produces is not: recorded digest "
            f"{entry.digest}, found {observed.digest}. This is the change a version string cannot "
            "catch, because the module's output moved without anybody announcing it."
        )

    return (
        f"The `{comparison.name}` entry does not match. {entry.covers} {detail} {entry.cost()} "
        "If the change was intended, `make manifest-write` records it, and that diff belongs in "
        "the pull request that explains why."
    )


def _canonical(document: Mapping[str, object]) -> str:
    """A mapping as one string that does not move when its keys are reordered."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), default=str)


def _digest(*parts: str) -> str:
    """SHA-256 over several strings, separated so that concatenation cannot collide."""
    digest = hashlib.sha256()

    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(_RECORD_SEPARATOR)

    return digest.hexdigest()
