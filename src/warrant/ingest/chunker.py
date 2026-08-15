"""Turn the catalog into the retrievable records the corpus is made of.

One chunk per live control and per live enhancement, following the catalog's own structure. That
is the whole design, and it is a decision rather than a default: chunk by character window and
the control identifier has to be recovered afterwards by heuristic, which is precisely what makes
a citation check unfalsifiable. Chunking on the document's own boundaries means `control_id` is
copied from the source rather than inferred from the text, so a chunk cannot be about a control
it is not labelled with.

This is the rule the embedding comparison was run under. `tools/bakeoff/prepare_corpus.py` is
frozen as the record of what that measured, and it is standard-library-only; this module is a
re-implementation of the same rule against the typed objects, so the evidence stays fixed while
the shipped chunker is free to change. The two agree byte-for-byte across all 1,014 documents,
which is asserted rather than assumed — see the tests.

Three decisions the shape of the corpus rests on:

- **Assessment objectives and methods are excluded.** Every control carries an
  `assessment-objective` part and, on average, three `assessment-method` parts. They restate the
  statement in assessment voice, doubling the corpus text while adding little vocabulary a
  question would match on.
- **Withdrawn controls are excluded**, 182 of the 1,196. Nearly all are empty shells pointing at
  the control that absorbed them, so embedding them adds documents that cannot be a right answer
  to anything but can still be returned instead of one. They still *resolve* — `ControlIndex`
  keeps them deliberately — so a citation naming one can be reported as stale rather than as
  invented.
- **Long controls are not split.** See `docs/decisions/chunking.md`; the assembly order below is
  what makes that safe, and it is load-bearing rather than cosmetic.

`CHUNKER_VERSION` is what declares a change to any of that. Chunk text determines the vectors,
the vectors determine which control ids come back, and the retrieved text goes into the prompt
that recorded model calls are keyed on — so a chunker change invalidates the corpus and every
fixture downstream of it. `chunker_fingerprint` is what stops the declaration going stale
quietly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, fields

from warrant.ingest.catalog import Catalog, CatalogError, Control, Part
from warrant.ingest.control_ids import ControlIndex
from warrant.ingest.parameters import ParameterResolver

# Bumped deliberately when the chunks this module produces change in any way — their text, their
# identifiers, or which controls get one. Everything downstream is a function of that.
CHUNKER_VERSION = "1"

# The parts that go into a chunk, the headings they are rendered under, and the order they are
# assembled in. The order is this mapping's rather than the document's: the pinned release
# publishes the statement before the guidance, and the no-split rule needs that to be true of
# every chunk rather than of every release.
#
# "Control" and "Discussion" are the headings the publication itself uses, so a chunk rendered
# back to a reviewer reads the way the document does.
INCLUDED_PARTS = {"statement": "Control", "guidance": "Discussion"}

# Part ids are the control id, an underscore, and the part's own path — `ac-2_smt`, `ac-2_gdn`.
# The suffix is what `part_path` is built from, so the value in a row composes back into an
# identifier that exists in the catalog.
_PART_ID_SEPARATOR = "_"

# Joins the part suffixes of a chunk covering more than one. Not a character any part id
# contains, so the composition stays reversible.
_PART_PATH_SEPARATOR = "+"

# Separates a chunk id from the part path inside it. Deliberately not `_`, which would make a
# chunk id indistinguishable from one of the catalog's own part ids.
_CHUNK_ID_SEPARATOR = ":"

# Written as an escape rather than as itself. A literal non-breaking space in source is invisible
# to a reader and to a diff, and this is a rule the pinned release does not exercise, so an editor
# or a formatter quietly turning it into an ordinary space would break it with nothing to notice.
_NON_BREAKING_SPACE = "\u00a0"


class ChunkingError(CatalogError):
    """A control cannot be turned into a chunk.

    A subclass of `CatalogError` for the reason `ParameterResolutionError` is: it means the
    document is not the one this code knows how to read, and callers already refusing to start
    on a malformed catalog should refuse on this without learning a second exception.
    """


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable piece of the catalog, before it is embedded.

    The fields are the `chunks` columns the chunker owns, in the order the table declares them.
    What is missing is what belongs to somebody else: the embedding and `chunker_version` are
    the ingest pipeline's to attach, and `created_at` is the database's.
    """

    # Derived from the catalog's own identifiers, so re-running ingest over an unchanged catalog
    # collides with the rows already present rather than doubling the corpus.
    chunk_id: str
    # Canonical form. For an enhancement this is the enhancement's own id, not its parent's.
    control_id: str
    base_control_id: str
    # The form a reader sees. Rendered in citations, never matched on.
    control_label: str
    title: str
    # Which parts of the control this text came from, as the catalog's own part-id suffixes.
    part_path: str
    text: str


def chunk_catalog(catalog: Catalog) -> tuple[Chunk, ...]:
    """Every chunk the catalog produces, in document order.

    Deterministic: the same catalog produces byte-identical chunks in the same order, because
    the order comes from the document and nothing here iterates a set or a dictionary.
    """
    index = ControlIndex.for_catalog(catalog)
    resolver = ParameterResolver.for_catalog(catalog)

    return tuple(
        _chunk_control(control, index, resolver)
        for control in catalog.iter_controls()
        # Withdrawn controls are in the document and out of the corpus. `ControlIndex` still
        # resolves them, which is what lets M0's citation check tell a stale citation from an
        # invented one.
        if not control.withdrawn
    )


def _chunk_control(control: Control, index: ControlIndex, resolver: ParameterResolver) -> Chunk:
    """Assemble one control into its chunk."""
    # By id rather than by any published form: this is a caller already holding a canonical
    # identifier, and accepting a label here would hide something un-normalised reaching the
    # corpus. A miss means the index and the catalog disagree about what exists.
    identity = index.get(control.id)

    if identity is None:
        raise ChunkingError(
            f"Control {control.id!r} is in the catalog but not in the index built from it. "
            "Chunks are keyed on the canonical id, so a control that cannot be resolved cannot "
            "be stored under an identifier anything downstream could join on."
        )

    sections = [(part, _prose(part, resolver)) for part in _included_parts(control)]
    sections = [(part, lines) for part, lines in sections if lines]

    if not sections:
        raise ChunkingError(
            f"Control {control.id!r} is not withdrawn but has no statement or guidance prose, so "
            "it would be embedded as its title alone. The pinned release gives every one of its "
            "1,014 live controls both parts; a control with neither is a document this chunker "
            "was not written against."
        )

    text = _normalise_whitespace(
        "\n\n".join(
            [f"{identity.label} {identity.title}"]
            + [f"{INCLUDED_PARTS[part.name]}:\n" + "\n".join(lines) for part, lines in sections]
        )
    )

    part_path = _PART_PATH_SEPARATOR.join(_part_suffix(part) for part, _ in sections)

    return Chunk(
        chunk_id=f"{identity.id}{_CHUNK_ID_SEPARATOR}{part_path}",
        control_id=identity.id,
        base_control_id=identity.base_id,
        # Through the same pass the text is, so the stored columns and the heading the chunk
        # opens with are the same string rather than two spellings of it. Five titles in this
        # release carry a trailing space, and a citation rendered from the column would carry it.
        control_label=_normalise_whitespace(identity.label),
        title=_normalise_whitespace(identity.title),
        part_path=part_path,
        text=text,
    )


def _included_parts(control: Control) -> list[Part]:
    """The control's top-level statement and guidance parts, in the order a chunk assembles them.

    Ordered by `INCLUDED_PARTS` rather than by the document. On the pinned release the two are
    the same list, but the no-split rule rests on the statement preceding the discussion in every
    chunk: a control published the other way round would be truncated at 512 tokens through its
    requirement instead of through the tail of its discussion, which is the one loss that rule
    exists to prevent. Imposing the order here makes the invariant structural rather than a
    property of the file. The sort is stable, so parts sharing a name keep their document order.

    Top-level only. `statement` and `guidance` are never nested inside one another, and walking
    the whole tree would pull in the assessment parts that hang beneath the same control.
    """
    order = list(INCLUDED_PARTS)

    return sorted(
        (part for part in control.parts if part.name in INCLUDED_PARTS),
        key=lambda part: order.index(part.name),
    )


def _prose(part: Part, resolver: ParameterResolver) -> list[str]:
    """Flatten a part subtree into resolved prose lines, each prefixed by its list label.

    The label — "a.", "1.", "(a)" — is kept because control text refers to its own clauses by
    lettering, and prose read without it loses the thing a reviewer is looking for.
    """
    lines: list[str] = []

    for descendant in part.walk():
        if not descendant.prose:
            continue

        prose = resolver.resolve(descendant.prose)
        lines.append(f"{descendant.label} {prose}" if descendant.label else prose)

    return lines


def _part_suffix(part: Part) -> str:
    """A part's own path within its control: `smt` from `ac-2_smt`.

    Falls back to the part's name, which is what the suffix abbreviates, so a release that stops
    giving these parts ids produces a readable `part_path` rather than an empty one.
    """
    if part.id:
        _, separator, path = part.id.partition(_PART_ID_SEPARATOR)
        if separator and path:
            return path

    return part.name


def _normalise_whitespace(text: str) -> str:
    """The document-wide whitespace pass, owned here rather than by the resolver.

    Parameter resolution deliberately tidies only the seam around its own substitutions, because
    a module named for parameter resolution quietly rewriting unrelated punctuation is the kind
    of thing discovered two milestones later as an unexplained diff. The sweep over a finished
    document is a chunker decision, it changes chunk text, and it therefore belongs under
    `CHUNKER_VERSION`.

    Two of these rules do work on the pinned release: 63 controls carry a space before
    punctuation that the source itself wrote — `[AC-25](#ac-25) .` — and 5 carry a trailing space
    on a line. The other three fire on nothing at all. They are kept for the reason the resolver
    keeps a marker pattern more tolerant than the file needs: a rule that costs nothing and
    covers a shape the next release may contain is cheaper than discovering the gap in a corpus.

    The frozen bake-off script collapses blank-line runs before stripping line ends rather than
    after, so it misses a blank line that carries a space. The pinned release contains none,
    which is why the two still agree byte-for-byte; a release containing one would make them
    differ, and the byte-identity test is where that would surface.
    """
    text = text.replace(_NON_BREAKING_SPACE, " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([;,.:])", r"\1", text)

    # Line ends are stripped before blank-line runs are collapsed, not after. A blank line in a
    # document that varies is as likely to carry a space as to be empty, and `\n{3,}` does not
    # match one that does — collapsing first would miss the shape the rule is kept for.
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunker_fingerprint(catalog: Catalog) -> str:
    """SHA-256 of every field of every chunk a catalog produces.

    `CHUNKER_VERSION` on its own is a comment, and a comment cannot notice that somebody changed
    how a chunk is assembled and did not bump it. This is the number that can: pin it beside the
    version, and an unannounced change fails with two digests rather than reaching the corpus and
    invalidating every vector and every recorded model call downstream of it.

    Covers the identifiers as well as the text, and the order they come in, so that a chunk
    silently changing which control it is labelled with is as visible as its prose changing.
    """
    digest = hashlib.sha256()
    names = [field.name for field in fields(Chunk)]

    for position, chunk in enumerate(chunk_catalog(catalog)):
        digest.update(str(position).encode())
        for name in names:
            digest.update(b"\x1f")
            digest.update(getattr(chunk, name).encode())
        digest.update(b"\x1e")

    return digest.hexdigest()
