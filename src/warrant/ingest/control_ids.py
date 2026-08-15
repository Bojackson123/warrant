"""Resolve every way of writing a control identifier to one canonical form.

The catalog publishes four identifiers for each control and two of them differ in a way that
reads as identical to a person. `AC-2` kept its conventional label when release 5.1.1 introduced
a zero-padded `ac-02` beside it, so anything keyed to one form that silently stops matching the
other is not a bug that shows up as an error — it shows up as a question set that quietly scores
zero, or a citation check that rejects a control the model got right.

    ac-2      OSCAL id             canonical: storage, matching, part-id prefixes, cross-links
    AC-2      label                citation: the form a reader sees
    AC-02     zero-padded label    alias
    ac-02     sort-id              alias

For an enhancement the same four read `ac-2.3`, `AC-2(3)`, `AC-02(03)`, `ac-02.03`.

The OSCAL id is canonical because the document already treats it as its primary key: part ids
are prefixed with it (`ac-2.3_smt.a`), prose cross-references link to it (`[AC-25](#ac-25)`),
and `chunks.control_id` was declared for it. The label is what gets rendered, and the two are
kept in separate fields on separate names so that mixing them is a thing somebody has to do on
purpose rather than a thing that happens.

Resolution is two layers, and the order between them is the design:

1. **The forms the catalog publishes**, registered as written and casefolded. Anchoring on the
   source rather than on a belief about how padding works means a release that renumbers
   differently still resolves its own identifiers correctly.
2. **A structural parse** for everything else — family, then numbers with their padding
   stripped, rebuilt as a canonical id. This is what accepts `AC-2 (3)`, `ac2` and the mixed
   `AC-2.3`, none of which any release publishes and all of which people type.

A parsed identifier is only ever a *string*. It becomes a control by dictionary lookup, so
`AC-99` parses cleanly and still misses. Nothing here returns a near match: resolving a citation
to a neighbouring control is the failure this module exists to prevent, and it would be worse
than returning nothing, because nothing is visible.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

from warrant.ingest.catalog import Catalog, CatalogError, Control, get_catalog

# One control identifier and nothing else: a two-letter family, a number, and optionally a
# second number in one of the two notations the catalog uses for it — `.3` as the id does, `(3)`
# as the label does. Anchored at both ends, so a string with prose around the identifier is not
# an identifier.
#
# Deliberately more permissive than the file about case, spacing and padding, because every form
# the pinned release publishes is matched by the published index before this is consulted and
# what remains for this pattern is precisely the shapes the release does not contain.
#
# It is not permissive about the two notations, which are alternatives rather than a menu of
# interchangeable characters: an opening parenthesis has to be closed. `AC-2(3` is a citation
# that got truncated somewhere, and accepting it would score a damaged identifier as a clean hit
# — the one outcome this module is written to make visible.
_CONTROL_ID = re.compile(
    r"""
    \A \s*
    ([A-Za-z]{2})               # family: ac, sc, pm
    \s* [-_ ]? \s*
    (\d{1,3})                   # the control number, padding and all
    (?:                         # the enhancement number, if there is one
        \s*
        (?:
            \( \s* (\d{1,3}) \s* \)     # the label's notation: (3), (03)
            |
            \.  \s* (\d{1,3})           # the id's notation: .3, .03
        )
    )?
    \s* \Z
    """,
    re.VERBOSE,
)

# Part ids are the control id, an underscore, and the part's own path: `ac-2.3_smt.a`. The
# separator is what makes the split unambiguous — no control id contains an underscore.
_PART_SEPARATOR = "_"


class ControlIdError(CatalogError):
    """An identifier cannot be resolved to a control in this catalog.

    A subclass of `CatalogError` for the reason `ParameterResolutionError` is: to ingest, an
    unresolvable identifier means the document is not the one this code knows how to read, and
    a caller already refusing to start on a malformed catalog should refuse on this too without
    learning a second exception. Callers asking a question rather than making an assertion —
    the citation check, which must handle "no" — use `resolve` and get `None`.
    """


@dataclass(frozen=True, slots=True)
class ControlIdentity:
    """Every identifier form for one control, with the roles named.

    Carries the title and the withdrawal flag as well, because the callers that resolve an
    identifier — citation checking, clause click-through — invariably want them next, and a
    lookup that hands back a bare string only moves the second lookup somewhere else.
    """

    # The canonical form. Store this, match on this, and put it in a column.
    id: str
    # The form a reader sees. Render this, and never match on it.
    label: str
    zero_padded_label: str
    sort_id: str
    # The control an enhancement belongs to; equal to `id` for a base control. Taken from the
    # document's own nesting rather than by splitting the id, so a release that stops deriving
    # one from the other does not quietly produce a wrong answer here.
    base_id: str
    group_id: str
    title: str
    is_enhancement: bool
    # Withdrawn controls resolve. They are excluded from the corpus, but they exist in the
    # document, and "cited a control that was withdrawn" is a different fact from "cited a
    # control that never existed" — one is a stale answer, the other is an invented one.
    withdrawn: bool


@dataclass(frozen=True, slots=True)
class ControlIndex:
    """Resolves identifiers against one catalog's controls.

    Built once and immutable. The two mappings are separate because they answer different
    questions: `by_form` is every string the catalog publishes, `by_id` is the canonical space a
    structural parse lands in. Read-only views, so a shared index cannot be edited by a caller.
    """

    by_form: Mapping[str, ControlIdentity]
    by_id: Mapping[str, ControlIdentity]

    @classmethod
    def for_catalog(cls, catalog: Catalog) -> ControlIndex:
        by_form: dict[str, ControlIdentity] = {}
        by_id: dict[str, ControlIdentity] = {}

        for group in catalog.groups:
            for control in group.controls:
                _register(control, group.id, base_id=control.id, by_form=by_form, by_id=by_id)

        if not by_id:
            raise ControlIdError("The catalog contains no controls, so there is nothing to index.")

        return cls(by_form=MappingProxyType(by_form), by_id=MappingProxyType(by_id))

    def resolve(self, text: str) -> ControlIdentity | None:
        """The control an identifier names, or `None` if this catalog has no such control.

        Published forms first, structural parse second, miss third. A miss is a miss: an
        identifier that parses but names nothing — `AC-99` — returns `None` rather than the
        control it most resembles.
        """
        if not text:
            return None

        published = self.by_form.get(text.strip().casefold())
        if published is not None:
            return published

        canonical = canonical_id(text)
        if canonical is None:
            return None

        return self.by_id.get(canonical)

    def require(self, text: str) -> ControlIdentity:
        """The control an identifier names, refusing rather than returning a miss.

        For ingest, where an unresolvable identifier means the catalog is not the document this
        code was written against. The message quotes the input, because the useful question on
        seeing this is always "what did it actually get".
        """
        identity = self.resolve(text)

        if identity is None:
            raise ControlIdError(
                f"{text!r} does not name a control in this catalog. Identifiers are accepted in "
                "any of the forms the catalog publishes — `ac-2`, `AC-2`, `AC-02`, `ac-02`, and "
                "the enhancement notations `ac-2.3`, `AC-2(3)`, `AC-02(03)`, `ac-02.03` — so a "
                "miss here means the control is absent rather than written differently. Nothing "
                "is resolved to a near match."
            )

        return identity

    def get(self, canonical: str) -> ControlIdentity | None:
        """Look up by canonical id alone, without normalising anything.

        For callers already holding a stored `control_id`, where accepting a label would hide
        the fact that something un-normalised had reached the database.
        """
        return self.by_id.get(canonical)

    def __contains__(self, item: object) -> bool:
        """Accepts an identifier in any form, or an identity this index produced.

        Both, because `__iter__` yields identities and `identity in index` is the obvious thing
        to write next. Answering that with a bare `False` would be a quiet wrong answer of
        exactly the kind the rest of this module refuses to give.
        """
        if isinstance(item, ControlIdentity):
            return self.by_id.get(item.id) is item

        return isinstance(item, str) and self.resolve(item) is not None

    def __len__(self) -> int:
        """The number of controls, not the number of forms they are known by."""
        return len(self.by_id)

    def __iter__(self) -> Iterator[ControlIdentity]:
        return iter(self.by_id.values())


def _register(
    control: Control,
    group_id: str,
    base_id: str,
    by_form: dict[str, ControlIdentity],
    by_id: dict[str, ControlIdentity],
) -> None:
    """Index one control and, recursively, its enhancements.

    Recursive rather than a pass over `Catalog.iter_controls`, which flattens the tree and so
    loses the parentage `base_id` is supposed to come from.
    """
    if not (control.label and control.zero_padded_label and control.sort_id):
        raise ControlIdError(
            f"Control {control.id!r} is missing one of its published identifier forms "
            f"(label {control.label!r}, zero-padded {control.zero_padded_label!r}, sort-id "
            f"{control.sort_id!r}). A control that cannot be rendered in a citation cannot be "
            "cited, so the index refuses rather than inventing the missing form from the id."
        )

    identity = ControlIdentity(
        id=control.id,
        label=control.label,
        zero_padded_label=control.zero_padded_label,
        sort_id=control.sort_id,
        base_id=base_id,
        group_id=group_id,
        title=control.title,
        is_enhancement=control.is_enhancement,
        withdrawn=control.withdrawn,
    )

    if identity.id in by_id:
        raise ControlIdError(
            f"Control id {control.id!r} appears more than once. Ids are the canonical form "
            "everything downstream stores and joins on, so a duplicate means two controls "
            "sharing one row's worth of identity."
        )

    by_id[identity.id] = identity

    for form in (identity.id, identity.label, identity.zero_padded_label, identity.sort_id):
        key = form.casefold()
        existing = by_form.get(key)

        # No collision exists in the pinned release — the four forms of all 1,196 controls
        # casefold to 3,961 distinct keys — and one would mean two controls answering to the
        # same string, which makes every lookup of it a coin toss.
        if existing is not None and existing.id != identity.id:
            raise ControlIdError(
                f"Controls {existing.id!r} and {identity.id!r} are both published as {form!r}. "
                "An identifier that names two controls cannot be resolved to either."
            )

        by_form[key] = identity

    for enhancement in control.enhancements:
        _register(enhancement, group_id, control.id, by_form, by_id)


def canonical_id(text: str) -> str | None:
    """The canonical form of an identifier, or `None` if the string is not one.

    Syntax only — this knows nothing about which controls exist, so `zz-1` normalises happily
    and `AC` does not normalise at all. Whether the result names anything is `ControlIndex`'s
    question, and keeping the two apart is what makes "parses but does not exist" a miss rather
    than a near match.

    Zero padding is stripped by reading each component as a number, which is why `ac-02.03`,
    `AC-2(3)` and `AC-02(3)` all arrive at `ac-2.3`.
    """
    match = _CONTROL_ID.match(text)
    if match is None:
        return None

    # The two enhancement notations are separate groups so that the parentheses can be required
    # to balance; at most one of them ever matches.
    family, control, parenthesised, dotted = match.groups()
    enhancement = parenthesised or dotted
    canonical = f"{family.lower()}-{int(control)}"

    return canonical if enhancement is None else f"{canonical}.{int(enhancement)}"


def control_id_of_part(part_id: str) -> str | None:
    """The canonical id of the control a part belongs to, or `None` if this is not a part id.

    Part ids are the control id with a suffix — `ac-2.3_smt.a`, `ac-2_obj` — and they use the
    non-padded form, so the split is the whole of the work. Separate from `resolve` because a
    part id is not a control id: something that asks for `ac-2_smt.a` and is handed `ac-2` has
    silently been given the whole control in place of one clause of it.
    """
    control, separator, path = part_id.partition(_PART_SEPARATOR)

    # A separator with nothing after it is a truncated part id, not a control id that happens to
    # end in an underscore. Refusing it is the same rule as refusing `ac-2_smt.a` above, applied
    # in the other direction: neither a part nor a fragment of one is silently a control.
    if not separator or not path:
        return None

    return canonical_id(control)


@lru_cache(maxsize=1)
def get_control_index() -> ControlIndex:
    """Build the index over the pinned catalog once per process."""
    return ControlIndex.for_catalog(get_catalog())
