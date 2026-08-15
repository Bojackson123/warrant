"""Read the OSCAL catalog into typed objects.

This is the bottom of the ingest stack and it does one thing: turn ten megabytes of nested
JSON into objects with names, failing loudly where the document is not shaped the way the rest
of the pipeline assumes. Nothing here resolves a parameter marker, normalises an identifier or
decides what belongs in a chunk — those are separate decisions with their own versions, and
folding them in here would mean every one of them was applied by anyone who merely wanted to
read the file.

What it does encode is the shape of the source, because the shape has three properties that
are easy to get wrong from a distance:

- Enhancements are controls nested inside controls, distinguished only by their class.
- Withdrawal is a property in a list, not a field, and 182 of the 1,196 records are withdrawn.
- Parameter identifiers use the zero-padded form while their owning control does not, so
  `ac-2` owns `ac-02_odp.01` and no parameter id can be derived from the control it sits in.

The parsing rules here were first worked out in the corpus preparation script under `tools/`,
which is deliberately standard-library-only and frozen as the record of what the embedding
comparison actually measured. This is a re-implementation of those rules rather than an import
of them, so that the evidence stays fixed while the shipped loader is free to change.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from warrant.catalog_pin import CatalogCounts
from warrant.settings import get_settings

# The class marking a control as an enhancement rather than a base control. The catalog gives
# no other signal: an enhancement is simply a control nested inside one.
ENHANCEMENT_CLASS = "SP800-53-enhancement"


class CatalogError(Exception):
    """The catalog is not the document this loader knows how to read.

    Raised for missing structure rather than for surprising content. A control without an id
    cannot be cited and cannot be stored, so it is worth refusing the whole file over; a
    control with an unfamiliar part name is just a part name.
    """


@dataclass(frozen=True, slots=True)
class Part:
    """One part of a control: a statement, its guidance, or an item nested in either."""

    id: str | None
    name: str
    title: str | None
    # Absent on the parts that exist only to hold other parts.
    prose: str | None
    # The list marker as published — "a.", "1.", "(a)". Kept because prose read without it
    # loses the lettering the control text refers to by name.
    label: str | None
    parts: tuple[Part, ...]

    def walk(self) -> Iterator[Part]:
        """Yield this part and every part beneath it, depth-first."""
        yield self
        for child in self.parts:
            yield from child.walk()


@dataclass(frozen=True, slots=True)
class Selection:
    """A parameter offering choices rather than a value."""

    # Absent in the source means one, per the OSCAL specification. Resolved here so that no
    # reader downstream has to know that.
    how_many: str
    choices: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Parameter:
    """An organisation-defined value a control's prose refers to by id."""

    id: str
    label: str | None
    # Empty for every parameter in the pinned release — all 1,600 are organisation-defined —
    # but the model permits concrete values and a later release may carry some.
    values: tuple[str, ...]
    select: Selection | None
    guidelines: tuple[str, ...]
    # The pre-5.1.1 identifier for this parameter, where the catalog records one. The same
    # renumbering that gave controls two identifier forms gave parameters two as well.
    alt_identifier: str | None


@dataclass(frozen=True, slots=True)
class Control:
    """A control or one of its enhancements, with its children nested beneath it."""

    id: str
    title: str
    class_: str | None
    # The three identifier forms the catalog publishes, lifted out of `props` because
    # everything downstream needs at least two of them and none of it should be re-scanning a
    # property list to find out which is which. For `ac-2`: "AC-2", "AC-02", "ac-02".
    label: str | None
    zero_padded_label: str | None
    sort_id: str | None
    group_id: str
    withdrawn: bool
    params: tuple[Parameter, ...]
    parts: tuple[Part, ...]
    enhancements: tuple[Control, ...]

    @property
    def is_enhancement(self) -> bool:
        return self.class_ == ENHANCEMENT_CLASS

    def walk(self) -> Iterator[Control]:
        """Yield this control and every enhancement beneath it, depth-first."""
        yield self
        for enhancement in self.enhancements:
            yield from enhancement.walk()

    def iter_parts(self) -> Iterator[Part]:
        """Yield every part of this control at every depth, in document order."""
        for part in self.parts:
            yield from part.walk()


@dataclass(frozen=True, slots=True)
class Group:
    """A control family."""

    id: str
    title: str
    class_: str | None
    controls: tuple[Control, ...]
    # One family carries prose of its own. Kept rather than dropped: content that exists in
    # the source and nowhere in the loader is content nobody will remember to look for.
    parts: tuple[Part, ...]


@dataclass(frozen=True, slots=True)
class Catalog:
    """The parsed catalog, with a global parameter index built alongside it."""

    uuid: str
    title: str
    version: str
    last_modified: str
    oscal_version: str
    groups: tuple[Group, ...]
    # Global because a parameter id cannot be derived from its control's id. Built once during
    # parsing; a read-only view so that a shared, cached catalog cannot be edited by a caller.
    parameters: Mapping[str, Parameter] = field(default_factory=lambda: MappingProxyType({}))

    def iter_controls(self) -> Iterator[Control]:
        """Yield every control and enhancement, depth-first, in document order."""
        for group in self.groups:
            for control in group.controls:
                yield from control.walk()

    def counts(self) -> CatalogCounts:
        """Count what was parsed, in the terms the pin declares."""
        base = enhancements = withdrawn = parts = 0

        for control in self.iter_controls():
            if control.is_enhancement:
                enhancements += 1
            else:
                base += 1
            if control.withdrawn:
                withdrawn += 1
            parts += sum(1 for _ in control.iter_parts())

        return CatalogCounts(
            groups=len(self.groups),
            base_controls=base,
            enhancements=enhancements,
            withdrawn=withdrawn,
            live=base + enhancements - withdrawn,
            parameters=len(self.parameters),
            parts=parts,
        )


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    """Read the pinned catalog once per process."""
    return load_catalog(get_settings().catalog_path)


@lru_cache(maxsize=2)
def load_catalog(path: Path) -> Catalog:
    """Read and parse a catalog from a specific file.

    Cached on the path. Parsing is around a second and produces immutable objects, so the
    alternative is every caller either paying it again or inventing somewhere to keep the
    result. Callers wanting to know whether the file is the pinned one ask
    `warrant.catalog_pin.verify_catalog`; this function does not hash what it reads.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CatalogError(f"{path} is not valid JSON: {error}") from error

    return parse_catalog(document)


def parse_catalog(document: Any) -> Catalog:
    """Parse an already-decoded OSCAL catalog document."""
    if not isinstance(document, dict) or "catalog" not in document:
        raise CatalogError(
            "The document has no top-level 'catalog' key. An OSCAL catalog wraps everything "
            "in one; a profile or a system security plan would look like this."
        )

    catalog = _require_mapping(document["catalog"], "catalog")
    metadata = _require_mapping(catalog.get("metadata"), "catalog.metadata")

    parameters: dict[str, Parameter] = {}
    groups = tuple(
        _parse_group(entry, index, parameters)
        for index, entry in enumerate(_list(catalog, "groups"))
    )

    if not groups:
        raise CatalogError("The catalog has no groups, so it contains no controls.")

    return Catalog(
        uuid=_require_str(catalog, "uuid", "catalog"),
        title=_require_str(metadata, "title", "catalog.metadata"),
        version=_require_str(metadata, "version", "catalog.metadata"),
        last_modified=_require_str(metadata, "last-modified", "catalog.metadata"),
        oscal_version=_require_str(metadata, "oscal-version", "catalog.metadata"),
        groups=groups,
        parameters=MappingProxyType(parameters),
    )


def _parse_group(entry: Any, index: int, parameters: dict[str, Parameter]) -> Group:
    where = f"catalog.groups[{index}]"
    group = _require_mapping(entry, where)
    group_id = _require_str(group, "id", where)

    return Group(
        id=group_id,
        title=_require_str(group, "title", where),
        class_=_optional_str(group, "class"),
        controls=tuple(
            _parse_control(control, group_id, parameters) for control in _list(group, "controls")
        ),
        parts=tuple(_parse_part(part, where) for part in _list(group, "parts")),
    )


def _parse_control(entry: Any, group_id: str, parameters: dict[str, Parameter]) -> Control:
    control = _require_mapping(entry, f"a control in group {group_id}")
    control_id = _require_str(control, "id", f"a control in group {group_id}")
    where = f"control {control_id}"

    params = tuple(_parse_parameter(param, where) for param in _list(control, "params"))

    for param in params:
        # Last definition wins, but there are none: ids are unique across the catalog, and a
        # duplicate would mean two controls disagreeing about what one identifier means.
        if param.id in parameters:
            raise CatalogError(
                f"Parameter id {param.id!r} is defined more than once; the second definition "
                f"is on {where}. Parameter ids are resolved globally, so a duplicate makes "
                "the resolution of every marker naming it ambiguous."
            )
        parameters[param.id] = param

    props = _list(control, "props")

    return Control(
        id=control_id,
        title=_require_str(control, "title", where),
        class_=_optional_str(control, "class"),
        label=_prop(props, "label"),
        zero_padded_label=_prop(props, "label", "zero-padded"),
        sort_id=_prop(props, "sort-id"),
        group_id=group_id,
        withdrawn=_prop(props, "status") == "withdrawn",
        params=params,
        parts=tuple(_parse_part(part, where) for part in _list(control, "parts")),
        enhancements=tuple(
            _parse_control(child, group_id, parameters) for child in _list(control, "controls")
        ),
    )


def _parse_part(entry: Any, where: str) -> Part:
    part = _require_mapping(entry, f"a part of {where}")

    return Part(
        id=_optional_str(part, "id"),
        name=_require_str(part, "name", f"a part of {where}"),
        title=_optional_str(part, "title"),
        prose=_optional_str(part, "prose"),
        label=_prop(_list(part, "props"), "label"),
        parts=tuple(_parse_part(child, where) for child in _list(part, "parts")),
    )


def _parse_parameter(entry: Any, where: str) -> Parameter:
    param = _require_mapping(entry, f"a parameter of {where}")
    param_id = _require_str(param, "id", f"a parameter of {where}")

    selection = param.get("select")
    select = None
    if isinstance(selection, dict):
        select = Selection(
            # Absent means one. Spelling it out here keeps the default in the parser rather
            # than in every place that renders a selection.
            how_many=_optional_str(selection, "how-many") or "one",
            choices=tuple(str(choice) for choice in _list(selection, "choice")),
        )

    return Parameter(
        id=param_id,
        label=_optional_str(param, "label"),
        values=tuple(str(value) for value in _list(param, "values")),
        select=select,
        guidelines=tuple(
            prose
            for guideline in _list(param, "guidelines")
            if isinstance(guideline, dict) and (prose := _optional_str(guideline, "prose"))
        ),
        alt_identifier=_prop(_list(param, "props"), "alt-identifier"),
    )


def _prop(props: Sequence[Any], name: str, cls: str | None = None) -> str | None:
    """The value of one property, matched on name and class.

    Class is matched exactly, absent included, because the catalog distinguishes the two label
    forms by nothing else: `AC-2` is the label with no class, `AC-02` the one classed
    `zero-padded`. Matching on name alone returns whichever comes first.
    """
    for entry in props:
        if isinstance(entry, dict) and entry.get("name") == name and entry.get("class") == cls:
            value = entry.get("value")
            return value if isinstance(value, str) else None
    return None


def _list(mapping: Mapping[str, Any], key: str) -> list[Any]:
    """A list-valued key, absent or wrong-typed reading as empty."""
    value = mapping.get(key)
    return value if isinstance(value, list) else []


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CatalogError(f"{where} is {type(value).__name__}, expected an object.")
    return value


def _require_str(mapping: Mapping[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise CatalogError(f"{where} has no '{key}'. It is required to identify the record.")
    return value


def _optional_str(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) else None
