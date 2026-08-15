"""Resolve the catalog's parameter markers into readable slots.

Control prose refers to its parameters by interpolation marker — `{{ insert: param,
ac-01_odp.05 }}` — and the catalog contains 3,023 of them. Left in place they end up in two
places that both matter: inside the text that gets embedded, where template braces are noise
in every vector, and inside the clause a reviewer clicks through to read, where they make the
corpus look like machine output nobody checked.

Resolution is kept out of the loader because it is a separate decision with its own version.
That version is load-bearing rather than decorative: a change in how a slot renders changes
chunk text, which changes embeddings, which changes retrieved identifiers, which changes the
prompt text that recorded model calls are keyed on. `RESOLUTION_VERSION` is what declares
that, and `resolution_fingerprint` is what stops the declaration from going stale quietly.

Three properties of the source shape what is here, and all three were measured rather than
assumed:

- **Every parameter in this release is organisation-defined.** None carries a concrete value,
  so resolution always renders a slot. The value path exists because the format permits one.
- **Marker spacing is uniform**, all 3,023 of them. The irregularity the prose actually
  contains is around the marker rather than inside it — `{{ … }} ; and` — so substitution
  introduces no whitespace defects of its own but does expose the source's.
- **Markers nest**, one level deep: a selection choice can itself contain a marker.

The rendering rules here were first worked out in the corpus preparation script under
`tools/`, which is frozen as the record of what the embedding comparison measured. This is a
re-implementation of them against the typed objects rather than an import, so that the
evidence stays fixed while the shipped resolver is free to change.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass

from warrant.ingest.catalog import Catalog, CatalogError, Parameter

# Bumped deliberately when the text this module produces changes, because everything
# downstream is a function of that text. `resolution_fingerprint` is the check that a change
# was noticed; this string is how it is announced.
RESOLUTION_VERSION = "1"

# The catalog's only interpolation syntax, with the whitespace on either side of the marker
# absorbed into the match so the replacement can decide what spacing to put back. Tolerant of
# spacing the pinned release does not contain: every marker in it is byte-identical, and a
# pattern that only accepts today's exact form would fail silently on a release that varied.
MARKER = re.compile(r"[ \t]*\{\{\s*insert:\s*param,\s*([^\s}]+)\s*\}\}[ \t]*")

# A rendered slot never keeps a space in front of these. The source writes `{{ … }} ; and` and
# `{{ … }} , and` in places, and the space only becomes conspicuous once the braces are gone.
_NO_SPACE_BEFORE = ";,.:)]"

# Absorbed leading whitespace is not put back after these, so that swallowing an indent or a
# space inside a bracket cannot turn into a stray one.
_NO_SPACE_AFTER = ("\n", "(", "[")

# Guards a malformed catalog against recursing forever. Nesting observed in the pinned release
# is one level deep; this is an order of magnitude of headroom.
MAX_RESOLUTION_DEPTH = 8

_ORGANIZATION_DEFINED = "organization-defined"


class ParameterResolutionError(CatalogError):
    """A marker cannot be resolved against the parameters the catalog defines.

    A subclass of `CatalogError` because it is the same kind of fact: the document is not the
    one this code knows how to read. Callers that already refuse to start on a malformed
    catalog refuse on an unresolvable marker too, without having to learn a second exception.
    """


@dataclass(frozen=True, slots=True)
class ParameterResolver:
    """Renders parameter markers against a catalog's global parameter index.

    Holds the index rather than the catalog: resolution needs nothing else, and a resolver
    built from a mapping can be pointed at a subset in a test without constructing a document
    around it. Construction is a dataclass literal over an already-parsed mapping, so there is
    no cache here — a second one would only add a way for two callers to disagree about which
    catalog they are resolving against.
    """

    parameters: Mapping[str, Parameter]

    @classmethod
    def for_catalog(cls, catalog: Catalog) -> ParameterResolver:
        return cls(catalog.parameters)

    def resolve(self, text: str) -> str:
        """Replace every marker in a string with the slot it stands for.

        Idempotent, because the output contains no markers: resolving resolved text is a no-op
        rather than a second pass that finds something new.
        """
        return self._resolve(text, depth=0)

    def render(self, parameter: Parameter) -> str:
        """Render one parameter as the text a reader should see in place of its marker.

        Square brackets mean "you have to decide this" and are used for nothing else, which is
        why a parameter carrying a concrete value renders without them.
        """
        return self._render(parameter, depth=0)

    def _resolve(self, text: str, depth: int) -> str:
        if depth > MAX_RESOLUTION_DEPTH:
            raise ParameterResolutionError(
                f"Parameter resolution went more than {MAX_RESOLUTION_DEPTH} levels deep, which "
                "means a parameter refers to itself through its choices or its guidance. The "
                "pinned catalog nests one level; a cycle is a defect in the file."
            )

        def replace(match: re.Match[str]) -> str:
            parameter_id = match.group(1)
            parameter = self.parameters.get(parameter_id)

            if parameter is None:
                raise ParameterResolutionError(
                    f"A marker refers to parameter {parameter_id!r}, which this catalog does not "
                    "define. Parameter identifiers use the zero-padded form regardless of their "
                    "control's own identifier — `ac-2` owns `ac-02_odp.01` — and the "
                    "pre-renumbering form recorded in `alt-identifier` is deliberately not "
                    "accepted here, because resolving a marker to a near match is how a corpus "
                    "ends up describing a control nobody cited."
                )

            slot = self._render(parameter, depth)
            return self._respace(match, text, slot)

        return MARKER.sub(replace, text)

    @staticmethod
    def _respace(match: re.Match[str], text: str, slot: str) -> str:
        """Put back at most one space on each side of a substitution.

        The pattern absorbs the whitespace around the marker so that this is decided once, here,
        from what surrounds the substitution — rather than by sweeping the finished document
        with a regex that would also rewrite prose containing no parameters at all. Tidying the
        seam is this module's business; tidying the catalog's own punctuation is not.
        """
        matched = match.group(0)
        before = text[: match.start()]
        after = text[match.end() :]

        lead = (
            " " if matched[0] in " \t" and before and not before.endswith(_NO_SPACE_AFTER) else ""
        )
        trail = " " if matched[-1] in " \t" and after and after[0] not in _NO_SPACE_BEFORE else ""

        return f"{lead}{slot}{trail}"

    def _render(self, parameter: Parameter, depth: int) -> str:
        # Precedence, most specific first. Nothing in the pinned release carries a value, and
        # nothing carries both a selection and a label, so the order between the first three is
        # currently unobservable and is written down rather than inferred from behaviour.
        if parameter.values:
            # Not bracketed: a value that has been decided is not a slot to fill in.
            return ", ".join(parameter.values)

        if parameter.select is not None:
            choices = [
                self._resolve(choice, depth + 1).strip() for choice in parameter.select.choices
            ]
            if not choices:
                return f"[{_ORGANIZATION_DEFINED} selection]"

            # "one-or-more" reads as prose once the hyphens go; absent in the source means one,
            # which the loader has already resolved.
            how_many = parameter.select.how_many.replace("-", " ")
            return f"[selection ({how_many}): {'; '.join(choices)}]"

        if parameter.label:
            return f"[{_slot(parameter.label)}]"

        for guideline in parameter.guidelines:
            # Guidance is a sentence about the parameter rather than a name for it, so the
            # trailing punctuation goes. Markers inside it are resolved for the same reason
            # they are inside a choice: 5 parameters carry one, and all 5 also carry a label,
            # so today this is unreachable by precedence rather than by resolution.
            prose = self._resolve(guideline, depth + 1).rstrip(";. ").strip()
            if prose:
                return f"[{_slot(prose)}]"

        return f"[{_ORGANIZATION_DEFINED} value]"


def _slot(label: str) -> str:
    """Prefix a parameter's name with its qualifier, unless it already carries one.

    Two naming conventions coexist in the catalog: most labels are a bare noun phrase
    ("frequency"), but 141 of them spell out "organization-defined" themselves. Prefixing
    unconditionally produces "organization-defined organization-defined personnel or roles".
    """
    if label.lower().startswith(_ORGANIZATION_DEFINED):
        return label
    return f"{_ORGANIZATION_DEFINED} {label}"


def resolution_fingerprint(catalog: Catalog) -> str:
    """SHA-256 of everything this module produces from a catalog.

    `RESOLUTION_VERSION` on its own is a comment, and a comment cannot notice that somebody
    changed how a slot renders and did not bump it. This is the number that can: pin it beside
    the version, and a rendering change fails with two digests instead of shipping quietly and
    invalidating every vector and every recorded model call downstream.

    Covers every part at every depth, assessment procedures included, even though the chunker
    will not embed all of them. The subject is the resolver rather than the corpus, so a change
    that only shows up in text nothing currently reads should still turn this red. Identifiers
    and positions go into the digest alongside the text, so that reordering is not invisible.
    """
    resolver = ParameterResolver.for_catalog(catalog)
    digest = hashlib.sha256()

    for control in catalog.iter_controls():
        for index, part in enumerate(control.iter_parts()):
            if part.prose is None:
                continue
            resolved = resolver.resolve(part.prose)
            digest.update(f"{control.id}\x1f{index}\x1f{resolved}\x1e".encode())

    for parameter_id in sorted(catalog.parameters):
        rendered = resolver.render(catalog.parameters[parameter_id])
        digest.update(f"{parameter_id}\x1f{rendered}\x1e".encode())

    return digest.hexdigest()
