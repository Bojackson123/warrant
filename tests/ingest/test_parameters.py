"""Resolving parameter markers: the corpus-wide guarantees, and the rendering rules.

The guarantees run over the real vendored catalog rather than a miniature of it, for the reason
the loader's tests give: a hand-built document would only prove that the resolver handles the
document the test author imagined. Synthetic parameters appear here only for the paths the
pinned release does not contain — a concrete value, guidance without a label, a cycle, an
unknown identifier, irregular marker spacing — and each of those says so.
"""

from __future__ import annotations

import re

import pytest

from warrant.ingest.catalog import Catalog, Parameter, Selection, get_catalog
from warrant.ingest.parameters import (
    MARKER,
    RESOLUTION_VERSION,
    ParameterResolutionError,
    ParameterResolver,
    resolution_fingerprint,
)

# The digest of everything the resolver produces from the pinned catalog. Pinned so that a
# change to any rendering rule fails here rather than reaching the corpus unannounced; the fix
# is to bump RESOLUTION_VERSION deliberately, or to revert.
EXPECTED_FINGERPRINT = "c5569bba19d1bc0894cbec8e1c77c0845785291dce10c44898e2f674199cb6b1"

# Every marker in the file, counted across the three places they occur: control prose, the
# choices of a selection, and parameter guidance.
MARKERS_IN_CATALOG = 3023


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return get_catalog()


@pytest.fixture(scope="module")
def resolver(catalog: Catalog) -> ParameterResolver:
    return ParameterResolver.for_catalog(catalog)


def _marker_bearing_prose(catalog: Catalog) -> list[str]:
    return [
        part.prose
        for control in catalog.iter_controls()
        for part in control.iter_parts()
        if part.prose and MARKER.search(part.prose)
    ]


def _every_string(catalog: Catalog) -> list[str]:
    strings = [
        part.prose
        for control in catalog.iter_controls()
        for part in control.iter_parts()
        if part.prose
    ]
    for parameter in catalog.parameters.values():
        strings.extend(parameter.guidelines)
        if parameter.select:
            strings.extend(parameter.select.choices)
    return strings


def test_no_resolved_text_in_the_catalog_contains_template_braces(
    catalog: Catalog, resolver: ParameterResolver
) -> None:
    """The criterion the whole module exists for, asserted over the corpus and not a sample."""
    for text in _every_string(catalog):
        resolved = resolver.resolve(text)

        assert "{{" not in resolved
        assert "}}" not in resolved
        assert "insert:" not in resolved


def test_no_rendered_parameter_contains_template_braces(
    catalog: Catalog, resolver: ParameterResolver
) -> None:
    """Choices and guidance can hold markers of their own, so rendering recurses."""
    for parameter in catalog.parameters.values():
        rendered = resolver.render(parameter)

        assert "{{" not in rendered
        assert "}}" not in rendered


def test_substitution_leaves_no_whitespace_artefact(
    catalog: Catalog, resolver: ParameterResolver
) -> None:
    """Scoped to strings that contain a marker, which is the only thing this module tidies."""
    for text in _marker_bearing_prose(catalog):
        resolved = resolver.resolve(text)

        assert "  " not in resolved
        assert re.search(r"\]\s+[;,.:)\]]", resolved) is None


def test_every_marker_resolves_and_every_parameter_is_reached(
    catalog: Catalog, resolver: ParameterResolver
) -> None:
    """No marker is left over, and no parameter goes unreferenced."""
    found = [
        parameter_id for text in _every_string(catalog) for parameter_id in MARKER.findall(text)
    ]

    assert len(found) == MARKERS_IN_CATALOG
    assert set(found) == set(catalog.parameters)


def test_resolution_is_deterministic_and_idempotent(
    catalog: Catalog, resolver: ParameterResolver
) -> None:
    """Resolved text holds no markers, so a second pass finds nothing to do."""
    for text in _marker_bearing_prose(catalog):
        once = resolver.resolve(text)

        assert resolver.resolve(text) == once
        assert resolver.resolve(once) == once


def test_the_resolution_version_and_fingerprint_are_pinned(catalog: Catalog) -> None:
    """A rendering change that nobody announced fails here rather than downstream."""
    assert RESOLUTION_VERSION == "1"
    assert resolution_fingerprint(catalog) == EXPECTED_FINGERPRINT


def test_a_label_renders_as_an_organisation_defined_slot(resolver: ParameterResolver) -> None:
    parameter = resolver.parameters["ac-02_odp.01"]

    assert resolver.render(parameter) == "[organization-defined prerequisites and criteria]"


def test_a_label_that_already_says_organization_defined_is_not_doubled(
    resolver: ParameterResolver,
) -> None:
    """141 labels spell the qualifier out themselves; prefixing them again reads as a bug."""
    parameter = resolver.parameters["ac-1_prm_1"]

    assert resolver.render(parameter) == "[organization-defined personnel or roles]"


def test_an_aggregate_parameter_renders_its_own_label(resolver: ParameterResolver) -> None:
    """`ac-1_prm_1` rolls up two later parameters, and its label is already the slot a reader
    wants. Expanding it would print the same phrase twice."""
    resolved = resolver.resolve(
        "Develop, document, and disseminate to {{ insert: param, ac-1_prm_1 }}:"
    )

    assert (
        resolved
        == "Develop, document, and disseminate to [organization-defined personnel or roles]:"
    )


def test_a_selection_renders_its_choices(resolver: ParameterResolver) -> None:
    parameter = resolver.parameters["ac-02.02_odp.01"]

    assert resolver.render(parameter) == "[selection (one): remove; disable]"


def test_a_one_or_more_selection_says_so(resolver: ParameterResolver) -> None:
    parameter = resolver.parameters["ac-01_odp.03"]

    assert resolver.render(parameter) == (
        "[selection (one or more): organization-level; mission/business process-level; "
        "system-level]"
    )


def test_a_choice_containing_a_marker_is_resolved(resolver: ParameterResolver) -> None:
    """One level of nesting, and the choice carries a trailing space the source left behind."""
    parameter = resolver.parameters["ac-04.04_odp.02"]

    assert resolver.render(parameter).endswith(
        "terminating communications sessions attempting to pass encrypted information; "
        "[organization-defined procedure or method]]"
    )


def test_a_space_before_punctuation_closes_up_around_a_substitution(
    catalog: Catalog, resolver: ParameterResolver
) -> None:
    """The source writes `}} ; and`, and the space is only conspicuous once the braces go."""
    control = next(control for control in catalog.iter_controls() if control.id == "ac-1")
    prose = next(
        part.prose
        for part in control.iter_parts()
        if part.prose and part.prose.startswith("Policy {{")
    )

    assert resolver.resolve(prose) == (
        "Policy [organization-defined frequency] and following [organization-defined events]; and"
    )


def test_prose_without_a_marker_is_returned_unchanged(resolver: ParameterResolver) -> None:
    """Including the source's own spacing defects: tidying those is not this module's business,
    and a resolver that rewrote prose containing no parameters would be misnamed."""
    prose = "described in [AC-25](#ac-25) . The policy is bounded by"

    assert resolver.resolve(prose) == prose


def test_an_unknown_parameter_id_is_refused(resolver: ParameterResolver) -> None:
    """The pinned catalog resolves every marker it contains, so this can only be provoked."""
    with pytest.raises(ParameterResolutionError, match="ac-02_prm_9"):
        resolver.resolve("Review {{ insert: param, ac-02_prm_9 }} accounts.")


def test_a_pre_renumbering_identifier_is_not_silently_accepted(
    resolver: ParameterResolver,
) -> None:
    """`ac-02_odp.01` records `ac-2_prm_1` as its former id. Resolving a marker to a near match
    is how a corpus ends up describing a control nobody cited."""
    assert resolver.parameters["ac-02_odp.01"].alt_identifier == "ac-2_prm_1"

    with pytest.raises(ParameterResolutionError, match="alt-identifier"):
        resolver.resolve("{{ insert: param, ac-2_prm_1 }}")


def _resolver(*parameters: Parameter) -> ParameterResolver:
    return ParameterResolver({parameter.id: parameter for parameter in parameters})


def _parameter(
    parameter_id: str = "xx-01_odp.01",
    *,
    label: str | None = None,
    values: tuple[str, ...] = (),
    select: Selection | None = None,
    guidelines: tuple[str, ...] = (),
) -> Parameter:
    return Parameter(
        id=parameter_id,
        label=label,
        values=values,
        select=select,
        guidelines=guidelines,
        alt_identifier=None,
    )


def test_irregular_marker_spacing_is_absorbed() -> None:
    """Every marker in the pinned release is byte-identical, so this is the one rule the file
    cannot exercise. A pattern that only accepted today's exact form would fail silently."""
    resolver = _resolver(_parameter(label="frequency"))

    for marker in (
        "{{insert:param,xx-01_odp.01}}",
        "{{  insert:  param,   xx-01_odp.01   }}",
        "{{\tinsert: param,\txx-01_odp.01\t}}",
    ):
        assert resolver.resolve(f"Review {marker} and report.") == (
            "Review [organization-defined frequency] and report."
        )


def test_a_parameter_with_a_concrete_value_is_substituted_rather_than_slotted() -> None:
    """No parameter in this catalog carries one — all 1,600 are organisation-defined — but the
    format permits it, and brackets would then claim a decided value still needs deciding."""
    resolver = _resolver(_parameter(values=("30 days",), label="frequency"))

    assert resolver.resolve("Review {{ insert: param, xx-01_odp.01 }}.") == "Review 30 days."


def test_a_parameter_with_only_guidance_falls_back_to_it() -> None:
    """Unreachable in this release, where every parameter has a label or a selection."""
    resolver = _resolver(_parameter(guidelines=("the frequency of review is defined;",)))

    assert resolver.resolve("{{ insert: param, xx-01_odp.01 }}") == (
        "[organization-defined the frequency of review is defined]"
    )


def test_a_marker_inside_guidance_is_resolved() -> None:
    """Five parameters carry one, and all five also carry a label, so precedence rather than
    resolution is what keeps braces out of the corpus today. That stops being true one release
    later."""
    resolver = _resolver(
        _parameter(
            "xx-01_odp.01", guidelines=("locations where {{ insert: param, xx-01_odp.02 }} apply;",)
        ),
        _parameter("xx-01_odp.02", label="systems"),
    )

    assert resolver.resolve("{{ insert: param, xx-01_odp.01 }}") == (
        "[organization-defined locations where [organization-defined systems] apply]"
    )


def test_a_parameter_with_nothing_to_render_still_produces_a_slot() -> None:
    resolver = _resolver(_parameter())

    assert resolver.resolve("{{ insert: param, xx-01_odp.01 }}") == "[organization-defined value]"


def test_a_selection_with_no_choices_still_produces_a_slot() -> None:
    resolver = _resolver(_parameter(select=Selection(how_many="one", choices=())))

    assert resolver.resolve("{{ insert: param, xx-01_odp.01 }}") == (
        "[organization-defined selection]"
    )


def test_a_cycle_is_refused_rather_than_recursed_forever() -> None:
    """Nesting in the pinned release is one level deep. A parameter that refers back to itself
    is a defect in the file, and the resolver has to say so rather than exhaust the stack."""
    resolver = _resolver(
        _parameter("xx-01_odp.01", guidelines=("see {{ insert: param, xx-01_odp.02 }};",)),
        _parameter("xx-01_odp.02", guidelines=("see {{ insert: param, xx-01_odp.01 }};",)),
    )

    with pytest.raises(ParameterResolutionError, match="levels deep"):
        resolver.resolve("{{ insert: param, xx-01_odp.01 }}")
