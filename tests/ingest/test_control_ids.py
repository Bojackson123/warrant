"""Resolving control identifiers: the corpus-wide round-trip, and where a miss has to stay one.

The round-trip runs over every control and enhancement in the real vendored catalog rather than
a sample, because a spot check is exactly what would pass while a whole family of identifiers
quietly stopped resolving. Synthetic controls appear only for the three failures the pinned
release cannot produce — a missing published form, a duplicate control id, and one form naming
two controls — and each says so.
"""

from __future__ import annotations

import pytest

from warrant.ingest.catalog import Catalog, Control, Group, get_catalog
from warrant.ingest.control_ids import (
    ControlIdentity,
    ControlIdError,
    ControlIndex,
    canonical_id,
    control_id_of_part,
    get_control_index,
)

# Controls and enhancements in the pinned release, and the number of distinct strings their four
# published forms casefold to. The second is well short of 4 × 1,196 because the forms collapse
# into each other wherever their differences have nothing to bite on: zero-padding is a no-op on
# a number that is already two digits, and casefolding erases the difference between a label and
# an id. So `AC-2` contributes two keys and `AC-10` contributes one.
CONTROLS_IN_CATALOG = 1196
PUBLISHED_FORMS = 3961


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return get_catalog()


@pytest.fixture(scope="module")
def index() -> ControlIndex:
    return get_control_index()


def _published_forms(identity: ControlIdentity) -> tuple[str, ...]:
    return (identity.id, identity.label, identity.zero_padded_label, identity.sort_id)


def test_every_control_round_trips_through_every_published_form(index: ControlIndex) -> None:
    """The criterion the module exists for, asserted over the corpus and not a handful of ids.

    Both notations for every control, each in the case the catalog publishes it in and in both
    of the cases it does not, all reaching the same record.
    """
    assert len(index) == CONTROLS_IN_CATALOG

    for identity in index:
        for form in _published_forms(identity):
            for variant in (form, form.upper(), form.lower()):
                assert index.resolve(variant) is identity


def test_enhancements_round_trip_in_both_notations(index: ControlIndex) -> None:
    """Padded and non-padded, dotted and parenthesised, over every enhancement in the file."""
    enhancements = [identity for identity in index if identity.is_enhancement]

    assert enhancements

    for identity in enhancements:
        assert index.resolve(identity.id) is identity  # ac-2.3
        assert index.resolve(identity.sort_id) is identity  # ac-02.03
        assert index.resolve(identity.label) is identity  # AC-2(3)
        assert index.resolve(identity.zero_padded_label) is identity  # AC-02(03)


def test_the_notations_are_the_ones_this_module_claims(index: ControlIndex) -> None:
    """Spelled out once, so the corpus-wide assertions above are not the only description of
    what the four forms actually look like."""
    identity = index.require("ac-2.3")

    assert _published_forms(identity) == ("ac-2.3", "AC-2(3)", "AC-02(03)", "ac-02.03")


def test_the_published_forms_and_the_structural_parser_agree(index: ControlIndex) -> None:
    """The parser is consulted only for strings the catalog does not publish, so nothing would
    otherwise notice it drifting away from the forms that are published. This is what fails on a
    release where padding stops being derivable, rather than the drift reaching a lookup."""
    assert len(index.by_form) == PUBLISHED_FORMS

    for identity in index:
        for form in _published_forms(identity):
            assert canonical_id(form) == identity.id


def test_the_four_published_forms_are_derivable_from_the_id(index: ControlIndex) -> None:
    """The regularity the decision record claims, asserted rather than assumed.

    Weaker than it looks if left to the parser test above: that one only checks that each form
    normalises back to its own id, which a release publishing `sort-id` unpadded or a label as
    `AC-2 (3)` would still satisfy. This states the actual shape of each form, so a release that
    stops deriving them the same way fails here — which is what makes the two-layer resolution
    order load-bearing rather than decorative.
    """
    for identity in index:
        family, _, numbers = identity.id.partition("-")
        components = numbers.split(".")
        padded = ".".join(component.zfill(2) for component in components)
        parenthesised = "".join(f"({component})" for component in components[1:])
        parenthesised_padded = "".join(f"({component.zfill(2)})" for component in components[1:])

        assert identity.sort_id == f"{family}-{padded}"
        assert identity.label == f"{family.upper()}-{components[0]}{parenthesised}"
        assert identity.zero_padded_label == (
            f"{family.upper()}-{components[0].zfill(2)}{parenthesised_padded}"
        )


def test_no_published_form_names_two_controls(index: ControlIndex) -> None:
    """Every key was registered by exactly one control, which is what makes the alias table safe
    to build densely rather than only over the forms that happen not to collide."""
    for key, identity in index.by_form.items():
        assert key in {form.casefold() for form in _published_forms(identity)}


def test_the_storage_form_and_the_citation_form_are_kept_apart(index: ControlIndex) -> None:
    """Silently mixing them is the failure this module exists to prevent, so they are different
    strings under different names and a caller has to choose."""
    identity = index.require("AC-2(3)")

    assert identity.id == "ac-2.3"
    assert identity.label == "AC-2(3)"
    assert identity.id != identity.label


def test_base_id_comes_from_the_documents_own_nesting(index: ControlIndex) -> None:
    """Asserted for every control rather than derived, because the whole point of taking it from
    the tree is not to trust that splitting the id gives the same answer."""
    for identity in index:
        base = index.get(identity.base_id)

        assert base is not None
        assert not base.is_enhancement
        assert base.group_id == identity.group_id

        if identity.is_enhancement:
            assert identity.base_id != identity.id
        else:
            assert identity.base_id == identity.id


def test_an_enhancement_resolves_to_its_own_control_not_its_parent(index: ControlIndex) -> None:
    """`AC-2` and `AC-2(3)` are different answers, and a lookup that folded the second into the
    first would make every enhancement uncitable."""
    assert index.require("AC-2(3)").id == "ac-2.3"
    assert index.require("AC-2").id == "ac-2"
    assert index.require("AC-2(3)") is not index.require("AC-2")


def test_a_withdrawn_control_resolves_and_says_so(index: ControlIndex) -> None:
    """Withdrawn controls are excluded from the corpus but not from the document. Citing one is
    a stale answer; citing something that never existed is an invented one, and a lookup that
    returned a miss for both would make the two indistinguishable."""
    identity = index.require("AC-2(10)")

    assert identity.id == "ac-2.10"
    assert identity.withdrawn


def test_human_variants_resolve(index: ControlIndex) -> None:
    """Case, surrounding space, a missing separator, and the mixed notations somebody produces
    by copying half of one form and half of another. None of these is published by any release,
    which is why the structural parser exists."""
    for text in ("  AC-2  ", "ac2", "AC 2", "Ac-2"):
        assert index.resolve(text) is index.require("ac-2")

    for text in ("AC-2.3", "ac-02(3)", "AC 2 (3)", "ac-2 (03)", "AC-02.3"):
        assert index.resolve(text) is index.require("ac-2.3")


def test_an_identifier_that_names_nothing_is_a_miss_not_a_near_match(index: ControlIndex) -> None:
    """`ac-2.14` sits directly beside the real `ac-2.13`, which is the case where returning the
    nearest control would look most like being helpful."""
    for text in ("AC-99", "ac-2.14", "ac-2(99)", "zz-1", "ZZ-1(1)"):
        assert canonical_id(text) is not None
        assert index.resolve(text) is None
        assert text not in index


def test_a_string_that_is_not_an_identifier_is_a_miss(index: ControlIndex) -> None:
    """Including a family on its own and prose with an identifier inside it. A control family is
    not a control, and a sentence mentioning one is not a citation."""
    for text in ("", "   ", "AC", "ac", "access control", "ac-2 accounts", "see AC-2", "2"):
        assert index.resolve(text) is None


def test_a_damaged_identifier_is_a_miss_rather_than_a_repaired_one(index: ControlIndex) -> None:
    """The two enhancement notations are alternatives, not a menu of interchangeable characters,
    so an opening parenthesis has to be closed. A citation truncated in transit is the case that
    matters: accepting `AC-2(3` would score a damaged identifier as a clean hit."""
    for text in ("AC-2(3", "AC-2 3)", "ac-23)", "AC-2(3))", "AC-2()"):
        assert index.resolve(text) is None
        assert canonical_id(text) is None


def test_an_identity_this_index_produced_is_in_it(index: ControlIndex) -> None:
    """`__iter__` yields identities, so `identity in index` is the obvious next thing to write
    and must not answer with a quiet `False`."""
    identity = index.require("ac-2.3")

    assert identity in index
    assert all(candidate in index for candidate in index)
    assert object() not in index


def test_require_refuses_a_miss_and_quotes_what_it_was_given(index: ControlIndex) -> None:
    with pytest.raises(ControlIdError, match="AC-99"):
        index.require("AC-99")


def test_a_part_id_is_not_a_control_id(index: ControlIndex) -> None:
    """Handing back the whole control for a request naming one clause of it is a silent
    substitution, so the two questions are asked with two functions."""
    assert index.resolve("ac-2.3_smt.a") is None
    assert control_id_of_part("ac-2.3_smt.a") == "ac-2.3"
    assert control_id_of_part("ac-2_obj") == "ac-2"
    assert control_id_of_part("ac-2") is None

    # A separator with nothing after it is a truncated part id, and reporting it as a
    # well-formed control is the same silent substitution in the other direction.
    assert control_id_of_part("ac-2_") is None
    assert control_id_of_part("_smt.a") is None


def test_every_part_id_in_the_catalog_names_a_control_the_index_knows(
    catalog: Catalog, index: ControlIndex
) -> None:
    """Parts are prefixed with the non-padded control id, so this also confirms that the padding
    problem stops at the control and does not reappear inside the clause identifiers that clause
    click-through will be resolving."""
    part_ids = [
        part.id
        for control in catalog.iter_controls()
        for part in control.iter_parts()
        if part.id is not None
    ]

    assert part_ids

    for part_id in part_ids:
        owner = control_id_of_part(part_id)

        assert owner is not None
        assert index.get(owner) is not None


def test_canonical_id_normalises_without_knowing_what_exists() -> None:
    """Syntax and existence are separate questions on purpose: a parse that also decided
    existence would have to guess at a string it could not place, and guessing is the behaviour
    this module refuses."""
    assert canonical_id("AC-02(03)") == "ac-2.3"
    assert canonical_id("zz-9") == "zz-9"
    assert canonical_id("AC") is None


def _catalog(*controls: Control) -> Catalog:
    return Catalog(
        uuid="uuid",
        title="title",
        version="version",
        last_modified="last-modified",
        oscal_version="oscal-version",
        groups=(Group(id="xx", title="Example", class_=None, controls=controls, parts=()),),
    )


def _control(
    control_id: str = "xx-1",
    *,
    label: str | None = "XX-1",
    zero_padded_label: str | None = "XX-01",
    sort_id: str | None = "xx-01",
) -> Control:
    return Control(
        id=control_id,
        title="Example",
        class_="SP800-53",
        label=label,
        zero_padded_label=zero_padded_label,
        sort_id=sort_id,
        group_id="xx",
        withdrawn=False,
        params=(),
        parts=(),
        enhancements=(),
    )


def test_a_control_missing_a_published_form_is_refused() -> None:
    """Every control in this release carries all four, so this needs a constructed one. Deriving
    the missing form from the id would work today and would be a guess about the next release."""
    with pytest.raises(ControlIdError, match="cannot be cited"):
        ControlIndex.for_catalog(_catalog(_control(label=None)))


def test_a_duplicate_control_id_is_refused() -> None:
    """Also unreachable: ids are unique across the pinned catalog."""
    with pytest.raises(ControlIdError, match="more than once"):
        ControlIndex.for_catalog(_catalog(_control(), _control()))


def test_one_published_form_naming_two_controls_is_refused() -> None:
    """A distinct refusal from the one above, and it needs its own case to reach: two controls
    with different ids, one of which is published under a form belonging to the other. The four
    forms casefold to 3,961 keys in this release with no control sharing one, so an identifier
    that names two controls cannot arise here — and it would make every lookup of it a coin
    toss, which is worse than the loud failure."""
    trespasser = _control("xx-1", label="YY-1")
    victim = _control("yy-1", label="YY-1", zero_padded_label="YY-01", sort_id="yy-01")

    with pytest.raises(ControlIdError, match="both published as"):
        ControlIndex.for_catalog(_catalog(trespasser, victim))
