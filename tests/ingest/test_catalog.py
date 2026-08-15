"""Parsing the catalog: the counts, and the three shapes that are easy to get wrong.

These run against the real vendored file rather than a miniature of it. A hand-built fixture
would test that the parser reads the document the test author imagined, and the whole reason
this loader exists is that the document has properties nobody would have imagined.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from warrant.catalog_pin import get_catalog_pin
from warrant.ingest.catalog import Catalog, CatalogError, Control, get_catalog, parse_catalog


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    """Parsed once for the module. It is immutable, and parsing 10 MB per test is waste."""
    return get_catalog()


@pytest.fixture(scope="module")
def by_id(catalog: Catalog) -> dict[str, Control]:
    return {control.id: control for control in catalog.iter_controls()}


def test_counts_match_the_pin(catalog: Catalog) -> None:
    """The check that a swapped-but-well-formed catalog cannot pass."""
    counts = catalog.counts()
    expected = get_catalog_pin().expected_counts

    assert counts.differences(expected) == {}


def test_metadata_matches_the_pin(catalog: Catalog) -> None:
    pin = get_catalog_pin()

    assert catalog.version == pin.catalog_version
    assert catalog.oscal_version == pin.oscal_version
    assert catalog.last_modified == pin.last_modified


def test_base_control_carries_every_identifier_form(by_id: dict[str, Control]) -> None:
    """The forms disagree with each other, which is why all of them are kept."""
    ac2 = by_id["ac-2"]

    assert ac2.label == "AC-2"
    assert ac2.zero_padded_label == "AC-02"
    assert ac2.sort_id == "ac-02"
    assert ac2.group_id == "ac"
    assert not ac2.is_enhancement
    assert not ac2.withdrawn


def test_enhancements_are_nested_under_their_control(by_id: dict[str, Control]) -> None:
    ac2 = by_id["ac-2"]

    assert "ac-2.3" in {enhancement.id for enhancement in ac2.enhancements}

    ac2_3 = by_id["ac-2.3"]

    assert ac2_3.is_enhancement
    assert ac2_3.label == "AC-2(3)"
    assert ac2_3.zero_padded_label == "AC-02(03)"
    assert ac2_3.sort_id == "ac-02.03"


def test_withdrawal_is_read_out_of_the_property_list(by_id: dict[str, Control]) -> None:
    """182 records are withdrawn and the catalog says so only in `props`."""
    assert by_id["sc-19"].withdrawn
    assert sum(1 for control in by_id.values() if control.withdrawn) == 182


def test_a_withdrawn_control_can_still_carry_prose(by_id: dict[str, Control]) -> None:
    """Which is why nothing downstream may infer status from the presence of a statement."""
    withdrawn = by_id["sc-19"]

    assert withdrawn.withdrawn
    assert any(part.name == "statement" for part in withdrawn.parts)


def test_parameters_are_indexed_globally(catalog: Catalog, by_id: dict[str, Control]) -> None:
    """A parameter id cannot be derived from its control's id: `ac-2` owns `ac-02_odp.01`."""
    assert len(catalog.parameters) == 1600
    assert "ac-02_odp.01" in catalog.parameters
    assert "ac-02_odp.01" in {param.id for param in by_id["ac-2"].params}

    for control in by_id.values():
        for param in control.params:
            assert catalog.parameters[param.id] is param


def test_parameter_carries_its_label_guidance_and_former_id(catalog: Catalog) -> None:
    param = catalog.parameters["ac-02_odp.01"]

    assert param.label == "prerequisites and criteria"
    assert param.guidelines and param.guidelines[0].startswith("prerequisites and criteria")
    assert param.alt_identifier == "ac-2_prm_1"
    assert param.select is None
    assert param.values == ()


def test_selection_parameters_parse_with_their_choices(catalog: Catalog) -> None:
    selection = catalog.parameters["ac-01_odp.03"].select

    assert selection is not None
    assert selection.how_many == "one-or-more"
    assert "organization-level" in selection.choices


def test_absent_how_many_means_one(catalog: Catalog) -> None:
    """The OSCAL default, resolved in the parser so no caller has to know it."""
    selection = catalog.parameters["ac-02.02_odp.01"].select

    assert selection is not None
    assert selection.how_many == "one"
    assert selection.choices == ("remove", "disable")


def test_every_parameter_in_the_catalog_is_reachable(catalog: Catalog) -> None:
    """Nothing is dropped: the index is built from the controls, and only controls carry
    parameters in this release."""
    from_controls = sum(len(control.params) for control in catalog.iter_controls())

    assert from_controls == len(catalog.parameters)


def test_assessment_parts_are_kept(by_id: dict[str, Control]) -> None:
    """The loader represents the document; deciding what to embed belongs to the chunker."""
    names = [part.name for part in by_id["ac-2"].parts]

    assert names[:2] == ["statement", "guidance"]
    assert "assessment-objective" in names
    assert "assessment-method" in names


def test_nested_parts_keep_their_list_labels(by_id: dict[str, Control]) -> None:
    """Prose read without its marker loses the lettering the control text cites."""
    statement = next(part for part in by_id["ac-2"].parts if part.name == "statement")
    items = [child for child in statement.parts if child.label]

    assert items, "the AC-2 statement is a lettered list"
    assert items[0].label == "a."
    assert items[0].prose


def test_controls_come_out_in_document_order(catalog: Catalog) -> None:
    ids = [control.id for control in catalog.iter_controls()]

    assert ids[0] == "ac-1"
    assert ids.index("ac-2") < ids.index("ac-2.1") < ids.index("ac-3")


def test_the_parameter_index_cannot_be_edited(catalog: Catalog) -> None:
    """The catalog is cached and shared; a caller must not be able to change it for everyone."""
    with pytest.raises(TypeError):
        catalog.parameters["ac-02_odp.01"] = None  # type: ignore[index]


def test_the_catalog_is_parsed_once(catalog: Catalog) -> None:
    assert get_catalog() is catalog


def _minimal_document() -> dict[str, Any]:
    return {
        "catalog": {
            "uuid": "8c1e2b7a-0000-4000-8000-000000000000",
            "metadata": {
                "title": "Example",
                "version": "1.0.0",
                "last-modified": "2026-01-01T00:00:00.00000-00:00",
                "oscal-version": "1.2.2",
            },
            "groups": [
                {
                    "id": "xx",
                    "title": "Example family",
                    "controls": [{"id": "xx-1", "title": "Example control"}],
                }
            ],
        }
    }


def test_a_profile_is_refused_rather_than_half_parsed() -> None:
    with pytest.raises(CatalogError, match="no top-level 'catalog' key"):
        parse_catalog({"profile": {}})


def test_a_control_without_an_id_fails_the_whole_file() -> None:
    """It could not be cited or stored, so the file is wrong rather than partly usable."""
    document = _minimal_document()
    del document["catalog"]["groups"][0]["controls"][0]["id"]

    with pytest.raises(CatalogError, match="'id'"):
        parse_catalog(document)


def test_missing_metadata_is_named() -> None:
    document = _minimal_document()
    del document["catalog"]["metadata"]["version"]

    with pytest.raises(CatalogError, match="catalog.metadata has no 'version'"):
        parse_catalog(document)


def test_a_catalog_with_no_groups_is_refused() -> None:
    document = _minimal_document()
    document["catalog"]["groups"] = []

    with pytest.raises(CatalogError, match="no groups"):
        parse_catalog(document)


def test_a_duplicated_parameter_id_is_refused() -> None:
    """Resolution is global, so two definitions make every marker naming it ambiguous."""
    document = _minimal_document()
    param = {"id": "xx-01_odp.01", "label": "frequency"}
    document["catalog"]["groups"][0]["controls"] = [
        {"id": "xx-1", "title": "One", "params": [param]},
        {"id": "xx-2", "title": "Two", "params": [param]},
    ]

    with pytest.raises(CatalogError, match="defined more than once"):
        parse_catalog(document)


def test_unparseable_json_names_the_file(tmp_path: Path) -> None:
    from warrant.ingest.catalog import load_catalog

    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(CatalogError, match="not valid JSON"):
        load_catalog(path)
