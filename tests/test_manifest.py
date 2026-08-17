"""The record of what this build's inputs are, and the check that refuses when one has moved.

Every mutation below is the same shape: change one input, run the check, and assert it fails
naming that input rather than failing generally. That is the ticket the manifest exists to close —
a green build over a corpus and a set of recorded calls that were made from something else.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from warrant import manifest as manifest_module
from warrant import manifest_check
from warrant.catalog_pin import CatalogPin, get_catalog_pin
from warrant.embedder_config import EmbedderConfig, get_embedder_config
from warrant.manifest import (
    ENTRY_NAMES,
    Manifest,
    ManifestEntry,
    ManifestError,
    ManifestMismatchError,
    Observation,
    check,
    get_manifest,
    load_manifest,
    observe,
    verify_manifest,
    write_manifest,
)
from warrant.settings import get_settings


@pytest.fixture
def manifest() -> Manifest:
    return get_manifest()


@pytest.fixture
def pin() -> CatalogPin:
    return get_catalog_pin()


@pytest.fixture
def config() -> EmbedderConfig:
    return get_embedder_config()


@pytest.fixture
def manifest_path() -> Path:
    return get_settings().manifest_path


@pytest.fixture(scope="module")
def observations() -> dict[str, Observation]:
    """What this build's inputs actually are, computed once.

    Both fingerprints walk the whole catalog, so recomputing them per test would be most of this
    file's runtime. Handed out as a copy below, so no test can leave a mutation for the next one.
    """
    return dict(observe())


@pytest.fixture
def observed(observations: dict[str, Observation]) -> dict[str, Observation]:
    return dict(observations)


@pytest.fixture
def committed_bytes(manifest_path: Path) -> bytes:
    """The manifest exactly as committed, read and never written."""
    return manifest_path.read_bytes()


@pytest.fixture
def manifest_copy(
    tmp_path: Path, manifest_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """A copy of the committed manifest, read and written by this process in place of the real one.

    Several tests below run the command that is allowed to write. Redirected to a copy rather than
    restored afterwards, because restoring depends on teardown actually running: an interrupted run
    would leave the repository's own manifest holding a digest a test made up, which is both a
    corrupted copy of the artefact under test and a failure the next `make manifest` would report
    as a real one.
    """
    copy = tmp_path / "manifest.json"
    shutil.copyfile(manifest_path, copy)
    monkeypatch.setattr(get_settings(), "manifest_path", copy)

    # Read once per process, so the copy is only what a caller sees if the cached read is dropped —
    # on the way out as well, because what is cached at that point came from a path in a temporary
    # directory that is about to stop existing.
    get_manifest.cache_clear()
    yield copy
    get_manifest.cache_clear()


def test_the_committed_manifest_describes_this_build(manifest: Manifest) -> None:
    """The load-bearing assertion: what is recorded is what this build actually produces.

    This is the test that goes red when somebody changes a chunking rule, a rendering rule or the
    pinned model and does not re-record. Everything else in this file checks that it goes red for
    the right reason and says something useful when it does.
    """
    comparisons = verify_manifest(manifest)

    assert {comparison.name for comparison in comparisons} == set(ENTRY_NAMES)
    assert all(comparison.matches for comparison in comparisons)


def test_a_different_catalog_fails_naming_the_catalog(
    tmp_path: Path, manifest: Manifest, pin: CatalogPin
) -> None:
    """One appended byte, which is the smallest change a swapped release could amount to."""
    copy = tmp_path / pin.file
    shutil.copyfile(get_settings().catalog_path, copy)
    copy.write_bytes(copy.read_bytes() + b" ")

    with pytest.raises(ManifestMismatchError) as raised:
        verify_manifest(manifest, catalog_path=copy)

    assert "`catalog` entry" in str(raised.value)


def test_a_changed_pin_fails_even_when_the_file_matches_it(
    tmp_path: Path, manifest: Manifest, pin: CatalogPin
) -> None:
    """A pin edited to point somewhere else, with a catalog that still satisfies its hash.

    The hash alone cannot see this: the bytes on disk are unchanged, and only the record of where
    they came from moved. It still invalidates everything, because the release a corpus was built
    against is part of what a citation claims.
    """
    moved = pin.model_copy(update={"release_tag": "v1.6.0"})

    with pytest.raises(ManifestMismatchError) as raised:
        verify_manifest(manifest, pin=moved)

    message = str(raised.value)
    assert "`catalog` entry" in message
    assert pin.release in message, "the recorded release has to be in the message"


def test_an_announced_chunker_change_fails_on_its_identity(
    monkeypatch: pytest.MonkeyPatch, manifest: Manifest
) -> None:
    """A bumped version with no re-record: the change was declared and nothing was rebuilt."""
    monkeypatch.setattr(manifest_module, "CHUNKER_VERSION", "2")

    with pytest.raises(ManifestMismatchError) as raised:
        verify_manifest(manifest)

    message = str(raised.value)
    assert "`chunker` entry" in message
    assert "'chunker 1'" in message and "'chunker 2'" in message
    assert "announced" in message


def test_an_unannounced_chunker_change_fails_on_its_digest(
    monkeypatch: pytest.MonkeyPatch, manifest: Manifest
) -> None:
    """The failure this whole mechanism exists for.

    Chunk assembly moved and the version did not, which is exactly what a table of version strings
    cannot notice. The message has to distinguish it from the announced case above, because the two
    call for different work: one is a re-record, the other is a question about what changed.
    """
    monkeypatch.setattr(manifest_module, "chunker_fingerprint", lambda catalog: "0" * 64)

    with pytest.raises(ManifestMismatchError) as raised:
        verify_manifest(manifest)

    message = str(raised.value)
    assert "`chunker` entry" in message
    assert "'chunker 1' is unchanged" in message
    assert "announced" not in message.replace("announcing", ""), (
        "this is the case a version string cannot catch, and the message must not read as the "
        "announced one"
    )


def test_an_announced_resolution_change_fails_on_its_identity(
    monkeypatch: pytest.MonkeyPatch, manifest: Manifest
) -> None:
    monkeypatch.setattr(manifest_module, "RESOLUTION_VERSION", "2")

    with pytest.raises(ManifestMismatchError) as raised:
        verify_manifest(manifest)

    assert "`parameter_resolution` entry" in str(raised.value)


def test_an_unannounced_resolution_change_fails_on_its_digest(
    monkeypatch: pytest.MonkeyPatch, manifest: Manifest
) -> None:
    """A rendering rule moved without the version moving — the resolver's own docstring's worry."""
    monkeypatch.setattr(manifest_module, "resolution_fingerprint", lambda catalog: "0" * 64)

    with pytest.raises(ManifestMismatchError) as raised:
        verify_manifest(manifest)

    message = str(raised.value)
    assert "`parameter_resolution` entry" in message
    assert "'resolution 1' is unchanged" in message


def test_a_swapped_embedder_fails_naming_everything_downstream(
    manifest: Manifest, config: EmbedderConfig
) -> None:
    """The widest change there is, and the message has to say so rather than mention the corpus."""
    swapped = config.model_copy(update={"revision": "f" * 40})

    with pytest.raises(ManifestMismatchError) as raised:
        verify_manifest(manifest, config=swapped)

    message = str(raised.value)
    assert "`embedder` entry" in message
    assert "corpus vectors" in message
    assert "recorded model answer" in message
    assert "no partial re-record" in message


def test_a_query_prefix_change_moves_the_embedder_digest(
    manifest: Manifest, config: EmbedderConfig
) -> None:
    """Same model, same revision, different instruction prefix.

    The identity string does not move, because it names the model. The vectors do move, because an
    asymmetric model embeds a differently prefixed query differently — so this has to be caught by
    the digest or not at all.
    """
    tweaked = config.model_copy(update={"query_prefix": "Find: "})

    with pytest.raises(ManifestMismatchError) as raised:
        verify_manifest(manifest, config=tweaked)

    message = str(raised.value)
    assert "`embedder` entry" in message
    assert f"'{config.name} @ {config.revision[:8]}' is unchanged" in message


def test_the_corpus_expectations_do_not_move_the_embedder_digest(
    manifest: Manifest, config: EmbedderConfig
) -> None:
    """Those two fields describe a corpus rather than a model, and are excluded on purpose."""
    tweaked = config.model_copy(update={"expected_chunks": 1, "expected_corpus_bytes": 1})

    assert all(comparison.matches for comparison in verify_manifest(manifest, config=tweaked))


def test_the_unbuilt_slots_exist_and_are_declared_absent(manifest: Manifest) -> None:
    """Named before they are built, so that building one is a regeneration rather than a surprise.

    A mutation test for the judge prompt arrives with the judge prompt; there is nothing to edit
    yet. What can be asserted now is that the slot is present, says what it will cover, and already
    declares what changing it will cost.
    """
    for name in ("judge_prompt",):
        entry = manifest.entries[name]

        assert not entry.built, f"{name} is recorded as built"
        assert entry.covers.strip(), f"{name} does not say what it covers"
        assert entry.invalidates, f"{name} does not say what changing it costs"


def test_an_input_that_starts_existing_is_a_mismatch(
    manifest: Manifest, observed: dict[str, Observation]
) -> None:
    """Filling a slot without re-recording fails, which is how a new input announces itself."""
    observed["judge_prompt"] = Observation("judge prompt 1", "0" * 64)

    with pytest.raises(ManifestMismatchError) as raised:
        check(manifest, observed)

    message = str(raised.value)
    assert "`judge_prompt` entry" in message
    assert "It exists now, so its cost applies now" in message


def test_an_input_that_stops_existing_is_a_mismatch(
    manifest: Manifest, observed: dict[str, Observation]
) -> None:
    """An input going away invalidates what was made with it exactly as a changed one does."""
    observed["chunker"] = Observation(None, None)

    with pytest.raises(ManifestMismatchError) as raised:
        check(manifest, observed)

    assert "does not produce it at all" in str(raised.value)


def test_the_costliest_difference_is_reported_first(
    manifest: Manifest, observed: dict[str, Observation]
) -> None:
    """Two inputs moved; the one that invalidates everything is the one a reader is shown."""
    observed["prompt_template"] = Observation("prompt template 1", "0" * 64)
    observed["embedder"] = Observation("something/else @ 00000000", "1" * 64)

    with pytest.raises(ManifestMismatchError) as raised:
        check(manifest, observed)

    assert "`embedder` entry" in str(raised.value)


def test_the_costliest_kind_outranks_the_larger_count() -> None:
    """Ranked by the worst thing an entry destroys, not by how many things it destroys.

    Counting them puts the entry that destroys the corpus second, behind one that destroys two
    cheaper things — and the corpus is the change with nothing downstream of it left standing.
    """
    ranked = Manifest(
        entries={
            "wide": ManifestEntry(covers="Two cheap ones.", invalidates=("generation", "judge")),
            "deep": ManifestEntry(covers="One expensive one.", invalidates=("corpus",)),
        }
    )
    moved = {name: Observation("moved", "0" * 64) for name in ranked.entries}

    with pytest.raises(ManifestMismatchError) as raised:
        check(ranked, moved)

    assert "`deep` entry" in str(raised.value)


def test_reordering_the_file_does_not_change_which_failure_is_reported(
    tmp_path: Path, manifest_path: Path, manifest: Manifest, observed: dict[str, Observation]
) -> None:
    """Four entries invalidate the same four things, so something has to break the tie.

    Whatever breaks it must not be the order the entries happen to sit in, or moving one up while
    editing the file changes which failure a reader is shown and nothing says it did.
    """
    for name in ("catalog", "parameter_resolution", "chunker", "embedder"):
        observed[name] = Observation("something/else @ 00000000", "0" * 64)

    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["entries"] = dict(reversed(list(document["entries"].items())))

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ManifestMismatchError) as as_committed:
        check(manifest, observed)

    with pytest.raises(ManifestMismatchError) as reordered:
        check(load_manifest(path), observed)

    assert str(as_committed.value) == str(reordered.value)


def test_an_input_nothing_observes_is_refused(
    manifest: Manifest, observed: dict[str, Observation]
) -> None:
    """An entry is added by hand in two places, and this is the half the file cannot see.

    `load_manifest` catches the file naming an input no code computes. Nothing catches the reverse
    — the file and `ENTRY_NAMES` both updated, the observation forgotten — and without this the
    entry is skipped by an indexing error rather than reported.
    """
    del observed["chunker"]

    with pytest.raises(ManifestError, match="no check looks at") as raised:
        check(manifest, observed)

    assert "chunker" in str(raised.value)


def test_costs_differ_and_are_recorded_in_the_file(manifest: Manifest) -> None:
    """The differing-cost note as data rather than as prose that can rot away from it."""
    embedder = set(manifest.entries["embedder"].invalidates)
    template = set(manifest.entries["prompt_template"].invalidates)

    assert template < embedder, "a prompt change must cost strictly less than an embedder swap"
    assert "corpus" in embedder and "corpus" not in template
    assert "no partial re-record" in manifest.entries["embedder"].cost()
    assert "no partial re-record" not in manifest.entries["prompt_template"].cost()


def test_the_file_documents_the_differing_cost_in_prose_too(manifest_path: Path) -> None:
    """The ticket's requirement is that the note is in the file, not only in a plan."""
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    comment = " ".join(document["_comment"]).lower()

    assert "embedder change invalidates everything" in comment
    assert "no partial re-record" in comment
    assert "cannot repair" in comment


def test_a_manifest_missing_an_input_is_refused(tmp_path: Path, manifest_path: Path) -> None:
    """An input the file omits is an input nothing checks, which is worse than a loud failure."""
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    del document["entries"]["chunker"]

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ManifestError, match="chunker"):
        load_manifest(path)


def test_a_manifest_naming_an_unknown_input_is_refused(tmp_path: Path, manifest_path: Path) -> None:
    """An entry nothing computes is a record of nothing, and reads as a check that is happening."""
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["entries"]["reranker"] = {"covers": "Something.", "invalidates": []}

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ManifestError, match="reranker"):
        load_manifest(path)


def test_an_unknown_invalidation_is_refused() -> None:
    """A misspelt kind must not become an entry that reads as costing nothing.

    `cost()` builds its sentence out of the kinds it recognises, so an unrecognised one does not
    make the message wrong in an obvious direction — it makes it reassuring, which is the one
    direction this file exists to rule out.
    """
    with pytest.raises(ValidationError, match="corpuss"):
        ManifestEntry(covers="Something.", invalidates=("corpuss",))


def test_a_manifest_naming_an_unknown_cost_is_refused(tmp_path: Path, manifest_path: Path) -> None:
    """The same typo through the file, which is where it would actually be made."""
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["entries"]["chunker"]["invalidates"] = ["corpuss"]

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ManifestError, match="corpuss"):
        load_manifest(path)


def test_a_manifest_that_is_not_json_is_refused(tmp_path: Path) -> None:
    """A trailing comma is what a hand edit to this file most often produces."""
    path = tmp_path / "manifest.json"
    path.write_text('{"entries": {},}', encoding="utf-8")

    with pytest.raises(ManifestError, match="not readable as JSON"):
        load_manifest(path)


def test_a_manifest_entry_missing_its_prose_is_refused(tmp_path: Path, manifest_path: Path) -> None:
    """An entry added by hand without saying what it covers, which is half of what an entry is."""
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    del document["entries"]["chunker"]["covers"]

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ManifestError, match="not usable as a manifest"):
        load_manifest(path)


def test_the_command_refuses_a_broken_manifest_without_a_traceback(
    manifest_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Adding an input is a hand edit, so a malformed file is the ordinary failure here.

    What meets it has to be the line the module promises. A traceback through a JSON parser names
    a position in a file and nothing about what the file is for.
    """
    manifest_copy.write_text('{"entries": {},}', encoding="utf-8")

    assert manifest_check.main([]) == 1
    assert capsys.readouterr().err.startswith("error: ")


def test_writing_refuses_an_entry_it_cannot_place(tmp_path: Path) -> None:
    """What an entry covers and what it costs are judgements, so this can only fill in numbers."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"entries": {}}), encoding="utf-8")

    with pytest.raises(ManifestError, match="no entry named"):
        write_manifest(path, {"chunker": Observation("chunker 1", "0" * 64)})


def test_writing_rewrites_only_the_recorded_values(
    manifest_copy: Path, committed_bytes: bytes
) -> None:
    """Comments, prose and ordering are authored; regeneration leaves all of it alone."""
    before = json.loads(committed_bytes.decode("utf-8"))

    write_manifest(manifest_copy, {"chunker": Observation("chunker 9", "0" * 64)})
    after = json.loads(manifest_copy.read_text(encoding="utf-8"))

    assert after["_comment"] == before["_comment"]
    assert list(after["entries"]) == list(before["entries"])
    assert after["entries"]["chunker"]["covers"] == before["entries"]["chunker"]["covers"]
    assert after["entries"]["chunker"]["identity"] == "chunker 9"
    assert after["entries"]["embedder"] == before["entries"]["embedder"]


def test_writing_is_stable_over_the_committed_file(
    manifest_copy: Path, committed_bytes: bytes
) -> None:
    """Regenerating an unchanged build produces the file byte-for-byte.

    Without this, every regeneration is a whole-file diff and the one-line diff a reviewer is
    supposed to read does not exist. Line endings are asserted separately from the bytes, because
    the platform default translates them and a byte mismatch would not say so.

    Run against a copy, and the copy is what the committed bytes are compared to — the assertion is
    about what regeneration produces, and producing it in the working tree is not part of it.
    """
    assert manifest_check.main(["--write"]) == 0

    written = manifest_copy.read_bytes()

    assert b"\r" not in written, "the file is checked out with line feeds and must be rewritten so"
    assert written == committed_bytes


def test_the_check_never_writes(
    monkeypatch: pytest.MonkeyPatch, manifest_copy: Path, committed_bytes: bytes
) -> None:
    """The whole mechanism: automatic repair would make this a record of whatever is on disk."""
    monkeypatch.setattr(manifest_module, "CHUNKER_VERSION", "2")

    assert manifest_check.main([]) == 1
    assert manifest_copy.read_bytes() == committed_bytes


def test_the_check_passes_on_this_build() -> None:
    assert manifest_check.main([]) == 0


def test_an_unknown_argument_is_refused(manifest_copy: Path, committed_bytes: bytes) -> None:
    """`--wrote` must not be read as a request to write."""
    assert manifest_check.main(["--wrote"]) == 1
    assert manifest_copy.read_bytes() == committed_bytes


def test_manifest_is_read_once() -> None:
    assert get_manifest() is get_manifest()


def test_manifest_cannot_be_edited_in_memory(manifest: Manifest) -> None:
    """A record a caller can change is not a record."""
    with pytest.raises(ValidationError, match="frozen"):
        manifest.entries["chunker"].digest = "0" * 64  # type: ignore[misc]


def test_an_entry_with_nothing_downstream_says_so() -> None:
    entry = ManifestEntry(covers="Something.", invalidates=())

    assert entry.cost() == "Nothing recorded depends on this yet."
