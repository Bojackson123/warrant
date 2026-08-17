"""The recorder's refusals, and the fact that re-running it costs nothing.

Both properties are about what the command does *not* do. It does not record against inputs the
manifest no longer recognises, and the ordering of that refusal matters as much as the refusal
itself -- a check that ran after the first provider call would have paid for a recording it then
declined to trust. And it does not re-record what is already there, because a recorder that
rewrote its own output would put a date change into every committed file on every run.

The command tests at the end are about which stages a given set of flags reaches, and they stub the
stages themselves out. That is the subject: a query stage that reached a database, or an answer
re-record that reached the question vectors, would each be a stage running where its cost was
documented as not applying.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from warrant import manifest as manifest_module
from warrant.embedder_config import EmbedderConfig, get_embedder_config
from warrant.embedding import VECTOR_DTYPE, Encoder, ProgressCallback
from warrant.fixtures import __main__ as record_command
from warrant.fixtures.queries import read_query_vectors
from warrant.fixtures.questions import QuestionSet, load_question_set
from warrant.fixtures.recorder import QUERIES_DIRECTORY, QueryReport, record_query_vectors
from warrant.settings import get_settings

TODAY = date(2026, 8, 17)


class StubEncoder:
    """Vectors without weights, and a record of what it was asked to embed."""

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.embedded: list[str] = []

    def embed_documents(
        self,
        texts: Sequence[str],
        progress: ProgressCallback | None = None,
    ) -> np.ndarray:
        return np.vstack([self.embed_query(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        self.embedded.append(text)

        generator = np.random.default_rng(abs(hash(text)) % (2**32))

        return np.asarray(generator.random(self.dimensions), dtype=VECTOR_DTYPE)


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Run as if on a machine that has never configured this project.

    Load-bearing rather than tidy: a developer with a key exported would otherwise have the
    command reach a provider in the tests that assert it does not.
    """
    monkeypatch.delenv("WARRANT_MODEL_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    yield

    get_settings.cache_clear()


@pytest.fixture
def config() -> EmbedderConfig:
    return get_embedder_config()


@pytest.fixture
def encoder(config: EmbedderConfig) -> StubEncoder:
    return StubEncoder(config.dimensions)


@pytest.fixture
def questions() -> QuestionSet:
    """The committed list, read through the loader that the command uses."""
    return load_question_set(get_settings().fixtures_path / "questions.json")


@pytest.fixture
def query_stage(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Stand in for the query stage, collecting whether it was told to re-embed.

    The database and the embedder are replaced along with it. What the tests using this are about is
    which stages a set of flags reaches and what it asks of them, and a connection that raises when
    it is opened is how the "needs no database" claim gets checked rather than restated.
    """
    forced: list[bool] = []

    def record(
        root: Path,
        questions: QuestionSet,
        encoder: Encoder,
        config: EmbedderConfig,
        today: date,
        force: bool = False,
    ) -> QueryReport:
        forced.append(force)

        return QueryReport(
            questions=len(questions.questions),
            dimensions=config.dimensions,
            vector_bytes=0,
            rewritten=force,
        )

    def refuse() -> None:
        raise AssertionError("the command opened a database connection")

    monkeypatch.setattr(record_command, "connection", refuse)
    monkeypatch.setattr(
        record_command, "load_embedder", lambda pinned: StubEncoder(pinned.dimensions)
    )
    monkeypatch.setattr(record_command, "record_query_vectors", record)

    return forced


def test_every_question_gets_a_vector(
    tmp_path: Path, questions: QuestionSet, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """One recorded vector per question, embedded through the query side of the model."""
    report = record_query_vectors(tmp_path, questions, encoder, config, TODAY)

    assert report.rewritten
    assert report.questions == len(questions.questions)
    assert encoder.embedded == list(questions.texts)

    recorded = read_query_vectors(tmp_path / QUERIES_DIRECTORY, config)

    assert set(recorded.vectors) == set(questions.texts)


def test_the_reported_size_is_the_arithmetic(
    tmp_path: Path, questions: QuestionSet, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """Count times width times four, which is the figure the storage note is written against."""
    report = record_query_vectors(tmp_path, questions, encoder, config, TODAY)

    assert report.vector_bytes == len(questions.questions) * config.dimensions * 4


def test_re_running_embeds_nothing(
    tmp_path: Path, questions: QuestionSet, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """The idempotence property: the same list under the same pin is already recorded.

    A recorder that rewrote its output would change `recorded_on` in a committed file on every
    run, which turns the diff a re-record is supposed to be into noise.
    """
    record_query_vectors(tmp_path, questions, encoder, config, TODAY)
    encoder.embedded.clear()

    report = record_query_vectors(tmp_path, questions, encoder, config, TODAY)

    assert not report.rewritten
    assert encoder.embedded == []


def test_re_running_leaves_the_files_untouched(
    tmp_path: Path, questions: QuestionSet, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """Byte for byte, so a second run produces no diff at all."""
    record_query_vectors(tmp_path, questions, encoder, config, TODAY)

    root = tmp_path / QUERIES_DIRECTORY
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    record_query_vectors(tmp_path, questions, encoder, config, date(2027, 1, 1))

    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_forcing_re_embeds_everything(
    tmp_path: Path, questions: QuestionSet, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """What a scheduled re-record uses: the keys have not moved and the recording is renewed."""
    record_query_vectors(tmp_path, questions, encoder, config, TODAY)
    encoder.embedded.clear()

    report = record_query_vectors(tmp_path, questions, encoder, config, TODAY, force=True)

    assert report.rewritten
    assert encoder.embedded == list(questions.texts)


def test_a_changed_question_list_is_re_embedded(
    tmp_path: Path, questions: QuestionSet, encoder: StubEncoder, config: EmbedderConfig
) -> None:
    """Adding a question is not covered by what is already stored, so the stage runs again."""
    record_query_vectors(tmp_path, questions, encoder, config, TODAY)

    grown = QuestionSet(
        version=questions.version,
        questions=(
            *questions.questions,
            questions.questions[0].model_copy(
                update={"id": "newly-added", "text": "Something nobody has recorded."}
            ),
        ),
    )

    assert record_query_vectors(tmp_path, grown, encoder, config, TODAY).rewritten


@pytest.mark.tokenizer
def test_a_stale_manifest_stops_the_command_before_it_reaches_a_provider(
    cached_encoding: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The refusal this command exists to make, and the ordering it has to make it in.

    A recording made from inputs this build no longer carries is not slightly stale: the control
    text inside its prompt came from a corpus that has been superseded. Checking after the first
    call would mean paying for a recording and then declining to trust it, so the assertion is that
    nothing ever asked for a client at all.

    The encoding is required because the manifest check counts a sample with it. Without it the
    command still refuses, and it refuses for a reason that has nothing to do with the chunker --
    which is a pass this test would not have earned.
    """
    asked: list[object] = []

    monkeypatch.setattr(manifest_module, "CHUNKER_VERSION", "2")
    monkeypatch.setattr(
        record_command,
        "get_model_client",
        lambda *arguments, **keywords: asked.append(arguments),
    )

    assert record_command.main([]) == 1

    assert asked == [], "the command reached for a model client despite a stale manifest"

    error = capsys.readouterr().err
    assert "`chunker` entry" in error
    assert "error: " in error


@pytest.mark.tokenizer
def test_the_query_stage_reaches_no_database(
    cached_encoding: None, query_stage: list[bool]
) -> None:
    """`--queries` is documented as needing no API key and no database, so it must need neither.

    Only the stage that asks the provider reads the corpus, so only a run that will reach that stage
    needs the corpus checked. Verifying it before the early return instead would make the half of
    this command that costs nothing require a running, ingested Postgres.
    """
    assert record_command.main(["--queries"]) == 0
    assert query_stage == [False]


@pytest.mark.tokenizer
def test_forcing_the_answers_leaves_the_recorded_vectors_alone(
    cached_encoding: None, query_stage: list[bool]
) -> None:
    """The flag that renews an answer must not rewrite the vectors its key was computed through.

    Re-embedding the questions on a machine other than the one that recorded them moves their last
    bits, which can reorder two near-tied chunks and change the prompt -- so the monthly re-record
    would orphan every recording it was meant to renew, on the strength of running elsewhere.
    """
    assert record_command.main(["--queries", "--force"]) == 0
    assert query_stage == [False]


@pytest.mark.tokenizer
def test_the_vectors_are_re_embedded_only_when_that_is_what_was_asked_for(
    cached_encoding: None, query_stage: list[bool]
) -> None:
    """The capability still exists, behind a flag that says what it does.

    It is for the change the query stage cannot notice by itself: an embedding library moving
    underneath a model pin that has not.
    """
    assert record_command.main(["--queries", "--force-queries"]) == 0
    assert query_stage == [True]


def test_an_unknown_argument_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    """A misspelt flag does not quietly record everything with the default behaviour."""
    assert record_command.main(["--queries-only"]) == 1

    assert "usage:" in capsys.readouterr().err


def test_the_committed_question_list_covers_all_three_classes(questions: QuestionSet) -> None:
    """The classes exist from the start, so a picker groups without a later redesign.

    The two beyond `answerable` are the ones the project is about: a question the catalog cannot
    answer, and one whose obvious answer the catalog contradicts. A list of only answerable
    questions demonstrates retrieval and demonstrates nothing about warrant.
    """
    for kind in ("answerable", "out_of_corpus", "prior_conflict_trap"):
        assert questions.of_class(kind), f"no {kind} questions are recorded"


def test_every_question_says_why_it_is_in_its_class(questions: QuestionSet) -> None:
    """A trap with no statement of the prior it conflicts with is an untested guess."""
    for question in questions.questions:
        assert question.because.strip(), f"{question.id} does not say why it is where it is"
