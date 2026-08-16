"""The generation pin, centred on what it must refuse.

The pin's job is that every value deciding an answer is both sent and hashed. A value the file
carries and the loader drops satisfies neither while looking exactly like one that satisfies both,
so the tests below are mostly about rejection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from warrant.model_config import ModelConfig, get_model_config, load_model_config
from warrant.settings import get_settings


@pytest.fixture
def document() -> dict[str, object]:
    """The committed pin, parsed, as a mutable starting point."""
    return json.loads(get_settings().model_config_path.read_text(encoding="utf-8"))


def _written(path: Path, document: dict[str, object]) -> ModelConfig:
    path.write_text(json.dumps(document), encoding="utf-8")

    return load_model_config(path)


def test_the_committed_pin_loads(document: dict[str, object]) -> None:
    config = get_model_config()

    assert config.name
    assert config.version.startswith(config.name)
    assert config.sampling.max_output_tokens > 0


def test_comment_blocks_reach_a_reader_and_not_the_model(document: dict[str, object]) -> None:
    """Both of them: the file explains its cost at the top level and inside `sampling`."""
    assert "_comment" in document
    assert "_comment" in document["sampling"]  # type: ignore[operator, index]

    config = get_model_config()

    assert not hasattr(config, "_comment")
    assert not hasattr(config.sampling, "_comment")


def test_an_unknown_sampling_parameter_is_refused(
    document: dict[str, object],
    tmp_path: Path,
) -> None:
    """The failure this pin exists to prevent, arriving through a hand edit.

    A knob written into the file that no field matches would be accepted, dropped, never sent and
    never hashed -- so whoever added it believes it is in effect and in the key, and it is in
    neither. Refusing is the only outcome that tells them.
    """
    document["sampling"]["frequency_penalty"] = 0.5  # type: ignore[index]

    with pytest.raises(ValidationError):
        _written(tmp_path / "model.json", document)


def test_an_unknown_top_level_key_is_refused(
    document: dict[str, object],
    tmp_path: Path,
) -> None:
    document["provider"] = "somewhere"

    with pytest.raises(ValidationError):
        _written(tmp_path / "model.json", document)


def test_a_pin_can_be_read_from_somewhere_else(
    document: dict[str, object],
    tmp_path: Path,
) -> None:
    """How a test keys a request against a deliberately different model."""
    document["version"] = "gpt-4.1-mini-2099-01-01"

    assert _written(tmp_path / "model.json", document).version == "gpt-4.1-mini-2099-01-01"


def test_the_pin_cannot_be_changed_in_memory() -> None:
    """A model swapped at runtime would key its calls against fixtures made with another."""
    with pytest.raises(ValidationError):
        get_model_config().version = "something-else"  # type: ignore[misc]


def test_get_model_config_is_cached() -> None:
    assert get_model_config() is get_model_config()
