"""Settings behaviour, centred on the property the whole replay story rests on."""

from __future__ import annotations

from pathlib import Path

import pytest

from warrant.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run each test as if on a machine that has never configured this project.

    Two things would otherwise leak in: a `WARRANT_*` variable exported in the developer's
    shell, and a `.env` file sitting in the repository root during live-mode work. Changing
    directory to an empty temporary path removes the second.
    """
    monkeypatch.delenv("WARRANT_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("WARRANT_DATABASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()


def test_constructs_with_nothing_set() -> None:
    """No key, no database, no env file: an object, not an exception."""
    settings = Settings()

    assert settings.model_api_key is None
    assert settings.database_url.startswith("postgresql://")


def test_absent_key_means_replay() -> None:
    assert Settings().mode == "replay"


def test_present_key_means_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WARRANT_MODEL_API_KEY", "sk-not-a-real-key")

    assert Settings().mode == "live"


def test_empty_key_still_means_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exported-but-blank variable is the common shape of "no key", not of a key."""
    monkeypatch.setenv("WARRANT_MODEL_API_KEY", "")

    assert Settings().mode == "replay"


def test_environment_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WARRANT_DATABASE_URL", "postgresql://elsewhere:5432/other")

    assert Settings().database_url == "postgresql://elsewhere:5432/other"


def test_default_paths_do_not_depend_on_the_working_directory() -> None:
    """Defaults are anchored to the package, so running from anywhere resolves the same."""
    settings = Settings()

    assert settings.catalog_path.is_absolute()
    assert settings.catalog_path.name == "NIST_SP-800-53_rev5_catalog.json"
    assert settings.embedder_config_path.name == "embedder.json"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
