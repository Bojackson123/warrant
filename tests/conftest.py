"""Fixtures shared by tests in more than one module.

What lives here is the one-off fetches: a machine that has not run them is a machine that has not
been set up, which is not the same thing as a broken build and must not be reported as one.
"""

from __future__ import annotations

import pytest

from warrant.tokenizer import TokenizerError, load_tokenizer


@pytest.fixture(scope="session")
def cached_encoding() -> None:
    """Skip unless the pinned tokenizer encoding is in the local cache.

    Requested by every test that observes the pinned inputs as a set, because observing them counts
    a sample with this encoding -- so `make tokenizer` is a prerequisite of the manifest check and
    not only of the paths that report a prompt size.

    The same shape `test_embedding` and `test_tokenizer` use for their own caches, and for the same
    reason: an absent cache says nothing about whether the code under test is correct, and a suite
    that fails over it teaches people to read a red run as normal.
    """
    try:
        load_tokenizer()
    except TokenizerError as error:
        pytest.skip(f"the pinned encoding is not cached: {error}")
