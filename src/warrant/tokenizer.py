"""The generation model's own tokenizer, and the size of a prompt measured with it.

    python -m warrant.tokenizer

run as a command, fetches the pinned encoding into the local cache. That is the second and last
thing in this project allowed to touch the network, and it is separated from everything else for
the reason `warrant.embedding` separates its own fetch: a library that downloads what it needs on
first use turns "counts a prompt with no network and no API cost" into a claim that happens to
hold on machines where somebody already ran it.

**Why the model's own tokenizer and not an approximate one.** The number this produces is what
gates prompt bloat before a change is merged, and a gate is only worth having if it counts what
the provider bills. A generic tokenizer gives a number that tracks the real one closely enough to
look right and is not it. The alternative -- asking the provider for an official count -- is an
API endpoint needing a key, which would put a per-pull-request check behind a credential and
produce nothing at all for a prompt nobody has recorded.

**Why the encoding is named in the pin rather than derived from the model.** `tiktoken` can map a
model name to an encoding, and that mapping lives in the library and moves with its version. A
prompt counted under a silently repointed encoding is the failure this whole file is arranged
against, so `data/model.json` names the encoding outright and this module asks for exactly that.

**Why counting is here and rendering is in `warrant.prompt`.** The two invalidate different
things. A changed template invalidates every recorded answer and every grade over them; a changed
tokenizer invalidates no stored artefact at all and makes every size previously reported wrong.
Keeping them apart also keeps `tiktoken` off the replay path, which renders a prompt to compute a
key and must not depend on an encoding being cached to do it.
"""

from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from warrant.model_config import ModelConfig, get_model_config
from warrant.settings import get_settings

if TYPE_CHECKING:
    from tiktoken import Encoding

# What the fingerprint is taken over. Deliberately mixed: ordinary words, a control identifier in
# its published form, a resolved parameter slot, punctuation and a digit group -- the shapes the
# real prompt is made of, so that an encoding that tokenises them differently moves the number.
_FINGERPRINT_SAMPLE = (
    "AC-2(3): Disable accounts within [Assignment: organization-defined time period] "
    "when the accounts have expired, 30 days.\n"
)

# How the fetch command re-enters this module to load what it just downloaded in a process with no
# network. Not a documented option, for the reason `warrant.embedding` gives about its own.
_VERIFY_FLAG = "--verify-offline"


class TokenizerError(Exception):
    """The pinned encoding could not be loaded, or is not the encoding that was pinned."""


class TokenCounter(Protocol):
    """What anything recording or reporting the size of a prompt needs.

    One method wide, so that a caller which only wants a number does not depend on an encoding
    being cached -- and so a test of something else can supply a number without half a megabyte of
    merge ranks. The seam is narrow deliberately: anything wider would start to be somewhere a
    stand-in and the real encoding could differ in ways that matter.
    """

    def count(self, text: str) -> int:
        """How many tokens this text is."""
        ...


class Tokenizer:
    """The pinned encoding, and the one thing this project asks of it."""

    def __init__(self, encoding: Encoding) -> None:
        self._encoding = encoding

    @property
    def name(self) -> str:
        """The encoding's own name for itself, which is what the manifest records."""
        return self._encoding.name

    @property
    def vocabulary_size(self) -> int:
        """How many tokens this encoding can represent at all."""
        return self._encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        """The tokens this text becomes. Exposed for the fingerprint, which needs the ids."""
        return self._encoding.encode(text)

    def count(self, text: str) -> int:
        """How many tokens this text is, as the provider would count it.

        Takes an already-rendered string rather than the pieces of one. There is exactly one
        renderer, in `warrant.prompt`, and a counting path that assembled its own input would be a
        second one -- counting a prompt that differs from the prompt actually sent.
        """
        return len(self.encode(text))


def load_tokenizer(config: ModelConfig | None = None) -> Tokenizer:
    """Load the pinned encoding from the local cache, and refuse to download it.

    The config is a parameter for the reason `load_embedder`'s is: it is how a test hands this a
    deliberately different pin without editing the file the whole project reads.
    """
    pinned = config if config is not None else get_model_config()
    cache = _prepare_cache()

    # Checked before `tiktoken` is asked for anything, because `tiktoken` has no offline switch:
    # a cache miss is a download, silently, on any machine that happens to have a connection. This
    # is a guard rather than a proof -- it establishes that a fetch has run here, not that this
    # particular encoding is the file in it -- and the proof is `_verify_offline`, which loads with
    # sockets refused.
    if not cache.is_dir() or not any(cache.iterdir()):
        raise TokenizerError(
            f"The pinned encoding {pinned.tokenizer.encoding} is not in the local cache at "
            f"{cache}, and this process will not download it. Fetch it once with "
            "`make tokenizer`. Counting a prompt is meant to need no network, and an encoding "
            "that fetches itself on first use would make that true only of machines where "
            "somebody had already run it."
        )

    return Tokenizer(_encoding(pinned))


def fetch_encoding(config: ModelConfig | None = None) -> Path:
    """Download the pinned encoding into the local cache; return the cache directory.

    The one function here allowed a network. It asks for the encoding exactly the way
    `load_tokenizer` will, so what lands in the cache is what the loader later looks for.
    """
    pinned = config if config is not None else get_model_config()
    cache = _prepare_cache()
    cache.mkdir(parents=True, exist_ok=True)

    _encoding(pinned)

    return cache


def tokenizer_fingerprint(config: ModelConfig | None = None) -> str:
    """SHA-256 of what the pinned encoding makes of a fixed sample.

    A behavioural number rather than a version label, and for the reason the chunker's fingerprint
    is one: the encoding's name is a claim maintained by whoever repoints it, and this is what
    notices that the name stayed still while the token boundaries moved. That change stores nothing
    and invalidates nothing -- it simply makes every size this project has ever reported wrong,
    which is a failure with no artefact left behind to reveal it.

    The vocabulary size is folded in beside the ids. Two encodings can agree on a short sample and
    differ in how much they can represent at all, and the count is cheap to include.
    """
    tokenizer = load_tokenizer(config)
    ids = tokenizer.encode(_FINGERPRINT_SAMPLE)

    digest = hashlib.sha256()
    digest.update(f"{tokenizer.name}\x1e{tokenizer.vocabulary_size}\x1e".encode())
    digest.update(",".join(str(token) for token in ids).encode("utf-8"))

    return digest.hexdigest()


def _encoding(pinned: ModelConfig) -> Encoding:
    """Ask `tiktoken` for the encoding the pin names, and say so plainly when it cannot."""
    # Imported here rather than at module scope so that importing this module -- which
    # `warrant.manifest` does, and which therefore everything does -- costs nothing on a path that
    # never counts anything.
    import tiktoken

    try:
        return tiktoken.get_encoding(pinned.tokenizer.encoding)
    except ValueError as error:
        raise TokenizerError(
            f"`tiktoken` does not know an encoding called {pinned.tokenizer.encoding!r}, which is "
            f"what data/model.json pins for {pinned.version}: {error}"
        ) from error
    except OSError as error:
        raise TokenizerError(
            f"The pinned encoding {pinned.tokenizer.encoding} is not cached and could not be "
            f"fetched: {error}. `make tokenizer` fetches it once."
        ) from error


def _prepare_cache() -> Path:
    """Point `tiktoken` at this project's cache, and return where that is.

    Set here rather than left to the library's default, which is a directory under the system
    temporary path -- swept on a schedule nobody controls, so "runs once per machine" would be true
    until it silently was not.

    Read by `tiktoken` at each cache access rather than at import, so unlike the embedding model's
    environment there is no ordering hazard to protect.
    """
    cache = get_settings().tokenizer_cache_dir
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)

    return cache


def main() -> int:
    """Fetch the pinned encoding, then confirm it loads the way a counting path will load it."""
    if sys.argv[1:] == [_VERIFY_FLAG]:
        return _verify_offline()

    pinned = get_model_config()

    print(f"Fetching the {pinned.tokenizer.encoding} encoding for {pinned.version}.")

    try:
        cache = fetch_encoding(pinned)
    except (TokenizerError, OSError) as error:
        print(
            f"error: could not fetch the {pinned.tokenizer.encoding} encoding: {error}\n"
            "This is one of the two steps that need a network. Everything else runs from the "
            "cache.",
            file=sys.stderr,
        )
        return 1

    # Flushed, because the check below writes to this same stream from another process.
    print(f"Cached in {cache}.", flush=True)

    # In a fresh process, so that the load below is a load and not a lookup of something this one
    # already holds in memory. `tiktoken` caches constructed encodings per process, which would
    # otherwise make the offline check pass on the strength of the fetch that preceded it.
    return subprocess.run(
        [sys.executable, "-m", "warrant.tokenizer", _VERIFY_FLAG],
        check=False,
    ).returncode


def _verify_offline() -> int:
    """Load the pinned encoding with every socket refused, and count something with it."""
    real_socket = socket.socket

    def refuse(*arguments: object, **keywords: object) -> socket.socket:
        raise AssertionError(
            "loading the pinned encoding opened a socket, so it was not read from the cache"
        )

    socket.socket = refuse  # type: ignore[assignment]

    try:
        tokenizer = load_tokenizer()
        counted = tokenizer.count(_FINGERPRINT_SAMPLE)
    except (AssertionError, TokenizerError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        socket.socket = real_socket  # type: ignore[assignment]

    print(
        f"Loads with no network and counts that sample at {counted} tokens. Sizing a prompt "
        "costs nothing and reaches nothing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
