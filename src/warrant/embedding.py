"""Loading the pinned embedding model, and turning text into vectors with it.

    python -m warrant.embedding

run as a command, fetches the pinned weights into the local cache. That is the only thing in this
project allowed to touch the network, and it is separated from everything else on purpose: the
promise is that a reviewer with no API key and no connection can ask an arbitrary question and
have it embedded for real, and a model that downloads itself on first use quietly breaks that.

Kept at the top level, beside `embedder_config`, for the reason that module gives: the ingest
pipeline and the retrieval path need the same encoder, and neither should have to import the
other — or a storage module — to embed a string.

**Loading is offline and pinned to a revision.** Both matter, and for different reasons.
`local_files_only=True` is what turns a missing model into an error rather than a several-hundred
megabyte download nobody asked for. `HF_HUB_OFFLINE` is what stops the *slow* failure documented
in `docs/decisions/embedder.md`: with the network broken rather than absent, the library issues a
`HEAD` per optional config file before falling back to the cache, and each one retries with
backoff — so the process runs, eventually, after minutes of invisible timeouts. That looks like
slowness rather than like misconfiguration, which is what makes it worth removing explicitly.

**Documents and queries are embedded differently**, because the pinned model is asymmetric: its
card specifies an instruction prefix on the query side and none on the document side. The prefixes
live in `data/embedder.json` and are applied here, so no caller has to remember which side it is
on. Embedding a question with the document path would silently cost retrieval quality with nothing
to notice.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from warrant.embedder_config import EmbedderConfig, get_embedder_config
from warrant.settings import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Called with the number of texts embedded so far and the total, after each batch. A callback
# rather than printing, so that the module that owns the output decides what a progress line
# looks like and a library caller can pass nothing.
ProgressCallback = Callable[[int, int], None]

# float32 throughout: it is what the model produces, what pgvector's `vector` type stores, and
# what the storage arithmetic in `data/embedder.json` was computed against. Stated once here so
# nothing downstream has to guess whether a stored vector is the model's output or a narrowing of
# it.
VECTOR_DTYPE = np.float32

# How the fetch command re-enters this module to run its check in a process that has downloaded
# nothing. Not a documented option: there is nothing a person would want it for that `make model`
# does not already do.
_VERIFY_FLAG = "--verify-offline"


class EmbedderError(Exception):
    """The pinned model could not be loaded, or is not the model that was pinned."""


class Encoder(Protocol):
    """What the ingest pipeline and the retrieval path need from an embedding model.

    A protocol rather than the class below, so that the pipeline's tests can exercise storage,
    idempotence and the transaction boundary against a real Postgres without half a gigabyte of
    weights. The seam is narrow deliberately: anything wider would start to be a place where a
    test double and the real model could differ in ways that matter.
    """

    @property
    def dimensions(self) -> int:
        """Width of the vectors this encoder produces."""
        ...

    def embed_documents(
        self,
        texts: Sequence[str],
        progress: ProgressCallback | None = None,
    ) -> np.ndarray:
        """Embed corpus text, as an `(len(texts), dimensions)` float32 array."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question, as a `(dimensions,)` float32 array."""
        ...


class Embedder:
    """The pinned sentence-transformers model, with its prefixes and normalisation applied."""

    def __init__(self, model: SentenceTransformer, config: EmbedderConfig, batch_size: int) -> None:
        self._model = model
        self._config = config
        self._batch_size = batch_size

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    @property
    def config(self) -> EmbedderConfig:
        """The pin this encoder was loaded from, for whoever is recording what produced a vector."""
        return self._config

    def embed_documents(
        self,
        texts: Sequence[str],
        progress: ProgressCallback | None = None,
    ) -> np.ndarray:
        """Embed corpus text in batches, reporting after each one.

        Batched here rather than by handing the whole list to the library, which offers no
        per-batch callback. It costs some throughput — the library groups by length to minimise
        padding, and slicing in document order gives that up — and buys the only progress output
        there is for a run that takes minutes. Deterministic either way: the slices are a function
        of the input order, so two runs over the same corpus batch identically.
        """
        if not texts:
            return np.zeros((0, self.dimensions), dtype=VECTOR_DTYPE)

        prefix = self._config.document_prefix
        batches: list[np.ndarray] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            batches.append(self._encode([prefix + text for text in batch]))

            if progress is not None:
                progress(min(start + self._batch_size, len(texts)), len(texts))

        return np.vstack(batches)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one question, alone.

        Alone rather than as a batch of one within something larger, because padding to a longer
        neighbour can move the last bits of a result. A query embedded on the request path should
        be the same vector as the same query embedded anywhere else.
        """
        return self._encode([self._config.query_prefix + text])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=len(texts),
            convert_to_numpy=True,
            normalize_embeddings=self._config.normalize,
            show_progress_bar=False,
        )

        return np.asarray(vectors, dtype=VECTOR_DTYPE)


def load_embedder(config: EmbedderConfig | None = None) -> Embedder:
    """Load the pinned model from the local cache, verifying it is the one that was pinned.

    The config is a parameter so a test can hand this a deliberately wrong pin without editing the
    file the whole project reads.
    """
    pinned = config if config is not None else get_embedder_config()
    settings = get_settings()

    model = _load_model(pinned)
    dimensions = model.get_embedding_dimension()
    window = model.max_seq_length

    # Checked at load rather than trusted, because both numbers are written down twice — here in
    # the pin, and in the weights themselves — and the failure from a disagreement is not a crash.
    # A model of the wrong width is caught by the database column; a model of the right width and
    # the wrong window silently truncates more of the corpus than the comparison measured.
    if dimensions != pinned.dimensions:
        raise EmbedderError(
            f"{pinned.name} at revision {pinned.revision} produces {dimensions}-dimensional "
            f"vectors, but the pin in data/embedder.json says {pinned.dimensions}. The stored "
            "corpus, the stored query vectors and every recorded model call downstream were "
            "produced at the pinned width, so this is a re-ingest rather than a mismatch to "
            "work around."
        )

    if window != pinned.max_sequence_length:
        raise EmbedderError(
            f"{pinned.name} at revision {pinned.revision} has a {window}-token window, but the "
            f"pin says {pinned.max_sequence_length}. The window decides how much of a long "
            "control is read at all, so a corpus embedded through a different one is not the "
            "corpus the model was chosen on."
        )

    return Embedder(model, pinned, settings.embed_batch_size)


def fetch_weights(config: EmbedderConfig | None = None) -> Path:
    """Download the pinned weights into the local cache; return the cache directory.

    The one function here that is allowed a network. It loads the model the same way
    `load_embedder` does, minus the offline flags, so what lands in the cache is exactly what the
    loader will later ask for rather than a superset guessed from the repository's file list.
    """
    pinned = config if config is not None else get_embedder_config()

    cache = _prepare_environment(offline=False)

    from sentence_transformers import SentenceTransformer

    SentenceTransformer(
        pinned.name,
        revision=pinned.revision,
        device="cpu",
        cache_folder=str(cache) if cache is not None else None,
    )

    if cache is not None:
        return cache

    # Only when nothing is configured. `cache_folder` above is handed straight to the hub as its
    # `cache_dir`, which puts the weights in that directory itself — not in the `hub`
    # subdirectory that `HF_HUB_CACHE` derives from `HF_HOME`. Returning the derived path would
    # name an empty directory, which matters most in the case the override exists for: a
    # container image built by copying whatever this printed.
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def _load_model(pinned: EmbedderConfig) -> SentenceTransformer:
    cache = _prepare_environment(offline=True)

    # Imported here rather than at module scope. The environment above is read by the hub library
    # when it is first imported, so setting it afterwards would have no effect; and this module is
    # imported by things that only want the protocol, which should not pay for loading torch.
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(
            pinned.name,
            revision=pinned.revision,
            device="cpu",
            local_files_only=True,
            cache_folder=str(cache) if cache is not None else None,
        )
    except OSError as error:
        raise EmbedderError(
            f"The pinned model {pinned.name} at revision {pinned.revision} is not in the local "
            f"cache, and this process will not download it: {error}. Fetch it once with "
            "`make model`. A model that downloads itself on first use would make the first run "
            "on a new machine need a network, which is the property this project promises it "
            "does not."
        ) from error


def _prepare_environment(offline: bool) -> Path | None:
    """Set the hub library's environment before it is imported; return the cache override.

    Returns `None` when no override is configured, which means the machine's ordinary cache —
    where a developer who has used these models before already has them.
    """
    settings = get_settings()
    cache = settings.model_cache_dir

    if cache is not None:
        os.environ["HF_HOME"] = str(cache)

    # Assigned rather than defaulted. `local_files_only` below already makes a download an error;
    # this is about the retry storm on the way to that error, and an ambient variable saying
    # otherwise should not be able to reintroduce it.
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

        # Only offline. Reading cached files is fast enough that a bar for it is noise in a log
        # that also carries the ingest's own progress; a bar for a several-hundred-megabyte
        # download is the one thing the fetch command has to say while it works.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # The tokenizer forks worker processes and warns about it on every load otherwise. Nothing
    # here is fast enough for the parallelism to matter.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    return cache


def main() -> int:
    """Fetch the pinned weights, then confirm they load the way ingest will load them."""
    if sys.argv[1:] == [_VERIFY_FLAG]:
        return _verify_offline()

    pinned = get_embedder_config()

    print(f"Fetching {pinned.name} at revision {pinned.revision}.")

    try:
        cache = fetch_weights(pinned)
    except OSError as error:
        print(
            f"error: could not fetch {pinned.name}: {error}\n"
            "This is the one step that needs a network. Everything else runs from the cache.",
            file=sys.stderr,
        )
        return 1

    # Flushed, because the check below writes to this same stream from another process. Python
    # buffers stdout when it is not a terminal, so a run piped to a log would otherwise report
    # where the weights are only after reporting that they load.
    print(f"Cached in {cache}.", flush=True)

    # Loading it back through the offline path is the point of the check: it confirms that what
    # was fetched is what a run with no network will find, rather than that a download succeeded.
    #
    # In a fresh process, because `HF_HUB_OFFLINE` is read by the hub library once, when it is
    # first imported — and the fetch above has already imported it. Loading here would set the
    # variable too late to do anything, leaving `local_files_only` as the only thing under test
    # and the slow failure this guards against undetectable in the one process guaranteed to
    # have just downloaded everything.
    return subprocess.run(
        [sys.executable, "-m", "warrant.embedding", _VERIFY_FLAG],
        check=False,
    ).returncode


def _verify_offline() -> int:
    """Load the pinned model with no network reachable, and confirm it really was offline."""
    try:
        embedder = load_embedder()
    except EmbedderError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    # Checked rather than assumed. The ordering that makes the flag take effect — nothing imports
    # the hub library before the loader sets the environment — is invisible at the import it
    # depends on, so an import added at module scope later would silently turn this back into the
    # check that cannot fail.
    from huggingface_hub import constants

    if not constants.HF_HUB_OFFLINE:
        print(
            "error: the model loaded, but the hub library was already imported with its network "
            "enabled, so this proved nothing about a machine with no network. Something imports "
            "sentence_transformers or huggingface_hub before load_embedder sets the environment.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Loads offline and produces {embedder.dimensions}-dimensional vectors. "
        "Nothing else in this project downloads anything."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
