"""Catalog parsing, parameter resolution, chunking, and the embed-and-store pipeline.

The loader is the floor everything else here stands on: it reads the pinned OSCAL file into
typed controls, parts and parameters, and does nothing else to them. The resolver is the first
thing to change what it read, turning the prose's parameter markers into readable slots, and it
carries a version because everything downstream is a function of the text it produces. The
control index resolves every published form of an identifier to one canonical id. The chunker
assembles all three into the records the corpus is made of — one per live control and per live
enhancement, so that a chunk's control id is copied from the source rather than recovered from
its text — and carries a version of its own for the same reason the resolver does. The pipeline
is the last step: it embeds those records with the pinned model and writes them, keyed on the
chunker's identifiers so that running it twice leaves one corpus, and records what produced them
so that a corpus built under a superseded pin can be refused rather than searched.
"""

from warrant.ingest.catalog import (
    ENHANCEMENT_CLASS,
    Catalog,
    CatalogError,
    Control,
    Group,
    Parameter,
    Part,
    Selection,
    get_catalog,
    load_catalog,
    parse_catalog,
)
from warrant.ingest.chunker import (
    CHUNKER_VERSION,
    Chunk,
    ChunkingError,
    chunk_catalog,
    chunker_fingerprint,
)
from warrant.ingest.control_ids import (
    ControlIdentity,
    ControlIdError,
    ControlIndex,
    canonical_id,
    control_id_of_part,
    get_control_index,
)
from warrant.ingest.parameters import (
    RESOLUTION_VERSION,
    ParameterResolutionError,
    ParameterResolver,
    resolution_fingerprint,
)
from warrant.ingest.pipeline import (
    CorpusProvenance,
    IngestError,
    IngestReport,
    corpus_fingerprint,
    ingest,
    read_provenance,
)

__all__ = [
    "CHUNKER_VERSION",
    "ENHANCEMENT_CLASS",
    "RESOLUTION_VERSION",
    "Catalog",
    "CatalogError",
    "Chunk",
    "ChunkingError",
    "Control",
    "ControlIdError",
    "ControlIdentity",
    "ControlIndex",
    "CorpusProvenance",
    "Group",
    "IngestError",
    "IngestReport",
    "Parameter",
    "ParameterResolutionError",
    "ParameterResolver",
    "Part",
    "Selection",
    "canonical_id",
    "chunk_catalog",
    "chunker_fingerprint",
    "control_id_of_part",
    "corpus_fingerprint",
    "get_catalog",
    "get_control_index",
    "ingest",
    "load_catalog",
    "parse_catalog",
    "read_provenance",
    "resolution_fingerprint",
]
