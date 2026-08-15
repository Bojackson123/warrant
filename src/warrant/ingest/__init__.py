"""Catalog parsing, parameter resolution, chunking, and the embed-and-store pipeline.

The loader is the floor everything else here stands on: it reads the pinned OSCAL file into
typed controls, parts and parameters, and does nothing else to them. The resolver is the first
thing to change what it read, turning the prose's parameter markers into readable slots, and it
carries a version because everything downstream is a function of the text it produces.
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
from warrant.ingest.parameters import (
    RESOLUTION_VERSION,
    ParameterResolutionError,
    ParameterResolver,
    resolution_fingerprint,
)

__all__ = [
    "ENHANCEMENT_CLASS",
    "RESOLUTION_VERSION",
    "Catalog",
    "CatalogError",
    "Control",
    "Group",
    "Parameter",
    "ParameterResolutionError",
    "ParameterResolver",
    "Part",
    "Selection",
    "get_catalog",
    "load_catalog",
    "parse_catalog",
    "resolution_fingerprint",
]
