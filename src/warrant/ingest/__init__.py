"""Catalog parsing, parameter resolution, chunking, and the embed-and-store pipeline.

The loader is the floor everything else here stands on: it reads the pinned OSCAL file into
typed controls, parts and parameters, and does nothing else to them.
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

__all__ = [
    "ENHANCEMENT_CLASS",
    "Catalog",
    "CatalogError",
    "Control",
    "Group",
    "Parameter",
    "Part",
    "Selection",
    "get_catalog",
    "load_catalog",
    "parse_catalog",
]
