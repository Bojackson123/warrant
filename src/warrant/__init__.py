"""Warrant — retrieval question answering over the NIST SP 800-53 catalog.

The submodules are the seams the system is built along: `ingest` turns the catalog into
stored chunks, `retrieval` finds them again, `fixtures` records and replays model calls,
and `api` serves the result.
"""

__version__ = "0.1.0"
