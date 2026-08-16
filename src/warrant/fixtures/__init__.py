"""Recorded model exchanges and the replay path that serves them.

The record of what they were produced from is `warrant.manifest`, one level up: it covers the
catalog, the resolver, the chunker and the embedding model as well as the prompt, and none of
those should have to import the replay package to find out what they are pinned to.
"""
