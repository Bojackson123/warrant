"""Recorded model exchanges and the replay path that serves them.

The record of what they were produced from is `warrant.manifest`, one level up: it covers the
catalog, the resolver, the chunker and the embedding model as well as the prompt, and none of
those should have to import the replay package to find out what they are pinned to.

Three modules, and the split is the argument:

- `request` -- what a call is made of, and the SHA-256 of *all* of it that a recording is filed
  under. Keying on a subset produces a hit with stale token counts after a prompt-template edit,
  which is a green build over a wrong answer.
- `store` -- what a recording holds, and the one-question interface it is read through. Exact key
  or nothing; there is no nearest match, because serving the nearest recorded answer to an unasked
  question is the failure this project exists to detect.
- `client` -- live and replay behind one `Protocol`, and the two opposite things a miss means. The
  console degrades to showing retrieval without an answer; anything measuring quality fails the
  run. A miss is never representable as a refusal.

Absent from this package is where fixtures live on disk. That is storage, and the client depends
on an interface so it does not depend on a layout.
"""
