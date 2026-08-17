"""Recorded model exchanges and the replay path that serves them.

The record of what they were produced from is `warrant.manifest`, one level up: it covers the
catalog, the resolver, the chunker and the embedding model as well as the prompt, and none of
those should have to import the replay package to find out what they are pinned to.

The modules, and the split is the argument. The first three are the mechanism and the last three
are where it lands on disk; the client depends on the interfaces in `store` rather than on any of
the layout, which is what let the layout be decided second.

- `request` -- what a call is made of, and the SHA-256 of *all* of it that a recording is filed
  under. Keying on a subset produces a hit with stale token counts after a prompt-template edit,
  which is a green build over a wrong answer.
- `store` -- what a recording holds, and the one-question interface it is read through. Exact key
  or nothing; there is no nearest match, because serving the nearest recorded answer to an unasked
  question is the failure this project exists to detect.
- `client` -- live and replay behind one `Protocol`, and the two opposite things a miss means. The
  console degrades to showing retrieval without an answer; anything measuring quality fails the
  run. A miss is never representable as a refusal.
- `disk` -- one file per recording, answers stored as lines so a re-record is a readable diff, and
  a filename that has to agree with the key inside it.
- `queries` -- the recorded question vectors, kept binary and apart, because embedding is
  deterministic on a machine and not across them and the difference reaches every key.
- `questions` and `recorder` -- what gets recorded, and the pass that records it. `__main__` is the
  command; it refuses before it writes anything if a pinned input has moved.
"""
