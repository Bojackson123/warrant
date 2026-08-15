# Pinned catalog

The corpus is the NIST SP 800-53 Rev 5 control catalog in OSCAL JSON, vendored into this
repository rather than fetched at build or run time. Vendoring is what lets the stack come up
with no network and no credentials; a fetch step would make that promise conditional on
GitHub being reachable.

NIST publishes new releases, so an unpinned catalog means the corpus can change under a green
build and any evaluation scored against it is silently measuring a different document.

**The pin itself lives in `pinned.json`** — release, source commit, content hash, and the
counts a correct copy parses to. Those values appear there and nowhere else, this file
included: two records of the same number is one record and one stale copy. This file is for
what a person needs to know that a machine does not check.

```
make catalog
```

reads the vendored file, verifies it against the pin, and prints what it contains.

The non-minified file is vendored deliberately: it is human-diffable, so a catalog change
shows up as readable prose rather than as one changed line of 5 MB. `.gitattributes` marks it
`-text` so that no checkout can rewrite its line endings — the recorded hash is quoted here,
upstream and in the integrity check, and a platform-dependent hash would be worthless.

## What the file contains

Twenty control families, and about twelve hundred records of which roughly one in six is
withdrawn. Every live control carries four top-level parts: `statement`, `guidance`,
`assessment-objective`, and `assessment-method`. The last two are SP 800-53A assessment
procedures published inside the same file, not control text — they restate the statement in
assessment voice, which roughly triples the length of a control.

Two withdrawn controls (`cp-10.3`, `sc-19`) still carry a `statement` part. Anything deciding
what to ingest has to read the withdrawal property rather than infer status from the presence
of prose.

No parameter in this catalog carries a concrete value. All of them are organisation-defined,
so parameter resolution is always rendering a slot, never substituting a value.

## Reproducing the download

```
curl -sSL -o data/catalog/<file> \
  https://raw.githubusercontent.com/<source_repository>/<source_commit>/<source_path>
sha256sum data/catalog/<file>
```

The four bracketed values are the fields of the same name in `pinned.json`, and the hash
printed should be its `sha256`. Pinning to the commit rather than the release tag matters:
tags can be moved, commits cannot.

## Identifier forms present in the file

Each control carries three `label` props and a `sort-id`, and they do not agree with each
other. Anything matching on identifiers has to choose one and say so.

| Form | Where it lives | Example (base) | Example (enhancement) |
|---|---|---|---|
| Object id | `id` | `ac-2` | `ac-2.3` |
| Human label | `props[label]`, no class | `AC-2` | `AC-2(3)` |
| Zero-padded label | `props[label]` class `zero-padded` | `AC-02` | `AC-02(03)` |
| Assessment label | `props[label]` class `sp800-53a` | `AC-02` | `AC-02(03)` |
| Sort id | `props[sort-id]` | `ac-02` | `ac-02.03` |

Parameter ids use the zero-padded form regardless of the control's own id — control `ac-2`
owns parameter `ac-02_odp.01` — so a resolver cannot derive a parameter id from its control's
id and must index parameters globally. Parameters carry the same split: an `alt-identifier`
property records the pre-renumbering id, so `ac-02_odp.01` was once `ac-2_prm_1`.
