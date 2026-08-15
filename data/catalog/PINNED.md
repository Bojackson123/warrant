# Pinned catalog

The corpus is the NIST SP 800-53 Rev 5 control catalog in OSCAL JSON, vendored into this
repository rather than fetched at build or run time. Vendoring is what lets the stack come up
with no network and no credentials; a fetch step would make that promise conditional on
GitHub being reachable.

NIST publishes new releases, so an unpinned catalog means the corpus can change under a green
build and any evaluation scored against it is silently measuring a different document.

| Field | Value |
|---|---|
| File | `NIST_SP-800-53_rev5_catalog.json` |
| Source repository | `github.com/usnistgov/oscal-content` |
| Release tag | `v1.5.0` (published 2026-05-13) |
| Source commit | `78650f02ad9321bb7b817846f8fbd4f2bcd620de` |
| Source path | `nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json` |
| Catalog `metadata.version` | `5.2.0` |
| Catalog `metadata.last-modified` | `2026-05-11T16:01:09.00000-00:00` |
| OSCAL model version | `1.2.2` |
| Size | 10,442,037 bytes |
| SHA-256 | `01f37cf90ea99d92242c936cbfbdebcc338eef1f71454e2acac36cc56e9bc062` |

The non-minified file is vendored deliberately: it is human-diffable, so a catalog change
shows up as readable prose rather than as one changed line of 5 MB.

## What the file contains

| | Count |
|---|---|
| Groups (control families) | 20 |
| Base controls | 324 |
| Control enhancements | 872 |
| Total controls and enhancements | 1,196 |
| Withdrawn | 182 |
| Live (not withdrawn) | 1,014 |
| Parameters | 1,600 |

Every live control carries four top-level parts: `statement`, `guidance`,
`assessment-objective`, and `assessment-method`. The last two are SP 800-53A assessment
procedures published inside the same file, not control text.

## Reproducing the download

```
curl -sSL -o data/catalog/NIST_SP-800-53_rev5_catalog.json \
  https://raw.githubusercontent.com/usnistgov/oscal-content/78650f02ad9321bb7b817846f8fbd4f2bcd620de/nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json
sha256sum data/catalog/NIST_SP-800-53_rev5_catalog.json
```

Pinning to the commit rather than the tag matters: tags can be moved, commits cannot.

## Identifier forms present in the file

Each control carries three `label` props and a `sort-id`, and they do not agree with each
other. Anything matching on identifiers has to choose one and say so.

| Form | Where it lives | Example (base) | Example (enhancement) |
|---|---|---|---|
| Object id | `id` | `ac-2` | `ac-2.3` |
| Human label | `props[label]`, no class | `AC-2` | `AC-2(3)` |
| Zero-padded label | `props[label]` class `zero-padded` | `AC-02` | `AC-02(03)` |
| Sort id | `props[sort-id]` | `ac-02` | `ac-02.03` |

Parameter ids use the zero-padded form regardless of the control's own id — control `ac-2`
owns parameter `ac-02_odp.01` — so a resolver cannot derive a parameter id from its control's
id and must index parameters globally.
