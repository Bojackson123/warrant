# The canonical control identifier

*Status: decided. This is the form stored in `chunks.control_id`, matched on, and used as the
key of any question set written against this corpus.*

## The answer

**The catalog's own OSCAL `id`: lower case, no zero padding, enhancements written with a dot.**

```
ac-2        a base control
ac-2.3      an enhancement of it
```

**Citations rendered to a reader use the `label` instead** — `AC-2`, `AC-2(3)`. Storage and
matching use the canonical form; display uses the label. They are separate fields with separate
names, and mixing them is the defect this decision exists to prevent.

Implemented in [`src/warrant/ingest/control_ids.py`](../../src/warrant/ingest/control_ids.py).

## Why a decision is needed at all

The catalog publishes four identifiers for every control, and two of them differ in a way that
reads as identical to a person:

| | base control | enhancement | role |
|---|---|---|---|
| OSCAL `id` | `ac-2` | `ac-2.3` | **canonical** |
| `label` | `AC-2` | `AC-2(3)` | **citation** |
| zero-padded label | `AC-02` | `AC-02(03)` | alias |
| `sort-id` | `ac-02` | `ac-02.03` | alias |

Release 5.1.1 introduced the zero-padded forms while keeping the conventional label for
backwards compatibility. Nothing about that is an error, and that is exactly the problem: a
question set keyed to `AC-2` that silently stops matching `ac-02` does not fail, it scores zero
and looks like a retrieval result. A corrupted evaluation is worse than no evaluation, because
it is believed.

## Why the `id` and not one of the others

The document already treats it as its primary key.

- Part identifiers are the control id with a suffix — `ac-2.3_smt.a`, `ac-2_obj` — using the
  **non-padded** form. All 9,798 of them. Choosing a padded canonical form would mean every
  clause identifier in the catalog disagreed with its own control.
- Prose cross-references link to it: `[AC-25](#ac-25)`.
- Parameter identifiers are the one place the padded form is primary — `ac-2` owns
  `ac-02_odp.01` — which is a good reason to keep the padded form addressable and a poor one to
  make it canonical, since no parameter id is derivable from its control's id in either
  direction.

The label is the wrong choice for a different reason: it is the form that gets rendered, so
making it also the form that gets matched removes the distinction this decision is trying to
preserve.

## What a lookup accepts

Any of the four published forms, in any case, is resolved from an index built out of the strings
the catalog itself publishes — 3,961 distinct casefolded keys over 1,196 controls, with no key
naming two controls.

Anything else is parsed structurally: family, then numbers with their padding stripped. That
covers the variants no release publishes and people type anyway — `AC-2.3`, `ac2`, `AC 2 (3)`,
`ac-02(3)`, surrounding whitespace.

Three things it deliberately does **not** do:

- **No near matches.** A parse produces a string, and that string becomes a control only by
  dictionary lookup. `AC-99` parses cleanly and resolves to nothing. `ac-2.14` sits directly
  beside the real `ac-2.13` and still resolves to nothing.
- **A part id is not a control id.** `ac-2.3_smt.a` returns a miss rather than the whole control;
  walking from a clause to its control is a separate, explicit call.
- **A family is not a control.** `AC` returns a miss.

**Withdrawn controls resolve, and report that they are withdrawn.** 182 of the 1,196 are, and
they are excluded from the embedded corpus — but citing one is a stale answer while citing
something that never existed is an invented one, and a lookup that returned a miss for both
would make the two indistinguishable to the citation check.

## For a question set written against this corpus

Key expected answers on the canonical form — `ac-2`, `ac-2.3` — or on any published form, and
resolve through `ControlIndex` before comparing. What must not happen is a set that stores
`AC-2` as a bare string and compares it against a stored `control_id` with `==`.

The forms are mutually derivable in this release: `sort-id` is the id with each component
zero-padded, and the label is the id uppercased with enhancement numbers parenthesised, with
zero exceptions across all 1,196 controls. That regularity is asserted by a test rather than
assumed, so a later release that breaks it fails loudly instead of resolving to a near match —
which is the whole reason resolution is anchored on the published strings first and the
structural parse second.
