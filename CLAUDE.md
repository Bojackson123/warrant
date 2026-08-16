# Warrant — working conventions

## `internal/` is never committed

`internal/` holds private planning material. It belongs in `.gitignore` and must never enter
git history — not in the initial commit, not later.

The shipped artifact is public; the planning vocabulary is not. Anything a reader needs in
order to understand the system belongs in the repo-root `README.md` or in the code itself.

## Comments and commit messages never reference milestones or tickets

No milestone identifiers, no ticket numbers, no "as specified in the plan" — not in code
comments, docstrings, commit messages, PR bodies, or test names.

**Why:** a history threaded with internal identifiers points at documents no reader can open.
It reads as a private conversation happening in a public place, and it ages badly — the
identifier outlives the document it referred to.

**How to apply:** describe what the code does and why, in terms that stand alone.

```
# Bad
# Chunk by control and enhancement per M0-07.
git commit -m "M0-07: add chunker"

# Good
# Chunk on the catalog's own control and enhancement boundaries so identifiers
# come from the source rather than being recovered by heuristic.
git commit -m "Add chunker following catalog control and enhancement structure"
```

The same rule holds in the other direction: do not name the planning documents, their
filenames, or their directory in committed text.

## Commit messages state the feature, not the reasoning

Conventional prefix (`feat:`, `fix:`, `docs:`, `refactor:`), then the change in plain terms.
At most one short body line, and only when it adds a fact the subject cannot carry. No
rationale essays, no restated docstrings, no explanation of what would have gone wrong.

**Why:** the reasoning belongs in the code, where it stays next to what it explains. A commit
message repeating it is a second copy that drifts, and it buries the one thing a `git log`
reader is scanning for — what changed.

```
# Bad
feat: answer from recorded model calls keyed on the whole request

Keying on a subset returns a stale answer after a prompt edit; hashing the
complete request makes that a miss, which the console degrades on and anything
measuring quality fails on.

# Good
feat: add model client with live and replay paths

Records are keyed on a SHA-256 of the complete request.
```

## Related repositories

The embedding bake-off harness is a standalone project in a sibling repository
(`embed-bakeoff`), deliberately agnostic to this one. It knows nothing about OSCAL or
compliance corpora; it takes an already-chunked labelled corpus and ranks embedding models
over it. When its results are used here, record the harness's commit alongside the chosen
model — reproducibility spans both repositories.
