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

## Related repositories

The embedding bake-off harness is a standalone project in a sibling repository
(`embed-bakeoff`), deliberately agnostic to this one. It knows nothing about OSCAL or
compliance corpora; it takes an already-chunked labelled corpus and ranks embedding models
over it. When its results are used here, record the harness's commit alongside the chosen
model — reproducibility spans both repositories.
