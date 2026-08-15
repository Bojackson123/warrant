#!/usr/bin/env python3
"""Turn the pinned OSCAL catalog into a labelled retrieval corpus.

Standard library only, on purpose: this has to run under a bare interpreter so that anyone
can regenerate and inspect the corpus without installing a model backend.

The chunking rule is one document per control and per enhancement, following the catalog's
own structure, so that a document id is a control id and retrieval quality means exactly
"did it find the right control". See README.md in this directory for what this is for and
what it is not.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# The catalog's only interpolation syntax. Spacing around the id varies across the file,
# so the pattern absorbs it rather than assuming one form.
MARKER = re.compile(r"\{\{\s*insert:\s*param,\s*([^\s}]+)\s*\}\}")

# Parts to include in a document. The catalog also carries SP 800-53A assessment objectives
# and methods under every control; those are excluded. They restate the statement in
# assessment voice, which roughly triples the text and pushes most controls past a 512-token
# window while adding little vocabulary a question would match on.
INCLUDED_PARTS = ("statement", "guidance")

# Guards against a malformed catalog sending resolution into a cycle. Nesting observed in
# the pinned release is one level deep; this is an order of magnitude of headroom.
MAX_RESOLUTION_DEPTH = 8


class CatalogError(Exception):
    """Raised when the catalog does not have the shape this script relies on."""


def iter_controls(catalog: dict[str, Any]):
    """Yield every control and enhancement, depth-first, in document order."""

    def walk(control: dict[str, Any]):
        yield control
        for child in control.get("controls", []):
            yield from walk(child)

    for group in catalog.get("groups", []):
        for control in group.get("controls", []):
            yield from walk(control)


def prop(control: dict[str, Any], name: str, cls: str | None = None) -> str | None:
    for entry in control.get("props", []):
        if entry.get("name") == name and entry.get("class") == cls:
            return entry.get("value")
    return None


def is_withdrawn(control: dict[str, Any]) -> bool:
    return any(
        p.get("name") == "status" and p.get("value") == "withdrawn"
        for p in control.get("props", [])
    )


def index_parameters(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index every parameter by id, across all controls.

    Global rather than per-control because parameter ids use the zero-padded identifier form
    while their owning control does not — control `ac-2` owns parameter `ac-02_odp.01` — so
    a parameter id cannot be derived from the control it appears in.
    """
    params: dict[str, dict[str, Any]] = {}
    for control in iter_controls(catalog):
        for param in control.get("params", []):
            params[param["id"]] = param
    return params


def render_parameter(param: dict[str, Any], params: dict[str, dict[str, Any]], depth: int) -> str:
    """Render one parameter as the slot a reader should see.

    No parameter in this catalog carries a concrete value — all 1,600 are organisation-defined
    — so every one of these is a slot rather than a substitution. Rendering it as an explicit
    bracketed slot keeps the prose readable and keeps template braces out of the embedded
    text, where they would be noise in every vector.
    """
    selection = param.get("select")
    if selection:
        choices = [
            resolve_markers(choice, params, depth + 1).strip()
            for choice in selection.get("choice", [])
        ]
        # Absent how-many means one, per the OSCAL specification.
        how_many = "one or more" if selection.get("how-many") == "one-or-more" else "one"
        if choices:
            return f"[selection ({how_many}): {'; '.join(choices)}]"
        return "[organization-defined selection]"

    label = param.get("label")
    if label:
        return f"[{_slot(label)}]"

    guidelines = param.get("guidelines") or []
    for guideline in guidelines:
        if guideline.get("prose"):
            return f"[{_slot(guideline['prose'].rstrip(';. '))}]"

    return "[organization-defined value]"


def _slot(label: str) -> str:
    """Prefix a parameter label with its qualifier, unless it already carries one.

    Two naming conventions coexist in the catalog: most labels are the bare noun phrase
    ("frequency"), but 141 of them spell out "organization-defined" themselves. Prefixing
    unconditionally produces "organization-defined organization-defined personnel or roles".
    """
    if label.lower().startswith("organization-defined"):
        return label
    return f"organization-defined {label}"


def resolve_markers(text: str, params: dict[str, dict[str, Any]], depth: int = 0) -> str:
    """Replace every parameter marker in a string, recursing into selection choices.

    An unknown parameter id is an error rather than a passthrough: leaving the raw marker in
    place would put template braces into the corpus, which is the exact defect this function
    exists to remove, and it would do so silently.
    """
    if depth > MAX_RESOLUTION_DEPTH:
        raise CatalogError(f"parameter resolution exceeded {MAX_RESOLUTION_DEPTH} levels")

    def replace(match: re.Match[str]) -> str:
        param_id = match.group(1)
        param = params.get(param_id)
        if param is None:
            raise CatalogError(f"marker references unknown parameter id {param_id!r}")
        return render_parameter(param, params, depth)

    return MARKER.sub(replace, text)


def collect_prose(parts: list[dict[str, Any]] | None, params: dict[str, dict[str, Any]]) -> list[str]:
    """Flatten a part subtree into prose lines, prefixed by their list labels."""
    lines: list[str] = []
    for part in parts or []:
        prose = part.get("prose")
        if prose:
            label = prop(part, "label")
            resolved = resolve_markers(prose, params)
            lines.append(f"{label} {resolved}" if label else resolved)
        lines.extend(collect_prose(part.get("parts"), params))
    return lines


def build_text(control: dict[str, Any], params: dict[str, dict[str, Any]]) -> str:
    """Assemble the document text for one control.

    Titled sections rather than bare concatenation, so that a chunk rendered back to a
    reviewer reads as the control does in the publication.
    """
    label = prop(control, "label") or control["id"]
    sections = [f"{label} {control['title']}"]

    for part in control.get("parts", []):
        if part.get("name") not in INCLUDED_PARTS:
            continue
        lines = collect_prose([part], params)
        if not lines:
            continue
        heading = "Discussion" if part["name"] == "guidance" else "Control"
        sections.append(heading + ":\n" + "\n".join(lines))

    return "\n\n".join(sections)


def normalise_whitespace(text: str) -> str:
    """Collapse the artefacts substitution leaves behind.

    Marker spacing is inconsistent in the source, so substitution produces doubled spaces and
    spaces before punctuation in places. Left alone these are cosmetic in a diff and real in
    an embedding.
    """
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([;,.:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def build_corpus(catalog: dict[str, Any]) -> tuple[list[dict[str, str]], Counter[str]]:
    params = index_parameters(catalog)
    counts: Counter[str] = Counter()
    documents: list[dict[str, str]] = []

    for control in iter_controls(catalog):
        is_enhancement = control.get("class") == "SP800-53-enhancement"
        counts["enhancements" if is_enhancement else "base_controls"] += 1

        # Withdrawn controls are excluded. Most carry no prose at all, only a link to the
        # control that absorbed them, so embedding them would add documents that cannot be a
        # correct answer to any question but can still be returned instead of one.
        if is_withdrawn(control):
            counts["withdrawn_skipped"] += 1
            continue

        text = normalise_whitespace(build_text(control, params))
        if not text:
            raise CatalogError(f"{control['id']}: produced no text")

        documents.append(
            {
                "id": control["id"],
                "text": text,
                # Ignored by the bake-off harness, which reads only id and text. Present so
                # the file can be read and checked by a person.
                "label": prop(control, "label") or control["id"],
                "title": control["title"],
            }
        )

    counts["parameters"] = len(params)
    counts["parameters_with_values"] = sum(1 for p in params.values() if p.get("values"))
    counts["parameters_selection"] = sum(1 for p in params.values() if p.get("select"))
    counts["documents"] = len(documents)
    return documents, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--catalog",
        type=Path,
        default=here.parents[1] / "data" / "catalog" / "NIST_SP-800-53_rev5_catalog.json",
    )
    parser.add_argument("--output", type=Path, default=here / "corpus.jsonl")
    args = parser.parse_args(argv)

    try:
        raw = json.loads(args.catalog.read_text(encoding="utf-8"))
        documents, counts = build_corpus(raw["catalog"])
    except CatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {args.catalog}: {exc}", file=sys.stderr)
        return 2

    # Written with sorted keys and an explicit newline so that two runs are byte-identical
    # and a regenerated corpus produces an empty diff when nothing changed.
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")

    for key in (
        "base_controls",
        "enhancements",
        "withdrawn_skipped",
        "documents",
        "parameters",
        "parameters_selection",
        "parameters_with_values",
    ):
        print(f"{key:>24}: {counts[key]}", file=sys.stderr)
    print(f"\nWrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
