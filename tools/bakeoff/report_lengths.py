#!/usr/bin/env python3
"""Report how much of the corpus each candidate model would truncate.

Most small sentence-transformers accept 512 tokens and silently drop the rest. A model
comparison that never measures this is comparing models on a corpus it has not established
they can read, and the case for a long-context candidate stays an assumption instead of a
number.

Needs the tokenizers, so run it with the bake-off harness environment:

    cd ../embed-bakeoff
    uv run --extra models python ../warrant/tools/bakeoff/report_lengths.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Tokenizer and window per candidate. The window is the model's configured
# max_seq_length rather than its architectural limit: it is the former that truncates.
CANDIDATES = [
    ("MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", 256),
    ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", 512),
    ("bge-base-en-v1.5", "BAAI/bge-base-en-v1.5", 512),
    ("nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1.5", 8192),
]


def percentile(values: list[int], fraction: float) -> int:
    return values[min(len(values) - 1, int(len(values) * fraction))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--corpus", type=Path, default=here / "corpus.jsonl")
    parser.add_argument("--output", type=Path, default=here / "lengths.json")
    args = parser.parse_args(argv)

    # Imported here rather than at module scope, and not a project dependency: the model
    # backend is installed ad hoc to re-derive these numbers, never by the running system.
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    texts = [
        json.loads(line)["text"] for line in args.corpus.read_text(encoding="utf-8").splitlines()
    ]
    report: dict[str, object] = {"corpus": str(args.corpus), "documents": len(texts)}
    rows = []

    for label, name, window in CANDIDATES:
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
        lengths = sorted(len(tokenizer.encode(t, add_special_tokens=True)) for t in texts)
        over = sum(1 for n in lengths if n > window)
        rows.append(
            {
                "label": label,
                "window": window,
                "median": percentile(lengths, 0.5),
                "p90": percentile(lengths, 0.9),
                "p99": percentile(lengths, 0.99),
                "max": lengths[-1],
                "over_window": over,
                "over_window_pct": round(over / len(lengths) * 100, 1),
            }
        )
        row = rows[-1]
        print(
            f"{label:<24} window={window:<5} median={row['median']:<5} p90={row['p90']:<5} "
            f"p99={row['p99']:<5} max={row['max']:<5} truncated={over} ({row['over_window_pct']}%)"
        )

    report["candidates"] = rows
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
