import type { ChunkView, CitationView } from "../api/types.ts";

interface Props {
  selected: CitationView | null;
  chunks: ChunkView[];
}

// The clause a citation rests on. Clicking a citation opens this to the retrieved chunk(s) for that
// control -- and when a citation has no clause to show, this says why rather than sitting blank,
// which is where an invalid citation becomes legible beyond its colour: an invented identifier
// resolves to nothing, and a real control the answer cited without it being retrieved has no text
// here at all.
export function ClausePanel({ selected, chunks }: Props) {
  if (selected === null) {
    return (
      <p className="text-sm text-slate-500 dark:text-slate-400">
        Click a citation in the answer to read the control it rests on.
      </p>
    );
  }

  const matches =
    selected.control_id === null
      ? []
      : chunks.filter((chunk) => chunk.control_id === selected.control_id);

  if (matches.length === 0) {
    const reason = !selected.exists
      ? `${selected.cited_as} does not name a control in the catalog, so there is no clause to read.`
      : `${selected.cited_as} is a real control, but it was not among the clauses retrieved for this ` +
        `question — so the answer cited it without the retrieval supporting it, and there is no ` +
        `retrieved text to show.`;
    return (
      <div className="rounded-md border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-200">
        {reason}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {matches.map((chunk) => (
        <article
          key={chunk.chunk_id}
          className="rounded-md border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-800"
        >
          <header className="mb-2 flex items-baseline justify-between gap-3">
            <h3 className="font-semibold text-slate-900 dark:text-slate-100">
              {chunk.control_label} — {chunk.title}
            </h3>
            <span className="shrink-0 text-xs text-slate-400 dark:text-slate-500">
              rank {chunk.rank} · score {chunk.score.toFixed(3)}
            </span>
          </header>
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-700 dark:text-slate-200">
            {chunk.text}
          </p>
        </article>
      ))}
    </div>
  );
}
