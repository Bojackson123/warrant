import type { CitationView } from "../api/types.ts";

// Why a citation is invalid, in the fewest words that still say which half failed. `exists=false` is
// an invented identifier; a real control that was not retrieved is the prior-knowledge case.
function invalidReason(citation: CitationView): string {
  if (!citation.exists) return "not a control in the catalog";
  return "cited but not retrieved";
}

interface Props {
  citation: CitationView;
  selected: boolean;
  onSelect: (citation: CitationView) => void;
}

// One inline citation. A valid one is quiet; an invalid one is unmistakably marked, because the
// whole point of the project is that a citation which does not stand up looks different from one
// that does. Both are clickable: clicking reads the clause, or is told why there is none.
export function CitationChip({ citation, selected, onSelect }: Props) {
  const valid = citation.valid;

  const base =
    "inline-flex items-baseline gap-1 rounded px-1.5 py-0.5 text-sm font-medium " +
    "cursor-pointer transition-colors focus:outline-none focus-visible:ring-2 " +
    "focus-visible:ring-offset-1 focus-visible:ring-offset-white dark:focus-visible:ring-offset-slate-900";

  const tone = valid
    ? "text-emerald-800 bg-emerald-50 hover:bg-emerald-100 focus-visible:ring-emerald-500 " +
      "dark:text-emerald-200 dark:bg-emerald-500/10 dark:hover:bg-emerald-500/20"
    : "text-amber-900 bg-amber-50 border border-dashed border-amber-500 hover:bg-amber-100 " +
      "focus-visible:ring-amber-500 dark:text-amber-200 dark:bg-amber-500/10 dark:hover:bg-amber-500/20";

  const ring = selected ? " ring-2 ring-slate-400 dark:ring-slate-500" : "";

  return (
    <button
      type="button"
      onClick={() => onSelect(citation)}
      aria-pressed={selected}
      data-valid={valid}
      title={valid ? citation.title ?? undefined : `Invalid citation — ${invalidReason(citation)}`}
      className={base + " " + tone + ring}
    >
      {!valid && (
        <span aria-hidden="true" className="text-amber-600 dark:text-amber-400">
          ⚠
        </span>
      )}
      <span>{citation.cited_as}</span>
      {!valid && <span className="sr-only"> (invalid citation: {invalidReason(citation)})</span>}
    </button>
  );
}
