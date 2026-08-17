import type { DeclineView } from "../api/types.ts";

interface Props {
  decline: DeclineView;
}

// The second state, stated plainly: retrieval ran and the clauses below are real, but no recorded
// answer matched this question and none was invented. The server's own wording is shown verbatim.
export function DeclineNotice({ decline }: Props) {
  return (
    <div className="rounded-md border border-slate-300 bg-slate-50 p-4 dark:border-slate-600 dark:bg-slate-800/60">
      <p className="mb-1 text-sm font-semibold text-slate-800 dark:text-slate-100">
        No answer generated
      </p>
      <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-300">
        {decline.detail}
      </p>
    </div>
  );
}
