import type { QuestionClass, QuestionView } from "../api/types.ts";

interface Props {
  version: string;
  questions: QuestionView[];
  onPick: (text: string) => void;
}

// The order the groups are shown in, and what each group is for. Traps sit above the out-of-corpus
// questions because they are the ones worth clicking first -- the answer most people expect, and the
// system declining to invent it. Copy lives here rather than coming from the server: the server
// carries what a question is, the console decides how to introduce each class.
const GROUPS: { kind: QuestionClass; heading: string; blurb: string }[] = [
  {
    kind: "answerable",
    heading: "Answerable",
    blurb: "The catalog states an answer, and a good one cites the control that warrants it.",
  },
  {
    kind: "prior_conflict_trap",
    heading: "Prior-conflict traps",
    blurb:
      "The answer almost everyone expects is not what the catalog says. Watch the system decline " +
      "to invent it rather than repeat the remembered number.",
  },
  {
    kind: "out_of_corpus",
    heading: "Outside the catalog",
    blurb: "The catalog does not address these. The honest answer names the boundary, not a guess.",
  },
];

// The recorded questions, grouped by class so a reviewer arriving cold has something to click and
// can see the traps for what they are. Picking one fills the query box rather than asking outright,
// so a reviewer can send it verbatim -- which key-matches a recording -- or edit it first and watch
// the paraphrase decline. The list is provisional and says so.
export function QuestionPicker({ version, questions, onPick }: Props) {
  return (
    <section className="mt-6 rounded-lg border border-slate-200 bg-slate-50/60 p-4 dark:border-slate-700 dark:bg-slate-800/40">
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
          Recorded questions
        </h2>
        <span className="text-xs text-slate-400 dark:text-slate-500">
          Provisional list ({version}) — a set to click through, not a measured one
        </span>
      </div>

      <div className="space-y-4">
        {GROUPS.map((group) => {
          const inGroup = questions.filter((question) => question.class === group.kind);
          if (inGroup.length === 0) return null;

          const isTrap = group.kind === "prior_conflict_trap";

          return (
            <div key={group.kind}>
              <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-200">
                {group.heading}
              </h3>
              <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">{group.blurb}</p>
              <ul className="space-y-1.5">
                {inGroup.map((question) => (
                  <li key={question.id}>
                    <button
                      type="button"
                      onClick={() => onPick(question.text)}
                      className="group flex w-full items-start gap-2 rounded-md border border-transparent px-2 py-1.5 text-left text-sm text-slate-700 transition-colors hover:border-slate-300 hover:bg-white dark:text-slate-200 dark:hover:border-slate-600 dark:hover:bg-slate-800"
                    >
                      {isTrap && (
                        <span className="mt-0.5 shrink-0 rounded bg-amber-100 px-1.5 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide text-amber-800 dark:bg-amber-500/15 dark:text-amber-200">
                          Trap
                        </span>
                      )}
                      <span>
                        {question.text}
                        {isTrap && (
                          <span className="mt-0.5 block text-xs font-normal text-slate-400 dark:text-slate-500">
                            {question.because}
                          </span>
                        )}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </div>
    </section>
  );
}
