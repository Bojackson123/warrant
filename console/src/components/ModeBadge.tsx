import type { Mode } from "../api/types.ts";

interface Props {
  mode: Mode;
  recordedOn: string | null;
}

// Which path served this answer, stated rather than inferred. Replay means the answer came from a
// recording and carries the day it was made; live means the full pipeline ran just now.
export function ModeBadge({ mode, recordedOn }: Props) {
  const replay = mode === "replay";
  return (
    <span className="inline-flex items-center gap-2 text-xs">
      <span
        className={
          "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-medium " +
          (replay
            ? "bg-sky-100 text-sky-800 dark:bg-sky-500/15 dark:text-sky-200"
            : "bg-emerald-100 text-emerald-800 dark:bg-emerald-500/15 dark:text-emerald-200")
        }
      >
        <span
          aria-hidden="true"
          className={
            "h-1.5 w-1.5 rounded-full " + (replay ? "bg-sky-500" : "bg-emerald-500")
          }
        />
        {replay ? "Replay" : "Live"}
      </span>
      {replay && recordedOn && (
        <span className="text-slate-400 dark:text-slate-500">recorded {recordedOn}</span>
      )}
    </span>
  );
}
