import { type FormEvent } from "react";

interface Props {
  value: string;
  onChange: (question: string) => void;
  pending: boolean;
  onAsk: (question: string) => void;
}

// The query box. Its text is owned by the app so the picker can fill it; this component reads and
// writes it through props rather than holding its own. A blank submission is held back here rather
// than sent for the server to refuse; everything else, including a question with no recording, is a
// real request the server answers or declines.
export function QueryBox({ value, onChange, pending, onAsk }: Props) {
  const submit = (event: FormEvent) => {
    event.preventDefault();
    const trimmed = value.trim();
    if (trimmed && !pending) onAsk(trimmed);
  };

  return (
    <form onSubmit={submit} className="flex gap-2">
      <input
        type="text"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Ask about a control, e.g. how audit records are protected"
        aria-label="Question"
        className="flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm placeholder:text-slate-400 focus:border-slate-400 focus:outline-none focus:ring-2 focus:ring-slate-300 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100 dark:placeholder:text-slate-500"
      />
      <button
        type="submit"
        disabled={pending || value.trim() === ""}
        className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white"
      >
        {pending ? "Asking…" : "Ask"}
      </button>
    </form>
  );
}
