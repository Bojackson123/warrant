import { useState } from "react";
import { postAnswer, AnswerError } from "./api/client.ts";
import type { AnswerResponse, CitationView } from "./api/types.ts";
import { QueryBox } from "./components/QueryBox.tsx";
import { AnswerView } from "./components/AnswerView.tsx";
import { ClausePanel } from "./components/ClausePanel.tsx";
import { ModeBadge } from "./components/ModeBadge.tsx";
import { DeclineNotice } from "./components/DeclineNotice.tsx";
import { MetaLine } from "./components/MetaLine.tsx";

export function App() {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnswerResponse | null>(null);
  const [selected, setSelected] = useState<CitationView | null>(null);

  const ask = async (question: string) => {
    setPending(true);
    setError(null);
    setSelected(null);
    try {
      const response = await postAnswer(question);
      setResult(response);
    } catch (caught) {
      setResult(null);
      setError(
        caught instanceof AnswerError || caught instanceof Error
          ? caught.message
          : "Something went wrong.",
      );
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="mx-auto max-w-3xl px-4 py-10">
      <header className="mb-8">
        <div className="flex items-baseline justify-between gap-4">
          <h1 className="text-2xl font-semibold tracking-tight text-slate-900 dark:text-slate-50">
            Warrant
          </h1>
          {result && <ModeBadge mode={result.mode} recordedOn={result.recorded_on} />}
        </div>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Answers over NIST SP 800-53, with every claim tagged to the control that warrants it.
        </p>
      </header>

      <QueryBox pending={pending} onAsk={ask} />

      {error && (
        <div
          role="alert"
          className="mt-6 rounded-md border border-red-300 bg-red-50 p-4 text-sm text-red-800 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-200"
        >
          {error}
        </div>
      )}

      {result && (
        <main className="mt-8 space-y-8">
          <section className="space-y-3">
            {result.answered && result.answer !== null ? (
              <AnswerView
                answer={result.answer}
                citations={result.citations}
                selected={selected}
                onSelect={setSelected}
              />
            ) : (
              result.decline && <DeclineNotice decline={result.decline} />
            )}
            <MetaLine promptTokenCount={result.prompt_token_count} />
          </section>

          <section>
            <h2 className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
              Clause
            </h2>
            <ClausePanel selected={selected} chunks={result.chunks} />
          </section>
        </main>
      )}
    </div>
  );
}
