interface Props {
  promptTokenCount: number;
}

// The small print under an answer: what the prompt cost to render, counted locally with the pinned
// encoding. Present whichever state the request ended in.
export function MetaLine({ promptTokenCount }: Props) {
  return (
    <p className="text-xs text-slate-400 dark:text-slate-500">
      Prompt: {promptTokenCount.toLocaleString()} tokens
    </p>
  );
}
