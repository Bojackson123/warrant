import { Fragment } from "react";
import type { CitationView } from "../api/types.ts";
import { parseAnswer } from "../lib/parseAnswer.ts";
import { CitationChip } from "./CitationChip.tsx";

interface Props {
  answer: string;
  citations: CitationView[];
  selected: CitationView | null;
  onSelect: (citation: CitationView) => void;
}

// The answer, rendered so that each claim carries the citation it rests on inline -- the citation is
// attached to the claim, not dropped in a footer. Blank lines in the recorded text become paragraph
// breaks so multi-sentence answers keep their shape.
export function AnswerView({ answer, citations, selected, onSelect }: Props) {
  const nodes = parseAnswer(answer, citations);

  return (
    <div className="space-y-4 whitespace-pre-wrap text-[0.95rem] leading-relaxed text-slate-800 dark:text-slate-100">
      <p>
        {nodes.map((node, i) =>
          node.kind === "text" ? (
            <Fragment key={i}>{node.text}</Fragment>
          ) : (
            <CitationChip
              key={i}
              citation={node.citation}
              selected={
                selected !== null &&
                selected.cited_as === node.citation.cited_as
              }
              onSelect={onSelect}
            />
          ),
        )}
      </p>
    </div>
  );
}
