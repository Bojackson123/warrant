import type { CitationView } from "../api/types.ts";

// The answer arrives as prose with citations written inline as bracketed control identifiers, e.g.
// "... upon detection [AU-9].". A node is either a run of literal text or one such citation.
export type AnswerNode =
  | { kind: "text"; text: string }
  | { kind: "citation"; citation: CitationView };

// Matches a single bracketed token with no nested brackets, e.g. "[AU-9]" or "[AU-9(2)]".
const BRACKET_TOKEN = /\[([^[\]]+)\]/g;

// Split the answer into text and citation nodes. A bracketed token becomes a citation node only
// when its inner text is exactly one of the server's `cited_as` values; any other bracketed prose
// stays literal text. This mirrors the server, which decides what is a citation by syntax and then
// resolves it -- so the console never invents a citation the check did not run over, nor drops one
// it did.
export function parseAnswer(answer: string, citations: CitationView[]): AnswerNode[] {
  const byCitedAs = new Map(citations.map((citation) => [citation.cited_as, citation]));

  const nodes: AnswerNode[] = [];
  let pending = "";
  let lastIndex = 0;

  const flush = () => {
    if (pending) {
      nodes.push({ kind: "text", text: pending });
      pending = "";
    }
  };

  for (const match of answer.matchAll(BRACKET_TOKEN)) {
    const [token, inner] = match;
    const start = match.index;
    pending += answer.slice(lastIndex, start);
    lastIndex = start + token.length;

    const citation = byCitedAs.get(inner);
    if (citation) {
      flush();
      nodes.push({ kind: "citation", citation });
    } else {
      // Not a citation the server reported; leave the brackets in place as ordinary text.
      pending += token;
    }
  }

  pending += answer.slice(lastIndex);
  flush();
  return nodes;
}
