import { describe, it, expect } from "vitest";
import { parseAnswer } from "../lib/parseAnswer.ts";
import type { CitationView } from "../api/types.ts";

function citation(cited_as: string, overrides: Partial<CitationView> = {}): CitationView {
  return {
    cited_as,
    control_id: cited_as.toLowerCase(),
    control_label: cited_as,
    title: "Some Control",
    exists: true,
    retrieved: true,
    valid: true,
    ...overrides,
  };
}

describe("parseAnswer", () => {
  it("turns a bracketed token that matches a cited_as into a citation node", () => {
    const nodes = parseAnswer("Records are protected [AU-9].", [citation("AU-9")]);

    expect(nodes).toEqual([
      { kind: "text", text: "Records are protected " },
      { kind: "citation", citation: citation("AU-9") },
      { kind: "text", text: "." },
    ]);
  });

  it("leaves a bracketed token with no matching citation as literal text", () => {
    const nodes = parseAnswer("An aside [see below] and a cite [AU-9].", [citation("AU-9")]);

    // The unmatched "[see below]" stays inside a text node; only "[AU-9]" becomes a citation.
    const kinds = nodes.map((n) => n.kind);
    expect(kinds).toEqual(["text", "citation", "text"]);
    expect(nodes[0]).toEqual({ kind: "text", text: "An aside [see below] and a cite " });
  });

  it("matches on the exact spelling, so an enhancement is not folded into its base", () => {
    const nodes = parseAnswer("Base [AU-9] and enhancement [AU-9(2)].", [
      citation("AU-9"),
      citation("AU-9(2)", { control_id: "au-9.2" }),
    ]);

    const cited = nodes.filter((n) => n.kind === "citation");
    expect(cited).toHaveLength(2);
    expect(cited.map((n) => (n.kind === "citation" ? n.citation.cited_as : ""))).toEqual([
      "AU-9",
      "AU-9(2)",
    ]);
  });

  it("returns a single text node when there are no citations", () => {
    const nodes = parseAnswer("Plain prose, nothing bracketed.", []);
    expect(nodes).toEqual([{ kind: "text", text: "Plain prose, nothing bracketed." }]);
  });
});
