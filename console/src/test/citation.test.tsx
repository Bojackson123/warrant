import { useState } from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ChunkView, CitationView } from "../api/types.ts";
import { AnswerView } from "../components/AnswerView.tsx";
import { ClausePanel } from "../components/ClausePanel.tsx";

// The answer view and the clause panel share one selection, exactly as the app wires them. Driving
// them together is what proves the click-through: selecting a citation reads its clause, or is told
// why there is none.
function Harness({
  answer,
  citations,
  chunks,
}: {
  answer: string;
  citations: CitationView[];
  chunks: ChunkView[];
}) {
  const [selected, setSelected] = useState<CitationView | null>(null);
  return (
    <>
      <AnswerView
        answer={answer}
        citations={citations}
        selected={selected}
        onSelect={setSelected}
      />
      <ClausePanel selected={selected} chunks={chunks} />
    </>
  );
}

const valid: CitationView = {
  cited_as: "AU-9",
  control_id: "au-9",
  control_label: "AU-9",
  title: "Protection of Audit Information",
  exists: true,
  retrieved: true,
  valid: true,
};

// A real control the answer cited without it having been retrieved: exists, but not retrieved.
const invalid: CitationView = {
  cited_as: "SC-28",
  control_id: "sc-28",
  control_label: "SC-28",
  title: "Protection of Information at Rest",
  exists: true,
  retrieved: false,
  valid: false,
};

const chunk: ChunkView = {
  rank: 1,
  score: 0.42,
  chunk_id: "au-9#a",
  control_id: "au-9",
  base_control_id: "au-9",
  control_label: "AU-9",
  title: "Protection of Audit Information",
  part_path: "a",
  text: "Audit information is protected from unauthorized modification and deletion.",
};

const ANSWER = "Records are protected [AU-9]. Data at rest is separate [SC-28].";

describe("citations", () => {
  it("marks an invalid citation differently from a valid one", () => {
    render(<Harness answer={ANSWER} citations={[valid, invalid]} chunks={[chunk]} />);

    const validChip = screen.getByRole("button", { name: /AU-9/ });
    const invalidChip = screen.getByRole("button", { name: /SC-28/ });

    expect(validChip).toHaveAttribute("data-valid", "true");
    expect(invalidChip).toHaveAttribute("data-valid", "false");
    // The invalid one carries a spelled-out reason for anyone not seeing the colour.
    expect(invalidChip).toHaveAccessibleName(/invalid citation/i);
  });

  it("shows the clause text when a valid citation is clicked", async () => {
    render(<Harness answer={ANSWER} citations={[valid, invalid]} chunks={[chunk]} />);

    await userEvent.click(screen.getByRole("button", { name: /AU-9/ }));

    expect(
      screen.getByText(/Audit information is protected from unauthorized modification/),
    ).toBeInTheDocument();
  });

  it("explains the absence when a cited-but-not-retrieved citation is clicked", async () => {
    render(<Harness answer={ANSWER} citations={[valid, invalid]} chunks={[chunk]} />);

    await userEvent.click(screen.getByRole("button", { name: /SC-28/ }));

    expect(screen.getByText(/was not among the clauses retrieved/)).toBeInTheDocument();
  });
});
