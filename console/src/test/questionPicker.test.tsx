import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { QuestionView } from "../api/types.ts";
import { QuestionPicker } from "../components/QuestionPicker.tsx";

// One of each class, so grouping and the trap label are both exercised.
const QUESTIONS: QuestionView[] = [
  {
    id: "accounts",
    class: "answerable",
    text: "What happens to accounts nobody uses?",
    because: "The catalog states it directly.",
  },
  {
    id: "passwords",
    class: "prior_conflict_trap",
    text: "How often must passwords be changed?",
    because: "Ninety days is remembered and not in the catalog.",
  },
  {
    id: "budget",
    class: "out_of_corpus",
    text: "What should a SIEM cost?",
    because: "A control catalog is not about cost.",
  },
];

describe("question picker", () => {
  it("shows the provisional version so the list is not mistaken for a measured set", () => {
    render(<QuestionPicker version="provisional-1" questions={QUESTIONS} onPick={() => {}} />);

    expect(screen.getByText(/provisional/i)).toHaveTextContent("provisional-1");
  });

  it("groups the questions under their class headings", () => {
    render(<QuestionPicker version="v" questions={QUESTIONS} onPick={() => {}} />);

    expect(screen.getByRole("heading", { name: /answerable/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /prior-conflict traps/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /outside the catalog/i })).toBeInTheDocument();
  });

  it("labels a prior-conflict trap so a reviewer sees it for what it is", () => {
    render(<QuestionPicker version="v" questions={QUESTIONS} onPick={() => {}} />);

    const trap = screen.getByRole("button", { name: /how often must passwords be changed/i });
    expect(trap).toHaveTextContent(/trap/i);
    // The reasoning is shown, not just the tag, so the label is not an unexplained word.
    expect(trap).toHaveTextContent(/ninety days/i);
  });

  it("fills the box with the exact recorded text when a question is picked", async () => {
    const onPick = vi.fn();
    render(<QuestionPicker version="v" questions={QUESTIONS} onPick={onPick} />);

    await userEvent.click(
      screen.getByRole("button", { name: /what happens to accounts nobody uses/i }),
    );

    // The exact text, not a normalised or nearest form -- a verbatim pick is what key-matches a
    // recording, and an edit away from it is what the server declines.
    expect(onPick).toHaveBeenCalledWith("What happens to accounts nobody uses?");
  });
});
