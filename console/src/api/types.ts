// The HTTP contract, mirrored from the server's Pydantic models in src/warrant/api/schemas.py.
// One response shape covers both states a question can end in; `answered` says which.

export type Mode = "live" | "replay";

// One control identifier an answer cited, and whether it stands up. The two halves of validity are
// reported separately because they fail for different reasons: `exists=false` is an invented
// identifier; `exists=true, retrieved=false` is a real control cited without having been retrieved
// -- the case that catches an answer drawing on prior knowledge rather than the text it was given.
export interface CitationView {
  // The identifier exactly as written in the answer, e.g. "AU-9" or "AU-9(2)". This is the token
  // that appears bracketed in the answer text, and the one the console renders.
  cited_as: string;
  // The canonical control id it resolves to, e.g. "au-9", or null when it names no control. This is
  // the join key to a chunk's own `control_id`.
  control_id: string | null;
  control_label: string | null;
  title: string | null;
  exists: boolean;
  retrieved: boolean;
  valid: boolean;
}

// One retrieved chunk, carrying the full clause text a click-through opens to -- no second request.
export interface ChunkView {
  rank: number;
  score: number;
  chunk_id: string;
  control_id: string;
  base_control_id: string;
  control_label: string;
  title: string;
  part_path: string;
  text: string;
}

// Why no answer was generated, in words the console shows as its second state.
export interface DeclineView {
  reason: string;
  detail: string;
}

export interface AnswerResponse {
  question: string;
  mode: Mode;
  // Counted over the rendered prompt with the pinned encoding. Present in both states.
  prompt_token_count: number;
  answered: boolean;
  answer: string | null;
  // The day a replayed answer was recorded (ISO date), or null for a live answer and when declined.
  recorded_on: string | null;
  citations: CitationView[];
  chunks: ChunkView[];
  decline: DeclineView | null;
}

// What a recorded question is for. `prior_conflict_trap` is the one the picker labels: not a
// question the catalog fails to answer, but one whose answer most people already believe they know
// and where the catalog says something else.
export type QuestionClass = "answerable" | "out_of_corpus" | "prior_conflict_trap";

// One recorded question, as the picker offers it. `class` mirrors the server's wire name, the
// vocabulary the picker groups by; `because` is why it is in its class, shown for a trap.
export interface QuestionView {
  id: string;
  class: QuestionClass;
  text: string;
  because: string;
}

// The recorded question list, and which list it is. `version` is provisional and shown as such.
export interface QuestionSetView {
  version: string;
  questions: QuestionView[];
}
