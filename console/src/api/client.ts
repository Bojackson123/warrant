import type { AnswerResponse, QuestionSetView } from "./types.ts";

// A request that reached the server but was refused for a reason the server named. The message is
// meant to be shown to the reviewer as-is.
export class AnswerError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "AnswerError";
    this.status = status;
  }
}

// The server returns FastAPI's `{ "detail": ... }` on a 4xx/5xx. `detail` is a string for our own
// HTTPExceptions and a list of objects for a 422 validation error; reduce either to one line.
function readDetail(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const first = detail[0];
      if (first && typeof first === "object" && "msg" in first) {
        return String((first as { msg: unknown }).msg);
      }
    }
  }
  return fallback;
}

// Ask one question. Resolves with the full response in either state -- a declined answer is a 200
// and a normal return, not an error. Rejects with an AnswerError only when the request itself was
// refused (empty/blank question, or a server fault), or with a plain Error when the API is
// unreachable.
export async function postAnswer(question: string): Promise<AnswerResponse> {
  let response: Response;
  try {
    response = await fetch("/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
  } catch {
    throw new Error(
      "Could not reach the API. Is it running? Start it with `make serve`.",
    );
  }

  if (!response.ok) {
    let body: unknown = null;
    try {
      body = await response.json();
    } catch {
      // A response without a JSON body; fall back to the status text below.
    }
    throw new AnswerError(
      response.status,
      readDetail(body, `The API returned ${response.status} ${response.statusText}.`),
    );
  }

  return (await response.json()) as AnswerResponse;
}

// The recorded question list the picker offers. Rejects with a plain Error when the API is
// unreachable or answers with an error status; the caller treats either as "no picker" and leaves
// the query box working, so a question can still be typed without the list.
export async function getQuestions(): Promise<QuestionSetView> {
  let response: Response;
  try {
    response = await fetch("/questions");
  } catch {
    throw new Error("Could not reach the API to load the question list.");
  }

  if (!response.ok) {
    throw new Error(`The API returned ${response.status} ${response.statusText}.`);
  }

  return (await response.json()) as QuestionSetView;
}
