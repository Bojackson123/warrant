# Warrant

Retrieval question answering over the NIST SP 800-53 catalog, where every claim in an
answer is tagged to the control that warrants it and the clause can be read in one click.

The primary artefact is the evaluation harness, not the console. The console exists to make
the retrieval visible; the thing being built is a measured account of what the retrieval does
and does not get right. This document describes `v0.1`, a walking skeleton that runs end to
end. Where something is planned rather than built, it says so.

## Running it

```
docker compose up
```

brings up Postgres with the corpus already in it and serves the console at
<http://localhost:8000>. There is no `.env`, no API key, no AWS account, and no network after
the images are built. This is the default — **replay mode**: a fixed list of questions has
been answered once by the real pipeline and recorded, and those recordings are what the
console serves. Ask one of them, get an answer citing the controls that warrant it, and click
a citation to read the clause.

Setting `WARRANT_MODEL_API_KEY` and nothing else switches to **live mode**, which calls the
model provider instead of replaying — the same corpus and the same retrieval, only the
generation call changes. The developer commands under [Development](#development) below are
for regenerating those recordings and rebuilding the corpus.

## The catalog, and two things wrong with it

The corpus is the NIST SP 800-53 Rev 5 control catalog, release 5.2.0, in OSCAL JSON,
vendored into the repository rather than fetched at build time — vendoring is what lets the
stack come up with no network. The pin lives in [`data/catalog/PINNED.md`](data/catalog/PINNED.md).
About 1,200 controls, of which one in six is withdrawn; the 1,014 live ones are what gets
embedded.

Two things in the source needed handling, and both are recorded here because they are the
evidence that the data was read rather than pointed at a loader.

**Parameter placeholders.** Control prose ships with unresolved markers of the form
`{{ insert: param, ac-02_odp.01 }}`. Every parameter in this release is organisation-defined
— none carries a concrete value — so resolution always renders a labelled slot such as
`[organization-defined frequency]`, never substitutes a value; embedding the raw braces would
put machine noise into every vector and into every clause a reader clicks through to. The
markers themselves are uniform in shape, but the spacing *around* them is not — the source
carries a space before the punctuation that follows, `}} ; and` — so the resolver tidies that
seam locally at each substitution site rather than sweeping the whole document, and a module
named for parameter resolution is not left quietly rewriting unrelated punctuation.

**Control identifiers.** Release 5.1.1 introduced a zero-padded label form (`AC-02`,
`AC-02(03)`) alongside the non-padded one (`AC-2`, `AC-2(3)`) it kept for continuity, and
5.2.0 carries both. The file is internally consistent — each control publishes several
identifier forms and each is globally unique — but code, or an evaluation set, that keys on
one form silently stops matching the other, and a silently corrupted eval is worse than none.
Every identifier is normalised to the catalog's own non-padded object id (`ac-2`) for storage
and matching, the human label (`AC-2`) is kept for what a reader sees, and the mapping is
recorded so the two never blur. The canonical choice is written down in
[`docs/decisions/control-ids.md`](docs/decisions/control-ids.md).

## The embedding model

`BAAI/bge-base-en-v1.5`, 768 dimensions. It was chosen by measurement rather than default
because it is the one component with no cheap migration path — changing it re-embeds the
corpus and re-records every fixture downstream. Four candidates were compared on CPU over the
1,014-document corpus with 42 hand-written questions, retrieval only. It scored recall@5 =
0.679, level with the best candidate (`nomic-embed-text-v1.5` at 0.690, a gap of half a
question) while embedding the corpus faster and answering a query in under 80 ms. A decision
rule written *before* the run takes the cheaper model where two are not measurably different.
768 dimensions is the ceiling worth paying for: retrieval quality flattens above it and every
dimension is fixture bytes committed twice. The full table, the pre-registered rule, and the
harness commit needed to reproduce the numbers are in
[`docs/decisions/embedder.md`](docs/decisions/embedder.md).

## Development

Requires [`uv`](https://docs.astral.sh/uv/), GNU `make`, and Docker.

```
make sync       # create the environment from the lockfile
make lint       # ruff check and format check
make typecheck  # pyright
make test       # pytest — the integration tests start a real Postgres, so Docker must be running
make db-up      # start Postgres with pgvector
make migrate    # bring the database up to this build's schema
make db-down    # stop it and delete the volume

make manifest   # check every pinned input against the record of what the stored corpus and
                # the recorded model calls were built from
make model      # fetch the pinned embedding weights — one of two steps that need a network
make tokenizer  # fetch the generation model's own tokenizer encoding — the other one
make ingest     # embed the catalog and build the corpus; migrates first, so an empty
                # database is fine. Runs offline, on CPU, and takes minutes.
make ask Q="how are inactive accounts disabled?"
                # ask the corpus a question and print the chunks it retrieves

make record-queries
                # embed each recorded question and store the vector replay retrieves it with.
                # Needs no API key and no database.
make record     # run the real pipeline over data/fixtures/questions.json and write down what
                # came back. The only command here that spends money.
make record-again
                # re-record every answer whether or not one is stored. The monthly cadence.
```

`make manifest` is the one that refuses. Everything stored here — the corpus vectors and the
recorded model answers — is a function of the catalog, the parameter resolver, the chunker,
the embedding model, the prompt template and the tokenizer, so changing one of those and
forgetting to rebuild what was made with it would otherwise be a green build over stale artefacts.
`data/manifest.json` records what each of those currently hashes to, and the check exits non-zero
naming the one that moved and what the move invalidates. Recording an intended change is
`make manifest-write`, a separate command on purpose: a check that repaired what it found would be
a record of nothing.

Not every entry costs the same to change, and the file says so per entry. Moving the prompt
template re-records the model answers and leaves the corpus alone. Moving the embedding model
invalidates everything, because different vectors retrieve different control identifiers and those
identifiers sit inside the prompt every recorded answer is keyed on. Moving the tokenizer is the
quiet one: it invalidates nothing at all and makes every prompt size ever reported wrong.

`make model` and `make tokenizer` run once per machine and are the only two commands here that
touch a network. Everything after them — embedding a question typed into the console, counting
what a prompt would cost — runs from the local cache with no network and no API key, which is the
property the whole thing is arranged around. `make ask` is where that is easiest to check: unplug
the machine, ask it something, and watch real retrieval answer. Both fetch commands prove the point
rather than assert it: each re-loads what it just fetched in a second process with every socket
refused, so a path that quietly reached out fails there instead of on somebody else's machine.

Re-running `make ingest` rebuilds the same corpus rather than a second copy of it, and prints a
fingerprint you can compare across runs to see that it did.

## Recorded answers

Asking a model costs money and returns something slightly different every time, so the answers to a
fixed list of questions are recorded once and replayed after that. That is what makes this runnable
with no API key and what makes a test over it deterministic. Everything is under `data/fixtures`,
in two directories that are kept apart on purpose: the answers are text, one file each, so a
re-record is a diff somebody can read; the question vectors are binary and never appear inline in
them.

A recording is filed under a SHA-256 of the **complete** request — the model, its pinned version,
the fully rendered prompt, and every sampling parameter. Never a subset. Editing one character of
the prompt template therefore produces a **miss**, not a stale answer served with stale token
counts, and **nothing is ever substituted**: a question that reads like a recorded one and is not
one has no recorded answer, and gets told so. Serving the nearest recorded answer would be an
answer whose warrant does not cover the question asked, which is exactly the failure this project
exists to detect.

The question vectors are recorded for a less obvious reason. Embedding is deterministic on a
machine and not across machines — floating-point results depend on the kernels torch picks for the
hardware — so two machines can embed one question into vectors that differ in the last bit, retrieve
two near-tied chunks the other way round, and produce a different prompt for the same question.
Recording the vectors confines that to questions typed freely, where there is nothing recorded to
miss. [`docs/decisions/fixtures.md`](docs/decisions/fixtures.md) works through the layout, the
storage arithmetic, and the **re-record cadence: monthly, and on any manifest change**, each one its
own pull request.

The question list in `data/fixtures/questions.json` is provisional and labelled as such. It groups
into the three classes a measured set will use — answerable, out of corpus, and questions whose
obvious answer the catalog contradicts — but it has not been sized or validated against anything.

## What this is not

Some things are left out on purpose, each for a reason rather than for lack of time:

- **Kubernetes.** One stateless service does not need an orchestrator; the deployment target
  is managed containers, and the orchestration complexity was not justified.
- **Multi-tenancy and authentication.** There are no users to keep apart.
- **Agent tool-calling, fine-tuning, a second corpus, and an in-workflow critic loop.**
- **LangChain.** LangGraph only, later, and conditional on there being a workflow that earns
  a graph.
- **Any vector database other than Postgres, and any ORM.** pgvector carries a few thousand
  vectors without a second datastore, and the SQL is written directly.
- **A hosted observability vendor** — a subscription for a stack torn down nightly.
- **A reranker,** until a golden set shows retrieval is the bottleneck. Added before that it
  is an unmeasured guess that also invalidates fixtures.

And some claims are deliberately not made, stated here as commitments:

- **It is not a compliance tool.** It answers questions about a public document. It does not
  assess systems, produce SSPs, or determine whether anything is compliant.
- **It offers no authoritative interpretation.** Where the catalog is ambiguous, so is the
  answer.
- **No scale claims.** A few thousand controls — this is a quality problem, not a volume one.
- **No users, no production, no uptime.**
- **CI does not gate answer quality at the pull request.** The recorded tests check that the
  pipeline runs and that citations are valid, not that an answer is good. Measuring answer
  quality is planned as a two-tier structure and is not built yet.
- **Replay scores are reproducible, not live.** They are what a fixed pipeline produced once,
  replayed — not a measurement taken fresh on each run.

## Licence

Source code is licensed under the [Apache License 2.0](LICENSE).

The NIST SP 800-53 catalog vendored under `data/catalog/` is not covered by that
licence: it is a U.S. Government work, not subject to domestic copyright, and is
reproduced unmodified. [`NOTICE`](NOTICE) records what came from where, and
[`data/catalog/PINNED.md`](data/catalog/PINNED.md) records exactly which release.
