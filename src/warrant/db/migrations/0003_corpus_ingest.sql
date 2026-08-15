-- What produced the corpus currently in this database.
--
-- The embedding column's declared width is already checked against the pinned model at
-- startup, but a width cannot tell two 768-dimensional models apart: a database carrying
-- vectors from a superseded pin passes that check and returns confidently wrong control ids.
-- This table is what makes the difference visible, and it records the chunker and the catalog
-- alongside the model because a corpus is a function of all three.
--
-- Immutable now that it has shipped. A column this table turns out to need goes in a new
-- numbered file, never an edit here.

CREATE TABLE IF NOT EXISTS corpus_ingest (
    -- One row, enforced. This describes the corpus that is in the database now, not a log of
    -- runs: with a history table, "did re-running ingest change anything?" stops being a
    -- question about the whole database and becomes a question about which row to read.
    id                  smallint         PRIMARY KEY DEFAULT 1 CHECK (id = 1),

    -- Both halves of the model pin. The name alone is not an identity -- published weights
    -- move under a tag -- and the revision alone is not readable.
    embedder_name       text             NOT NULL,
    embedder_revision   text             NOT NULL,
    dimensions          integer          NOT NULL,

    -- The declared version and the digest of what the chunker actually produced. The version is
    -- a statement of intent that a person maintains; the fingerprint is what notices when the
    -- statement went stale because somebody changed how a chunk is assembled and did not bump
    -- it. Recording only the first would record the claim rather than the fact.
    chunker_version     text             NOT NULL,
    chunker_fingerprint text             NOT NULL,

    -- Parameter resolution runs before chunking and changes the prose that gets embedded, so it
    -- is part of what produced these vectors even though it leaves no other trace in the row.
    resolution_version  text             NOT NULL,

    -- The catalog file, by content rather than by release name. Two builds can name the same
    -- release and read different bytes; only one of them produced what is stored here.
    catalog_sha256      text             NOT NULL,

    chunk_count         integer          NOT NULL,

    -- SHA-256 over every stored vector and the id it belongs to. Re-running ingest and getting
    -- the same digest is what turns "it is idempotent" into one number to compare.
    corpus_fingerprint  text             NOT NULL,

    -- Wall-clock for the embedding pass. Kept because it is the number that says whether a
    -- re-ingest is a coffee break or an afternoon, and it is otherwise nowhere.
    embed_seconds       double precision NOT NULL,

    ingested_at         timestamptz      NOT NULL DEFAULT now()
);

COMMENT ON TABLE corpus_ingest IS
    'Provenance for the embedded corpus: which model, chunker and catalog produced the rows in '
    'chunks. Read to refuse a corpus built under a pin this build no longer carries.';
