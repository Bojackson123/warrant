-- The corpus: one row per retrievable piece of the catalog, with its embedding.
--
-- Immutable now that it has shipped. A column this table turns out to need goes in a new
-- numbered file, never an edit here.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    -- Supplied by the chunker and derived from the catalog's own identifiers, so re-running
    -- ingest over an unchanged catalog collides with the rows already present instead of
    -- doubling the corpus. A synthetic key would make every ingest look like new data.
    chunk_id        text        NOT NULL PRIMARY KEY,

    -- A real column, not a key in a metadata blob. Everything downstream -- the citation
    -- check, the click-through, the eval -- joins or filters on this, and a JSON path would
    -- make the one identifier the whole project is about the hardest thing in the row to
    -- query. Canonical form; for an enhancement this is the enhancement's own id.
    control_id      text        NOT NULL,

    -- The control an enhancement belongs to, equal to control_id for a base control. Lets a
    -- question about a control reach its enhancements without parsing identifiers back apart.
    base_control_id text        NOT NULL,

    -- The form a reader sees. The catalog carries both a zero-padded identifier and the
    -- conventional label, and mixing them silently is the failure this pair of columns exists
    -- to prevent: storage and matching use the canonical form above, rendered citations use
    -- this one.
    control_label   text        NOT NULL,

    title           text        NOT NULL,

    -- Where in the control this text came from, precise enough to render the clause a
    -- reviewer clicks through to rather than the whole control.
    part_path       text        NOT NULL,

    text            text        NOT NULL,

    -- The width is the pinned embedding model's, and it is a literal here because a column
    -- type cannot be parameterised. data/embedder.json remains the single source of truth:
    -- the application reads this column's declared width back out of the catalog at startup
    -- and refuses to run if the two disagree, so the duplication cannot survive unnoticed.
    -- A model change is a new migration and a full re-ingest, which is what it always was.
    embedding       vector(768) NOT NULL,

    -- Which chunker produced this row. Changing the chunker invalidates the embeddings and
    -- every recorded generation downstream of them, so the value is recorded per row and read
    -- by the fixture manifest rather than inferred.
    chunker_version text        NOT NULL,

    created_at      timestamptz NOT NULL DEFAULT now()
);

-- Click-through and citation checking both look rows up by identifier, and both are on the
-- request path.
CREATE INDEX IF NOT EXISTS chunks_control_id_idx ON chunks (control_id);
CREATE INDEX IF NOT EXISTS chunks_base_control_id_idx ON chunks (base_control_id);

-- No vector index, deliberately.
--
-- The catalog produces roughly a thousand chunks -- about 3 MB of vectors -- and an exact
-- scan over that is a few milliseconds. An approximate index would not be measurably faster
-- here, and it would cost something that matters more: recall would become a property of the
-- index's search parameters rather than of the embedding model, so every retrieval number
-- this project reports would be measuring the two together without saying so.
--
-- Revisit at the point where the corpus is large enough for the scan to show up in a request
-- -- on the order of a hundred thousand rows -- and treat the index as a change that has to
-- be measured against exact search, not assumed to be free.

COMMENT ON TABLE chunks IS
    'The embedded corpus. One row per control or enhancement chunk, searched by exact vector '
    'distance; see the migration that created it for why there is no vector index.';
