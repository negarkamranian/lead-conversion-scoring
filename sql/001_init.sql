CREATE TABLE IF NOT EXISTS ingestion_batches (
    source_hash TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    row_count INTEGER NOT NULL CHECK (row_count >= 0),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw_leads (
    source_hash TEXT NOT NULL REFERENCES ingestion_batches(source_hash),
    source_row_number INTEGER NOT NULL CHECK (source_row_number > 0),
    lead_id TEXT,
    created_at TEXT,
    product_type TEXT,
    channel TEXT,
    device TEXT,
    partner TEXT,
    city TEXT,
    insurance_company TEXT,
    payment_type TEXT,
    minutes_since_abandonment TEXT,
    days_to_policy_expiry TEXT,
    price TEXT,
    discount_percent TEXT,
    has_previous_purchase TEXT,
    visited_offer_page TEXT,
    incoming_call_last_24h TEXT,
    sessions_last_7d TEXT,
    offer_views_last_7d TEXT,
    price_comparisons_last_7d TEXT,
    days_since_last_visit TEXT,
    expected_margin TEXT,
    completed_purchase TEXT,
    PRIMARY KEY (source_hash, source_row_number)
);
CREATE INDEX IF NOT EXISTS idx_raw_leads_lead_id ON raw_leads (lead_id);
CREATE INDEX IF NOT EXISTS idx_raw_leads_created_at ON raw_leads (created_at);

CREATE TABLE IF NOT EXISTS data_dictionary (
    column_name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scoring_batches (
    scoring_batch_id UUID PRIMARY KEY,
    model_version TEXT NOT NULL,
    source_hash TEXT NOT NULL REFERENCES ingestion_batches(source_hash),
    data_as_of TIMESTAMPTZ NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    row_count INTEGER CHECK (row_count >= 0),
    error_message TEXT
);
CREATE INDEX IF NOT EXISTS idx_scoring_batches_latest
    ON scoring_batches (completed_at DESC) WHERE status = 'succeeded';

CREATE TABLE IF NOT EXISTS lead_scores (
    scoring_batch_id UUID NOT NULL REFERENCES scoring_batches(scoring_batch_id),
    lead_id TEXT NOT NULL,
    purchase_probability DOUBLE PRECISION NOT NULL
        CHECK (purchase_probability >= 0 AND purchase_probability <= 1),
    priority_rank INTEGER NOT NULL CHECK (priority_rank > 0),
    priority_tier TEXT NOT NULL CHECK (priority_tier IN ('call', 'backlog')),
    scored_at TIMESTAMPTZ NOT NULL,
    model_version TEXT NOT NULL,
    data_as_of TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (scoring_batch_id, lead_id),
    UNIQUE (scoring_batch_id, priority_rank)
);
CREATE INDEX IF NOT EXISTS idx_lead_scores_top_n
    ON lead_scores (scoring_batch_id, priority_rank)
    INCLUDE (lead_id, purchase_probability, priority_tier);

