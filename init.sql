-- Initialize database schema for teams_profil_ki
CREATE TABLE IF NOT EXISTS processed_documents (
    id SERIAL PRIMARY KEY,
    pdf_id INTEGER NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    anonymized_filename VARCHAR(255),
    processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS extracted_entities (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES processed_documents(id),
    entity_type VARCHAR(50) NOT NULL,
    original_text TEXT NOT NULL,
    anonymized_text VARCHAR(100),
    position_start INTEGER,
    position_end INTEGER,
    confidence_score FLOAT,
    detection_method VARCHAR(50) DEFAULT 'spacy_ner'
);

CREATE TABLE IF NOT EXISTS document_metadata (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES processed_documents(id),
    page_count INTEGER,
    word_count INTEGER,
    processing_time_seconds FLOAT,
    file_size_bytes BIGINT
);

-- Create indexes for better performance
CREATE INDEX idx_processed_documents_status ON processed_documents(status);
CREATE INDEX idx_extracted_entities_document_id ON extracted_entities(document_id);
CREATE INDEX idx_extracted_entities_type ON extracted_entities(entity_type);
