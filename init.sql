CREATE TABLE IF NOT EXISTS extracted_entities (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,
    original_text TEXT NOT NULL,
    anonymized_text VARCHAR(100),
    detection_method VARCHAR(50) DEFAULT 'spacy_ner'
);
