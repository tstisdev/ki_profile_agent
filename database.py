import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'teams_profil_ki'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password123')
}

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def test_connection():
    """Test database connection"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()
                print(f"Connected to PostgreSQL: {version[0]}")
                return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

def insert_processed_document(pdf_id, original_filename, anonymized_filename=None, status='completed'):
    """Insert a processed document record"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO processed_documents (pdf_id, original_filename, anonymized_filename, status)
                VALUES (%s, %s, %s, %s)
                RETURNING id;
            """, (pdf_id, original_filename, anonymized_filename, status))
            return cur.fetchone()[0]

def insert_extracted_entity(document_id, entity_type, original_text, anonymized_text, position_start=None, position_end=None, confidence_score=None, detection_method='spacy_ner'):
    """Insert an extracted entity record"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO extracted_entities (document_id, entity_type, original_text, anonymized_text, position_start, position_end, confidence_score, detection_method)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """, (document_id, entity_type, original_text, anonymized_text, position_start, position_end, confidence_score, detection_method))

def insert_document_metadata(document_id, page_count=None, word_count=None, processing_time=None, file_size=None):
    """Insert document metadata"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO document_metadata (document_id, page_count, word_count, processing_time_seconds, file_size_bytes)
                VALUES (%s, %s, %s, %s, %s);
            """, (document_id, page_count, word_count, processing_time, file_size))

def get_processed_documents():
    """Get all processed documents"""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT pd.*, COUNT(ee.id) as entity_count
                FROM processed_documents pd
                LEFT JOIN extracted_entities ee ON pd.id = ee.document_id
                GROUP BY pd.id, pd.processing_date
                ORDER BY pd.processing_date DESC;
            """)
            return cur.fetchall()

def get_entities_by_document(document_id):
    """Get all entities for a specific document"""
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM extracted_entities
                WHERE document_id = %s
                ORDER BY position_start;
            """, (document_id,))
            return cur.fetchall()
