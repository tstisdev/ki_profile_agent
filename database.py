import os
import psycopg2
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'teams_profil_ki'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password123')
}

@contextmanager
def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
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
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                print(f"Connected to PostgreSQL: {cur.fetchone()[0]}")
                return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

def insert_extracted_entity(entity_type, original_text, anonymized_text, detection_method='spacy_ner'):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO extracted_entities (entity_type, original_text, anonymized_text, detection_method)
                VALUES (%s, %s, %s, %s);
            """, (entity_type, original_text, anonymized_text, detection_method))

def get_entities_for_deanonymization():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT ON (anonymized_text) anonymized_text, original_text 
                FROM extracted_entities 
                ORDER BY anonymized_text, id;
            """)
            return {row[0]: row[1] for row in cur.fetchall()}
