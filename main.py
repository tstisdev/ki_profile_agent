import re
import time
import os
from pathlib import Path
import spacy
from spacy_layout import spaCyLayout
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from database import (
    test_connection,
    insert_processed_document,
    insert_extracted_entity,
    insert_document_metadata,
    get_processed_documents
)

INPUT_DIR = Path(__file__).parent / "Data"
OUTPUT_DIR = Path(__file__).parent / "Anonymized"
OUTPUT_DIR.mkdir(exist_ok=True)

nlp = spacy.load("de_core_news_lg")
layout = spaCyLayout(nlp)

styles = getSampleStyleSheet()
text_style = ParagraphStyle(name="TextStyle", fontSize=10, leading=12)


def is_header_or_footer(span, page_height):
    y, height = span._.layout.y, span._.layout.height
    return y < page_height * 0.1 or (y + height) > page_height * 0.9


def anonymize(span_doc, pdf_id):
    name_re = re.compile(r"^[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?$")
    initial_re = re.compile(r"^[A-ZÄÖÜ]\.$")

    mask = [False] * len(span_doc)

    for ent in span_doc.ents:
        if ent.label_ == "PER" and all(tok.pos_ == "PROPN" for tok in ent):
            for i in range(ent.start, ent.end):
                mask[i] = True

    for i in range(len(span_doc) - 1):
        tok1, tok2 = span_doc[i], span_doc[i + 1]
        if tok1.pos_ == "PROPN" and name_re.match(tok1.text) and tok2.pos_ == "PROPN" and initial_re.match(tok2.text):
            mask[i] = mask[i + 1] = True

    out = []
    i = 0
    while i < len(span_doc):
        if mask[i]:
            while i < len(span_doc) and mask[i]:
                i += 1
            out.append(f"PER_{pdf_id:02d}" + span_doc[i - 1].whitespace_)
        else:
            out.append(span_doc[i].text + span_doc[i].whitespace_)
            i += 1

    return "".join(out).strip()


def render_table(df, width):
    df.columns = df.iloc[0].astype(str)
    df = df.drop(df.index[0]).reset_index(drop=True)

    data = [[Paragraph(str(c), text_style) for c in df.columns]] + [
        [Paragraph(str(v or ""), text_style) for v in row] for row in df.itertuples(index=False)
    ]

    col_widths = [width / len(df.columns)] * len(df.columns)
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.black), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return table


def clean_text(txt):
    return (
        str(txt)
        .replace("+++", "Experte in")
        .replace("++", "Fundierte Kenntnisse in")
        .replace("+", "Grundkenntnisse in"))


def process_pdf(pdf_path, pdf_id):
    start_time = time.time()

    # Get file size
    file_size = pdf_path.stat().st_size

    # Insert document record into database
    document_id = insert_processed_document(
        pdf_id=pdf_id,
        original_filename=pdf_path.name,
        anonymized_filename=f"PER_{pdf_id:02d}.pdf",
        status='processing'
    )

    try:
        doc = nlp(layout(str(pdf_path)))
        story = []
        page_count = len(doc._.pages)
        word_count = 0

        # Track extracted entities for database logging
        extracted_entities = []

        for page_id, (page_layout, spans) in enumerate(doc._.pages):
            for span in spans:
                if is_header_or_footer(span, page_layout.height):
                    continue

                if span.label_ == "table" and span._.data is not None:
                    story.append(render_table(span._.data, A4[0] - 144))
                else:
                    # Process text and track entities
                    span_doc = span.as_doc()
                    word_count += len(span_doc)

                    # Log entities found in this span
                    for ent in span_doc.ents:
                        if ent.label_ == "PER":
                            extracted_entities.append({
                                'entity_type': 'PER',
                                'original_text': ent.text,
                                'anonymized_text': f'PER_{pdf_id:02d}',
                                'position_start': ent.start,
                                'position_end': ent.end,
                                'confidence_score': getattr(ent, 'confidence', None)
                            })

                    anonymized_text = anonymize(span_doc, pdf_id)
                    story.append(Paragraph(clean_text(anonymized_text), text_style))

                story.append(Spacer(1, 4))

            if page_id < len(doc._.pages) - 1:
                story.append(PageBreak())

        # Generate PDF
        output_path = OUTPUT_DIR / f"PER_{pdf_id:02d}.pdf"
        SimpleDocTemplate(str(output_path), pagesize=A4).build(story)

        processing_time = time.time() - start_time

        # Insert extracted entities into database
        for entity in extracted_entities:
            insert_extracted_entity(
                document_id=document_id,
                **entity
            )

        # Insert document metadata
        insert_document_metadata(
            document_id=document_id,
            page_count=page_count,
            word_count=word_count,
            processing_time=processing_time,
            file_size=file_size
        )

        print(f"Saved PER_{pdf_id:02d}.pdf (processed in {processing_time:.2f}s, {len(extracted_entities)} entities found)")

    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
        raise


def display_processing_stats():
    """Display statistics from the database"""
    try:
        documents = get_processed_documents()
        if documents:
            print("\n=== Processing Statistics ===")
            for doc in documents:
                print(f"Document: {doc['original_filename']}")
                print(f"  Status: {doc['status']}")
                print(f"  Entities found: {doc['entity_count']}")
                print(f"  Processed: {doc['processing_date']}")
                print("-" * 40)
        else:
            print("No processed documents found in database.")
    except Exception as e:
        print(f"Could not retrieve statistics: {e}")


def main():
    # Test database connection first
    print("Testing database connection...")
    if not test_connection():
        print("Warning: Database connection failed. Processing will continue without database logging.")
        print("Make sure PostgreSQL is running: docker-compose up -d")
        return

    print("Database connection successful!")

    # Process PDFs
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF files to process...")

    for i, pdf in enumerate(pdf_files, start=1):
        print(f"\nProcessing {i}/{len(pdf_files)}: {pdf.name}")
        process_pdf(pdf, i)

    # Display processing statistics
    display_processing_stats()


if __name__ == "__main__":
    main()