import re
import time
from pathlib import Path
import spacy
from spacy_layout import spaCyLayout
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from database import test_connection, insert_extracted_entity, get_entities_for_deanonymization

INPUT_DIR = Path(__file__).parent / "Data"

ANON_DIR = Path(__file__).parent / "Anonymized"
ANON_DIR.mkdir(exist_ok=True)

DEANON_DIR = Path(__file__).parent / "Deanonymized"
DEANON_DIR.mkdir(exist_ok=True)

nlp = spacy.load("de_core_news_lg")
layout = spaCyLayout(nlp)

text_style = ParagraphStyle(name="TextStyle", fontSize=10, leading=12)

def is_header_or_footer(span, page_height):
    y, height = span._.layout.y, span._.layout.height
    return y < page_height * 0.1 or (y + height) > page_height * 0.9

def is_person_name(entity_text, entity_tokens):
    if len(entity_text.strip()) < 2 or entity_text.isupper():
        return False
    if any(char.isdigit() for char in entity_text):
        return False

    valid_tokens = [tok for tok in entity_tokens if tok.text.strip()]
    if len(valid_tokens) == 1:
        token = valid_tokens[0]
        return (token.text[0].isupper() and token.pos_ == "PROPN" and
                2 <= len(token.text) <= 15 and token.text.isalpha())

    return all(tok.pos_ == "PROPN" for tok in valid_tokens)

def anonymize(span_doc, pdf_id):
    name_re = re.compile(r"^[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)?$")
    initial_re = re.compile(r"^[A-ZÄÖÜ]\.$")

    mask = [False] * len(span_doc)
    found_entities = []

    # Process spaCy entities
    for ent in span_doc.ents:
        if ent.label_ == "PER" and is_person_name(ent.text.strip(), list(ent)):
            for i in range(ent.start, ent.end):
                mask[i] = True
            found_entities.append({
                'entity_type': 'PER',
                'original_text': ent.text.strip(),
                'anonymized_text': f'PER_{pdf_id:02d}',
                'detection_method': 'spacy_ner'
            })

    # Process name patterns
    for i in range(len(span_doc) - 1):
        tok1, tok2 = span_doc[i], span_doc[i + 1]
        if (tok1.pos_ == "PROPN" and name_re.match(tok1.text) and
            tok2.pos_ == "PROPN" and initial_re.match(tok2.text) and not mask[i]):
            mask[i] = mask[i + 1] = True
            found_entities.append({
                'entity_type': 'PER',
                'original_text': f"{tok1.text} {tok2.text}",
                'anonymized_text': f'PER_{pdf_id:02d}',
                'detection_method': 'pattern_matching'
            })

    # Generate anonymized text
    out = []
    i = 0
    while i < len(span_doc):
        if mask[i]:
            while i < len(span_doc) and mask[i]:
                i += 1
            out.append(f"PER_{pdf_id:02d}" + (span_doc[i - 1].whitespace_ if i > 0 else ""))
        else:
            out.append(span_doc[i].text + span_doc[i].whitespace_)
            i += 1

    return "".join(out).strip(), found_entities

def render_table(df, width):
    df.columns = df.iloc[0].astype(str)
    df = df.drop(df.index[0]).reset_index(drop=True)
    data = [[Paragraph(str(c), text_style) for c in df.columns]] + [
        [Paragraph(str(v or ""), text_style) for v in row] for row in df.itertuples(index=False)
    ]
    col_widths = [width / len(df.columns)] * len(df.columns)
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                              ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return table

def clean_text(txt):
    return (str(txt)
            .replace("+++", "Experte in")
            .replace("++", "Fundierte Kenntnisse in")
            .replace("+", "Grundkenntnisse in"))

def process_pdf(pdf_path, pdf_id):
    start_time = time.time()

    try:
        doc = nlp(layout(str(pdf_path)))
        story = []
        extracted_entities = []

        for page_id, (page_layout, spans) in enumerate(doc._.pages):
            for span in spans:
                if is_header_or_footer(span, page_layout.height):
                    continue

                if span.label_ == "table" and span._.data is not None:
                    story.append(render_table(span._.data, A4[0] - 144))
                else:
                    anonymized_text, found_entities = anonymize(span.as_doc(), pdf_id)
                    story.append(Paragraph(clean_text(anonymized_text), text_style))
                    extracted_entities.extend(found_entities)

                story.append(Spacer(1, 4))

            if page_id < len(doc._.pages) - 1:
                story.append(PageBreak())

        SimpleDocTemplate(str(ANON_DIR / f"PER_{pdf_id:02d}.pdf"), pagesize=A4).build(story)

        for entity in extracted_entities:
            insert_extracted_entity(**entity)

        print(f"Saved PER_{pdf_id:02d}.pdf ({len(extracted_entities)} entities)")

    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
        raise

def deanonymize_text(text, entity_mapping):
    for anonymized, original in entity_mapping.items():
        text = text.replace(anonymized, original)
    return text

def deanonymize_pdf(pdf_path, entity_mapping):
    try:
        doc = nlp(layout(str(pdf_path)))
        story = []

        for page_id, (page_layout, spans) in enumerate(doc._.pages):
            for span in spans:
                if span.label_ == "table" and span._.data is not None:
                    df = span._.data.copy()
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).apply(lambda x: deanonymize_text(x, entity_mapping))
                    story.append(render_table(df, A4[0] - 144))
                else:
                    original_text = span.as_doc().text
                    deanonymized_text = deanonymize_text(original_text, entity_mapping)
                    story.append(Paragraph(clean_text(deanonymized_text), text_style))

                story.append(Spacer(1, 4))

            if page_id < len(doc._.pages) - 1:
                story.append(PageBreak())

        return story
    except Exception as e:
        print(f"Error deanonymizing {pdf_path.name}: {e}")
        raise

def process_deanonymization():
    print("Starting deanonymization process...")

    entity_mapping = get_entities_for_deanonymization()
    if not entity_mapping:
        print("No entities found in database for deanonymization.")
        return

    print(f"Found {len(entity_mapping)} entities for deanonymization")

    anonymized_files = list(ANON_DIR.glob("PER_*.pdf"))
    if not anonymized_files:
        print(f"No anonymized PDF files found in {ANON_DIR}")
        return

    print(f"Processing {len(anonymized_files)} anonymized PDF files...")

    for pdf_file in anonymized_files:
        print(f"Deanonymizing: {pdf_file.name}")

        try:
            story = deanonymize_pdf(pdf_file, entity_mapping)

            output_name = pdf_file.name.replace("PER_", "RESTORED_")
            output_path = DEANON_DIR / output_name

            SimpleDocTemplate(str(output_path), pagesize=A4).build(story)
            print(f"Saved deanonymized: {output_name}")

        except Exception as e:
            print(f"Failed to deanonymize {pdf_file.name}: {e}")

    print(f"Deanonymization complete! Files saved to {DEANON_DIR}")

def main():
    print("Testing database connection...")
    if not test_connection():
        print("Database connection failed. Make sure PostgreSQL is running: docker-compose up -d")
        return

    print("\nChoose an option:")
    print("1. Anonymize PDFs (Data/ -> Anonymized/)")
    print("2. Deanonymize PDFs (Anonymized/ -> Deanonymized/)")

    choice = input("\nEnter choice: ").strip()

    if choice == "1":
        pdf_files = list(INPUT_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {INPUT_DIR}")
            return

        print(f"Processing {len(pdf_files)} PDF files...")
        for i, pdf in enumerate(pdf_files, start=1):
            print(f"Processing {i}/{len(pdf_files)}: {pdf.name}")
            process_pdf(pdf, i)

    elif choice == "2":
        process_deanonymization()

    else:
        print("Invalid choice. Please run again and select 1 or 2.")

if __name__ == "__main__":
    main()