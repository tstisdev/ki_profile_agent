import spacy
from spacy_layout import spaCyLayout
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from io import BytesIO

INPUT_DIR = Path(__file__).parent / "Data"
OUTPUT_DIR = Path(__file__).parent / "Anonymized"
OUTPUT_DIR.mkdir(exist_ok=True)

nlp = spacy.load("de_core_news_sm")
layout = spaCyLayout(nlp)

def anonymize_text(text, per_id: int):
    doc = nlp(text)
    for ent in reversed(doc.ents):
        if ent.label_ == "PER":
            text = text[:ent.start_char] + f"[{ent.label_}_{per_id:02d}]" + text[ent.end_char:]
    return text

def save_layout_as_pdf(doc, output_path: Path, per_id: int):
    output_buffer = BytesIO()
    c = canvas.Canvas(output_buffer, pagesize=A4)
    width, height = A4

    current_page = 1

    for span in doc.spans["layout"]:
        layout_info = span._.layout
        if layout_info.page_no != current_page:
            c.showPage()
            current_page = layout_info.page_no

        if span.label_ not in {"section_header", "page_footer"}:
            x = layout_info.x
            y = height - layout_info.y
            font_size = getattr(layout_info, "font_size", 9) or 9
            c.setFont("Helvetica", font_size)
            text = anonymize_text(span.text, per_id)
            c.drawString(x, y, text)

    c.save()
    with open(output_path, 'wb') as f:
        f.write(output_buffer.getvalue())

def anonymize_files():
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    for per_id, doc in enumerate(layout.pipe(pdf_files)):
        output_path = OUTPUT_DIR / f"{pdf_files[per_id].stem}.pdf"
        save_layout_as_pdf(doc, output_path, per_id)

if __name__ == "__main__":
    anonymize_files()
