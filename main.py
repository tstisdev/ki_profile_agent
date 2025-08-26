from pathlib import Path
import spacy
from spacy_layout import spaCyLayout
import pdfplumber
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import black

INPUT_DIR = Path(__file__).parent / "Data"
OUTPUT_DIR = Path(__file__).parent / "Anonymized"
OUTPUT_DIR.mkdir(exist_ok=True)

nlp = spacy.load("de_core_news_lg")
layout = spaCyLayout(nlp)

def anonymize_pdf(input_path, output_path):
    doc = layout(str(input_path))
    doc = nlp(doc)

    person_mapping = {}
    person_counter = 1

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            if ent.text not in person_mapping:
                person_mapping[ent.text] = f"PER_{person_counter:02d}"
                person_counter += 1

    with pdfplumber.open(input_path) as pdf:
        c = canvas.Canvas(str(output_path), pagesize=letter)

        for page_num, page in enumerate(pdf.pages):
            page_width = float(page.width)
            page_height = float(page.height)
            c.setPageSize((page_width, page_height))

            page_layout, page_spans = doc._.pages[page_num]

            header_threshold = page_height * 0.9
            footer_threshold = page_height * 0.1

            for span in page_spans:
                if span.label_ in ["page_header", "page_footer"]:
                    continue

                if (span._.layout.y > header_threshold or
                        span._.layout.y + span._.layout.height < footer_threshold):
                    continue

                text = span.text
                for original_name, replacement in person_mapping.items():
                    text = text.replace(original_name, replacement)

                x = span._.layout.x
                y = page_height - span._.layout.y - span._.layout.height

                font_size = min(span._.layout.height * 0.8, 12)
                c.setFont("Helvetica", font_size)
                c.setFillColor(black)

                max_width = span._.layout.width
                lines = []
                words = text.split()
                current_line = []

                for word in words:
                    test_line = " ".join(current_line + [word])
                    if c.stringWidth(test_line, "Helvetica", font_size) <= max_width:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(" ".join(current_line))
                            current_line = [word]
                        else:
                            lines.append(word)

                if current_line:
                    lines.append(" ".join(current_line))

                for i, line in enumerate(lines):
                    line_y = y + i * font_size * 1.2
                    if line_y > footer_threshold and line_y < header_threshold:
                        if x + c.stringWidth(line, "Helvetica", font_size) <= page_width:
                            c.drawString(x, line_y, line)

            c.showPage()

        c.save()


if __name__ == "__main__":
    for file in list(INPUT_DIR.glob("*.pdf")):
        anonymize_pdf(INPUT_DIR/file.name, OUTPUT_DIR/file.name)