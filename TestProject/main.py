import pdfplumber
import spacy
from pathlib import Path
from reportlab.pdfgen import canvas
from io import BytesIO

INPUT_DIR = Path(__file__).parent/"Data"
OUTPUT_DIR = Path(__file__).parent/"Anonymized"
OUTPUT_DIR.mkdir(exist_ok=True)

try:
    nlp = spacy.load("de_core_news_lg")
except ModuleNotFoundError:
    print("spaCy package 'de_core_news_lg' not found.")

def anonymize(text: str, per_id: int) -> str:
    doc = nlp(text)
    anonymized_text = text
    for ent in reversed(doc.ents):
        if ent.label_ in {"PER"}:
            anonymized_text = (
                anonymized_text[:ent.start_char] + f"[{ent.label_}]" + anonymized_text[ent.end_char:]
            )
    return anonymized_text

def anonymize_files():
    for per_id, file in INPUT_DIR.glob("*.pdf"):
        with pdfplumber.open(file) as pdf:
            output_buffer = BytesIO()

            for page_num, page in enumerate(pdf.pages):
                page_width, page_height = page.width, page.height

                if page_num == 0:
                    c = canvas.Canvas(output_buffer, page_width, page_height)
                else:
                    c.showPage()

                words = page.extract_words()

                for word in words:
                    text = word['text']
                    anonymized_text = anonymize(text, per_id)

                    c.setFont("Helvetica", 9)

                    x = word['x0']
                    y = page_height - word['top']

                    c.drawString(x, y, anonymized_text)

            c.save()

            output_file = OUTPUT_DIR / file.name
            with open(output_file, 'wb') as f:
                f.write(output_buffer.getvalue())


if __name__ == "__main__":
    anonymize_files()