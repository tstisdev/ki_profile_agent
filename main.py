import pdfplumber
from pathlib import Path
from reportlab.pdfgen import canvas
from io import BytesIO
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

INPUT_DIR = Path(__file__).parent/"Data"
OUTPUT_DIR = Path(__file__).parent/"Anonymized"
OUTPUT_DIR.mkdir(exist_ok=True)

analyzer = AnalyzerEngine()
engine = AnonymizerEngine()

def anonymize(text: str, per_id: int) -> str:
    analyzer_results = analyzer.analyze(text, "de", ["PERSON"], )
    result = engine.anonymize(text, analyzer_results, {"PERSON": OperatorConfig("replace", {"new_value": f"PER_{per_id}"})})
    return result.text

def anonymize_files():
    for per_id, file in enumerate(INPUT_DIR.glob("*.pdf")):
        with pdfplumber.open(file) as pdf:
            output_buffer = BytesIO()

            for page_num, page in enumerate(pdf.pages):
                page_width, page_height = page.width, page.height

                if page_num == 0:
                    c = canvas.Canvas(output_buffer, pagesize=(page_width, page_height))
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