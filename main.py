import requests
from pathlib import Path
from fpdf import FPDF
from PyPDF2 import PdfReader


OUTPUT_DIR = Path(__file__).parent/"generated_pdfs"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_pdfs():
    for i in range(0, 30):
        data = requests.get(f"https://jsonplaceholder.typicode.com/posts/{i+1}").json()
        title = data.get("title").strip().capitalize()
        body = data.get("body").strip()
        filename = f"{i+1:02d} - {title}.pdf"

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, text=title)
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 7, text=body)

        pdf.output(str(OUTPUT_DIR/filename))


def read_pdfs():
    for pdf in sorted(OUTPUT_DIR.glob("*.pdf")):
        print(f"\n{pdf.name}")
        for ln in PdfReader(pdf).pages[0].extract_text().splitlines()[:2]:
            print(" ", ln.strip())


if __name__ == "__main__":
    create_pdfs()
    read_pdfs()