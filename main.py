from pathlib import Path
import spacy
from spacy_layout import spaCyLayout
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak
from reportlab.platypus import Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from xml.sax.saxutils import escape

INPUT_DIR = Path(__file__).parent / "Data"
OUTPUT_DIR = Path(__file__).parent / "Anonymized"
OUTPUT_DIR.mkdir(exist_ok=True)

nlp = spacy.load("de_core_news_lg")
layout = spaCyLayout(nlp)

styles = getSampleStyleSheet()
text_style = ParagraphStyle("Body9", parent=styles["Normal"], fontSize=9, leading=10)


def is_header_or_footer(span, page_height, cutoff=0.1):
    y, h = span._.layout.y, span._.layout.height
    return (y < page_height * cutoff) or ((y + h) > page_height * (1 - cutoff))


def anonymize(span_doc, pdf_id):
    return " ".join(
        f"PER_{pdf_id:02d}" if tok.ent_type_ == "PER" else tok.text
        for tok in span_doc
    )


def render_table(df, available_width):
    if all(isinstance(c, (int, float)) for c in df.columns):
        df.columns = df.iloc[0].astype(str)
        df = df.drop(df.index[0]).reset_index(drop=True)

    data = [[Paragraph(str(c), text_style) for c in df.columns]] + [
        [Paragraph(str(v) if v is not None else "", text_style) for v in row]
        for row in df.itertuples(index=False, name=None)
    ]

    number_cols = len(data[0]) if data else 1
    table = Table(data, colWidths=[available_width / number_cols] * number_cols)
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.black),
        ("VALIGN", (0,0), (-1,-1), "TOP")
    ]))
    return table


def process_pdf(pdf_path, pdf_id):
    doc = layout(str(pdf_path))
    doc = nlp(doc)

    story = []
    num_pages = len(doc._.pages)

    margin = 36
    available_width = A4[0] - 2 * margin

    for page_idx, (page_layout, spans) in enumerate(doc._.pages):
        for span in spans:
            if is_header_or_footer(span, page_layout.height):
                continue

            if span.label_ == "table" and span._.data is not None:
                story.append(render_table(span._.data, available_width))
            else:
                txt = anonymize(span.as_doc(), pdf_id)
                story.append(Paragraph(escape(txt), text_style))

            story.append(Spacer(1, 4))

        if page_idx < num_pages - 1:
            story.append(PageBreak())

    out_path = OUTPUT_DIR / f"PER_{pdf_id:02d}.pdf"
    SimpleDocTemplate(
        str(out_path), pagesize=A4, leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin
    ).build(story)

    print(f"Saved {out_path}")


def main():
    for i, pdf in enumerate(list(INPUT_DIR.glob("*.pdf")), start=1):
        process_pdf(pdf, i)


if __name__ == "__main__":
    main()