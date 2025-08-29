import re
from pathlib import Path
import spacy
from spacy_layout import spaCyLayout
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

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
            out.append(f"PER_{pdf_id:02d}")
        else:
            out.append(span_doc[i].text)
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
    doc = nlp(layout(str(pdf_path)))
    story = []

    for page_id, (page_layout, spans) in enumerate(doc._.pages):
        for span in spans:
            if is_header_or_footer(span, page_layout.height):
                continue

            if span.label_ == "table" and span._.data is not None:
                story.append(render_table(span._.data, A4[0] - 144))
            else:
                story.append(Paragraph(clean_text(anonymize(span.as_doc(), pdf_id)), text_style))

            story.append(Spacer(1, 4))

        if page_id < len(doc._.pages) - 1:
            story.append(PageBreak())

    SimpleDocTemplate(str(OUTPUT_DIR / f"PER_{pdf_id:02d}.pdf"), pagesize=A4).build(story)
    print(f"Saved PER_{pdf_id:02d}.pdf")


def main():
    for i, pdf in enumerate(INPUT_DIR.glob("*.pdf"), start=1):
        process_pdf(pdf, i)


if __name__ == "__main__":
    main()