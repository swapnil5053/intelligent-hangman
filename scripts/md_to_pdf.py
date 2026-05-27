import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


def write_wrapped(c, text, x, y, max_width, leading=14, font_name="Helvetica", font_size=11):
    from textwrap import wrap
    c.setFont(font_name, font_size)
    lines = []
    for para in text.split("\n"):
        if para.strip() == "":
            lines.append("")
            continue
        # estimate characters per line by font size; be conservative
        wrapped = wrap(para, width=max(10, int(max_width / (font_size * 0.55))))
        lines.extend(wrapped if wrapped else [""])
    for line in lines:
        if y < 72:
            c.showPage()
            y = A4[1] - 72
            c.setFont(font_name, font_size)
        c.drawString(x, y, line)
        y -= leading
    return y


def md_to_pdf(md_path: str, pdf_path: str):
    with open(md_path, 'r', encoding='utf-8') as f:
        md = f.read()
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    left_margin = 60
    right_margin = 60
    top_margin = 60
    bottom_margin = 60
    y = height - top_margin
    x = left_margin
    # Simple markdown handling for #, ##, ### headers and lists
    pending_table = []
    def flush_table(c, rows, x, y, max_width):
        # render table rows as wrapped monospaced text, one row per line
        for row in rows:
            y = write_wrapped(c, row, x, y, max_width=max_width, leading=12, font_name="Courier", font_size=9)
        return y

    for raw_line in md.splitlines():
        line = raw_line.rstrip()
        if line.startswith('# '):
            if pending_table:
                y = flush_table(c, pending_table, x, y, max_width=width-left_margin-right_margin)
                pending_table = []
            c.setFont("Helvetica-Bold", 16)
            c.drawString(x, y, line[2:].strip())
            y -= 20
        elif line.startswith('## '):
            if pending_table:
                y = flush_table(c, pending_table, x, y, max_width=width-left_margin-right_margin)
                pending_table = []
            c.setFont("Helvetica-Bold", 13)
            c.drawString(x, y, line[3:].strip())
            y -= 16
        elif line.startswith('### '):
            if pending_table:
                y = flush_table(c, pending_table, x, y, max_width=width-left_margin-right_margin)
                pending_table = []
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x, y, line[4:].strip())
            y -= 14
        elif line.startswith('|'):
            # collect table rows to wrap together
            pending_table.append(line)
        elif line.startswith('- '):
            # wrapped bullet with indent
            bullet_text = '• ' + line[2:].strip()
            y = write_wrapped(c, bullet_text, x+12, y, max_width=width-left_margin-right_margin-12, leading=14, font_name="Helvetica", font_size=11)
        else:
            if pending_table:
                y = flush_table(c, pending_table, x, y, max_width=width-left_margin-right_margin)
                pending_table = []
            y = write_wrapped(c, line, x, y, max_width=width-left_margin-right_margin, leading=14)
    if pending_table:
        y = flush_table(c, pending_table, x, y, max_width=width-left_margin-right_margin)
    c.save()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python scripts/md_to_pdf.py <input.md> <output.pdf>")
        sys.exit(1)
    md_to_pdf(sys.argv[1], sys.argv[2])
