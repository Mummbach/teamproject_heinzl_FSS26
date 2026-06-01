"""
Render presentation_2.html to a landscape PDF (one slide per page).

Usage:  python3 render_pdf.py
Requires: playwright + chromium  (pip install playwright && python3 -m playwright install chromium)

It injects a print stylesheet so every .slide becomes its own 1280x720 page,
renders via headless Chromium, and writes presentation_2.pdf next to the source.
"""
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
SRC = HERE / "presentation_2.html"
OUT = (HERE / "presentation_2.pdf").resolve()

PRINT_CSS = """
<style id="pdfprint">
  @page { size: 1280px 720px; margin: 0; }
  html, body { background:#fff !important; overflow:visible !important; display:block !important; min-height:0 !important; }
  .slideshow-container { width:1280px !important; height:auto !important; box-shadow:none !important; margin:0 !important; position:static !important; }
  .slide { display:flex !important; flex-direction:column !important; position:relative !important; top:auto !important; left:auto !important;
           width:1280px !important; height:720px !important; overflow:hidden !important;
           break-after:page; page-break-after:always; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
  .slide:last-child { break-after:auto; page-break-after:avoid; }
  .nav-bar { display:none !important; }
</style>
</head>"""

html = SRC.read_text().replace("</head>", PRINT_CSS, 1)

# write the print variant alongside the source so relative image paths resolve
tmp = HERE / ".presentation_2_print.tmp.html"
tmp.write_text(html)
try:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(tmp.resolve().as_uri())
        page.wait_for_timeout(800)  # let images/layout settle
        page.pdf(
            path=str(OUT),
            width="1280px",
            height="720px",
            print_background=True,
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
            prefer_css_page_size=True,
        )
        browser.close()
    print("PDF written:", OUT)
finally:
    tmp.unlink(missing_ok=True)
