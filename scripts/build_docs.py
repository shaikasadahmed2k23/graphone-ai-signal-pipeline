"""Build docs: render docs/architecture.mmd -> docs/architecture.png, then docs/architecture.md -> architecture.pdf.

    python scripts/build_docs.py

Uses the Playwright Chromium we already depend on (mermaid is loaded from a pinned CDN version).
"""
from __future__ import annotations

import sys
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"

CSS = """
@page { size: A4; margin: 14mm 14mm; }
body { font: 9.4pt/1.38 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: #1a1a1a; }
h1 { font-size: 16pt; margin: 0 0 4pt; } h2 { font-size: 11.5pt; margin: 10pt 0 3pt; border-bottom: 1px solid #ccc; padding-bottom: 1pt; }
p { margin: 3pt 0; } ul { margin: 3pt 0 3pt 14pt; padding: 0; } li { margin: 1.5pt 0; }
code { font: 8.4pt Consolas, monospace; background: #f1f1f1; padding: 0 2pt; border-radius: 2px; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0; font-size: 8.8pt; } td, th { border: 1px solid #ccc; padding: 2pt 5pt; text-align: left; }
th { background: #f4f4f4; } img { max-height: 92mm; max-width: 100%; display: block; margin: 4pt auto; }
"""


def render_diagram(page) -> Path:
    src = (DOCS / "architecture.mmd").read_text(encoding="utf-8")
    html = f"""<html><body style="margin:0;background:white"><div id="g" class="mermaid">{src}</div>
    <script src="{MERMAID}"></script><script>mermaid.initialize({{startOnLoad:true,theme:'neutral',flowchart:{{htmlLabels:true,useMaxWidth:false}}}});</script></body></html>"""
    page.set_viewport_size({"width": 1100, "height": 900})
    page.set_content(html, wait_until="networkidle")
    page.wait_for_selector("#g svg", timeout=30000)
    out = DOCS / "architecture.png"
    page.locator("#g svg").screenshot(path=str(out))
    return out


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(device_scale_factor=2)
        png = render_diagram(pg)
        print("diagram ->", png.relative_to(ROOT))
        body = markdown.markdown((DOCS / "architecture.md").read_text(encoding="utf-8"), extensions=["tables"])
        body = body.replace('src="architecture.png"', f'src="{png.as_uri()}"')
        pg.set_content(f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>")
        pdf = ROOT / "architecture.pdf"
        pg.pdf(path=str(pdf), format="A4", print_background=True, margin={"top": "14mm", "bottom": "14mm", "left": "14mm", "right": "14mm"})
        b.close()
    print("pdf ->", pdf.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
