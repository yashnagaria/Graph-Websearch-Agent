"""Render interview.md to interview.pdf.

Uses headless Chrome/Edge as the print engine, so the only Python dependency is the
``markdown`` package:

    pip install markdown
    python scripts/build_interview_pdf.py
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "interview.md"
OUT = REPO / "interview.pdf"
HTML = REPO / "build" / "interview.html"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]

CSS = """
@page { size: A4; margin: 17mm 15mm 16mm 15mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 10pt; line-height: 1.55; color: #1b1f24; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
.cover { page-break-after: always; padding-top: 58mm; text-align: center; }
.cover h1 { font-size: 30pt; line-height: 1.15; margin: 0 0 10mm; color: #0d1117; border: 0; }
.cover .sub { font-size: 12.5pt; color: #4a5560; font-style: italic; max-width: 135mm;
              margin: 0 auto 22mm; line-height: 1.5; }
.cover .rule { width: 46mm; height: 3px; background: #4F9CF9; margin: 0 auto 22mm; }
.cover .meta { font-size: 9.5pt; color: #6a737d; line-height: 1.9; }

h2 { font-size: 16pt; color: #0d1117; margin: 0 0 6mm; padding-bottom: 2.5mm;
     border-bottom: 2px solid #4F9CF9; break-after: avoid; }
h2.chapter { page-break-before: always; }
h3 { font-size: 12pt; color: #1b1f24; margin: 8mm 0 3mm; break-after: avoid; }
p { margin: 0 0 3.6mm; text-align: justify; hyphens: auto; }
strong { color: #0d1117; }
a { color: #1f6feb; text-decoration: none; }

ul, ol { margin: 0 0 4mm; padding-left: 6.5mm; }
li { margin-bottom: 1.6mm; }

code { font-family: Consolas, "Cascadia Mono", monospace; font-size: 8.8pt;
       background: #f2f4f7; padding: 0.4mm 1.2mm; border-radius: 2px; color: #24292f; }
pre { background: #f7f9fb; border: 1px solid #dfe4ea; border-left: 3px solid #4F9CF9;
      border-radius: 3px; padding: 3.5mm 4mm; margin: 0 0 4.5mm; overflow: visible;
      break-inside: avoid; page-break-inside: avoid; }
pre code { font-size: 7.6pt; line-height: 1.38; background: none; padding: 0;
           white-space: pre; display: block; color: #24292f; }

table { border-collapse: collapse; width: 100%; margin: 0 0 5mm; font-size: 8.6pt; }
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }
th { background: #eef2f7; text-align: left; font-weight: 600; color: #0d1117; }
th, td { border: 1px solid #d7dde5; padding: 1.7mm 2.2mm; vertical-align: top; }
tr:nth-child(even) td { background: #fafbfc; }
td code, th code { font-size: 8pt; background: #eef1f5; }

blockquote { margin: 0 0 4.5mm; padding: 2.5mm 4mm; border-left: 3px solid #c8d1dc;
             background: #fbfcfd; color: #37414c; }
blockquote p { margin: 0 0 2mm; }
blockquote p:last-child { margin: 0; }
blockquote ul { margin-bottom: 0; }

hr { border: 0; border-top: 1px solid #e3e8ee; margin: 7mm 0; }
"""


def find_chrome():
    for name in ("chrome", "google-chrome", "chromium", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    for path in CHROME_CANDIDATES:
        if os.path.exists(path):
            return path
    sys.exit("No Chrome/Chromium/Edge found - install one, or convert interview.md by hand.")


def main():
    text = SRC.read_text(encoding="utf-8")
    lines = text.split("\n")

    # The H1 and the italic strapline become the cover; drop them from the body flow.
    title = lines[0].lstrip("# ").strip()
    subtitle = next((ln.strip("*").strip() for ln in lines[1:6]
                     if ln.startswith("*") and ln.endswith("*")), "")
    body_md = "\n".join(lines[1:])
    if subtitle:
        body_md = body_md.replace(f"*{subtitle}*", "", 1)
    body_md = body_md.lstrip("\n-").lstrip()

    html_body = markdown.markdown(
        body_md, extensions=["tables", "fenced_code", "sane_lists", "toc"]
    )

    # Every numbered section except the first starts on a fresh page.
    html_body = re.sub(
        r'<h2 id="([^"]*)">(\d+)\.',
        lambda m: f'<h2 class="{"first" if m.group(2) == "1" else "chapter"}" '
                  f'id="{m.group(1)}">{m.group(2)}.',
        html_body,
    )

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>{CSS}</style></head><body>
<div class="cover">
  <h1>{title}</h1>
  <div class="rule"></div>
  <div class="sub">{subtitle}</div>
  <div class="meta">
    A multi-agent web research system built on LangGraph<br>
    github.com/yashnagaria/Graph-Websearch-Agent
  </div>
</div>
{html_body}
</body></html>"""

    HTML.parent.mkdir(parents=True, exist_ok=True)
    HTML.write_text(page, encoding="utf-8")

    subprocess.run(
        [find_chrome(), "--headless=new", "--disable-gpu", "--no-sandbox",
         "--no-pdf-header-footer", f"--print-to-pdf={OUT}", HTML.as_uri()],
        capture_output=True, text=True, timeout=180, check=False,
    )

    if not OUT.exists():
        sys.exit("Chrome did not produce a PDF.")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
