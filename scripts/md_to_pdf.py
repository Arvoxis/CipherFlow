"""Render a Markdown document to a print-quality PDF.

Markdown goes to HTML with a print stylesheet, then headless Chrome prints it. Chrome is used
because it is the only renderer on this machine that handles page breaks, repeating table headers
and widow control properly, and it needs nothing installed beyond what is already here.

Run:
    python scripts/md_to_pdf.py cipherflow/deliverables/PRD.md
    python scripts/md_to_pdf.py cipherflow/deliverables/PRD.md --out docs/PRD.pdf --title "CipherFlow PRD"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import markdown
import requests
import websockets

REPO_ROOT = Path(__file__).resolve().parents[1]

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: Letter; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", Calibri, system-ui, sans-serif;
  font-size: 10.5pt; line-height: 1.5; color: #14181f; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 20pt; margin: 0 0 4pt; color: #0f1b2d; letter-spacing: -0.2pt; }
h2 {
  font-size: 13.5pt; margin: 22pt 0 7pt; padding-bottom: 4pt; color: #0f1b2d;
  border-bottom: 1.5px solid #2f5496; break-after: avoid; break-inside: avoid;
}
h3 { font-size: 11.5pt; margin: 14pt 0 5pt; color: #2f5496; break-after: avoid; }
h4 { font-size: 10.5pt; margin: 11pt 0 4pt; break-after: avoid; }
p { margin: 0 0 7pt; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 8pt; padding-left: 20pt; }
li { margin-bottom: 3pt; }
strong { color: #0f1b2d; }
code {
  font-family: Consolas, "Cascadia Mono", monospace; font-size: 9pt;
  background: #eef1f6; padding: 1px 4px; border-radius: 3px; color: #1c3a5e;
}
pre {
  background: #f6f8fb; border: 1px solid #d8dee9; border-left: 3px solid #2f5496;
  border-radius: 4px; padding: 9pt 11pt; overflow-x: auto; break-inside: avoid;
  margin: 0 0 10pt;
}
pre code { background: none; padding: 0; font-size: 8.6pt; color: #14181f; line-height: 1.45; }
blockquote {
  margin: 0 0 10pt; padding: 8pt 12pt; background: #fdf8ec;
  border-left: 3px solid #c8962a; break-inside: avoid;
}
blockquote p:last-child { margin-bottom: 0; }
table {
  width: 100%; border-collapse: collapse; margin: 0 0 11pt; font-size: 9pt;
  break-inside: auto;
}
thead { display: table-header-group; }   /* repeat the header on every page */
h1 { break-after: avoid; }
tr { break-inside: avoid; }
th {
  background: #2f5496; color: #fff; text-align: left; font-weight: 600;
  padding: 5pt 7pt; border: 1px solid #2f5496;
}
td { padding: 4.5pt 7pt; border: 1px solid #c9d1de; vertical-align: top; }
tbody tr:nth-child(even) td { background: #f5f7fa; }
hr { border: 0; border-top: 1px solid #d8dee9; margin: 16pt 0; }
a { color: #2f5496; text-decoration: none; }
.title-block { border-bottom: 2.5px solid #2f5496; padding-bottom: 10pt; margin-bottom: 4pt; }
"""

HEADER = """
<div style="font-size:7.5pt;color:#8a93a3;width:100%;padding:0 0.75in;
            font-family:'Segoe UI',Calibri,sans-serif;">
  <span style="float:left;">__TITLE__</span>
  <span style="float:right;">__SUB__</span>
</div>
"""

FOOTER = """
<div style="font-size:7.5pt;color:#8a93a3;width:100%;padding:0 0.75in;
            font-family:'Segoe UI',Calibri,sans-serif;">
  <span style="float:left;">__FOOT__</span>
  <span style="float:right;">Page <span class="pageNumber"></span>
    of <span class="totalPages"></span></span>
</div>
"""


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    found = shutil.which("chrome") or shutil.which("msedge")
    if found:
        return found
    raise SystemExit("No Chrome or Edge found. Edit CHROME_CANDIDATES.")


def build_html(md_path: Path, title: str) -> str:
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "md_in_html"],
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )


async def render(html_path: Path, out_pdf: Path, title: str, subtitle: str, footer: str):
    chrome = find_chrome()
    profile = tempfile.mkdtemp(prefix="cf_pdf_")
    proc = subprocess.Popen(
        [
            chrome,
            "--headless=new",
            "--remote-debugging-port=9334",
            f"--user-data-dir={profile}",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--allow-file-access-from-files",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        ws_url = None
        for _ in range(40):
            try:
                tabs = requests.get("http://127.0.0.1:9334/json", timeout=1).json()
                ws_url = next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"]
                break
            except Exception:
                time.sleep(0.5)
        if not ws_url:
            raise SystemExit("Chrome debug endpoint never came up.")

        async with websockets.connect(ws_url, max_size=200 * 1024 * 1024) as ws:
            n = 0

            async def send(method, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == n:
                        if "error" in msg:
                            raise RuntimeError(f"{method}: {msg['error']}")
                        return msg.get("result", {})

            await send("Page.enable")
            await send("Page.navigate", url=html_path.resolve().as_uri())
            await asyncio.sleep(2.5)  # let fonts and layout settle before printing
            res = await send(
                "Page.printToPDF",
                printBackground=True,
                paperWidth=8.5,
                paperHeight=11.0,
                marginTop=0.95,
                marginBottom=0.62,
                marginLeft=0.75,
                marginRight=0.75,
                displayHeaderFooter=True,
                headerTemplate=HEADER.replace("__TITLE__", title).replace("__SUB__", subtitle),
                footerTemplate=FOOTER.replace("__FOOT__", footer),
                preferCSSPageSize=False,
            )
            out_pdf.parent.mkdir(parents=True, exist_ok=True)
            out_pdf.write_bytes(base64.b64decode(res["data"]))
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("markdown", help="path to the .md file")
    ap.add_argument("--out", default=None, help="output .pdf (default: same name beside it)")
    ap.add_argument("--title", default=None, help="running header, left")
    ap.add_argument("--subtitle", default="", help="running header, right")
    ap.add_argument("--footer", default="Confidential - pre-filing", help="running footer, left")
    args = ap.parse_args()

    md_path = Path(args.markdown)
    if not md_path.exists():
        raise SystemExit(f"Not found: {md_path}")
    out_pdf = Path(args.out) if args.out else md_path.with_suffix(".pdf")
    title = args.title or md_path.stem

    html = build_html(md_path, title)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tf:
        tf.write(html)
        html_path = Path(tf.name)
    try:
        asyncio.run(render(html_path, out_pdf, title, args.subtitle, args.footer))
        print(f"Wrote {out_pdf}  ({out_pdf.stat().st_size // 1024} KB)")
    finally:
        html_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
