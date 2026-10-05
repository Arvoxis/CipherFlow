"""Capture screenshots of the running Streamlit demo for the invention disclosure form.

Drives headless Chrome over the DevTools Protocol, so nothing extra has to be installed:
websockets and requests are already in the `ml` env, and Chrome ships with Windows.
Captures at deviceScaleFactor=2 so the images stay sharp when Word scales them to page width.

The demo must already be running:
    python -m streamlit run cipherflow/serve/app.py --server.port 8531

Then:
    python scripts/capture_demo_shots.py --port 8531
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

import requests
import websockets

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts" / "screenshots"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

# (filename, tab label to click or None for the default tab, extra JS, settle seconds)
CLICK_START = """
(() => {
  const b = [...document.querySelectorAll('button')]
    .find(e => (e.innerText || '').trim() === 'Start');
  if (b) { b.click(); return true; }
  return false;
})()
"""

SHOTS = [
    # Press Start and let it stream, so the capture shows live predictions and a running
    # accuracy rather than an idle "press Start" panel, which proves nothing.
    # Keep the sidebar on the first shot only; it establishes which model and dataset are
    # loaded, and after that it just squeezes the content.
    ("01_live_classification", "Live classification", CLICK_START, 11, False),
    ("02_tokenization_claimA", "Tokenization (Claim A)", None, 4, True),
    ("03_embedding_map_claimB", "Embedding map (Claim B)", None, 9, True),
    ("04_evasion_claimC", "Evasion (Claim C)", None, 14, True),
    ("05_provenance", "Provenance", None, 4, True),
]

CLICK_TAB = """
(() => {
  const t = [...document.querySelectorAll('button[role="tab"],[role="tab"]')]
    .find(e => (e.innerText || '').includes(%s));
  if (t) { t.click(); return true; }
  return false;
})()
"""

# Streamlit renders a fixed toolbar and a "Deploy" button that mean nothing in a disclosure
# document. Hide them so the captures show only the application.
HIDE_CHROME = """
(() => {
  const css = document.createElement('style');
  css.textContent = `
    [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stStatusWidget"], #MainMenu, header {display:none !important;}
    [data-testid="stAppViewContainer"] {padding-top:0 !important;}
  `;
  document.head.appendChild(css);
  return true;
})()
"""


MEASURE_HEIGHT = """
(() => {
  let h = document.body.scrollHeight;
  for (const el of document.querySelectorAll('section, div[data-testid]')) {
    if (el.scrollHeight > h && el.scrollHeight < 12000) h = el.scrollHeight;
  }
  return h;
})()
"""


def crop_sidebar(path: Path) -> None:
    """Drop the left navigation column.

    The sidebar is worth showing once, to establish what model and dataset are loaded. On the
    content tabs it only squeezes the part that matters, which is what makes the embedded
    screenshot hard to read once Word scales it to the text width.
    """
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    px = im.load()
    y = min(400, h // 3)  # near the top, where the sidebar is present on every tab
    left = px[4, y]
    edge = 0
    for x in range(4, w // 2):  # first column that stops matching the sidebar background
        if px[x, y] != left:
            edge = x
            break
    if 100 < edge < w // 2:
        im.crop((edge, 0, w, h)).save(path)


def trim_bottom(path: Path, pad: int = 28) -> None:
    """Crop the empty background below the last real content.

    Streamlit reserves a tall run of blank page under short tabs, and that blank space is what
    makes an embedded screenshot shrink to illegibility once Word scales it to the text width.
    """
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    bg = im.getpixel((w - 3, h - 3))  # page background, sampled from the bottom-right corner
    px = im.load()
    last = 0
    for y in range(h - 1, -1, -1):
        step = max(1, w // 260)  # sampling the row is plenty and keeps this fast
        if any(px[x, y] != bg for x in range(w // 3, w, step)):
            last = y
            break
    if last and last + pad < h:
        im.crop((0, 0, w, min(h, last + pad))).save(path)


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    found = shutil.which("chrome") or shutil.which("msedge")
    if found:
        return found
    raise SystemExit("No Chrome or Edge found. Edit CHROME_CANDIDATES.")


class CDP:
    """Minimal DevTools Protocol client: enough to navigate, evaluate and screenshot."""

    def __init__(self, ws):
        self.ws = ws
        self.n = 0

    async def send(self, method: str, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})


async def capture(port: int, width: int, height: int, scale: int):
    chrome = find_chrome()
    profile = tempfile.mkdtemp(prefix="cf_shots_")
    proc = subprocess.Popen(
        [
            chrome,
            "--headless=new",
            "--remote-debugging-port=9333",
            f"--user-data-dir={profile}",
            f"--window-size={width},{height}",
            "--hide-scrollbars",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        ws_url = None
        for _ in range(40):  # Chrome needs a moment before the debug endpoint answers.
            try:
                tabs = requests.get("http://127.0.0.1:9333/json", timeout=1).json()
                page = next(t for t in tabs if t["type"] == "page")
                ws_url = page["webSocketDebuggerUrl"]
                break
            except Exception:
                time.sleep(0.5)
        if not ws_url:
            raise SystemExit("Chrome debug endpoint never came up.")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        async with websockets.connect(ws_url, max_size=120 * 1024 * 1024) as ws:
            cdp = CDP(ws)
            await cdp.send("Page.enable")
            await cdp.send("Runtime.enable")
            await cdp.send(
                "Emulation.setDeviceMetricsOverride",
                width=width,
                height=height,
                deviceScaleFactor=scale,
                mobile=False,
            )
            await cdp.send(
                "Emulation.setEmulatedMedia",
                media="screen",
                features=[{"name": "prefers-color-scheme", "value": "dark"}],
            )
            await cdp.send("Page.navigate", url=f"http://localhost:{port}")
            await asyncio.sleep(22)  # Streamlit's first paint is slow.
            await cdp.send("Runtime.evaluate", expression=HIDE_CHROME)

            written = []
            for name, tab, extra, settle, nosidebar in SHOTS:
                if tab:
                    r = await cdp.send("Runtime.evaluate", expression=CLICK_TAB % json.dumps(tab))
                    if not r.get("result", {}).get("value"):
                        print(f"  !! tab not found: {tab}")
                if extra:
                    await cdp.send("Runtime.evaluate", expression=extra)
                await asyncio.sleep(settle)
                # Grow the viewport to the rendered height so nothing is cut off at the fold.
                # captureBeyondViewport on its own leaves Streamlit's sticky header floating
                # over the middle of the image.
                # Streamlit scrolls an inner container, so Page.getLayoutMetrics understates the
                # real height. Ask the DOM for the tallest scrollHeight on the page instead.
                r = await cdp.send("Runtime.evaluate", expression=MEASURE_HEIGHT)
                full_h = min(int(r["result"]["value"]) + 32, 3600)
                await cdp.send(
                    "Emulation.setDeviceMetricsOverride",
                    width=width,
                    height=full_h,
                    deviceScaleFactor=scale,
                    mobile=False,
                )
                await asyncio.sleep(1.5)
                shot = await cdp.send("Page.captureScreenshot", format="png", captureBeyondViewport=False)
                await cdp.send(
                    "Emulation.setDeviceMetricsOverride",
                    width=width,
                    height=height,
                    deviceScaleFactor=scale,
                    mobile=False,
                )
                path = OUT_DIR / f"{name}.png"
                path.write_bytes(base64.b64decode(shot["data"]))
                if nosidebar:
                    crop_sidebar(path)
                trim_bottom(path)
                written.append(path)
                print(f"  [x] {path.name}  ({path.stat().st_size // 1024} KB)")
            return written
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8531, help="port the Streamlit demo is on")
    ap.add_argument("--width", type=int, default=1340)
    ap.add_argument("--height", type=int, default=840)
    ap.add_argument("--scale", type=int, default=2, help="deviceScaleFactor; 2 = retina")
    args = ap.parse_args()
    print(f"Capturing demo at localhost:{args.port} -> {OUT_DIR}")
    shots = asyncio.run(capture(args.port, args.width, args.height, args.scale))
    print(f"\n{len(shots)} screenshots written.")


if __name__ == "__main__":
    main()
