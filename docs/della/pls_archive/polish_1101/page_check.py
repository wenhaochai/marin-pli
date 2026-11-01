"""Render exps/per-layer-contribution.html of a site worktree in headless Chromium and check the polish pass:
page errors, charts drawn, horizontal overflow at 400 px, and the number of lines of every figure's short caption at
1280 px in English and in Chinese (owner rule: a short caption fits in two lines at desktop width).
Usage: page_check.py SITE [SHOT_DIR fig-id ...]  (screenshots of the named figures, both languages, 1280 px)
Run on a vis node with the marin-pls-speed venv (playwright)."""
import functools, http.server, sys, threading
from pathlib import Path
from playwright.sync_api import sync_playwright
site = Path(sys.argv[1]); shots = Path(sys.argv[2]) if len(sys.argv) > 2 else None; figs = sys.argv[3:]
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
H = functools.partial(Q, directory=str(site))
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
URL = f"http://127.0.0.1:{port}/exps/per-layer-contribution.html"
LINES = """() => [...document.querySelectorAll('.ar-main figure.viz')].map(f => {
  const c = f.querySelector('figcaption'); if (!c) return [f.id, -1, ''];
  const cs = getComputedStyle(c), lh = parseFloat(cs.lineHeight);
  const h = c.getBoundingClientRect().height - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
  return [f.id, Math.round(h / lh * 10) / 10, c.innerText.slice(0, 60)]; })"""
with sync_playwright() as p:
    b = p.chromium.launch()
    for width in (1280, 400):
        pg = b.new_page(viewport={"width": width, "height": 900}, device_scale_factor=2)
        errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(URL); pg.wait_for_timeout(3500)
        n = pg.evaluate("document.querySelectorAll('.ar-main figure.viz svg').length")
        sw = pg.evaluate("document.documentElement.scrollWidth")
        print(f"{width}px: svgs in figures {n}, scrollWidth {sw}, errors {errs[:3]}")
        if width == 1280:
            for lang in ("en", "zh"):
                pg.click(f".ar-lang button[data-lang='{lang}']"); pg.wait_for_timeout(1200)
                rows = pg.evaluate(LINES)
                over = [(i, l, t) for i, l, t in rows if l > 2.2]
                print(f"  {lang}: {len(rows)} captions, lines max {max(l for _, l, _ in rows)}, over two lines: {len(over)}")
                for i, l, t in over: print(f"    {i} {l} {t}")
                if shots:
                    shots.mkdir(exist_ok=True)
                    for fid in figs:
                        el = pg.locator(f"figure#{fid}").first; el.scroll_into_view_if_needed(); pg.wait_for_timeout(700)
                        el.screenshot(path=str(shots / f"{fid}_{lang}.png"))
        pg.close()
    b.close()
