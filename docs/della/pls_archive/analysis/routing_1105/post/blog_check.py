"""Render blogs/residual-paths-not-routing.html of a site worktree in headless Chromium: page errors, charts drawn, overflow at 400 px, screenshots of every figure and of the whole page in both languages. Usage: blog_check.py SITE OUTDIR. Run on a vis node with the marin-pls-speed venv."""
import functools, http.server, sys, threading
from pathlib import Path
from playwright.sync_api import sync_playwright
site, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Q, directory=str(site)))
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_address[1]}/blogs/residual-paths-not-routing.html"
with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ("en", "zh"):
        pg = b.new_page(viewport={"width": 1280, "height": 900}); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e))); pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto(url); pg.wait_for_timeout(2500)
        if lang == "zh":
            pg.click('.ar-lang button[data-lang="zh"]'); pg.wait_for_timeout(1200)
        n = pg.evaluate("document.querySelectorAll('figure.viz svg').length"); nf = pg.evaluate("document.querySelectorAll('figure.viz').length")
        print(lang, "charts drawn", n, "of", nf, "errors", errs[:5], "scrollWidth", pg.evaluate("document.documentElement.scrollWidth"))
        for fid in ("overview", "rr-removal", "rr-depth", "rr-scale", "rr-cost", "rr-paths"):
            el = pg.query_selector(f"#{fid}")
            if el: el.scroll_into_view_if_needed(); pg.wait_for_timeout(400); el.screenshot(path=str(out / f"{fid}_{lang}.png"))
        el = pg.query_selector("figure.rr-schema"); el.scroll_into_view_if_needed(); el.screenshot(path=str(out / f"schema_{lang}.png"))
        pg.screenshot(path=str(out / f"full_{lang}.png"), full_page=True)
        pg.close()
    pg = b.new_page(viewport={"width": 400, "height": 900}); pg.goto(url); pg.wait_for_timeout(2000)
    print("400px scrollWidth", pg.evaluate("document.documentElement.scrollWidth"))
    b.close()
