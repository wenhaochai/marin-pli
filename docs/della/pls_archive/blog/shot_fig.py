"""Screenshot one figure element. Usage: shot_fig.py SITE OUT.png page fig-id"""
import functools, http.server, sys, threading
from pathlib import Path
from playwright.sync_api import sync_playwright
site, out, page_q, fid = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site)); H.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(f"http://127.0.0.1:{port}/{page_q}"); pg.wait_for_timeout(3000)
    el = pg.locator(f"figure:has([data-fig='{fid}']), figure:has(#{fid}), figure#{fid}").first
    el.scroll_into_view_if_needed(); pg.wait_for_timeout(800); el.screenshot(path=out); print("ok errors:", errs[:3])
    b.close()
