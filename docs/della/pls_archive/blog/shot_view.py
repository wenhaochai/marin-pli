"""Viewport screenshot at an anchor. Usage: shot_view.py SITE OUT.png page#anchor width"""
import functools, http.server, sys, threading
from pathlib import Path
from playwright.sync_api import sync_playwright
site, out, page_q, w = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site)); H.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": w, "height": 900})
    pg.goto(f"http://127.0.0.1:{port}/{page_q}"); pg.wait_for_timeout(3000); pg.screenshot(path=out); b.close()
print("ok")
