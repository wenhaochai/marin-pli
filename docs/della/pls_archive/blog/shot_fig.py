"""Screenshot one figure of a page: shot_fig.py SITE OUT.png PAGE FIG_ID WIDTH LANG"""
import functools, http.server, sys, threading
from pathlib import Path
from playwright.sync_api import sync_playwright
site, out, page, fid, w, lang = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site)); H.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch(); ctx = b.new_context(viewport={"width": w, "height": 900}, device_scale_factor=2)
    ctx.add_init_script(f"localStorage.setItem('wc-lang', '{lang}')")
    pg = ctx.new_page(); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(f"http://127.0.0.1:{port}/{page}"); pg.wait_for_timeout(2500)
    pg.evaluate("document.querySelectorAll('header, .skip-link, [class*=skip]').forEach(e => e.style.display='none')")
    pg.query_selector(f"#{fid}").screenshot(path=out); print("errors", errs)
    b.close()
