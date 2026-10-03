"""Screenshot website pages with headless Chromium (serves the site on 127.0.0.1 inside this process). Usage: shot.py SITE OUTDIR page[?query] ..."""
import functools, http.server, sys, threading
from pathlib import Path
from playwright.sync_api import sync_playwright
site, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(exist_ok=True)
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site))
H.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch()
    for page_q in sys.argv[3:]:
        for w in (1280, 420):
            pg = b.new_page(viewport={"width": w, "height": 900}, device_scale_factor=1)
            errs = []; pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None); pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto(f"http://127.0.0.1:{port}/{page_q}"); pg.wait_for_timeout(2500)
            name = page_q.replace("/", "_").replace("?", "_").replace("=", "-").replace("&", "_") + f"_{w}.png"
            pg.screenshot(path=str(out / name), full_page=True)
            print(name, "errors:", errs[:5])
    b.close()
