"""Record table at phone widths: for 420, 375, 360 px in en and zh, report whether the table needs sideways scrolling
(wrap scrollWidth > clientWidth) or the page does, and screenshot the table. Usage: check_record_table.py SITE OUTDIR PAGE"""
import functools, http.server, sys, threading
from pathlib import Path
from playwright.sync_api import sync_playwright
site, out, page = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]; out.mkdir(exist_ok=True)
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site)); H.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ("en", "zh"):
        for w in (420, 375, 360):
            ctx = b.new_context(viewport={"width": w, "height": 900}, device_scale_factor=2)
            ctx.add_init_script(f"localStorage.setItem('wc-lang', '{lang}')")
            pg = ctx.new_page(); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto(f"http://127.0.0.1:{port}/{page}"); pg.wait_for_timeout(2000)
            m = pg.evaluate("""() => { const t = document.getElementById('rec-table'), wr = t.closest('.rec-wrap');
                const tot = [...t.querySelectorAll('tr.total td')].map(td => [...td.querySelectorAll('span')].filter(s => getComputedStyle(s).display !== 'none').map(s => s.textContent).join(''));
                return {wrap: wr.clientWidth, table: t.scrollWidth, page: document.documentElement.scrollWidth, rows: t.querySelectorAll('tbody tr').length, tot}; }""")
            el = pg.query_selector(".rec-wrap"); el.screenshot(path=str(out / f"record_{lang}_{w}.png"))
            print(lang, w, "fits" if m["table"] <= m["wrap"] and m["page"] <= w else "SCROLLS", m, "errors", errs[:3])
            ctx.close()
    b.close()
