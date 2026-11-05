"""Render frames of the film for a look. Usage: check_film.py OUTDIR en|zh t1 t2 ...; run on a vis node with the marin-pls-speed venv (playwright)."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
out = Path(sys.argv[1]); lang = sys.argv[2]; ts = [float(x) for x in sys.argv[3:]]; out.mkdir(exist_ok=True)
JS = Path(__file__).with_name("residual-paths-film.js").read_text()
FONTS = "/scratch/gpfs/GROUP/USER/tmp/plc_site_wt/assets/fonts"
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1280, "height": 720})
    pg.set_content(f'<!doctype html><html data-lang="{lang}"><head><style>body{{margin:0}}svg{{width:1280px;height:720px;display:block}}</style></head><body><svg id="s" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1920 1080"></svg></body></html>')
    pg.add_script_tag(content=JS)
    print("T", pg.evaluate("RPFilm.T"))
    for t in ts:
        pg.evaluate(f"document.getElementById('s').innerHTML = RPFilm.frame({t})")
        pg.screenshot(path=str(out / f"f_{lang}_{t:05.1f}.png"))
    b.close()
