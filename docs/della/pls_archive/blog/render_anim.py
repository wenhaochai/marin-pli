"""Render an animation page (render(t), window.T) frame by frame with headless Chromium, then encode the site's three files:
<slug>.mp4 (H.264), <slug>-av1.mp4 (AV1) and <slug>-poster.jpg. Usage:
  render_anim.py page.html OUTDIR SLUG [--preview t1,t2,...] [--fps 30] [--poster T]"""
import argparse, subprocess, shutil
from pathlib import Path
from playwright.sync_api import sync_playwright
ap = argparse.ArgumentParser(); ap.add_argument("page"); ap.add_argument("out"); ap.add_argument("slug")
ap.add_argument("--preview"); ap.add_argument("--fps", type=int, default=30); ap.add_argument("--poster", type=float, default=6.0)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1920, "height": 1080})
    pg.goto(Path(a.page).resolve().as_uri()); pg.wait_for_timeout(500)
    if a.preview:
        for t in [float(x) for x in a.preview.split(",")]:
            pg.evaluate(f"render({t})"); pg.screenshot(path=str(out / f"{a.slug}-t{t:05.2f}.png"))
        b.close(); raise SystemExit
    T = pg.evaluate("window.T"); n = round(T * a.fps)
    fr = out / "frames"; shutil.rmtree(fr, ignore_errors=True); fr.mkdir()
    for i in range(n):
        pg.evaluate(f"render({i / a.fps})"); pg.screenshot(path=str(fr / f"f{i:04d}.png"))
    pg.evaluate(f"render({a.poster})"); pg.screenshot(path=str(out / f"{a.slug}-poster.jpg"), type="jpeg", quality=88)
    b.close()
src = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(fr / "f%04d.png")]
subprocess.run(src + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart", str(out / f"{a.slug}.mp4")], check=True)
subprocess.run(src + ["-c:v", "libsvtav1", "-pix_fmt", "yuv420p", "-crf", "34", "-preset", "6", "-movflags", "+faststart", str(out / f"{a.slug}-av1.mp4")], check=True)
shutil.rmtree(fr)
for f in sorted(out.glob(f"{a.slug}*")): print(f.name, f.stat().st_size)
