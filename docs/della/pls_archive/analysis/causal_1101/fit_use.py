"""Linear fit of separate-heads cost on the stream-difference use (sep checkpoint, mean over layers 1-16, mean and
shuffle modes) over the 10 designs measured before the causal test (one-stream designs at x = 0), and the residual of
each causal-test design. Inputs: tmp/pls_paths/ha[s]-sep-<tag>.json and d48g4.npz. Read-only.
Usage: fit_use.py [tag=cost ...] (extra designs, e.g. the HC shared-write pair once fetched)."""
import json, sys
from pathlib import Path
import numpy as np
O = Path("/scratch/gpfs/GROUP/USER/tmp/pls_paths")
d = dict(np.load("/scratch/gpfs/GROUP/USER/tmp/pls_blog/d48g4.npz", allow_pickle=True))
def cost(t):
    a, b = d.get(f"-sep-{t}/final"), d.get(f"-sep-bbfrozen-{t}/final")
    return None if a is None or b is None else float(a[47] - b[47])
def use(t, p):
    f = O / f"{p}-sep-{t}.json"
    if not f.exists():
        return None
    v = [x for k, x in json.loads(f.read_text())["dce"].items() if int(k) <= 16]
    return sum(v) / len(v)
OLD = ["ahc-vs-d48-g4", "ahc-vs-d48-s1", "ahc-hcrwsig-vs-d48", "ahc-hcs8-vs-d48", "ahc-hcs2-vs-d48", "amhc-vs-d48-g4",
       "amhc-mhcid-vs-d48", "amhc-mhcrwlin-vs-d48", "ahc-hcs1-vs-d48", "apreln-vs-d48-g4"]
NEW = ["amhc-mhcid-mhcrwlin-fa4thd-vs-d48", "ahc-hcwsh-fa4thd-vs-d48"]
for p in ("ha", "has"):
    xs = np.array([0.0 if t in ("ahc-hcs1-vs-d48", "apreln-vs-d48-g4") else use(t, p) for t in OLD])
    ys = np.array([cost(t) for t in OLD])
    k, b = np.polyfit(xs, ys, 1)
    res = ys - (k * xs + b)
    sd = np.sqrt((res ** 2).sum() / (len(xs) - 2))
    print(f"{p}: cost = {b:.4f} {k:+.4f} x   residual sd {sd:.4f} (n {len(xs)})")
    for t in NEW:
        x, y = use(t, p), cost(t)
        if x is None or y is None:
            print(f"   {t}: use {x} cost {y} (missing)"); continue
        print(f"   {t}: use {x:.3f} cost {y:.3f} predicted {k * x + b:.3f} residual {y - k * x - b:+.3f} ({(y - k * x - b) / sd:+.1f} sd)")
