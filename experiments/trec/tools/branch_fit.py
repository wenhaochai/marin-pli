"""Q8 at each branch point: the exponent p for which the fraction left f lines up across the three arms against
x = steps * lr^p. Spread = mean squared gap in log f between each arm and their mean, interpolated in log x over the
common range."""
import sys
import numpy as np
sys.path.insert(0, "/scratch/gpfs/GROUP/USER/project/marin-trec/experiments/trec")
import build_site_data as b
def spread(arms, p):
    cs = [(np.log(np.maximum(a[1:, 0], 1) * lr ** p), np.log(a[1:, 1])) for lr, a in arms]
    lo, hi = max(c[0].min() for c in cs), min(c[0].max() for c in cs)
    g = np.linspace(lo, hi, 50)
    ys = np.array([np.interp(g, x, y) for x, y in cs])
    return float(np.mean((ys - ys.mean(0)) ** 2))
for bp in (1240, 2480, 3720):
    cur = b.branch_curves(bp)
    if cur is None:
        print(bp, "not complete"); continue
    arms = [(lr, a) for _, lr, _, a in cur]
    ps = np.arange(0.5, 2.51, 0.01); sp = np.array([spread(arms, p) for p in ps]); k = int(sp.argmin()); ok = ps[sp <= 2 * sp[k]]
    hl = [float(np.interp(0.5, a[::-1, 1], a[::-1, 0])) if a[:, 1].min() < 0.5 else float("nan") for _, a in arms]
    print(f"{bp}: p {ps[k]:.2f} ({ok.min():.2f}-{ok.max():.2f}); spread p=1 {spread(arms, 1):.1e}, p=2 {spread(arms, 2):.1e}, steps {spread(arms, 0):.1e}, best {sp[k]:.1e};"
          f" left at the end {[round(float(a[-1, 1]), 3) for _, a in arms]}; half-life {[round(h) if h == h else None for h in hl]}")
